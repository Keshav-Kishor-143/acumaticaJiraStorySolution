#!/usr/bin/env python3
"""
Enhanced Hybrid Retrieval System

Combines multiple search strategies for optimal document retrieval:
1. Vector similarity search (using pre-computed vectors)
2. Keyword-based search
3. Domain-specific search
4. Title/section matching
5. Query expansion

No PDF processing dependencies - uses pre-computed vectors and metadata.
"""

import os
import json
import re
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Add src directory to Python path for imports when run directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent  # Go up from core/ to src/
    sys.path.insert(0, str(src_dir))

from openai import OpenAI
# Using lightweight similarity search instead of Pinecone
from collections import Counter, defaultdict
import math
from sklearn.metrics.pairwise import cosine_similarity  # Lightweight similarity search

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.utils.local_manager import LocalManager

def _get_huggingface_embedding(model_name: str, device: str = "cpu", logger=None):
    """Get HuggingFace embedding model with fallback support"""
    # Try multiple import paths for compatibility
    try:
        # Try old llama-index path (v0.9.x)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        if logger:
            logger.debug("Using llama-index HuggingFaceEmbedding (old path)")
        return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
    except ImportError:
        try:
            # Try new llama-index path (v0.10+)
            from llama_index.embeddings import HuggingFaceEmbedding
            if logger:
                logger.debug("Using llama-index HuggingFaceEmbedding (new path)")
            return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
        except ImportError:
            # Use sentence-transformers directly (preferred method)
            try:
                from sentence_transformers import SentenceTransformer
                if logger:
                    logger.info("Using sentence-transformers directly (HuggingFace embeddings)")
                
                # Create a wrapper class to match the interface
                class EmbeddingWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def get_text_embedding(self, text: str):
                        return self.model.encode(text, normalize_embeddings=True).tolist()
                
                return EmbeddingWrapper(SentenceTransformer(model_name, device=device))
            except ImportError as e:
                raise ImportError(
                    f"Failed to import HuggingFace embeddings. "
                    f"Please install: pip install sentence-transformers torch"
                ) from e

@dataclass
class SearchResult:
    """Enhanced search result with multiple relevance signals"""
    pdf_name: str
    page_number: int
    score: float
    text_content: str
    metadata: Dict[str, Any]
    relevance_signals: Dict[str, float]  # Multiple relevance scores
    search_strategy: str  # Which strategy found this result
    combined_score: float  # Final computed score
    content: str = ""
    image_path: str = ""
    vision_extracted_text: str = ""

    def get_image_path(self) -> str:
        """Get standardized image path"""
        from pathlib import Path
        from src.config.config import config
        return str(Path(config.LOCAL_BASE_PATH) / self.pdf_name / "images" / f"page{self.page_number}.jpg")

class HybridRetriever:
    """Advanced multi-strategy retrieval system for maximum accuracy"""
    
    def __init__(self):
        """Advanced multi-strategy retrieval system for maximum accuracy"""
        self.logger = get_logger("HYBRID_RETRIEVER")
        self.storage_mode = "local"  # GCS disabled
        
        # Using local storage only
        self.gcs_manager = None
        self.logger.info("Using local storage for retrieval")
        
        # Initialize caches
        self.vectors_cache = {}
        self.metadata_cache = {}
        self.content_index = {}
        self.text_index = {}
        self.keyword_index = defaultdict(list)
        
        # Initialize components
        self._initialize_clients()
        self._load_content_index()
        
        # Base search strategy weights (will be dynamically adjusted)
        self.base_strategy_weights = {
            'title_section_match': 0.35,  # Higher weight for finding specific sections
            'domain_specific': 0.25,      # Higher weight for Acumatica-specific terms
            'enhanced_keyword': 0.20,     # Exact keyword matching
            'vector_semantic': 0.15,      # Reduced weight for semantic similarity
            'query_expanded': 0.05        # Minimal weight for query expansion
        }
        
        # Process-specific weights (for procedure/workflow queries)
        self.process_strategy_weights = {
            'title_section_match': 0.40,  # Highest weight for finding specific procedures
            'domain_specific': 0.30,      # High weight for Acumatica domain terms
            'enhanced_keyword': 0.20,     # Exact term matching
            'vector_semantic': 0.07,      # Minimal semantic matching
            'query_expanded': 0.03        # Very low expansion weight
        }
        
        # Form-specific weights (for screen/form documentation)
        self.form_strategy_weights = {
            'domain_specific': 0.40,      # Highest weight for form IDs and fields
            'title_section_match': 0.30,  # High weight for form titles
            'enhanced_keyword': 0.20,     # Exact field matching
            'vector_semantic': 0.07,      # Low semantic matching
            'query_expanded': 0.03        # Very low expansion
        }
        
        self.strategy_weights = self.base_strategy_weights.copy()
        
        # Performance tracking
        self.search_analytics = {
            'total_searches': 0,
            'query_types': {},
            'strategy_performance': {},
            'accuracy_feedback': []
        }
        
        self.logger.info("Hybrid Multi-Strategy Retriever initialized", extra={
            "strategies": len(self.strategy_weights)
        })
    
    def _initialize_clients(self):
        """Initialize HuggingFace embeddings and OpenAI client"""
        try:
            # Validate OpenAI API key
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in environment variables")
            if not config.OPENAI_API_KEY.startswith("sk-"):
                raise ValueError("Invalid OpenAI API key format")
                
            # Initialize OpenAI client for LLM and Vision
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            # Initialize HuggingFace embedding model (FREE!)
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize embeddings with fallback support
            self.embedding_model = _get_huggingface_embedding(
                config.EMBEDDING_MODEL, 
                device=device, 
                logger=self.logger
            )
            
            # Initialize vectors and metadata cache for pre-computed data
            self.vectors_cache = {}
            self.metadata_cache = {}
            
            self.logger.info("HuggingFace + OpenAI clients initialized successfully", extra={
                "embedding_model": config.EMBEDDING_MODEL,
                "llm_model": config.LLM_MODEL,
                "approach": "pre_computed_vectors"
            })
        except Exception as e:
            self.logger.error("Failed to initialize clients", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def _load_content_index(self):
        """Load and build enhanced content index from local storage or GCS metadata"""
        try:
            self.content_index = {}
            self.text_index = {}
            self.keyword_index = defaultdict(list)
            
            if self.storage_mode == "gcs" and self.gcs_manager:
                # Load all metadata from GCS
                all_metadata = self.gcs_manager.list_all_metadata()
                for pdf_stem, metadata in all_metadata.items():
                    self._index_metadata_content(metadata)
                    self.logger.debug("Loaded metadata from GCS", extra={
                        "pdf_stem": pdf_stem
                    })
                
                # Also get PDF list from GCS for completeness
                pdf_list = self.gcs_manager.list_pdfs_in_gcs()
                for pdf_info in pdf_list:
                    pdf_stem = pdf_info["stem"]
                    if pdf_stem not in self.content_index:
                        self.content_index[pdf_stem] = {
                            "pdf_name": pdf_info["name"],
                            "size_mb": pdf_info["size_mb"],
                            "source": "gcs"
                        }
                
                self.logger.info("GCS content index built", extra={
                    "pdfs": len(self.content_index),
                    "pages": len(self.text_index),
                    "keywords": len(self.keyword_index)
                })
            else:
                # Local storage mode
                from pathlib import Path
                from src.config.config import config
                
                # Get list of PDF directories
                pdf_dirs = [d for d in Path(config.LOCAL_BASE_PATH).iterdir() if d.is_dir()]
                
                for pdf_dir in pdf_dirs:
                    pdf_name = pdf_dir.name
                    
                    # Load metadata if available
                    metadata_file = pdf_dir / "metadata" / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                # Add document name to metadata for indexing
                                if isinstance(metadata, dict):
                                    metadata['document_name'] = pdf_name
                                elif isinstance(metadata, list):
                                    for entry in metadata:
                                        if isinstance(entry, dict):
                                            entry['document_name'] = pdf_name
                                self._index_metadata_content(metadata)
                                # Store in metadata cache
                                self.metadata_cache[pdf_name] = metadata
                                self.logger.debug("Metadata loaded", extra={
                                    "pdf_name": pdf_name,
                                    "metadata_entries": len(metadata) if isinstance(metadata, list) else 1
                                })
                        except Exception as e:
                            self.logger.error("Failed to load metadata", extra={
                                "pdf_name": pdf_name,
                                "error_message": str(e),
                                "error_type": type(e).__name__
                            })
                    
                    # Load vectors if available
                    vectors_file = pdf_dir / "vectors" / "vectors.json"
                    if vectors_file.exists():
                        try:
                            self.logger.debug("Loading vectors", extra={
                                "pdf_name": pdf_name,
                                "vectors_file": str(vectors_file)
                            })
                            with open(vectors_file, 'r', encoding='utf-8') as f:
                                vectors_data = json.load(f)
                                self.vectors_cache[pdf_name] = np.array(vectors_data.get('embeddings', []))
                                self.logger.debug("Vectors loaded", extra={
                                    "pdf_name": pdf_name,
                                    "vector_count": len(self.vectors_cache[pdf_name])
                                })
                        except Exception as e:
                            self.logger.error("Failed to load vectors", extra={
                                "pdf_name": pdf_name,
                                "error_message": str(e),
                                "error_type": type(e).__name__
                            })
                    
                    # Add to content index
                    self.content_index[pdf_name] = {
                        "pdf_name": pdf_name,
                        "source": "local",
                        "has_metadata": metadata_file.exists(),
                        "has_vectors": vectors_file.exists()
                    }
                
                self.logger.info("Local content index built", extra={
                    "pdfs": len(self.content_index),
                    "pages": len(self.text_index),
                    "keywords": len(self.keyword_index),
                    "pdfs_with_vectors": len(self.vectors_cache)
                })
            
        except Exception as e:
            self.logger.error("Content index build failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            self.content_index = {}
            self.text_index = {}
            self.keyword_index = defaultdict(list)
            raise
    
    def _index_pdf_content(self, pdf_path: Path, pdf_name: str):
        """Extract and index text content directly from PDF"""
        try:
            # This function is no longer needed as we rely on pre-computed vectors.
            # Keeping it for now in case it's called elsewhere, but it will be removed.
            pass
            
        except Exception as e:
            self.logger.warning("PDF indexing failed", extra={"pdf": pdf_name, "error": str(e)})
    
    def _extract_structured_text(self, blocks_dict: Dict) -> Dict[str, Any]:
        """Extract structured text from PDF blocks"""
        titles = []
        content_parts = []
        tables = []
        
        for block in blocks_dict.get('blocks', []):
            if 'lines' in block:
                for line in block['lines']:
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        if not text:
                            continue
                        
                        font_size = span.get('size', 12)
                        font_flags = span.get('flags', 0)
                        
                        # Detect titles (larger font, bold)
                        if font_size > 14 or (font_flags & 16):  # Bold flag
                            titles.append(text)
                        else:
                            content_parts.append(text)
        
        return {
            'titles': titles,
            'content': ' '.join(content_parts),
            'tables': tables
        }
    
    def _build_keyword_index(self, page_id: str, text: str, structured_text: Dict):
        """Build keyword index for fast lookup"""
        # Extract keywords from different text sources
        all_text = text + ' ' + ' '.join(structured_text.get('titles', []))
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Add to keyword index
        for word in set(words):
            self.keyword_index[word].append(page_id)
        
        # Add domain-specific keywords
        domain_keywords = self._extract_domain_keywords(all_text)
        for keyword in domain_keywords:
            self.keyword_index[keyword].append(page_id)
    
    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract domain-specific keywords (Acumatica, business processes)"""
        domain_terms = []
        
        # Business process terms
        process_patterns = [
            r'\b(return|replacement|refund|exchange)\b',
            r'\b(sales order|purchase order|invoice)\b',
            r'\b(customer|vendor|supplier)\b',
            r'\b(workflow|process|procedure|step)\b',
            r'\b(configuration|setup|settings)\b',
            r'\b(acumatica|erp|system)\b',
        ]
        
        text_lower = text.lower()
        for pattern in process_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            domain_terms.extend(matches)
        
        return list(set(domain_terms))
    
    def _index_metadata_content(self, metadata: Dict):
        """Index content from metadata"""
        try:
            # Handle both single document and multi-document metadata
            if isinstance(metadata, dict):
                # Single document metadata
                for page_data in metadata.get('pages', []):
                    page_num = page_data.get('page_number', 0)
                    page_id = f"{metadata.get('document_name', 'unknown')}_page_{page_num}"
                    
                    if page_id not in self.text_index:
                        self.text_index[page_id] = {
                            'pdf_name': metadata.get('document_name', 'unknown'),
                            'page_number': page_num,
                            'raw_text': page_data.get('text', ''),
                            'structured_text': {'titles': [], 'content': page_data.get('text', '')}
                        }
            elif isinstance(metadata, list):
                # List of metadata entries
                for entry in metadata:
                    if isinstance(entry, dict):
                        page_num = entry.get('page_number', 0)
                        doc_name = entry.get('document_name', 'unknown')
                        page_id = f"{doc_name}_page_{page_num}"
                        
                        if page_id not in self.text_index:
                            self.text_index[page_id] = {
                                'pdf_name': doc_name,
                                'page_number': page_num,
                                'raw_text': entry.get('text', ''),
                                'structured_text': {'titles': [], 'content': entry.get('text', '')}
                            }
            
            self.logger.debug("Indexed metadata content", extra={
                "pages_indexed": len(self.text_index)
            })
            
        except Exception as e:
            self.logger.error("Failed to index metadata content", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
    
    def _load_precomputed_vectors_sync(self, document_id: str) -> bool:
        """Load pre-computed vectors from local storage"""
        try:
            if document_id in self.vectors_cache:
                return True  # Already loaded
                
            self.logger.info("Loading pre-computed vectors", extra={
                "document_id": document_id
            })
            
            # Load vectors from local storage
            from pathlib import Path
            vectors_file = Path(config.LOCAL_BASE_PATH) / document_id / "vectors" / "vectors.json"
            metadata_file = Path(config.LOCAL_BASE_PATH) / document_id / "metadata" / "metadata.json"
            
            # Load vectors
            if vectors_file.exists():
                with open(vectors_file, 'r', encoding='utf-8') as f:
                    vectors_data = json.load(f)
                    self.vectors_cache[document_id] = np.array(vectors_data.get('embeddings', []))
                self.logger.info("Loaded vectors", extra={
                    "document_id": document_id,
                    "vectors_count": len(self.vectors_cache[document_id])
                })
            else:
                self.logger.error("Vectors file not found", extra={
                    "document_id": document_id,
                    "path": str(vectors_file)
                })
                return False
            
            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_cache[document_id] = json.load(f)
                self.logger.info("Loaded metadata", extra={
                    "document_id": document_id,
                    "sections": len(self.metadata_cache[document_id])
                })
            else:
                self.logger.warning("Metadata file not found", extra={
                    "document_id": document_id,
                    "path": str(metadata_file)
                })
                self.metadata_cache[document_id] = []
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to load pre-computed vectors", extra={
                "document_id": document_id,
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def _get_query_embedding_sync(self, query: str) -> np.ndarray:
        """Get embedding for query using HuggingFace (keep existing approach)"""
        try:
            # Use HuggingFace embeddings (FREE!)
            if hasattr(self, 'embedding_model') and self.embedding_model:
                embedding = self.embedding_model.get_text_embedding(query)
            else:
                # Fallback: initialize HuggingFace embedding model
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.embedding_model = _get_huggingface_embedding(
                    config.EMBEDDING_MODEL,
                    device=device,
                    logger=self.logger
                )
                embedding = self.embedding_model.get_text_embedding(query)
            
            # Truncate to match expected dimension
            if len(embedding) > config.EMBEDDING_DIMENSION:
                embedding = embedding[:config.EMBEDDING_DIMENSION]
            
            embedding_array = np.array(embedding)
            self.logger.debug(f"Generated HuggingFace query embedding with dimension: {embedding_array.shape}")
            return embedding_array
            
        except Exception as e:
            self.logger.error("Failed to get query embedding", extra={"error": str(e)})
            raise
    
    def _similarity_search_sync(self, document_id: str, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Perform similarity search using lightweight cosine similarity"""
        try:
            # Get document vectors
            document_vectors = self.vectors_cache[document_id]
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                document_vectors
            )[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            metadata_list = self.metadata_cache.get(document_id, [])
            
            for idx in top_indices:
                # Find metadata entry with matching vector_index
                matching_metadata = None
                for page_metadata in metadata_list:
                    for chunk in page_metadata.get('chunks', []):
                        if chunk.get('vector_index') == idx:
                            matching_metadata = {
                                'page_number': page_metadata.get('page_number', 0),
                                'image_path': page_metadata.get('image_path', ''),
                                'coordinates': chunk.get('coordinates', {}),
                                'document_name': document_id
                            }
                            break
                    if matching_metadata:
                        break
                
                if matching_metadata:
                    result = SearchResult(
                        pdf_name=document_id,
                        page_number=matching_metadata['page_number'],
                        score=float(similarities[idx]),
                        text_content='',  # Will be filled by Vision API
                        metadata=matching_metadata,
                        relevance_signals={'vector_similarity': float(similarities[idx])},
                        search_strategy='vector_semantic',
                        combined_score=float(similarities[idx]),
                        content='',  # Will be filled by Vision API
                        image_path=matching_metadata['image_path']
                    )
                    results.append(result)
            
            self.logger.debug("Found similar sections", extra={
                "document_id": document_id,
                "results_count": len(results),
                "top_scores": [float(similarities[idx]) for idx in top_indices[:3]] if len(top_indices) >= 3 else []
            })
            return results
            
        except Exception as e:
            self.logger.error("Similarity search failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__,
                "document_id": document_id
            })
            return []

    def _create_context_aware_vision_prompt(self, result: SearchResult, query: str) -> str:
        """Create context-aware vision prompt based on query and result metadata"""
        query_lower = query.lower()
        
        # Detect query type for better extraction
        if any(keyword in query_lower for keyword in ['how to', 'steps', 'procedure', 'process', 'configure', 'setup']):
            prompt_type = "procedure"
        elif any(keyword in query_lower for keyword in ['error', 'problem', 'issue', 'fix', 'troubleshoot']):
            prompt_type = "troubleshooting"
        elif any(keyword in query_lower for keyword in ['form', 'field', 'screen', 'menu', 'option']):
            prompt_type = "configuration"
        else:
            prompt_type = "general"
        
        base_prompt = f"""Extract all text from this documentation page image.

Context:
- Document: {result.pdf_name}
- Page: {result.page_number}
- Query: {query}

Extraction Guidelines:
1. Extract ALL visible text accurately, preserving:
   - Technical terms and form names
   - Step-by-step instructions
   - Field names, menu paths, and configuration options
   - Code snippets and examples
   - Tables and structured data

2. Preserve formatting:
   - Maintain line breaks and indentation
   - Keep numbered/bulleted lists intact
   - Preserve table structures

3. Focus on:
"""
        
        if prompt_type == "procedure":
            base_prompt += """   - Step-by-step instructions
   - Prerequisites and requirements
   - Sequential actions and decision points
   - Verification steps"""
        elif prompt_type == "troubleshooting":
            base_prompt += """   - Error messages and descriptions
   - Problem symptoms
   - Resolution steps
   - Workarounds and solutions"""
        elif prompt_type == "configuration":
            base_prompt += """   - Form names and screen locations
   - Field names and values
   - Menu navigation paths
   - Configuration options and settings"""
        else:
            base_prompt += """   - All relevant information
   - Key concepts and definitions
   - Related procedures and references"""
        
        base_prompt += "\n\nReturn ONLY the extracted text without any commentary or interpretation."
        
        return base_prompt
    
    def _extract_from_metadata_fallback(self, result: SearchResult, query: str) -> str:
        """Fallback to metadata-based text extraction when vision fails"""
        try:
            page_id = f"{result.pdf_name}_page_{result.page_number}"
            
            # Try to get text from text_index
            if page_id in self.text_index:
                text_data = self.text_index[page_id]
                extracted_text = text_data.get('text', '') or text_data.get('content', '') or text_data.get('raw_text', '')
                
                if extracted_text:
                    self.logger.info("Using metadata fallback for text extraction", extra={
                        "pdf_name": result.pdf_name,
                        "page_number": result.page_number,
                        "text_length": len(extracted_text)
                    })
                    return extracted_text
            
            # Try to get from metadata file
            metadata_path = Path(config.LOCAL_BASE_PATH) / result.pdf_name / "metadata" / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    for page_data in metadata:
                        if page_data.get('page_number') == result.page_number:
                            # Try to extract from chunks or other metadata fields
                            chunks = page_data.get('chunks', [])
                            if chunks:
                                # Return a summary indicating metadata was used
                                return f"[Metadata-based content from {result.pdf_name}, page {result.page_number}]"
            
            return ""
        except Exception as e:
            self.logger.warning("Metadata fallback extraction failed", extra={
                "error": str(e),
                "pdf_name": result.pdf_name,
                "page_number": result.page_number
            })
            return ""
    
    def _enhance_with_vision_sync(self, search_results: List[SearchResult], query: str = "") -> List[SearchResult]:
        """Enhance results with OpenAI Vision API text extraction with retries and metadata fallback"""
        MAX_RETRIES = 3
        RETRY_DELAY = 1  # seconds
        
        import time
        import base64
        from pathlib import Path
        
        for result in search_results:
            if not result.image_path:
                self.logger.debug("Skipping result with no image path", extra={
                    "pdf_name": result.pdf_name,
                    "page_number": result.page_number
                })
                continue
                
            # Use standardized image path
            try:
                image_path = result.get_image_path()
                self.logger.debug("Attempting to access image", extra={
                    "pdf_name": result.pdf_name,
                    "page_number": result.page_number,
                    "full_path": str(image_path),
                    "base_path": str(config.LOCAL_BASE_PATH),
                    "exists": Path(image_path).exists()
                })
                
                if not Path(image_path).exists():
                    # Check if directory exists
                    image_dir = Path(image_path).parent
                    self.logger.warning("Image file not found", extra={
                        "pdf_name": result.pdf_name,
                        "page_number": result.page_number,
                        "full_path": str(image_path),
                        "dir_exists": image_dir.exists(),
                        "dir_path": str(image_dir),
                        "dir_contents": list(image_dir.glob("*.jpg")) if image_dir.exists() else []
                    })
                    continue
            except Exception as e:
                self.logger.error("Error accessing image path", extra={
                    "pdf_name": result.pdf_name,
                    "page_number": result.page_number,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                continue
            
            # Multiple retry attempts
            for attempt in range(MAX_RETRIES):
                try:
                    # Read image as base64
                    try:
                        with open(image_path, "rb") as image_file:
                            image_data = image_file.read()
                            image_size = len(image_data)
                            self.logger.debug("Read image file", extra={
                                "image_size_bytes": image_size,
                                "path": str(image_path)
                            })
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                    except Exception as e:
                        self.logger.error("Failed to read image file", extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "path": str(image_path)
                        })
                        raise
                        
                    try:
                        # Log the actual model in use
                        self.logger.debug("Calling Vision API", extra={
                            "model": config.VISION_MODEL,
                            "max_tokens": config.VISION_MAX_TOKENS,
                            "image_size_bytes": image_size
                        })
                        
                        # Check image size (OpenAI limit is 20MB)
                        MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
                        if image_size > MAX_IMAGE_SIZE_BYTES:
                            raise ValueError(f"Image size ({image_size} bytes) exceeds OpenAI's limit of 20MB")

                        # Enhanced vision prompt with context-aware extraction
                        vision_prompt = self._create_context_aware_vision_prompt(result, query)
                        
                        response = self.openai_client.chat.completions.create(
                            model=config.VISION_MODEL,
                            messages=[{
                                "role": "system",
                                "content": "You are an expert at extracting technical documentation from images. Extract text accurately and completely, preserving formatting and technical details."
                            }, {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": vision_prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }],
                            max_tokens=config.VISION_MAX_TOKENS,
                            temperature=0.1  # Lower temperature for more accurate extraction
                        )
                        
                        extracted_text = response.choices[0].message.content.strip()
                        
                        # Check for generic/unhelpful responses
                        generic_indicators = [
                            "i'm unable to extract",
                            "i cannot extract",
                            "unable to extract specific information",
                            "cannot see",
                            "image quality",
                            "unclear"
                        ]
                        is_generic = any(indicator in extracted_text.lower() for indicator in generic_indicators)
                        
                        if not extracted_text or is_generic:
                            # Fallback to metadata-based extraction
                            self.logger.warning("Vision extraction returned generic/empty response, using metadata fallback", extra={
                                "pdf_name": result.pdf_name,
                                "page_number": result.page_number,
                                "is_generic": is_generic
                            })
                            extracted_text = self._extract_from_metadata_fallback(result, query)
                            
                        if not extracted_text:
                            raise ValueError("Both vision extraction and metadata fallback failed")
                            
                        result.vision_extracted_text = extracted_text
                        self.logger.info("Text extraction successful", extra={
                            "pdf_name": result.pdf_name,
                            "page_number": result.page_number,
                            "text_length": len(result.vision_extracted_text),
                            "attempt": attempt + 1,
                            "image_size_bytes": image_size,
                            "used_fallback": is_generic if extracted_text else False
                        })
                        break  # Success - exit retry loop
                        
                    except Exception as e:
                        self.logger.error("Vision API call failed", extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "image_size_bytes": image_size
                        })
                        raise
                    
                except Exception as e:
                    self.logger.warning("Vision extraction attempt failed", extra={
                        "pdf_name": result.pdf_name,
                        "page_number": result.page_number,
                        "error": str(e),
                        "attempt": attempt + 1,
                        "max_retries": MAX_RETRIES
                    })
                    
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)  # Wait before retrying
                    else:
                        self.logger.error("Vision extraction failed after all retries, using metadata fallback", extra={
                            "pdf_name": result.pdf_name,
                            "page_number": result.page_number,
                            "total_attempts": MAX_RETRIES
                        })
                        # Final fallback to metadata
                        result.vision_extracted_text = self._extract_from_metadata_fallback(result, query) or ""
        
        return search_results

    def search_all_documents(self, query: str, top_k: int = 5, include_strategies: List[str] = None, search_params: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search across all available documents using pre-computed vectors"""
        try:
            # Get list of all available documents from content index
            document_ids = list(self.content_index.keys())
            if not document_ids:
                self.logger.error("No documents available for search")
                return []
            
            # Apply intent-directed filtering if search_params provided
            if search_params and search_params.get("target_directories"):
                target_docs = [Path(d["path"]).name for d in search_params["target_directories"]]
                
                # Use fuzzy matching for document IDs to handle variations
                filtered_ids = []
                for target_doc in target_docs:
                    # Try exact match first
                    if target_doc in document_ids:
                        filtered_ids.append(target_doc)
                    else:
                        # Try fuzzy match (partial/contains match)
                        for doc_id in document_ids:
                            target_normalized = target_doc.lower().replace("(", "").replace(")", "").strip()
                            doc_normalized = doc_id.lower().replace("(", "").replace(")", "").strip()
                            
                            # Check if target is contained in doc_id or vice versa
                            if target_normalized in doc_normalized or doc_normalized in target_normalized:
                                if doc_id not in filtered_ids:
                                    filtered_ids.append(doc_id)
                                    self.logger.debug("Fuzzy matched document ID", extra={
                                        "target": target_doc,
                                        "matched": doc_id
                                    })
                                break
                            # Word-based matching for multi-word IDs
                            target_words = set(target_normalized.split())
                            doc_words = set(doc_normalized.split())
                            if len(target_words & doc_words) >= max(2, len(target_words) * 0.7):
                                if doc_id not in filtered_ids:
                                    filtered_ids.append(doc_id)
                                    self.logger.debug("Word-matched document ID", extra={
                                        "target": target_doc,
                                        "matched": doc_id
                                    })
                                break
                
                if filtered_ids:
                    document_ids = filtered_ids
                    self.logger.info("Intent-directed search: filtering to specific documents", extra={
                        "original_count": len(self.content_index.keys()),
                        "filtered_count": len(document_ids),
                        "target_docs": target_docs,
                        "matched_docs": document_ids
                    })
                else:
                    self.logger.warning("Intent-directed search: no documents matched, using all documents", extra={
                        "target_docs": target_docs,
                        "available_docs": list(document_ids)[:5]
                    })
            
            self.logger.debug("Starting search", extra={
                "document_count": len(document_ids),
                "query": query,
                "top_k": top_k,
                "vectors_loaded": len(self.vectors_cache),
                "documents_with_vectors": list(self.vectors_cache.keys()),
                "has_search_params": bool(search_params)
            })
            
            all_results = []
            
            # Get query embedding once for all documents
            query_embedding = self._get_query_embedding_sync(query)
            
            # Search each document
            for document_id in document_ids:
                try:
                    # Load vectors if not already loaded
                    if document_id not in self.vectors_cache:
                        self.logger.debug("Loading vectors for document", extra={
                            "document_id": document_id
                        })
                        if not self._load_precomputed_vectors_sync(document_id):
                            continue
                    
                    # Search within document
                    document_results = self._similarity_search_sync(document_id, query_embedding, top_k)
                    all_results.extend(document_results)
                    
                    self.logger.debug("Document searched", extra={
                        "document_id": document_id,
                        "results_found": len(document_results)
                    })
                    
                except Exception as e:
                    self.logger.error("Document search failed", extra={
                        "document_id": document_id,
                        "error_message": str(e),
                        "error_type": type(e).__name__
                    })
                    continue
            
            # Sort all results by score and take top_k
            all_results.sort(key=lambda x: x.score, reverse=True)
            top_results = all_results[:top_k]
            
            # CONFIDENCE THRESHOLD CHECK: If best result has very low confidence, log warning
            # Trust intent layer's semantic analysis - no keyword-based fallback
            confidence_threshold = 0.1  # Very low threshold - trust semantic matching
            if top_results and top_results[0].score < confidence_threshold:
                self.logger.warning("Very low confidence results detected", extra={
                    "best_score": top_results[0].score,
                    "threshold": confidence_threshold,
                    "query_preview": query[:100],
                    "note": "This may indicate semantic mismatch - consider reviewing intent analysis"
                })
                # Don't do keyword-based fallback - trust the semantic intelligence
                # If scores are truly low, the issue is with semantic matching, not missing docs
            
            # Enhance with vision API if enabled (optional)
            if config.INCLUDE_VISION_ANALYSIS and top_results:
                top_results = self._enhance_with_vision_sync(top_results, query)
            
            self.logger.info("Multi-document search complete", extra={
                "total_results": len(all_results),
                "top_k_returned": len(top_results),
                "documents_searched": len(document_ids),
                "best_score": top_results[0].score if top_results else 0,
                "confidence_check_applied": top_results and top_results[0].score < confidence_threshold if top_results else False
            })
             
            return top_results
            
        except Exception as e:
            self.logger.error("Multi-document search failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            return []

    def search(self, query: str, document_id: str = None, top_k: int = 5, 
              include_strategies: List[str] = None, search_params: Dict[str, Any] = None) -> List[SearchResult]:
        """
        Dynamic multi-strategy search with intelligent analysis and ranking
        
        Args:
            query: Search query
            document_id: Optional specific document ID to search
            top_k: Number of results to return
            include_strategies: Optional list of strategies to use
            search_params: Optional parameters from intent analysis
            
        Returns:
            List of dynamically ranked search results
        """
        try:
            self.logger.info("🔍 Starting dynamic multi-strategy search", extra={
                "query": query,
                "document_id": document_id,
                "top_k": top_k,
                "has_intent_params": bool(search_params)
            })
            
            # Step 1: Analyze query for dynamic strategy selection
            query_analysis = self._analyze_query(query)
            
            # Step 2: Dynamically adjust strategy weights based on query analysis
            self._adjust_strategy_weights(query_analysis)
            
            # Step 3: Track analytics for continuous improvement
            self._track_search_analytics(query, query_analysis)
            
            # Step 4: Apply intent-directed search if parameters provided
            if search_params and search_params.get("target_directories"):
                target_docs = search_params["target_directories"]
                self.logger.info("Using intent-directed search", extra={
                    "target_docs": [Path(d).name for d in target_docs],
                    "search_focus": search_params.get("search_focus")
                })
                # Filter documents based on intent analysis
                filtered_docs = []
                for doc_path in target_docs:
                    doc_name = Path(doc_path).name
                    if doc_name in self.content_index:
                        filtered_docs.append(doc_name)
                query_analysis['target_documents'] = filtered_docs
            
            # Step 5: Execute multi-strategy search
            strategy_results = {}
            
            # Execute each strategy based on dynamic weights
            if self.strategy_weights.get('vector_semantic', 0) > 0:
                strategy_results['vector_semantic'] = self._vector_search(query, top_k)
            
            if self.strategy_weights.get('enhanced_keyword', 0) > 0:
                strategy_results['enhanced_keyword'] = self._keyword_search(query, query_analysis, top_k)
            
            if self.strategy_weights.get('title_section_match', 0) > 0:
                strategy_results['title_section_match'] = self._title_section_search(query, query_analysis, top_k)
            
            if self.strategy_weights.get('domain_specific', 0) > 0:
                strategy_results['domain_specific'] = self._domain_specific_search(query, query_analysis, top_k)
            
            if self.strategy_weights.get('query_expanded', 0) > 0:
                strategy_results['query_expanded'] = self._query_expansion_search(query, query_analysis, top_k)
            
            # Step 6: Merge and re-rank results using dynamic scoring
            final_results = self._merge_and_rerank(strategy_results, query_analysis, top_k)
            
            # Step 7: Apply dynamic boosting
            final_results = self._boost_customer_support_results(final_results, query)
            
            # Step 8: Enhance with vision analysis if needed
            if config.INCLUDE_VISION_ANALYSIS and final_results:
                final_results = self._enhance_with_vision_sync(final_results[:top_k], query)
            
            self.logger.info("✅ Dynamic multi-strategy search complete", extra={
                "results_found": len(final_results),
                "strategies_used": len(strategy_results),
                "query_type": query_analysis['query_type'],
                "final_weights": self.strategy_weights
            })
             
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error("Dynamic search failed", extra={"error": str(e)})
            # Fallback to basic vector search
            return self._vector_search(query, top_k)
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis for better understanding and document selection"""
        query_lower = query.lower()
        
        analysis = {
            'original_query': query,
            'query_type': self._determine_query_type(query_lower),
            'key_terms': [],
            'domain_terms': [],
            'process_terms': [],
            'required_context': [],
            'prerequisites': [],
            'expansion_needed': True
        }
        
        # Extract key terms with better domain understanding
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        analysis['key_terms'] = [word.lower() for word in words if word.lower() not in ['the', 'and', 'for', 'how', 'what', 'where', 'when']]
        
        # Identify required context
        if 'how' in query_lower or 'process' in query_lower or 'steps' in query_lower:
            analysis['required_context'].extend(['prerequisites', 'steps', 'verification'])
        if 'what' in query_lower or 'explain' in query_lower or 'describe' in query_lower:
            analysis['required_context'].extend(['definition', 'examples', 'related_concepts'])
        if 'where' in query_lower or 'screen' in query_lower or 'form' in query_lower:
            analysis['required_context'].extend(['navigation', 'ui_elements', 'form_fields'])
        
        # Dynamic domain pattern detection based on query content
        analysis['domain_patterns'] = self._detect_domain_patterns_dynamic(query, query_lower)
        
        # Extract domain terms dynamically
        for category, pattern_data in analysis['domain_patterns'].items():
            for term in pattern_data['terms']:
                if term.lower() in query_lower:
                    analysis['domain_terms'].append(term.lower())
        
        # Identify prerequisites
        prerequisite_indicators = [
            (r'before\s+([^,.]+)', 'precondition'),
            (r'requires?\s+([^,.]+)', 'requirement'),
            (r'need\s+to\s+([^,.]+)', 'requirement'),
            (r'first\s+([^,.]+)', 'initial_step')
        ]
        
        for pattern, prereq_type in prerequisite_indicators:
            matches = re.findall(pattern, query_lower)
            if matches:
                analysis['prerequisites'].extend([(match.strip(), prereq_type) for match in matches])
        
        return analysis
    
    def _detect_domain_patterns_dynamic(self, query: str, query_lower: str) -> Dict[str, Dict]:
        """Dynamically detect domain patterns based on query content and context"""
        detected_patterns = {}
        
        # User management pattern
        if any(term in query_lower for term in ['user', 'create', 'add', 'manage', 'setup']):
            detected_patterns['user_management'] = {
                'terms': ['user', 'create', 'add', 'setup', 'manage', 'administrator', 'permissions', 'access'],
                'boost': 5.0,
                'category': 'user_management'
            }
        
        # Form/screen pattern
        if any(term in query_lower for term in ['form', 'screen', 'page']) or re.search(r'[A-Z]{2}\d{6}', query):
            detected_patterns['forms'] = {
                'terms': ['form', 'screen', 'page', 'interface', 'field', 'button', 'menu'],
                'boost': 4.5,
                'category': 'forms'
            }
        
        # Process/procedure pattern
        if any(term in query_lower for term in ['how', 'process', 'steps', 'procedure', 'workflow']):
            detected_patterns['processes'] = {
                'terms': ['process', 'procedure', 'steps', 'workflow', 'sequence', 'method'],
                'boost': 4.0,
                'category': 'processes'
            }
        
        # Employee/labor pattern
        if any(term in query_lower for term in ['employee', 'labor', 'rate', 'cost']):
            detected_patterns['employee_management'] = {
                'terms': ['employee', 'labor', 'rate', 'cost', 'consultant', 'effective date'],
                'boost': 4.5,
                'category': 'employee_management'
            }
        
        # Configuration pattern
        if any(term in query_lower for term in ['configure', 'setup', 'setting', 'define']):
            detected_patterns['configuration'] = {
                'terms': ['configure', 'setup', 'setting', 'define', 'specify', 'parameter'],
                'boost': 4.0,
                'category': 'configuration'
            }
        
        # Document pattern
        if any(term in query_lower for term in ['document', 'invoice', 'order', 'shipment', 'return']):
            detected_patterns['documents'] = {
                'terms': ['document', 'invoice', 'order', 'shipment', 'return', 'receipt'],
                'boost': 3.5,
                'category': 'documents'
            }
        
        # If no specific patterns detected, use general pattern
        if not detected_patterns:
            detected_patterns['general'] = {
                'terms': ['acumatica', 'system', 'application', 'software'],
                'boost': 2.0,
                'category': 'general'
            }
        
        return detected_patterns

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query for better response structuring"""
        query_types = {
            'form_specific': [
                'form', 'screen', 'page', r'[A-Z]{2}\d{6}',  # Form identifiers
                'labor rates', 'employee rate', 'cost rate',  # Labor forms
                'add row', 'settings in the row', 'toolbar'   # Form actions
            ],
            'procedural': [
                'how to', 'how do i', 'steps', 'process', 'procedure',
                'define', 'configure', 'set up', 'create', 'add'
            ],
            'employee_related': [
                'employee', 'labor', 'rate', 'cost', 'effective date',
                'consultant', r'EP\d{8}', 'CONSULTJR', 'CONSULTSR'
            ],
            'conceptual': [
                'what is', 'what are', 'explain', 'describe', 'definition',
                'purpose', 'usage', 'when to use'
            ],
            'locational': [
                'where is', 'where can i', 'find', 'locate', 'navigation',
                'which screen', 'which form', 'access'
            ],
            'configuration': [
                'setup', 'configure', 'setting', 'preference', 'option',
                'define', 'specify', 'enter'
            ]
        }
        
        for query_type, indicators in query_types.items():
            if any(indicator in query.lower() for indicator in indicators):
                return query_type
        
        return 'general'

    def _adjust_strategy_weights(self, query_analysis: Dict[str, Any]):
        """Dynamically adjust search strategy weights based on query analysis"""
        query_type = query_analysis['query_type']
        
        # Base weights for different query types
        type_weights = {
            'procedural': {
                'title_section_match': 0.40,  # Higher for finding specific procedures
                'enhanced_keyword': 0.30,
                'vector_semantic': 0.15,
                'domain_specific': 0.10,
                'query_expanded': 0.05
            },
            'conceptual': {
                'vector_semantic': 0.40,      # Higher for understanding concepts
                'domain_specific': 0.25,
                'enhanced_keyword': 0.20,
                'title_section_match': 0.10,
                'query_expanded': 0.05
            },
            'locational': {
                'title_section_match': 0.35,  # Higher for finding UI elements
                'enhanced_keyword': 0.30,
                'domain_specific': 0.20,
                'vector_semantic': 0.10,
                'query_expanded': 0.05
            },
            'comparative': {
                'vector_semantic': 0.35,      # Higher for understanding relationships
                'domain_specific': 0.30,
                'enhanced_keyword': 0.20,
                'title_section_match': 0.10,
                'query_expanded': 0.05
            },
            'troubleshooting': {
                'enhanced_keyword': 0.35,     # Higher for finding specific issues
                'domain_specific': 0.25,
                'vector_semantic': 0.20,
                'title_section_match': 0.15,
                'query_expanded': 0.05
            },
            'configuration': {
                'title_section_match': 0.35,  # Higher for finding settings
                'domain_specific': 0.30,
                'enhanced_keyword': 0.20,
                'vector_semantic': 0.10,
                'query_expanded': 0.05
            }
        }
        
        # Use type-specific weights or fall back to base weights
        self.strategy_weights = type_weights.get(query_type, self.base_strategy_weights)
        
        # Adjust weights based on required context
        if 'prerequisites' in query_analysis['required_context']:
            self.strategy_weights['domain_specific'] *= 1.2
        if 'navigation' in query_analysis['required_context']:
            self.strategy_weights['title_section_match'] *= 1.2
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v/total_weight for k, v in self.strategy_weights.items()}
        
        self.logger.info("Strategy weights adjusted", extra={
            "query_type": query_type,
            "weights": self.strategy_weights
        })
    
    def _track_search_analytics(self, query: str, query_analysis: Dict[str, Any]):
        """Track search analytics for performance monitoring"""
        try:
            self.search_analytics['total_searches'] += 1
            
            query_type = query_analysis['query_type']
            if query_type not in self.search_analytics['query_types']:
                self.search_analytics['query_types'][query_type] = 0
            self.search_analytics['query_types'][query_type] += 1
            
            # Log key metrics every 10 searches
            if self.search_analytics['total_searches'] % 10 == 0:
                self.logger.info("📊 Search analytics update", 
                               total_searches=self.search_analytics['total_searches'],
                               query_types=self.search_analytics['query_types'])
                               
        except Exception as e:
            self.logger.warning("Analytics tracking failed", extra={"error": str(e)})
    
    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Enhanced vector search using pre-computed vectors"""
        try:
            # Create query embedding using HuggingFace (FREE!)
            query_embedding = self._get_query_embedding_sync(query)
            
            # Search across all documents
            all_results = []
            for document_id in list(self.content_index.keys()):
                if document_id not in self.vectors_cache:
                    if not self._load_precomputed_vectors_sync(document_id):
                        continue
                
                document_results = self._similarity_search_sync(document_id, query_embedding, top_k)
                all_results.extend(document_results)
            
            # Sort and return top_k
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            self.logger.warning("Vector search failed", extra={"error": str(e)})
            return []
    
    def _keyword_search(self, query: str, query_analysis: Dict, top_k: int) -> List[SearchResult]:
        """Enhanced keyword search with term frequency analysis"""
        try:
            # Extract search terms from query analysis
            search_terms = query_analysis['key_terms'] + query_analysis['domain_terms']
            
            # Score pages based on keyword matches
            page_scores = defaultdict(float)
            
            for term in search_terms:
                if term in self.keyword_index:
                    # TF-IDF style scoring
                    doc_freq = len(self.keyword_index[term])
                    idf = math.log(len(self.text_index) / (doc_freq + 1))
                    
                    for page_id in self.keyword_index[term]:
                        # Calculate term frequency in document
                        text_data = self.text_index.get(page_id, {})
                        text = text_data.get('raw_text', '').lower()
                        tf = text.count(term) / max(len(text.split()), 1)
                        
                        # TF-IDF score
                        page_scores[page_id] += tf * idf
            
            # Convert to SearchResult objects
            results = []
            for page_id, score in sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                if page_id in self.text_index:
                    text_data = self.text_index[page_id]
                    
                    result = SearchResult(
                        pdf_name=text_data['pdf_name'],
                        page_number=text_data['page_number'],
                        score=score,
                        text_content=text_data['raw_text'],
                        metadata={'source': 'keyword_search'},
                        relevance_signals={'keyword_tfidf': score},
                        search_strategy='enhanced_keyword',
                        combined_score=score
                    )
                    # Use standardized image path
                    result.image_path = result.get_image_path()
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.warning("Keyword search failed", extra={"error": str(e)})
            return []
    
    def _query_expansion_search(self, query: str, query_analysis: Dict, top_k: int) -> List[SearchResult]:
        """Search using LLM-expanded query for better semantic matching"""
        try:
            # Generate query expansion
            expansion_prompt = f"""
            Expand this search query to include related terms, synonyms, and domain-specific vocabulary for finding relevant content in business documentation:
            
            Original query: "{query}"
            
            Provide 5-10 related search terms or phrases that could help find the same information:
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            expanded_terms = response.choices[0].message.content.strip()
            
            # Combine original query with expanded terms
            enhanced_query = f"{query} {expanded_terms}"
            
            # Run vector search with enhanced query
            results = self._vector_search(enhanced_query, top_k)
            
            # Update strategy information
            for result in results:
                result.search_strategy = 'query_expanded'
                result.relevance_signals['query_expansion'] = result.score
            
            return results
            
        except Exception as e:
            self.logger.warning("Expanded query search failed", extra={"error": str(e)})
            return []
    
    def _title_section_search(self, query: str, query_analysis: Dict, top_k: int) -> List[SearchResult]:
        """Search based on document titles and section headers"""
        try:
            results = []
            query_terms = set(query_analysis['key_terms'])
            
            for page_id, text_data in self.text_index.items():
                # Check titles and structured content
                titles = text_data.get('structured_text', {}).get('titles', [])
                title_text = ' '.join(titles).lower()
                
                # Score based on title matches
                title_score = 0
                for term in query_terms:
                    if term in title_text:
                        title_score += 2.0  # Higher weight for title matches
                
                # Score based on content structure
                content = text_data.get('structured_text', {}).get('content', '').lower()
                content_score = 0
                for term in query_terms:
                    content_score += content.count(term) * 0.1
                
                total_score = title_score + content_score
                
                if total_score > 0:
                    result = SearchResult(
                        pdf_name=text_data['pdf_name'],
                        page_number=text_data['page_number'],
                        score=total_score,
                        text_content=text_data['raw_text'],
                        metadata={'titles': titles},
                        relevance_signals={'title_match': title_score, 'content_match': content_score},
                        search_strategy='title_section_match',
                        combined_score=total_score
                    )
                    # Use standardized image path
                    result.image_path = result.get_image_path()
                    results.append(result)
            
            # Sort by score and return top results
            results.sort(key=lambda x: x.combined_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.warning("Title section search failed", extra={"error": str(e)})
            return []
    
    def _domain_specific_search(self, query: str, query_analysis: Dict, top_k: int) -> List[SearchResult]:
        """Dynamic domain-specific search using intelligent pattern detection"""
        try:
            # Use dynamic domain patterns from query analysis
            domain_patterns = query_analysis.get('domain_patterns', {})
            
            results = []
            query_lower = query.lower()
            
            # Include Acumatica mappings in search
            search_terms = set([query_lower])
            if query_analysis.get('acumatica_mappings'):
                search_terms.update([term.lower() for term in query_analysis['acumatica_mappings']])
            
            for page_id, text_data in self.text_index.items():
                page_text = text_data['raw_text'].lower()
                page_text_original = text_data['raw_text']  # Keep original case for exact matching
                total_score = 0
                matched_patterns = []
                
                # Check each domain pattern
                for pattern_name, pattern_data in domain_patterns.items():
                    pattern_score = 0
                    required_count = 0
                    exact_phrase_bonus = 0
                    
                    # Check for exact phrases first (higher priority)
                    for exact_phrase in pattern_data.get('exact_phrases', []):
                        if exact_phrase in page_text_original:  # Case-sensitive exact match
                            exact_phrase_bonus += pattern_data['boost'] * 2.0  # Double boost for exact phrases
                        elif exact_phrase.lower() in page_text:  # Case-insensitive fallback
                            exact_phrase_bonus += pattern_data['boost'] * 1.5
                    
                    # Check regular terms
                    for term in pattern_data['terms']:
                        term_lower = term.lower()
                        # Check if term appears in any of our search contexts
                        term_found = any(term_lower in search_term for search_term in search_terms)
                        
                        if term_found and term_lower in page_text:
                            pattern_score += pattern_data['boost']
                            
                            if term_lower in [req.lower() for req in pattern_data['required_terms']]:
                                required_count += 1
                    
                    # Only count if required terms are present or exact phrases found
                    if required_count >= len(pattern_data['required_terms']) or exact_phrase_bonus > 0:
                        total_score += pattern_score + exact_phrase_bonus
                        if pattern_score > 0 or exact_phrase_bonus > 0:
                            matched_patterns.append(pattern_name)
                
                if total_score > 0:
                    result = SearchResult(
                        pdf_name=text_data['pdf_name'],
                        page_number=text_data['page_number'],
                        score=total_score,
                        text_content=text_data['raw_text'],
                        metadata={
                            'domain_patterns': matched_patterns,
                            'acumatica_mappings': query_analysis.get('acumatica_mappings', [])
                        },
                        relevance_signals={'domain_specific': total_score},
                        search_strategy='domain_specific',
                        combined_score=total_score
                    )
                    # Use standardized image path
                    result.image_path = result.get_image_path()
                    results.append(result)
            
            # Sort and return top results
            results.sort(key=lambda x: x.combined_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.warning("Domain specific search failed", extra={"error": str(e)})
            return []
    
    def _merge_and_rerank(self, strategy_results: Dict[str, List[SearchResult]], 
                         query_analysis: Dict, top_k: int) -> List[SearchResult]:
        """Merge results from multiple strategies and re-rank using ensemble scoring"""
        try:
            # Collect all unique results
            all_results = {}  # page_id -> SearchResult
            
            for strategy, results in strategy_results.items():
                weight = self.strategy_weights.get(strategy, 0.1)
                
                for result in results:
                    page_key = f"{result.pdf_name}_page_{result.page_number}"
                    
                    if page_key not in all_results:
                        # First time seeing this result
                        result.relevance_signals = {}
                        result.combined_score = 0
                        all_results[page_key] = result
                    
                    # Add this strategy's contribution
                    all_results[page_key].relevance_signals[strategy] = result.score
                    all_results[page_key].combined_score += result.score * weight
            
            # Apply query-specific boosting
            for result in all_results.values():
                boost_factor = self._calculate_query_boost(result, query_analysis)
                result.combined_score *= boost_factor
            
            # Sort by final combined score
            final_results = sorted(all_results.values(), 
                                key=lambda x: x.combined_score, 
                                reverse=True)
            
            # Update search strategy to show best contributing strategy
            for result in final_results:
                best_strategy = max(result.relevance_signals.items(), 
                                key=lambda x: x[1])[0]
                result.search_strategy = f"hybrid_{best_strategy}"
            
            self.logger.info("Results merged and reranked", extra={
                "total_results": len(all_results),
                "strategies_used": len(strategy_results),
                "top_k_returned": min(top_k, len(final_results))
            })
            
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error("Result merging failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            if strategy_results:
                return list(strategy_results.values())[0][:top_k]
            return []
    
    def _calculate_query_boost(self, result: SearchResult, query_analysis: Dict) -> float:
        """Calculate query-specific boosting factor"""
        boost = 1.0
        
        # Query type specific boosts
        if query_analysis['query_type'] == 'process_workflow':
            # Boost pages with process-related content
            if any(term in result.text_content.lower() for term in ['workflow', 'process', 'step', 'procedure']):
                boost *= 1.3
        
        # Domain term proximity boost
        domain_terms = query_analysis.get('domain_terms', [])
        if len(domain_terms) >= 2:
            text_lower = result.text_content.lower()
            proximity_boost = 1.0
            
            for i, term1 in enumerate(domain_terms[:-1]):
                for term2 in domain_terms[i+1:]:
                    if term1 in text_lower and term2 in text_lower:
                        # Find positions
                        pos1 = text_lower.find(term1)
                        pos2 = text_lower.find(term2)
                        
                        # Boost if terms are close to each other
                        distance = abs(pos1 - pos2)
                        if distance < 200:  # Within 200 characters
                            proximity_boost *= 1.2
            
            boost *= proximity_boost
        
        # Exact phrase boost
        if len(query_analysis['original_query']) > 10:
            # Check for exact phrase matches
            query_phrases = re.findall(r'"([^"]*)"', query_analysis['original_query'])
            for phrase in query_phrases:
                if phrase.lower() in result.text_content.lower():
                    boost *= 1.5
        
        return boost

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for customer support accuracy"""
        query_lower = query.lower()
        
        # Visual/diagram queries
        visual_terms = ["diagram", "chart", "image", "figure", "visual", "picture", "flow", "process flow"]
        if any(term in query_lower for term in visual_terms):
            return "visual_priority"
            
        # Process/procedure queries  
        process_terms = ["how to", "step", "process", "procedure", "workflow", "method", "instructions"]
        if any(term in query_lower for term in process_terms):
            return "process_priority"
            
        # Error/troubleshooting queries
        error_terms = ["error", "problem", "issue", "fix", "troubleshoot", "not working", "failed"]
        if any(term in query_lower for term in error_terms):
            return "error_priority"
            
        return "general"

    def _boost_customer_support_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Boost results based on customer support relevance"""
        query_type = self._classify_query_type(query)
        
        for result in results:
            content = result.text_content.lower()
            metadata = result.metadata
            
            # Boost based on query type
            if query_type == "visual_priority":
                # Boost if page has visual indicators
                visual_indicators = ["figure", "diagram", "chart", "image", "step 1", "step 2", "workflow"]
                boost = sum(0.1 for term in visual_indicators if term in content)
                result.relevance_signals['visual_boost'] = boost
                
            elif query_type == "process_priority":
                # Boost sequential/process content
                process_indicators = ["step", "first", "then", "next", "finally", "procedure", "process"]
                boost = sum(0.15 for term in process_indicators if term in content)
                result.relevance_signals['process_boost'] = boost
                
            elif query_type == "error_priority":
                # Boost troubleshooting content
                solution_indicators = ["solution", "fix", "resolve", "troubleshoot", "error", "problem"]
                boost = sum(0.2 for term in solution_indicators if term in content)
                result.relevance_signals['solution_boost'] = boost
            
            # Add exact term matching boost for customer support
            exact_match_boost = self._calculate_exact_term_boost(query, result.text_content)
            result.relevance_signals['exact_match_boost'] = exact_match_boost
            
            # Recalculate combined score
            total_boost = sum(result.relevance_signals.get(key, 0) for key in [
                'visual_boost', 'process_boost', 'solution_boost', 'exact_match_boost'
            ])
            result.combined_score += total_boost
            
        return sorted(results, key=lambda x: x.combined_score, reverse=True)

    def _calculate_exact_term_boost(self, query: str, content: str) -> float:
        """Enhanced exact term matching with intelligent context analysis"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        import re
        quoted_terms = re.findall(r'"([^"]*)"', query)
        
        boost = 0.0
        
        # High boost for exact quoted phrases (critical accuracy indicator)
        for term in quoted_terms:
            if term.lower() in content_lower:
                boost += 0.4  # Increased from 0.3
        
        # Enhanced technical term detection
        technical_terms = re.findall(r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b', query)  # CamelCase
        for term in technical_terms:
            if term in content:  # Exact case match
                boost += 0.3  # Increased from 0.2
                
        # Action word clustering (critical for procedural accuracy)
        action_words = ['create', 'add', 'setup', 'configure', 'manage', 'process', 'define', 'implement', 'navigate', 'click', 'enter', 'save']
        query_actions = [word for word in query_lower.split() if word in action_words]
        content_actions = [word for word in content_lower.split() if word in action_words]
        
        action_matches = len(set(query_actions) & set(content_actions))
        if action_matches > 0:
            boost += action_matches * 0.25  # Strong boost for procedural accuracy
        
        # Domain terminology clustering
        domain_terms = ['acumatica', 'screen', 'form', 'field', 'user', 'system', 'management', 'module', 'button', 'menu']
        query_domain = [word for word in query_lower.split() if word in domain_terms]
        content_domain = [word for word in content_lower.split() if word in domain_terms]
        
        domain_matches = len(set(query_domain) & set(content_domain))
        if domain_matches >= 2:
            boost += domain_matches * 0.2  # High boost for domain context accuracy
        
        # Enhanced sequential phrase matching
        query_words = query_lower.split()
        if len(query_words) >= 2:
            consecutive_matches = 0
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in content_lower:
                    consecutive_matches += 1
                    boost += 0.2  # Increased from 0.1
            
            # Bonus for multiple consecutive matches (indicates high relevance)
            if consecutive_matches >= 2:
                boost += consecutive_matches * 0.15
                    
        # Context depth analysis (3+ word phrases indicate high specificity)
        if len(query_words) >= 3:
            for i in range(len(query_words) - 2):
                three_phrase = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}"
                if three_phrase in content_lower:
                    boost += 0.35  # Very high boost for specific context matches
        
        # Exact sentence fragment matching (very high confidence indicator)
        if len(query_words) >= 4:
            for i in range(len(query_words) - 3):
                four_phrase = f"{query_words[i]} {query_words[i+1]} {query_words[i+2]} {query_words[i+3]}"
                if four_phrase in content_lower:
                    boost += 0.5  # Maximum boost for very specific matches
                    
        return min(boost, 0.8)  # Increased cap for higher confidence potential

def main():
    """Test the hybrid retrieval system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hybrid Retrieval System")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    retriever = HybridRetriever()
    results = retriever.search(args.query, args.top_k)
    
    print(f"\n🔍 Hybrid Search Results for: '{args.query}'")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.pdf_name} - Page {result.page_number}")
        print(f"   Strategy: {result.search_strategy}")
        print(f"   Score: {result.combined_score:.4f}")
        print(f"   Signals: {result.relevance_signals}")
        print(f"   Text Preview: {result.text_content[:200]}...")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
OpenAI-Powered PDF RAG Inference Script

Advanced inference system using OpenAI for embeddings and vision:
1. OpenAI embeddings for query processing (no PyTorch dependency)
2. Enhanced vision analysis with GPT-4 Vision
3. Pure vision-based text extraction and analysis
4. Multi-strategy processing for maximum accuracy
5. No local model dependencies for fast deployment
6. Adaptive error handling and recovery

Usage:
    python infer.py
    python infer.py --question "What is this document about?"
"""

import os
import sys
import base64
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add src directory to Python path for imports when run directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent.absolute()
    src_dir = current_dir.parent  # Go up from core/ to src/
    project_root = src_dir.parent  # Go up from src/ to project root
    sys.path.insert(0, str(src_dir.absolute()))
    
    # Set environment variables for organized structure
    import os
    os.environ["DATA_DIR"] = str(project_root / "data_storage" / "data")
    os.environ["IMAGES_DIR"] = str(project_root / "data_storage" / "images")
    os.chdir(str(project_root))

from openai import OpenAI
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity  # Lightweight similarity search
from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation
from src.core.hybrid_retriever import HybridRetriever, SearchResult
import numpy as np # Added for lightweight similarity search

# Ultra-concise Vision System Prompt for maximum cost efficiency
VISION_SYSTEM_PROMPT = """Extract relevant info from doc images. Be very brief and focused."""

def _get_huggingface_embedding(model_name: str, logger=None):
    """Get HuggingFace embedding model with fallback support"""
    # Try multiple import paths for compatibility
    try:
        # Try old llama-index path (v0.9.x)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        if logger:
            logger.debug("Using llama-index HuggingFaceEmbedding (old path)")
        return HuggingFaceEmbedding(model_name=model_name)
    except ImportError:
        try:
            # Try new llama-index path (v0.10+)
            from llama_index.embeddings import HuggingFaceEmbedding
            if logger:
                logger.debug("Using llama-index HuggingFaceEmbedding (new path)")
            return HuggingFaceEmbedding(model_name=model_name)
        except ImportError:
            # Use sentence-transformers directly (preferred method)
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                import os
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if logger:
                    logger.info("Using sentence-transformers directly (HuggingFace embeddings)")
                
                # Create a wrapper class to match the interface
                class EmbeddingWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def get_text_embedding(self, text: str):
                        return self.model.encode(text, normalize_embeddings=True).tolist()
                
                # Fix for meta tensor issue: explicitly disable device_map and use trust_remote_code
                # Set environment variable to prevent meta tensor issues
                os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
                
                # Load model without device_map to avoid meta tensor issues
                # Use device parameter directly instead of moving after loading
                try:
                    model = SentenceTransformer(
                        model_name, 
                        device=device,
                        trust_remote_code=True
                    )
                except Exception as e:
                    # Fallback: load without device parameter and move manually
                    if "meta tensor" in str(e).lower() or "to_empty" in str(e).lower():
                        logger.warning(f"Meta tensor issue detected, using fallback loading method: {e}")
                        model = SentenceTransformer(model_name, trust_remote_code=True)
                        # Only move if not already on correct device
                        if next(model.parameters()).device.type != device:
                            model = model.to(device)
                    else:
                        raise
                
                return EmbeddingWrapper(model)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import HuggingFace embeddings. "
                    f"Please install: pip install sentence-transformers torch"
                ) from e

class VDRInferencer:
    """HuggingFace + OpenAI PDF Q&A system (FREE embeddings!)"""
    
    def __init__(self):
        try:
            self.logger = get_logger("VDR_INFERENCER")
            
            self.logger.info("VDR Inferencer initializing", extra={
                "llm_model": config.LLM_MODEL,
                "vision_model": config.VISION_MODEL,
                "embedding_model": config.EMBEDDING_MODEL
            })
            
            # Initialize storage manager based on mode
            self._initialize_storage()
            
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            # Initialize embeddings with fallback support
            self.embedding_model = _get_huggingface_embedding(config.EMBEDDING_MODEL, self.logger)
            
            # Initialize vector cache
            self.vectors_cache = {}
            self.metadata_cache = {}
            
            # Initialize hybrid retriever
            self.use_hybrid_search = True
            from src.core.hybrid_retriever import HybridRetriever
            self.hybrid_retriever = HybridRetriever()  # Remove embedding_model parameter
            
            self.temp_dir = None
            
            self.logger.info("VDR Inferencer initialized")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error("Failed to initialize VDR Inferencer", extra={
                    "error_message": str(e),
                    "error_type": type(e).__name__
                })
            raise

    def _initialize_storage(self):
        """Initialize local storage manager"""
        try:
            self.logger.info("Initializing storage manager", extra={
                "storage_mode": "local",
                "local_base_path": config.LOCAL_BASE_PATH,
                "base_path_exists": os.path.exists(config.LOCAL_BASE_PATH)
            })
            
            from src.utils.local_manager import LocalManager
            self.local_manager = LocalManager()
            self.gcs_manager = None  # GCS disabled
            self.logger.info("Local Manager initialized", extra={
                "base_path": self.local_manager.base_path,
                "base_path_exists": os.path.exists(self.local_manager.base_path),
                "base_path_contents": os.listdir(self.local_manager.base_path) if os.path.exists(self.local_manager.base_path) else []
            })
                
        except Exception as e:
            self.logger.error("Failed to initialize storage manager", extra={
                "error_message": str(e),
                "error_type": type(e).__name__,
                "storage_mode": "local",
                "local_base_path": config.LOCAL_BASE_PATH
            })
            raise

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        # We'll initialize this when needed for queries
        pass

    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        # We'll initialize this when needed for queries
        pass

    def _initialize_vector_cache(self):
        """Initialize vector cache"""
        # We'll initialize this when needed for queries
        pass

    def _initialize_hybrid_retriever(self):
        """Initialize hybrid retrieval system"""
        # We'll initialize this when needed for queries
        pass
    
    def get_image_path(self, pdf_name: str, page_number: int) -> Optional[str]:
        """Get local image path"""
        try:
            # Get image path from local storage
            from pathlib import Path
            
            # Check if this is a DLL document
            if '_DLL' in pdf_name:
                # DLL documents use class_diagram.png and code_structure.png
                if page_number == 1:
                    image_path = Path(config.LOCAL_BASE_PATH) / pdf_name / "images" / "class_diagram.png"
                else:
                    image_path = Path(config.LOCAL_BASE_PATH) / pdf_name / "images" / "code_structure.png"
            else:
                # PDF documents use page{number}.jpg
                image_path = Path(config.LOCAL_BASE_PATH) / pdf_name / "images" / f"page{page_number}.jpg"
            
            if not image_path.exists():
                self.logger.warning("Image not found", extra={
                    "pdf_name": pdf_name,
                    "page_number": page_number,
                    "path": str(image_path)
                })
                return None
            
            return str(image_path)
            
        except Exception as e:
            self.logger.error("Failed to get image path", extra={
                "pdf_name": pdf_name,
                "page_number": page_number,
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            return None

    def list_available_documents(self) -> List[Dict[str, Any]]:
        """List available documents from local storage"""
        try:
            from pathlib import Path
            
            # Get list of document directories
            base_path = Path(config.LOCAL_BASE_PATH)
            document_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            
            documents = []
            for doc_dir in document_dirs:
                doc_info = {
                    "document_name": doc_dir.name,
                    "has_metadata": (doc_dir / "metadata" / "metadata.json").exists(),
                    "has_vectors": (doc_dir / "vectors" / "vectors.json").exists(),
                    "has_images": (doc_dir / "images").exists()
                }
                documents.append(doc_info)
            
            self.logger.info("Listed available documents", extra={
                "document_count": len(documents)
            })
            
            return documents
            
        except Exception as e:
            self.logger.error("Failed to list documents", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            return []
    

    
    def _should_use_section_extraction(self, question: str) -> bool:
        """
        Determine if a query is focused enough to warrant section extraction (Vision API cost)
        
        Args:
            question: User's question
            
        Returns:
            True if query is highly focused and worth the Vision API cost
        """
        question_lower = question.lower()
        
        # HIGHLY FOCUSED queries that justify Vision API cost
        focused_indicators = [
            # Specific technical terms
            'flowchart', 'diagram', 'chart', 'table', 'graph', 'workflow', 'process flow',
            'specific step', 'exact procedure', 'detailed process', 'step by step',
            # Specific UI elements  
            'button', 'field', 'screen', 'interface', 'menu', 'option', 'setting',
            # Specific data/configuration
            'configuration', 'parameter', 'value', 'setting', 'option', 'field name',
            # Complex analysis requests
            'analyze', 'compare', 'difference', 'relationship', 'connection'
        ]
        
        # BROAD/GENERAL queries that don't justify Vision API cost
        general_indicators = [
            'what is', 'tell me about', 'explain', 'describe', 'overview', 'summary',
            'general', 'basic', 'simple', 'introduction', 'help', 'understand'
        ]
        
        # Check for focused indicators
        focused_score = sum(1 for indicator in focused_indicators if indicator in question_lower)
        general_score = sum(1 for indicator in general_indicators if indicator in question_lower)
        
        # Decision logic
        if focused_score >= 2:  # Multiple focused terms
            return True
        elif focused_score >= 1 and general_score == 0:  # Focused term without general terms
            return True
        elif focused_score >= 1 and len(question.split()) >= 8:  # Focused + detailed question
            return True
        else:
            return False  # General question - no Vision API cost
    
    def _select_best_section(self, question: str, pdf_name: str, page_num: int) -> Optional[Dict[str, Any]]:
        """
        Select the best extracted section for a given question using temporary section extraction (ONLY for focused queries)
        
        Args:
            question: User's question
            pdf_name: PDF name
            page_num: Page number
            
        Returns:
            Best section metadata or None if no good match
        """
        try:
            # COST CONTROL: Only use section extraction for highly focused queries
            if not self._should_use_section_extraction(question):
                self.logger.debug("Query too general - skipping section extraction to save Vision API costs")
                return None
            
            self.logger.info("Focused query detected - using temporary section extraction")
            
            # Get the image for this page
            image_path = self.get_image_path(pdf_name, page_num)
            if not image_path:
                self.logger.debug("No image found for section extraction - will use full page")
                return None
            
            # Create temporary sections for this specific query
            temp_section_data = self._create_temporary_sections(image_path, question)
            if not temp_section_data:
                self.logger.debug("No sections extracted - will use full page")
                return None
            
            # Use the dedicated method to select the best section from temporary data
            best_section = self._select_best_section_from_temp(question, temp_section_data)
            
            # Clean up temporary sections after selection
            self._cleanup_temporary_sections(temp_section_data)
            
            if best_section:
                self.logger.info("Best temporary section selected", extra={
                    "page": page_num,
                    "section_type": best_section.get('type', 'unknown'),
                    "confidence": best_section.get('confidence', 0)
                })
                return best_section
                
            return None
            
        except Exception as e:
            self.logger.warning("Section selection failed", extra={"error": str(e), "page": page_num})
            return None
    
    def _calculate_ai_relevance(self, question: str, ai_description: str, section_type: str) -> float:
        """
        Calculate semantic relevance between question and AI description (Phase 2B)
        
        Args:
            question: User's question
            ai_description: AI-generated section description
            section_type: Type of section
            
        Returns:
            Relevance score 0-1
        """
        try:
            question_lower = question.lower()
            description_lower = ai_description.lower()
            
            # Extract key terms from question
            question_words = set(question_lower.split())
            description_words = set(description_lower.split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'how', 'what', 'where', 'when', 'why'}
            question_words -= stop_words
            description_words -= stop_words
            
            if not question_words or not description_words:
                return 0.0
            
            # Calculate word overlap
            common_words = question_words.intersection(description_words)
            overlap_score = len(common_words) / max(len(question_words), len(description_words))
            
            # Look for semantic connections beyond exact word matches
            semantic_score = 0
            
            # Process-related semantic matching
            if any(word in question_lower for word in ['how', 'process', 'steps', 'create', 'setup', 'procedure']):
                if any(word in description_lower for word in ['process', 'flow', 'step', 'procedure', 'workflow', 'sequence']):
                    semantic_score += 0.3
            
            # Data/configuration semantic matching
            if any(word in question_lower for word in ['data', 'information', 'settings', 'configuration', 'parameters']):
                if any(word in description_lower for word in ['data', 'table', 'information', 'configuration', 'field', 'column']):
                    semantic_score += 0.3
            
            # Visual/interface semantic matching
            if any(word in question_lower for word in ['screen', 'interface', 'view', 'display', 'show']):
                if any(word in description_lower for word in ['interface', 'screen', 'display', 'view', 'element']):
                    semantic_score += 0.3
            
            # Specific domain terms (Acumatica, sales, returns, etc.)
            domain_terms = ['acumatica', 'sales', 'return', 'order', 'customer', 'invoice', 'shipment', 'replacement']
            question_domain = sum(1 for term in domain_terms if term in question_lower)
            description_domain = sum(1 for term in domain_terms if term in description_lower)
            
            if question_domain > 0 and description_domain > 0:
                domain_score = min(question_domain, description_domain) / max(question_domain, description_domain, 1)
                semantic_score += domain_score * 0.2
            
            # Combine overlap and semantic scores
            total_relevance = (overlap_score * 0.6) + (semantic_score * 0.4)
            
            # Cap the maximum relevance score
            return min(total_relevance, 1.0)
            
        except Exception as e:
            self.logger.warning("AI relevance calculation failed", extra={"error": str(e)})
            return 0.0
    
    def create_enhanced_vision_prompt(self, question: str, context: str) -> str:
        """
        CREATE INTELLIGENT VISION PROMPT optimized for customer support scenarios
        
        Args:
            question: User's question
            context: Document context
            
        Returns:
            Smart vision prompt optimized for customer support accuracy
        """
        # Detect question intent for customer support scenarios
        question_lower = question.lower()
        
        # Diagram/visual workflow questions (HIGH PRIORITY for customer support)
        if any(keyword in question_lower for keyword in [
            'diagram', 'chart', 'flow', 'visual', 'workflow', 'flowchart', 'process flow'
        ]):
            base_prompt = f"""Q: {question}
Context: {context}

CUSTOMER SUPPORT - VISUAL ANALYSIS:
This is a customer support query about diagrams/visuals. Analyze carefully:

1. IDENTIFY VISUAL ELEMENTS:
   - Flowcharts, diagrams, process flows
   - Arrows, connections, decision points
   - Steps, phases, or sequences

2. EXTRACT VISUAL WORKFLOW:
   - Start/end points
   - Decision branches (if/then)
   - Sequential steps
   - Prerequisites or conditions

FORMAT RESPONSE:
VISUAL WORKFLOW:
1. [Step with visual context]
2. [Step with visual context]
DECISION POINTS: [Any yes/no or conditional branches]
IMPORTANT NOTES: [Critical details from diagram]
TEXT: "[Any text visible in diagram]"

Focus on actionable steps a customer can follow."""

        # Error/troubleshooting questions (CRITICAL for customer support)
        elif any(keyword in question_lower for keyword in [
            'error', 'problem', 'issue', 'fix', 'troubleshoot', 'not working', 'failed', 'resolve'
        ]):
            base_prompt = f"""Q: {question}
Context: {context}

CUSTOMER SUPPORT - TROUBLESHOOTING:
This is a customer support troubleshooting query. Focus on solutions:

1. IDENTIFY PROBLEM:
   - Error descriptions
   - Failure conditions
   - System states

2. FIND SOLUTIONS:
   - Resolution steps
   - Workarounds
   - Prevention measures

FORMAT:
PROBLEM: [Issue description]
SOLUTION STEPS:
1. [Action step]
2. [Action step]
PREVENTION: [How to avoid this issue]
TEXT: "[Exact error messages or instructions]"

Prioritize immediate actionable solutions."""

        # Process/procedure questions (COMMON in customer support)
        elif any(keyword in question_lower for keyword in [
            'steps', 'process', 'how to', 'procedure', 'method', 
            'create', 'setup', 'configure', 'install', 'implement'
        ]):
            base_prompt = f"""Q: {question}
Context: {context}

CUSTOMER SUPPORT - STEP-BY-STEP GUIDE:
This is a customer support procedure query. Extract clear instructions:

1. FIND PREREQUISITES:
   - Required access/permissions
   - Necessary tools/information

2. EXTRACT NUMBERED STEPS:
   - Sequential actions
   - Decision points
   - Verification steps

FORMAT:
PREREQUISITES: [What's needed first]
STEPS:
1. [Clear action step]
2. [Clear action step]
VERIFICATION: [How to confirm success]
NOTES: [Important warnings or tips]
TEXT: "[Exact instructions]"

Make steps crystal clear for non-technical users."""

        # Configuration/settings questions
        elif any(keyword in question_lower for keyword in [
            'setting', 'configure', 'setup', 'option', 'parameter', 'field'
        ]):
            base_prompt = f"""Q: {question}
Context: {context}

CUSTOMER SUPPORT - CONFIGURATION:
This is about system settings/configuration. Be specific:

FIND SETTINGS INFO:
- Menu locations/navigation paths
- Field names and values
- Default vs recommended settings

FORMAT:
LOCATION: [Where to find this setting]
FIELDS: [Specific field names and values]
STEPS: [How to configure]
DEFAULTS: [Default or recommended values]
TEXT: "[Exact field names and paths]"

Include exact menu paths and field names."""

        else:
            # General customer support format
            base_prompt = f"""Q: {question}
Context: {context}

CUSTOMER SUPPORT - GENERAL QUERY:
Provide clear, actionable information for customer support:

EXTRACT ANSWER:
- Direct response to question
- Supporting details
- Any relevant procedures

FORMAT:
ANSWER: [Clear, direct response]
DETAILS: [Supporting information]
RELATED PROCEDURES: [Any related steps/processes]
TEXT: "[Relevant quotes from document]"

Focus on accuracy and clarity for customer support."""
        
        return base_prompt
    
    async def analyze_document_image_enhanced(self, image_path: str, question: str, context: str = "") -> Dict[str, Any]:
        """
        ENHANCED DOCUMENT IMAGE ANALYSIS using pure vision-based processing
        
        Args:
            image_path: Path to the document image
            question: User's question
            context: Additional context (PDF name, page number)
            
        Returns:
            Enhanced analysis result with vision-based processing and cost metrics
        """
        analysis_result = {
            "vision_analysis": "",
            "success": False,
            "error": None,
            "strategies_used": [],
            "token_usage": {
                "vision_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }
        
        try:
            # Step 1: Verify image exists and is accessible
            if not os.path.exists(image_path):
                analysis_result["error"] = f"Image not found at {image_path}"
                return analysis_result
            
            # Step 2: Encode image for vision analysis
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                analysis_result["error"] = "Could not encode image for vision analysis"
                return analysis_result
            
            # Step 3: Create cost-efficient vision prompt
            vision_msg = self.create_enhanced_vision_prompt(
                question, context
            )
            
            # Step 4: Cost-optimized async vision analysis with system prompt
            image_format = self.get_image_format(image_path)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=config.VISION_MODEL,  
                    messages=[
                        {"role": "system", "content": VISION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_msg},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=config.TEMPERATURE,  # Cost-optimized temperature for accuracy
                    max_tokens=config.VISION_MAX_TOKENS  # Cost-optimized token limit
                )
            )
            
            analysis_result["vision_analysis"] = response.choices[0].message.content
            analysis_result["success"] = True
            analysis_result["strategies_used"].append("ENHANCED_VISION_ANALYSIS")
            
            # Track token usage for cost calculation
            if hasattr(response, 'usage'):
                usage = response.usage
                analysis_result["token_usage"] = {
                    "vision_tokens": getattr(usage, 'prompt_tokens', 0),  # Vision tokens counted as prompt tokens
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage, 'completion_tokens', 0)
                }
            
            self.logger.info("Enhanced vision analysis completed", extra={
                "strategies": len(analysis_result["strategies_used"]),
                "vision_response_length": len(analysis_result["vision_analysis"]),
                "prompt_tokens": analysis_result["token_usage"]["prompt_tokens"],
                "completion_tokens": analysis_result["token_usage"]["completion_tokens"]
            })
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"Enhanced analysis failed: {str(e)}"
            analysis_result["error"] = error_msg
            self.logger.error("Enhanced analysis failed", extra={"error": str(e), "image_path": image_path})
            return analysis_result
    
    def create_query_embedding(self, question: str) -> List[float]:
        """Create query embedding using HuggingFace embeddings (FREE!) with dimension matching"""
        try:
            self.logger.debug("Creating query embedding with HuggingFace", extra={
                "query": question[:100] + "..." if len(question) > 100 else question
            })
            
            # Use HuggingFace embeddings (FREE!)
            embedding = self.embedding_model.get_text_embedding(question)
            
            # Handle dimension mismatch by truncating to match Pinecone index
            if len(embedding) > config.EMBEDDING_DIMENSION:
                original_dim = len(embedding)
                embedding = embedding[:config.EMBEDDING_DIMENSION]
                self.logger.debug("Embedding truncated to match index", extra={
                    "original_dim": original_dim,
                    "truncated_dim": len(embedding)
                })
            
            self.logger.debug("Query embedding created successfully", extra={
                "dimension": len(embedding),
                "model": config.EMBEDDING_MODEL,
                "expected_dimension": config.EMBEDDING_DIMENSION
            })
            
            return embedding
            
        except Exception as e:
            self.logger.error("Query embedding failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__,
                "query": question[:100] + "..." if len(question) > 100 else question
            })
            return []
    
    def search_all_documents(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar content across all available documents using pre-computed vectors"""
        try:
            # Get list of all available documents
            if hasattr(self, 'gcs_manager') and self.gcs_manager:
                available_pdfs = self.gcs_manager.list_pdfs_in_gcs()
                document_ids = [pdf['document_name'] for pdf in available_pdfs]
            else:
                # Fallback to any cached documents
                document_ids = list(self.vectors_cache.keys())
                if not document_ids:
                    self.logger.warning("No documents available for search")
                    return []
            
            self.logger.info("Searching across documents", extra={
                "document_count": len(document_ids),
                "documents": document_ids
            })
            
            all_results = []
            
            # Search each document
            for document_id in document_ids:
                try:
                    # Load vectors if not already cached
                    if document_id not in self.vectors_cache:
                        self._load_precomputed_vectors(document_id)
                    
                    # Perform search in this document
                    if document_id in self.vectors_cache:
                        vectors = self.vectors_cache[document_id]
                        metadata = self.metadata_cache.get(document_id, [])
                        
                        # Ensure query embedding is a numpy array
                        query_embedding_np = np.array(query_embedding).reshape(1, -1)
                        
                        # Calculate cosine similarity
                        similarities = cosine_similarity(query_embedding_np, vectors).flatten()
                        
                        # Get all results with their document context
                        for idx, similarity in enumerate(similarities):
                            if idx < len(metadata):
                                match_metadata = metadata[idx]
                                result = {
                                    'id': match_metadata.get('id', 'Unknown'),
                                    'score': float(similarity),
                                    'metadata': match_metadata,
                                    'pdf_name': match_metadata.get('pdf_name', document_id),
                                    'page_number': match_metadata.get('page_number', 0),
                                    'image_path': match_metadata.get('image_path', ''),
                                    'original_path': match_metadata.get('image_path', ''),
                                    'document_id': document_id
                                }
                                all_results.append(result)
                
                except Exception as e:
                    self.logger.error("Failed to search document", extra={
                        "document_id": document_id,
                        "error_message": str(e),
                        "error_type": type(e).__name__
                    })
                    continue
            
            # Sort all results by similarity score and return top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = all_results[:top_k]
            
            self.logger.info("Multi-document search complete", extra={
                "total_results": len(all_results),
                "top_k_returned": len(top_results),
                "best_score": top_results[0]['score'] if top_results else 0
            })
            
            return top_results
            
        except Exception as e:
            self.logger.error("Multi-document search failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            raise Exception(f"Multi-document search failed: {str(e)}")
    
    def search_similar_documents(self, query_embedding: List[float], document_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using pre-computed vectors with lightweight similarity search"""
        try:
            self.logger.debug("Querying Pinecone index", extra={
                "embedding_dim": len(query_embedding),
                "expected_dim": config.EMBEDDING_DIMENSION,
                "top_k": top_k
            })
            
            # Load vectors if not already cached
            if document_id not in self.vectors_cache:
                self._load_precomputed_vectors(document_id)
            
            # Perform cosine similarity search
            if document_id in self.vectors_cache:
                vectors = self.vectors_cache[document_id]
                metadata = self.metadata_cache.get(document_id, [])
                
                # Ensure query embedding is a numpy array for cosine_similarity
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_embedding_np, vectors).flatten()
                
                # Get top_k indices
                top_k_indices = np.argsort(similarities)[-top_k:]
                
                documents = []
                for idx in top_k_indices:
                    match_score = similarities[idx]
                    match_metadata = metadata[idx]
                    
                    doc = {
                        'id': match_metadata.get('id', 'Unknown'),
                        'score': match_score,
                        'metadata': match_metadata,
                        'pdf_name': match_metadata.get('pdf_name', 'Unknown'),
                        'page_number': match_metadata.get('page_number', 0),
                        'image_path': match_metadata.get('image_path', ''),
                        'original_path': match_metadata.get('image_path', '')
                    }
                    documents.append(doc)
                
                self.logger.debug("Document search successful", extra={
                    "found_docs": len(documents)
                })
                return documents
            else:
                self.logger.error("Vectors not found in cache", extra={
                    "document_id": document_id
                })
                return []
            
        except Exception as e:
            self.logger.error("Document search failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            # Re-raise the exception to stop the retry loop
            raise Exception(f"Lightweight vector search failed: {str(e)}")
    
    def hybrid_search_documents(self, question: str, top_k: int = 3, search_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Enhanced document search using hybrid multi-strategy retrieval"""
        try:
            if not self.use_hybrid_search or not self.hybrid_retriever:
                self.logger.warning("Hybrid search not available, falling back to vector search")
                # Fallback: create embedding and use basic search across all documents
                query_embedding = self.create_query_embedding(question)
                return self.search_all_documents(query_embedding, top_k)
            
            self.logger.info("Starting hybrid multi-strategy search", extra={
                "query": question[:100] + "..." if len(question) > 100 else question
            })
            
            # Use hybrid retrieval system across all available documents
            # Pass search_params to enable intent-directed search
            hybrid_results = self.hybrid_retriever.search_all_documents(question, top_k, search_params=search_params)
            
            # Convert hybrid results to compatible format
            documents = []
            for result in hybrid_results:
                # Use combined_score if available, otherwise fall back to score
                final_score = result.combined_score if hasattr(result, 'combined_score') and result.combined_score > 0 else result.score
                
                # Ensure score is never zero if result was found (minimum threshold)
                if final_score == 0.0 and hasattr(result, 'relevance_signals') and result.relevance_signals:
                    # If score is 0 but we have relevance signals, use a minimum score
                    # This handles cases where semantic similarity is low but document was found
                    final_score = 0.05  # Minimum confidence for found documents
                
                # Log score information for debugging
                if final_score < 0.1:
                    self.logger.debug("Low score result", extra={
                        "pdf_name": result.pdf_name,
                        "page": result.page_number,
                        "score": final_score,
                        "combined_score": getattr(result, 'combined_score', 'N/A'),
                        "base_score": result.score,
                        "strategy": result.search_strategy,
                        "has_vision_text": bool(getattr(result, 'vision_extracted_text', ''))
                    })
                
                doc = {
                    'id': f"{result.pdf_name}_page_{result.page_number}",
                    'score': final_score,  # Use the best available score
                    'metadata': {
                        'pdf_name': result.pdf_name,
                        'page_number': result.page_number,
                        'image_path': result.image_path,
                        'search_strategy': result.search_strategy,
                        'relevance_signals': result.relevance_signals,
                        'base_score': result.score,
                        'combined_score': getattr(result, 'combined_score', result.score)
                    },
                    'pdf_name': result.pdf_name,
                    'page_number': result.page_number,
                    'image_path': result.image_path,
                    'original_path': result.image_path,
                    'text_content': result.text_content,
                    'search_strategy': result.search_strategy,
                    'relevance_signals': result.relevance_signals,
                    'vision_extracted_text': getattr(result, 'vision_extracted_text', ''),  # Pass vision-extracted text
                    'document_id': result.pdf_name  # Include document_id for DLL detection
                }
                documents.append(doc)
            
            self.logger.info("Hybrid search completed", extra={
                "found_docs": len(documents),
                "strategies_used": len(set(doc['search_strategy'] for doc in documents))
            })
            
            return documents
            
        except Exception as e:
            self.logger.error("Hybrid search failed, falling back to vector search", extra={
                "error_message": str(e),
                "error_type": type(e).__name__
            })
            # Fallback to basic vector search
            query_embedding = self.create_query_embedding(question)
            return self.search_similar_documents(query_embedding, top_k)
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI vision API with cost optimization"""
        try:
            # Open and resize image to reduce token cost
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to max dimensions for cost efficiency
                max_size = config.MAX_IMAGE_SIZE  # 1024 from config
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    self.logger.debug("Image resized for cost optimization", extra={
                        "original_size": f"{Image.open(image_path).size}",
                        "new_size": f"{img.size}",
                        "max_size": max_size
                    })
                
                # Save to bytes with compression
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=85, optimize=True)
                img_bytes = img_buffer.getvalue()
                
                # Encode to base64
                encoded_string = base64.b64encode(img_bytes).decode('utf-8')
                
                self.logger.debug("Image optimized for vision API", extra={
                    "original_size_kb": round(os.path.getsize(image_path)/1024, 1),
                    "optimized_size_kb": round(len(img_bytes)/1024, 1),
                    "estimated_tokens": round(len(encoded_string)/4)  # Rough token estimate
                })
                
                return encoded_string
                
        except Exception as e:
            self.logger.error("Image encoding failed", extra={
                "error_message": str(e),
                "error_type": type(e).__name__,
                "path": image_path
            })
            return ""
    
    def get_image_format(self, image_path: str) -> str:
        """Get image format from file extension"""
        try:
            img_path = Path(image_path)
            suffix = img_path.suffix[1:].lower()  # Remove dot and lowercase
            # Map common formats
            format_map = {
                'jpg': 'jpeg',
                'jpeg': 'jpeg', 
                'png': 'png',
                'gif': 'gif',
                'bmp': 'bmp',
                'webp': 'webp'
            }
            return format_map.get(suffix, 'png')  # Default to png
        except Exception:
            return 'png'  # Safe default
    
    def generate_comprehensive_answer(self, question: str, document_analyses: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
        """Generate comprehensive answer with enhanced synthesis and cost tracking"""
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        try:
            if not document_analyses:
                return "I'm sorry, but I couldn't find any relevant documents to answer your question. You might want to check if the right documents are uploaded, or try rephrasing your question in a different way.", token_usage
            
            # Combine all successful analyses
            valid_analyses = [doc for doc in document_analyses if doc.get('analysis_success', False)]
            
            if not valid_analyses:
                return "I found some relevant documents in your collection, but I'm having trouble accessing the content right now. This sometimes happens with complex document formats. Could you try re-uploading your PDF, or let me know if you'd like me to help with something else?", token_usage
            
            # Analyze query type for response structuring
            query_type = self.hybrid_retriever._determine_query_type(question)
            
            # Create enhanced synthesis based on query type
            combined_content = []
            
            # Group content by document and section
            content_by_doc = {}
            for analysis in valid_analyses:
                doc_name = analysis['source'].split(' - ')[0]
                if doc_name not in content_by_doc:
                    content_by_doc[doc_name] = []
                content_by_doc[doc_name].append(analysis)
            
            # Build response structure based on query type
            if query_type == 'procedural':
                synthesis_prompt = self._build_procedural_prompt(question, content_by_doc)
            elif query_type == 'conceptual':
                synthesis_prompt = self._build_conceptual_prompt(question, content_by_doc)
            elif query_type == 'locational':
                synthesis_prompt = self._build_locational_prompt(question, content_by_doc)
            elif query_type == 'comparative':
                synthesis_prompt = self._build_comparative_prompt(question, content_by_doc)
            elif query_type == 'troubleshooting':
                synthesis_prompt = self._build_troubleshooting_prompt(question, content_by_doc)
            else:
                synthesis_prompt = self._build_general_prompt(question, content_by_doc)
            
            # Smart token management
            max_response_tokens = min(800, config.MAX_TOKENS) if hasattr(config, 'MAX_TOKENS') else 800
            
            # Try preferred LLM first, then fall back to a safe model if unsupported
            models_to_try = []
            if config.LLM_MODEL:
                models_to_try.append(config.LLM_MODEL)
            if "gpt-4o-mini" not in models_to_try:
                models_to_try.append("gpt-4o-mini")

            last_error = None
            response = None
            for mdl in models_to_try:
                try:
                    self.logger.info("Attempting answer synthesis", extra={
                        "model": mdl,
                        "max_tokens": max_response_tokens
                    })
                    response = self.openai_client.chat.completions.create(
                        model=mdl,
                        messages=[{"role": "user", "content": synthesis_prompt}],
                        max_tokens=max_response_tokens,
                        temperature=config.TEMPERATURE
                    )
                    # If call succeeded, also update runtime model in case of fallback
                    if mdl != config.LLM_MODEL:
                        self.logger.warning("Fell back to alternate LLM model for synthesis", extra={
                            "from": config.LLM_MODEL,
                            "to": mdl
                        })
                    break
                except Exception as e:
                    last_error = e
                    self.logger.error("LLM synthesis attempt failed", extra={
                        "model": mdl,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    response = None
                    continue

            if response is None:
                raise last_error if last_error else Exception("Unknown synthesis error")
            
            answer = response.choices[0].message.content.strip()
            
            # Check if response was truncated and add completion indicator
            if response.choices[0].finish_reason == 'length':
                if not answer.endswith(('.', '!', '?', ':', ')')):
                    answer += "..."
                answer += "\n\n*[Response truncated for cost optimization. Ask for more details if needed!]*"
            
            # Calculate token usage
            usage = response.usage
            tokens_used = {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            }
            
            return answer, tokens_used
            
        except Exception as e:
            self.logger.error("Answer synthesis failed", extra={"error": str(e)})
            return "I apologize, but I ran into an unexpected issue while analyzing your documents. Could you try asking your question again? If the problem persists, you might want to try rephrasing your question or checking if your documents uploaded properly.", token_usage

    def _build_procedural_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for procedural (how-to) questions with explicit technical detail extraction"""
        return f"""QUESTION: {question}

TASK: Extract EXACT technical details FIRST, then provide a precise, step-by-step implementation guide based STRICTLY on the documentation below.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

CRITICAL EXTRACTION REQUIREMENTS:

1. **Extract Technical Components EXACTLY as written**:
   - DAC names: Extract exact class names (e.g., `Customer`, `SOOrder`)
   - Graph classes: Extract exact names (e.g., `CustomerMaint`, `SOOrderEntry`)
   - Form IDs: Extract exact IDs (e.g., `SM201020`, `CR301000`)
   - Field names: Extract exact names (e.g., `CustomerID`, `OrderNbr`)
   - Event handlers: Extract exact names (e.g., `FieldUpdated`, `RowSelected`)
   - Code elements: Extract exact code snippets, attributes, methods

2. **Validation Rules**:
   - If a technical detail is NOT in documentation, write: `[NOT FOUND IN DOCUMENTATION]`
   - Do NOT infer, guess, or assume technical details
   - Only include what is explicitly stated
   - Use code formatting for all technical names: `Customer`, `SM201020`

3. **Extraction Format**:
   - Use backticks for technical names: `CustomerID`, `FieldUpdated`
   - Extract code snippets exactly as written
   - Preserve exact capitalization and spelling

RESPONSE FORMAT:

## Technical Components Extracted

### DAC/Graph Classes
- **DAC**: `[Extract exact name]` or `[NOT FOUND IN DOCUMENTATION]`
- **Graph Class**: `[Extract exact name]` or `[NOT FOUND IN DOCUMENTATION]`
- **Extension Class**: `[Extract exact name]` or `[NOT FOUND IN DOCUMENTATION]`

### Forms and Screens
- **Form ID**: `[Extract exact ID]` or `[NOT FOUND IN DOCUMENTATION]`
- **Screen Name**: [Extract exact name] or `[NOT FOUND IN DOCUMENTATION]`
- **Navigation Path**: [Extract exact path verbatim] or `[NOT FOUND IN DOCUMENTATION]`

### Fields
- **Field Names**: `[Extract exact names]` or `[NOT FOUND IN DOCUMENTATION]`
- **Field Types**: [Extract types if mentioned] or `[NOT FOUND IN DOCUMENTATION]`
- **Field Attributes**: `[Extract attributes]` or `[NOT FOUND IN DOCUMENTATION]`

### Event Handlers
- **Event Names**: `[Extract exact names]` or `[NOT FOUND IN DOCUMENTATION]`
- **Method Signatures**: [Extract signatures if provided] or `[NOT FOUND IN DOCUMENTATION]`

### Code Elements
- **PXGraph Methods**: `[Extract methods]` or `[NOT FOUND IN DOCUMENTATION]`
- **Attributes**: `[Extract attributes]` or `[NOT FOUND IN DOCUMENTATION]`
- **Code Snippets**: 
```csharp
[Extract exact code if present]
```
OR `[NOT FOUND IN DOCUMENTATION]`

## Step-by-Step Implementation

### Step 1: [Action Title]
**Navigation Path**: [Extract EXACT path from documentation] or `[NOT FOUND IN DOCUMENTATION]`
**Form ID**: `[Extract exact ID]` or `[NOT FOUND IN DOCUMENTATION]`
**Screen Name**: [Extract exact name] or `[NOT FOUND IN DOCUMENTATION]`

**Actions Required**:
1. Navigate to: [Exact menu path from docs]
2. Open form: `[Form ID]` - [Screen name]
3. Locate field: `[Field name]` (if mentioned)
4. Configure: [Exact instructions from documentation]

**Code Implementation** (if mentioned):
```csharp
[Extract exact code snippet]
```
OR `[NOT FOUND IN DOCUMENTATION]`

**Event Handlers** (if mentioned):
- Event: `[Event name]` | Method: `[Method name]`
OR `[NOT FOUND IN DOCUMENTATION]`

### Step 2: [Next Step]
[Continue with same structure, extracting exact details]

## Important Notes
- ✅ This solution is based STRICTLY on the provided documentation
- ⚠️ Items marked `[NOT FOUND IN DOCUMENTATION]` require additional research
- 📋 Follow the exact steps and locations mentioned in the documentation
- 🔍 Verify each technical detail exists in the document content above
- ❌ Do NOT add information not present in the documentation

> 📚 Source: This information comes from [Document names]. All technical details extracted exactly as written in documentation."""

    def _build_conceptual_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for conceptual (what-is) questions with DAC/DLL awareness"""
        return f"""QUESTION: {question}

TASK: Provide a clear explanation of the concept based STRICTLY on the documentation.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

CRITICAL REQUIREMENTS:
1. Identify specific DACs, screens, forms mentioned in documentation (if any)
2. Only use information present in the documentation
3. If information is missing, state that clearly

RESPONSE FORMAT:
Let me explain this concept in Acumatica:

## Overview:
[Clear, concise explanation of the concept based on documentation]

## Related Components (if mentioned):
- **DAC**: [Name if mentioned, or "Not specified"]
- **Screen/Form**: [Form ID if mentioned, or "Not specified"]

## Key Points:
- [Important aspect 1 from documentation]
- [Important aspect 2 from documentation]
[Continue with key points from documentation...]

## Related Concepts:
- [Related concept 1]: [Brief connection from documentation]
- [Related concept 2]: [Brief connection from documentation]

## Practical Application:
[How this concept is used in practice - only if mentioned in docs]

## Important Notes:
- This explanation is based on the provided documentation
- If specific details are missing, additional documentation may be needed

> 📚 Source: This information is compiled from [Document names]."""

    def _load_domain_metadata(self) -> Dict[str, Any]:
        """Load domain.json metadata for enhanced context"""
        try:
            # domain.json is located in knowledge_base/manuals/domain.json
            domain_path = Path(config.LOCAL_BASE_PATH) / "domain.json"
            if domain_path.exists():
                import json
                with open(domain_path, encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning("Failed to load domain metadata", extra={"error": str(e)})
            return {}
    
    def _get_domain_context(self, doc_name: str) -> str:
        """Get domain-specific context for a document (supports both PDF and DLL documents)"""
        domain_metadata = self._load_domain_metadata()
        summaries = domain_metadata.get("document_summaries", {})
        
        # Try exact match first
        if doc_name in summaries:
            doc_info = summaries[doc_name]
        # Try with _DLL suffix if not found (for DLL documents)
        elif doc_name.endswith('_DLL') and doc_name[:-4] in summaries:
            doc_info = summaries[doc_name[:-4]]
        # Try without _DLL suffix if doc_name has it (for DLL documents stored with suffix)
        elif '_DLL' in doc_name:
            base_name = doc_name.replace('_DLL', '')
            if base_name in summaries:
                doc_info = summaries[base_name]
            else:
                return ""
        else:
            return ""
        
        context_parts = []
        
        if doc_info.get("summary"):
            context_parts.append(f"Summary: {doc_info['summary']}")
        
        if doc_info.get("core_topics"):
            topics = ", ".join(doc_info["core_topics"][:5])  # Limit to top 5
            context_parts.append(f"Key Topics: {topics}")
        
        if doc_info.get("forms"):
            forms = ", ".join(doc_info["forms"][:3])  # Limit to top 3
            context_parts.append(f"Relevant Forms: {forms}")
        
        if doc_info.get("key_operations"):
            operations = ", ".join(doc_info["key_operations"][:3])  # Limit to top 3
            context_parts.append(f"Key Operations: {operations}")
        
        return "\n".join(context_parts)
    
    def _format_document_content(self, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Format content from multiple documents for the prompt with domain context"""
        formatted_content = []
        
        for doc_name, analyses in content_by_doc.items():
            # Format document name - use DLL suffix if it's a DLL document
            display_name = doc_name
            if '_DLL' not in doc_name:
                # Check if this is a DLL by looking at metadata
                if analyses and isinstance(analyses[0], dict):
                    metadata = analyses[0].get('metadata', {})
                    if metadata.get('section_type') in ['dll_diagram', 'dll_text']:
                        dll_name = metadata.get('dll_name', '')
                        if dll_name:
                            display_name = f"{dll_name}_DLL"
            
            doc_content = [f"=== {display_name} ==="]
            
            # Add domain context if available (try both with and without _DLL suffix)
            domain_context = self._get_domain_context(doc_name)
            if not domain_context and '_DLL' not in doc_name:
                # Try with _DLL suffix for DLL documents
                domain_context = self._get_domain_context(f"{doc_name}_DLL")
            elif not domain_context and '_DLL' in doc_name:
                # Try without _DLL suffix
                domain_context = self._get_domain_context(doc_name.replace('_DLL', ''))
            
            if domain_context:
                doc_content.append(f"Document Context:\n{domain_context}\n")
            
            for analysis in analyses:
                if analysis.get('vision_content'):
                    doc_content.append(f"Content: {analysis['vision_content']}")
            
            formatted_content.append("\n".join(doc_content))
        
        return "\n\n".join(formatted_content)
    
    def _load_precomputed_vectors(self, document_id: str):
        """Load pre-computed vectors from local storage"""
        try:
            if document_id in self.vectors_cache:
                return  # Already loaded
            
            from pathlib import Path
            vectors_file = Path(config.LOCAL_BASE_PATH) / document_id / "vectors" / "vectors.json"
            metadata_file = Path(config.LOCAL_BASE_PATH) / document_id / "metadata" / "metadata.json"
            
            if vectors_file.exists():
                with open(vectors_file, 'r', encoding='utf-8') as f:
                    vectors_data = json.load(f)
                    self.vectors_cache[document_id] = np.array(vectors_data.get('embeddings', []))
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(metadata, list):
                        self.metadata_cache[document_id] = metadata
                    else:
                        # Convert dict to list format if needed
                        pages = metadata.get('pages', [])
                        self.metadata_cache[document_id] = pages if pages else [metadata]
        except Exception as e:
            self.logger.error("Failed to load pre-computed vectors", extra={
                "document_id": document_id,
                "error": str(e)
            })
    
    def _create_temporary_sections(self, image_path: str, question: str) -> Optional[Dict[str, Any]]:
        """Create temporary sections for focused queries (stub - returns None to skip section extraction)"""
        # This is a cost-saving measure - skip section extraction for now
        return None
    
    def _select_best_section_from_temp(self, question: str, temp_section_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select best section from temporary data (stub)"""
        return None
    
    def _cleanup_temporary_sections(self, temp_section_data: Dict[str, Any]):
        """Clean up temporary sections (stub)"""
        pass
    
    def _build_locational_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for locational (where-is) questions with exact navigation"""
        return f"""QUESTION: {question}

TASK: Provide clear navigation instructions and location information based STRICTLY on documentation.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

CRITICAL REQUIREMENTS:
1. Provide EXACT navigation path from documentation
2. Include specific form IDs and screen names if mentioned
3. Only use information present in documentation
4. If path is not specified, state that clearly

RESPONSE FORMAT:
Let me help you find that in Acumatica:

## Location:
**Navigation Path**: [Exact menu path from documentation, e.g., "System > Customization > Customization Projects"]
**Screen/Form**: [Form ID and name if mentioned in documentation]
**DAC**: [DAC name if mentioned, or "Not specified"]

## Steps to Access:
1. **Navigate to**: [First menu level from documentation]
   - [Specific instruction from documentation]

2. **Select**: [Next menu option from documentation]
   - [Detailed step from documentation]

## Additional Information:
[Any related details about the location from documentation]

## Important Notes:
- Navigation path is based on the provided documentation
- If specific path is not mentioned, you may need to consult additional documentation

> 📚 Source: This information comes from [Document names]."""
    
    def _build_comparative_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for comparative questions"""
        return f"""QUESTION: {question}

TASK: Compare and contrast the concepts or features mentioned.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

RESPONSE FORMAT:
Let me compare these for you:

## Comparison:
[Clear comparison of the concepts]

## Key Differences:
- [Difference 1]
- [Difference 2]

## Similarities:
- [Similarity 1]
- [Similarity 2]

> 📚 Source: This information comes from [Document names]."""
    
    def _build_troubleshooting_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for troubleshooting questions with DAC/DLL awareness"""
        return f"""QUESTION: {question}

TASK: Provide troubleshooting steps and solutions based STRICTLY on the documentation.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

CRITICAL REQUIREMENTS:
1. Identify specific screens/forms mentioned in documentation
2. Provide exact navigation paths if mentioned
3. Only use information from documentation
4. If information is missing, state that clearly

RESPONSE FORMAT:
Let me help you troubleshoot this issue:

## Problem:
[Description of the issue from documentation]

## Affected Components (if mentioned):
- **Screen/Form**: [Form ID if mentioned, or "Not specified"]
- **DAC**: [DAC name if mentioned, or "Not specified"]

## Solution Steps:
1. **Navigate to**: [Exact location if mentioned in docs]
   - [Specific troubleshooting step from documentation]

2. **Action**: [Next step from documentation]
   - [Detailed instructions from documentation]

## Prevention:
[How to avoid this issue in the future - only if mentioned in docs]

## Important Notes:
- These steps are based on the provided documentation
- If specific details are missing, additional documentation may be needed

> 📚 Source: This information comes from [Document names]."""
    
    def _build_general_prompt(self, question: str, content_by_doc: Dict[str, List[Dict]]) -> str:
        """Build prompt for general questions with DAC/DLL awareness"""
        return f"""QUESTION: {question}

TASK: Provide a comprehensive answer based STRICTLY on the documentation below.

DOCUMENT CONTENT:
{self._format_document_content(content_by_doc)}

CRITICAL REQUIREMENTS:
1. Identify specific DACs, screens, forms mentioned in documentation (if any)
2. Provide navigation paths if mentioned in documentation
3. Only use information present in the documentation
4. If specific details are missing, state that clearly

RESPONSE FORMAT:
Let me help you with that:

## Answer:
[Comprehensive answer to the question based STRICTLY on documentation]

## Required Components (if mentioned in docs):
- **DAC**: [Name if mentioned, or "Not specified"]
- **Screen/Form**: [Form ID if mentioned, or "Not specified"]

## Step-by-Step Instructions (if applicable):
[Only if procedural steps are in documentation]

## Details:
[Additional relevant information from documentation]

## Important Notes:
- This information is based on the provided documentation
- If specific details are missing, additional documentation may be needed

> 📚 Source: This information comes from [Document names]."""
    
    async def ask_question(self, question: str, top_k: int = 3, max_retries: int = 0, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ENHANCED QUESTION ANSWERING PIPELINE with Pure Dynamic Intelligence and Cost Tracking"""
        for attempt in range(max_retries + 1):
            try:
                self.logger.info("Starting enhanced question processing", extra={
                    "query": question[:100] + "..." if len(question) > 100 else question,
                    "attempt": attempt + 1,
                    "max_attempts": max_retries + 1
                })
                
                # Initialize cost tracking
                total_vision_tokens = 0
                total_prompt_tokens = 0
                total_completion_tokens = 0
                
                # Step 1: Enhanced multi-strategy document search with intent parameters
                similar_docs = self.hybrid_search_documents(question, top_k, search_params=search_params)
                if search_params:
                    self.logger.info("Using intent-based search parameters", extra={
                        "target_docs": search_params.get("target_directories", []),
                        "search_focus": search_params.get("search_focus", "general")
                    })
                if not similar_docs:
                    error_msg = "No similar documents found"
                    self.logger.warning("No similar documents found")
                    return {
                        "answer": "I couldn't find any relevant documents for your question.",
                        "sources": [],
                        "error": error_msg,
                        "cost_metrics": config.calculate_cost_inr()
                    }
            
                self.logger.info("Found relevant documents", extra={
                    "document_count": len(similar_docs)
                })
                
                # Step 2: Enhanced document analysis with dynamic path resolution
                document_analyses = []
                successful_analyses = 0
                
                # Process only the top_k documents for efficiency
                docs_to_process = similar_docs[:top_k]
                
                for i, doc in enumerate(docs_to_process, 1):
                    self.logger.info("Processing document", extra={
                        "index": f"{i}/{len(docs_to_process)}",
                        "pdf_name": doc['pdf_name'],
                        "page_number": doc['page_number'],
                        "has_pre_extracted_vision": bool(doc.get('vision_extracted_text', ''))
                    })
                    
                    # OPTIMIZATION: Check if vision text was already extracted by hybrid_retriever
                    pre_extracted_vision_text = doc.get('vision_extracted_text', '')
                    
                    if pre_extracted_vision_text and len(pre_extracted_vision_text.strip()) > 50:
                        # Use pre-extracted vision text - no need to call Vision API again
                        self.logger.info("Using pre-extracted vision text from hybrid_retriever", extra={
                            "pdf_name": doc['pdf_name'],
                            "page_number": doc['page_number'],
                            "text_length": len(pre_extracted_vision_text),
                            "cost_saved": "Vision API call skipped"
                        })
                        
                        successful_analyses += 1
                        
                        # No token usage for pre-extracted text (already counted in hybrid_retriever)
                        # Format source name - include DLL suffix if it's a DLL document
                        pdf_name = doc['pdf_name']
                        doc_id = doc.get('document_id', '')
                        if '_DLL' in doc_id and '_DLL' not in pdf_name:
                            source_name = f"{pdf_name}_DLL"
                        else:
                            source_name = pdf_name
                        
                        # For DLL documents, use more descriptive source format
                        if '_DLL' in doc_id or '_DLL' in pdf_name:
                            section_type = doc.get('metadata', {}).get('section_type', 'page')
                            if section_type == 'dll_diagram':
                                source_label = f"{source_name} - Class Diagram"
                            elif section_type == 'dll_text':
                                source_label = f"{source_name} - Text Content"
                            else:
                                source_label = f"{source_name} - Page {doc['page_number']}"
                        else:
                            source_label = f"{source_name} - Page {doc['page_number']}"
                        
                        document_analyses.append({
                            'source': source_label,
                            'document_name': source_name,
                            'pdf_name': pdf_name,
                            'dll_name': doc.get('metadata', {}).get('dll_name', ''),
                            'vision_content': pre_extracted_vision_text,
                            'score': doc['score'],
                            'metadata': doc['metadata'],
                            'analysis_success': True,
                            'strategies_used': ['hybrid_retriever_vision'],  # Indicate source
                            'token_usage': {
                                'vision_tokens': 0,  # Already counted in hybrid_retriever
                                'prompt_tokens': 0,
                                'completion_tokens': 0
                            },
                            'pre_extracted': True  # Flag to indicate this was pre-extracted
                        })
                        
                        self.logger.info("Document analysis successful (pre-extracted)", extra={
                            "pdf_name": doc['pdf_name'],
                            "page_number": doc['page_number']
                        })
                        continue  # Skip Vision API call
                    
                    # Fallback: Call Vision API if text wasn't pre-extracted
                    # Get image path
                    resolved_path = self.get_image_path(
                        doc['pdf_name'], 
                        doc['page_number']
                    )
                    
                    if not resolved_path:
                        self.logger.warning("Could not download image", extra={
                            "pdf_name": doc['pdf_name'],
                            "page_number": doc['page_number']
                        })
                        continue
                    
                    # Enhanced analysis (only if not pre-extracted)
                    context = f"Document: {doc['pdf_name']}, Page: {doc['page_number']}"
                    analysis_result = await self.analyze_document_image_enhanced(resolved_path, question, context)
                    
                    if analysis_result["success"]:
                        successful_analyses += 1
                        
                        # Aggregate token usage
                        total_vision_tokens += analysis_result["token_usage"]["vision_tokens"]
                        total_prompt_tokens += analysis_result["token_usage"]["prompt_tokens"]
                        total_completion_tokens += analysis_result["token_usage"]["completion_tokens"]
                        
                        # Format source name - include DLL suffix if it's a DLL document
                        pdf_name = doc['pdf_name']
                        doc_id = doc.get('document_id', '')
                        if '_DLL' in doc_id and '_DLL' not in pdf_name:
                            source_name = f"{pdf_name}_DLL"
                        else:
                            source_name = pdf_name
                        
                        # For DLL documents, use more descriptive source format
                        if '_DLL' in doc_id or '_DLL' in pdf_name:
                            section_type = doc.get('metadata', {}).get('section_type', 'page')
                            if section_type == 'dll_diagram':
                                source_label = f"{source_name} - Class Diagram"
                            elif section_type == 'dll_text':
                                source_label = f"{source_name} - Text Content"
                            else:
                                source_label = f"{source_name} - Page {doc['page_number']}"
                        else:
                            source_label = f"{source_name} - Page {doc['page_number']}"
                        
                        document_analyses.append({
                            'source': source_label,
                            'document_name': source_name,
                            'pdf_name': pdf_name,
                            'dll_name': doc.get('metadata', {}).get('dll_name', ''),
                            'vision_content': analysis_result["vision_analysis"],
                            'score': doc['score'],
                            'metadata': doc['metadata'],
                            'analysis_success': True,
                            'strategies_used': analysis_result["strategies_used"],
                            'token_usage': analysis_result["token_usage"],
                            'pre_extracted': False  # Flag to indicate Vision API was called
                        })
                        
                        self.logger.info("Document analysis successful (Vision API)", extra={
                            "strategies": len(analysis_result["strategies_used"]),
                            "prompt_tokens": analysis_result["token_usage"]["prompt_tokens"],
                            "completion_tokens": analysis_result["token_usage"]["completion_tokens"]
                        })
                    else:
                        self.logger.warning("Document analysis failed", extra={
                            "error_message": analysis_result.get("error")
                        })
                
                # Step 3: Generate comprehensive answer
                self.logger.info("Generating answer", extra={
                    "successful_analyses": successful_analyses,
                    "total_processed": len(docs_to_process),
                    "total_found": len(similar_docs)
                })
                
                final_answer, synthesis_tokens = self.generate_comprehensive_answer(question, document_analyses)
                
                # Add synthesis tokens to total
                total_prompt_tokens += synthesis_tokens["input_tokens"]
                total_completion_tokens += synthesis_tokens["output_tokens"]
                
                # Dynamic answer quality assessment
                answer_quality = self._assess_answer_quality(final_answer, document_analyses, question)
                
                # Calculate total cost in INR
                cost_metrics = config.calculate_cost_inr(
                    input_tokens=total_prompt_tokens,
                    output_tokens=total_completion_tokens,
                    vision_tokens=total_vision_tokens
                )
                
                result = {
                    "answer": final_answer,
                    "sources": self._prepare_sources(similar_docs, document_analyses),
                    "total_documents_analyzed": successful_analyses,
                    "total_documents_processed": len(docs_to_process),
                    "total_documents_found": len(similar_docs),
                    "method": "enhanced_vdr",
                    "success_rate": f"{successful_analyses}/{len(docs_to_process)}",
                    "response_type": self._determine_response_type(question),
                    "answer_quality": answer_quality,
                    "cost_metrics": cost_metrics
                }
                
                self.logger.info("Question answering complete", extra={
                    "success_rate": result["success_rate"],
                    "total_analyzed": successful_analyses,
                    "total_cost_inr": cost_metrics["total_cost_inr"],
                    "total_tokens": {
                        "vision": total_vision_tokens,
                        "prompt": total_prompt_tokens,
                        "completion": total_completion_tokens
                    }
                })
                
                return result
                
            except Exception as e:
                self.logger.error("Question processing failed", extra={
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "attempt": attempt + 1,
                    "max_attempts": max_retries + 1
                })
                
                if attempt >= max_retries:
                    return {
                        "answer": "I encountered an error while processing your question. Please try again.",
                        "sources": [],
                        "error": str(e),
                        "cost_metrics": config.calculate_cost_inr()
                    }
                
                wait_time = 2 ** attempt
                self.logger.info("Retrying question processing", extra={
                    "attempt": attempt + 1,
                    "wait_time": wait_time
                })
                await asyncio.sleep(wait_time)
        
        return {
            "answer": "Maximum retries exceeded",
            "sources": [],
            "error": "Maximum retries exceeded",
            "cost_metrics": config.calculate_cost_inr()
        }

    def _determine_response_type(self, question: str) -> str:
        """Determine the type of response based on the question"""
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['steps', 'process', 'how to', 'procedure', 'workflow']):
            return "Step-by-Step Process"
        elif any(keyword in question_lower for keyword in ['explain', 'describe', 'what is', 'details']):
            return "Detailed Information"
        return "Quick Answer"
    
    def _assess_answer_quality(self, answer: str, document_analyses: List[Dict], question: str) -> str:
        """Dynamically assess answer quality based on content analysis and relevance"""
        try:
            quality_score = 0.0
            
            # Factor 1: Answer completeness and structure
            answer_lower = answer.lower()
            question_lower = question.lower()
            
            # Check for complete procedural answers (if it's a procedural question)
            if any(word in question_lower for word in ['how', 'steps', 'process', 'create', 'setup']):
                if 'steps:' in answer_lower or 'step' in answer_lower:
                    quality_score += 2.0  # Strong procedural structure
                if 'prerequisites' in answer_lower or 'requirements' in answer_lower:
                    quality_score += 1.0  # Shows completeness
                if 'verification' in answer_lower or 'confirm' in answer_lower:
                    quality_score += 1.0  # Shows thorough process
            
            # Factor 2: Specific terminology and technical accuracy
            query_terms = set(question_lower.split())
            answer_terms = set(answer_lower.split())
            term_overlap = len(query_terms & answer_terms) / max(len(query_terms), 1)
            quality_score += term_overlap * 2.0
            
            # Factor 3: Source consistency and confidence
            if document_analyses:
                high_vision_quality = sum(1 for analysis in document_analyses 
                                        if analysis.get('vision_content') and 
                                        len(analysis['vision_content']) > 100)
                quality_score += min(high_vision_quality * 0.5, 2.0)
                
                # Check for multiple consistent sources
                source_count = len([a for a in document_analyses if a.get('analysis_success')])
                if source_count >= 3:
                    quality_score += 1.5
                elif source_count >= 2:
                    quality_score += 1.0
            
            # Factor 4: Answer depth and specificity
            if len(answer) > 200:  # Detailed answer
                quality_score += 1.0
            if '•' in answer or '-' in answer or 'Step' in answer:  # Well-structured
                quality_score += 1.0
            
            # Factor 5: Exact phrase and concept matching
            # Check if answer contains specific phrases from the question
            question_phrases = []
            words = question_lower.split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if phrase in answer_lower:
                    quality_score += 0.5
            
            # Quality classification based on score
            if quality_score >= 7.0:
                return "excellent"
            elif quality_score >= 5.5:
                return "very_good"
            elif quality_score >= 4.0:
                return "good"
            elif quality_score >= 2.5:
                return "fair"
            elif quality_score >= 1.0:
                return "poor"
            else:
                return "uncertain"
                
        except Exception as e:
            self.logger.warning("Answer quality assessment failed", extra={"error": str(e)})
            return "good"  # Safe default

    def _normalize_confidence_score(self, raw_score: float, has_vision_content: bool, analysis_success: bool) -> float:
        """
        Normalize and enhance confidence scores for better accuracy.
        Handles edge cases where raw_score might be very low or zero.
        
        IMPROVED: Properly handles cosine similarity scores which are typically in [0, 1] range
        for normalized embeddings, but can be very low for semantic matches.
        """
        # Log raw score for debugging
        if raw_score < 0.1:
            self.logger.debug("Very low raw score detected", extra={
                "raw_score": raw_score,
                "has_vision": has_vision_content,
                "analysis_success": analysis_success
            })
        
        # Base normalization: ensure score is between 0 and 1
        # Cosine similarity with normalized embeddings is typically [0, 1], but can be negative
        if raw_score < 0:
            # If negative (rare but possible with unnormalized embeddings), normalize to [0, 1]
            normalized = (raw_score + 1) / 2
        else:
            normalized = max(0.0, min(1.0, raw_score))
        
        # CRITICAL FIX: If score is 0.0 but document was found, apply minimum confidence
        # This handles cases where semantic similarity is very low but document is still relevant
        if normalized == 0.0:
            # If document was found through search, it should have minimum confidence
            # This prevents 0.00 scores from appearing in results
            normalized = 0.05  # Minimum 5% confidence for found documents
            self.logger.debug("Applied minimum confidence for zero score", extra={
                "raw_score": raw_score,
                "normalized": normalized
            })
        elif normalized < 0.01 and raw_score > 0:
            # Very low but non-zero score - ensure visibility
            normalized = max(0.01, normalized)
        
        # Apply non-linear scaling to better represent confidence levels
        # Low scores (< 0.3) are common for semantic search - scale them appropriately
        if normalized < 0.3:
            # Low confidence range: apply gentle scaling to make differences visible
            # Scale [0, 0.3] to [0.05, 0.4] to preserve relative differences
            normalized = 0.05 + (normalized / 0.3) * 0.35
        elif normalized < 0.7:
            # Medium confidence: keep as-is (already meaningful)
            pass
        else:
            # High confidence: slight boost
            normalized = min(1.0, normalized * 1.05)
        
        # Boost confidence if vision extraction was successful
        if has_vision_content and analysis_success:
            # Vision extraction adds credibility - indicates document was actually analyzed
            normalized = min(1.0, normalized * 1.15)
        
        # Final validation: ensure score is never 0 if document was found
        if normalized == 0.0:
            normalized = 0.05  # Absolute minimum
        
        # Log low confidence for analysis
        if normalized < 0.1:
            self.logger.debug("Low confidence score after normalization", extra={
                "normalized": normalized,
                "raw_score": raw_score,
                "has_vision": has_vision_content,
                "note": "Low semantic similarity - document may still be relevant"
            })
        
        return round(normalized, 4)  # More precision for low scores
    
    def _prepare_sources(self, similar_docs: List[Dict], document_analyses: List[Dict]) -> List[Dict]:
        """Prepare source references for the response with enhanced confidence scores"""
        sources = []
        for i, doc in enumerate(similar_docs):
            was_analyzed = i < len(document_analyses) and document_analyses[i].get('analysis_success', False)
            content_preview = ""
            has_vision_content = False
            
            if was_analyzed and i < len(document_analyses):
                analysis = document_analyses[i]
                if analysis.get('vision_content'):
                    content = analysis['vision_content'][:300]
                    content_preview = content.replace('\n', ' ')[:200] + "..."
                    has_vision_content = len(analysis['vision_content']) > 50
            
            # Normalize confidence score
            # Try to get the best available score (combined_score from metadata, or base score)
            metadata = doc.get('metadata', {})
            raw_score = metadata.get('combined_score', doc.get('score', 0.0))
            
            # If score is still 0 or very low, log for debugging
            if raw_score < 0.01:
                self.logger.warning("Very low or zero score detected in source preparation", extra={
                    "pdf_name": doc.get('pdf_name', 'Unknown'),
                    "page": doc.get('page_number', 0),
                    "raw_score": raw_score,
                    "base_score": doc.get('score', 0.0),
                    "combined_score": metadata.get('combined_score', 'N/A'),
                    "strategy": metadata.get('search_strategy', 'unknown'),
                    "relevance_signals": metadata.get('relevance_signals', {})
                })
            
            normalized_score = self._normalize_confidence_score(raw_score, has_vision_content, was_analyzed)
            
            # CRITICAL: Ensure we never have 0.00 scores - if document was found, it must have minimum confidence
            if normalized_score <= 0.0:
                self.logger.warning("Normalized score is still 0.00, applying minimum confidence", extra={
                    "pdf_name": doc.get('pdf_name', 'Unknown'),
                    "page": doc.get('page_number', 0),
                    "raw_score": raw_score,
                    "normalized_before": normalized_score
                })
                normalized_score = 0.05  # Absolute minimum for found documents
            
            # Include DLL information if available
            pdf_name = doc['pdf_name']
            doc_id = doc.get('document_id', '')
            doc_metadata = doc.get('metadata', {})
            
            # Determine document name - use DLL suffix if it's a DLL document
            if '_DLL' in doc_id and '_DLL' not in pdf_name:
                document_name = f"{pdf_name}_DLL"
            elif '_DLL' in pdf_name:
                document_name = pdf_name
            else:
                document_name = pdf_name
            
            sources.append({
                'document': document_name,
                'document_name': document_name,  # Add explicit document_name field
                'pdf_name': pdf_name,  # Keep original pdf_name
                'dll_name': doc_metadata.get('dll_name', ''),
                'page': doc['page_number'],
                'similarity_score': normalized_score,  # Use normalized score (guaranteed > 0)
                'confidence': normalized_score,  # Also add 'confidence' field for markdown formatter compatibility
                'raw_score': raw_score,  # Keep original for reference
                'image_path': doc.get('image_path'),
                'image_exists': doc.get('image_exists', False),
                'analyzed': was_analyzed,
                'content_preview': content_preview if content_preview else "Document located in database",
                'relevance_level': "High" if normalized_score > 0.7 else "Medium" if normalized_score > 0.5 else "Low"
            })
        
        return sources
    
    async def interactive_mode(self):
        """Start enhanced interactive question-answering session"""
        print(f"\n{'='*70}")
        print(f"🎯 ENHANCED VDR PDF Q&A - Pure Dynamic Intelligence")
        print(f"{'='*70}")
        print(f"🚀 Advanced Features:")
        print(f"   • Dynamic path resolution")
        print(f"   • Enhanced vision analysis")
        print(f"   • OCR text extraction backup")
        print(f"   • Multi-strategy processing")
        print(f"   • Intelligent error recovery")
        print(f"{'='*70}")
        print(f"Ask questions about your PDF documents!")
        print(f"Type 'quit', 'exit', or 'q' to stop.")
        print(f"{'='*70}\n")
        
        while True:
            try:
                question = input("\n💭 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thank you for using Enhanced VDR PDF Q&A!")
                    break
                
                result = await self.ask_question(question)
                
                print(f"\n🤖 Enhanced Answer:")
                print(f"{result['answer']}")
                
                if result.get('sources'):
                    print(f"\n📚 Sources (Success Rate: {result.get('success_rate', 'N/A')}):")
                    for i, source in enumerate(result['sources'], 1):
                        score = source.get('similarity_score', 0)
                        print(f"  {i}. {source['document']} - Page {source['page']} (Score: {score:.3f})")
                
                print(f"\n{'-'*60}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thank you for using Enhanced VDR PDF Q&A!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue

# VDRInferencer class is ready for use

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced VDR-based PDF Q&A System")
    parser.add_argument("--question", type=str, help="Ask a specific question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve (default: 3)")
    
    args = parser.parse_args()
    
    try:
        config.validate_config()
        inferencer = VDRInferencer()
        
        if args.question:
            result = await inferencer.ask_question(args.question, args.top_k)
            
            print(f"\n🤖 Enhanced Answer:")
            print(f"{result['answer']}")
            
            # Display cost metrics in INR
            if result.get('cost_metrics'):
                cost = result['cost_metrics']
                print(f"\n💰 Cost Breakdown (Indian Rupees):")
                print(f"   🔍 Vision Analysis: ₹{cost.get('vision_cost_inr', 0):.4f} ({cost.get('vision_tokens', 0):,} tokens)")
                print(f"   📝 Input Processing: ₹{cost.get('input_cost_inr', 0):.4f} ({cost.get('input_tokens', 0):,} tokens)")
                print(f"   🤖 Answer Generation: ₹{cost.get('output_cost_inr', 0):.4f} ({cost.get('output_tokens', 0):,} tokens)")
                print(f"   💸 Total Cost: ₹{cost.get('total_cost_inr', 0):.4f} (@ ₹{cost.get('usd_to_inr_rate', 83.5)}/USD)")
            
            if result.get('sources'):
                print(f"\n📚 Sources (Success Rate: {result.get('success_rate', 'N/A')}):")
                for i, source in enumerate(result['sources'], 1):
                    score = source.get('similarity_score', 0)
                    print(f"  {i}. {source['document']} - Page {source['page']} (Score: {score:.3f})")
        else:
            await inferencer.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 
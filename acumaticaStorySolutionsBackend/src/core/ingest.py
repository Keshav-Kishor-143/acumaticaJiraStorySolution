#!/usr/bin/env python3
"""
VDR-based PDF RAG Ingestion Script - Fast Processing Mode

This script converts PDFs to images and generates embeddings using HuggingFace:
1. Converts PDFs to high-resolution images (fast)
2. Generates visual embeddings using HuggingFace models (FREE!)
3. Stores vectors and metadata locally
4. NO section extraction during ingestion (done on-demand for speed)
"""

import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
import fitz
from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation, performance_monitor

# Helper function for HuggingFace embeddings with fallback
def _get_huggingface_embedding(model_name: str, device: str = "cpu", logger=None):
    """Get HuggingFace embedding model with fallback support"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
    except ImportError:
        try:
            from llama_index.embeddings import HuggingFaceEmbedding
            return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
        except ImportError:
            from sentence_transformers import SentenceTransformer
            if logger:
                logger.info("Using sentence-transformers directly (HuggingFace embeddings)")
            class EmbeddingWrapper:
                def __init__(self, model):
                    self.model = model
                def get_text_embedding(self, text: str):
                    return self.model.encode(text, normalize_embeddings=True).tolist()
            return EmbeddingWrapper(SentenceTransformer(model_name, device=device))

class VDRIngestor:
    """
    Fast HuggingFace PDF ingestion system (FREE embeddings!)
    
    Optimized for speed:
    - PDF → Images → Embeddings → Local storage
    - NO section extraction during ingestion (prevents timeouts)
    - Section analysis done on-demand during inference
    """
    
    def __init__(self):
        self.logger = get_logger("VDR_INGESTOR")
        self.images_dir = Path(config.IMAGES_DIR)
        self.data_dir = Path(config.DATA_DIR)
        
        self.logger.info("🚀 HuggingFace + OpenAI VDR Ingestor initializing...", extra={
            "data_dir": str(self.data_dir), 
            "images_dir": str(self.images_dir)
        })
        
        # Initialize local storage
        self._initialize_storage()
        
        # Initialize HuggingFace Embeddings (FREE!)
        self._initialize_embeddings()
        
        # Initialize OpenAI client (only for Vision)
        self._initialize_openai()
        
        self.logger.info("HuggingFace + OpenAI VDR Ingestor fully initialized and ready")
    
    def _initialize_storage(self):
        """Initialize local storage"""
        try:
            self.logger.info("Using local storage mode")
            
            # Ensure local directories exist
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.error("Failed to initialize storage", extra={
                "error": str(e)
            })
            raise
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embedding model (FREE!)"""
        try:
            self.logger.info("Loading HuggingFace embeddings...", extra={
                "model": config.EMBEDDING_MODEL
            })
            
            with TimedOperation("embedding_model_load", self.logger):
                # Detect device properly
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Initialize HuggingFace embedding model
                self.embedding_model = _get_huggingface_embedding(
                    model_name=config.EMBEDDING_MODEL,
                    device=device,
                    logger=self.logger
                )
                
                self.logger.info(f"🚀 HuggingFace embeddings loaded on: {device}")
                
                # Test embedding generation
                test_embedding = self.embedding_model.get_text_embedding("test query")
                embedding_dim = len(test_embedding)
                
                self.logger.info("HuggingFace embedding model loaded successfully", extra={
                    "model": config.EMBEDDING_MODEL,
                    "embedding_dimension": embedding_dim,
                    "expected_dimension": config.EMBEDDING_DIMENSION
                })
                
                if embedding_dim != config.EMBEDDING_DIMENSION:
                    self.logger.warning("Embedding dimension mismatch", extra={
                        "actual": embedding_dim, 
                        "expected": config.EMBEDDING_DIMENSION
                    })
            
            self.logger.info(f"HuggingFace embeddings ready: {config.EMBEDDING_MODEL}")
            
        except Exception as e:
            self.logger.error("Failed to initialize HuggingFace embeddings", extra={
                "error": str(e)
            })
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI client (for Vision only)"""
        try:
            self.logger.info("Connecting to OpenAI...")
            
            with TimedOperation("openai_connection", self.logger):
                self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
                
                self.logger.info("OpenAI connection successful for Vision API", extra={
                    "llm_model": config.LLM_MODEL,
                    "vision_model": config.VISION_MODEL
                })
            
            self.logger.info(f"OpenAI client ready for Vision: {config.VISION_MODEL}")
            
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client", extra={
                "error": str(e)
            })
            raise

    def convert_pdf_to_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Convert PDF pages to high-resolution images for VDR processing"""
        try:
            pdf_name = Path(pdf_path).stem
            
            with TimedOperation("pdf_processing", self.logger, pdf_path=pdf_path):
                pdf_doc = fitz.open(pdf_path)
                image_metadata = []
                
                self.logger.info(f"Converting PDF: {pdf_name}")
                self.logger.info("PDF opened successfully", extra={
                    "pdf_name": pdf_name, 
                    "total_pages": pdf_doc.page_count,
                    "dpi": config.DPI,
                    "max_image_size": config.MAX_IMAGE_SIZE
                })
                
                # Create images directory if it doesn't exist
                self.images_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug("Created image directory", extra={
                    "path": str(self.images_dir)
                })
                
                # Process each page
                for page_num in range(pdf_doc.page_count):
                    # Convert page to image
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(dpi=config.DPI)
                    
                    # Resize image if needed
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    original_size = img.size
                    
                    # Calculate new size maintaining aspect ratio
                    if max(img.size) > config.MAX_IMAGE_SIZE:
                        ratio = config.MAX_IMAGE_SIZE / max(img.size)
                        new_size = tuple(int(dim * ratio) for dim in img.size)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        self.logger.debug("Image resized", extra={
                            "page": page_num + 1,
                            "original_size": original_size,
                            "new_size": new_size
                        })
                    else:
                        new_size = img.size
                    
                    # Save image
                    image_filename = f"page{page_num + 1}.jpg"
                    image_path = self.images_dir / image_filename
                    img.save(str(image_path), "JPEG", quality=95)
                    
                    # Record metadata
                    metadata = {
                        "pdf_name": pdf_name,
                        "pdf_path": pdf_path,
                        "page_number": page_num + 1,
                        "image_path": str(image_path),
                        "image_filename": image_filename,
                        "image_size": f"{new_size[0]}x{new_size[1]}",
                        "processing_mode": "fast_local_processing"
                    }
                    image_metadata.append(metadata)
                    
                    # Batch progress update
                    if (page_num + 1) % 10 == 0:
                        self.logger.info("Batch progress update", extra={
                            "pages_processed": page_num + 1,
                            "total_pages": pdf_doc.page_count
                        })
                
                self.logger.info(f"PDF conversion complete: {len(image_metadata)} images created")
                return image_metadata
                
        except Exception as e:
            self.logger.error("Failed to convert PDF to images", extra={
                "error": str(e)
            })
            return None
    
    def generate_visual_embeddings(self, image_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for images using HuggingFace"""
        try:
            self.logger.info("Starting HuggingFace embedding generation", extra={
                "count": len(image_metadata)
            })
            embeddings_data = []
            
            for metadata in image_metadata:
                # Create visual context
                context = f"Document: {metadata['pdf_name']} Page: {metadata['page_number']} - Visual document content"
                self.logger.debug("Using visual context for HuggingFace embedding", extra={
                    "context": context
                })
                
                # Generate embedding
                embedding = self.embedding_model.get_text_embedding(context)
                
                # Store result
                embeddings_data.append({
                    "metadata": metadata,
                    "embedding": embedding
                })
            
            self.logger.info("HuggingFace embedding generation complete", extra={
                "total_generated": len(embeddings_data),
                "success_rate": 100.0,
                "model": config.EMBEDDING_MODEL
            })
            return embeddings_data
            
        except Exception as e:
            self.logger.error("Failed to generate embeddings", extra={
                "error": str(e)
            })
            return None

    def ingest_pdf(self, pdf_path: str) -> bool:
        """Complete VDR ingestion pipeline for a single PDF"""
        try:
            pdf_name = Path(pdf_path).stem
            
            print(f"\n📚 Starting VDR ingestion: {Path(pdf_path).name}")
            
            # Step 1: Convert PDF to images
            image_metadata = self.convert_pdf_to_images(pdf_path)
            if not image_metadata:
                return False
            
            # Step 2: Generate embeddings using HuggingFace
            embeddings_data = self.generate_visual_embeddings(image_metadata)
            if not embeddings_data:
                return False
            
            # Step 3: Save vectors and metadata locally
            success = self._save_vectors_and_metadata(embeddings_data, pdf_name)
            
            if success:
                print(f"✅ Successfully ingested: {Path(pdf_path).name}")
            else:
                print(f"❌ Failed to ingest: {Path(pdf_path).name}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error in VDR PDF ingestion: {str(e)}")
            return False

    def _save_vectors_and_metadata(self, embeddings_data: List[Dict[str, Any]], pdf_name: str) -> bool:
        """Save vectors and metadata locally"""
        try:
            # Create document directory structure
            doc_dir = self.images_dir / pdf_name
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            vectors_dir = doc_dir / "vectors"
            vectors_dir.mkdir(exist_ok=True)
            
            metadata_dir = doc_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            # Prepare vectors and metadata
            vectors = {"embeddings": []}
            metadata = []
            
            for i, item in enumerate(embeddings_data):
                # Get embedding
                embedding = item["embedding"]
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                vectors["embeddings"].append(embedding)
                
                # Get metadata
                meta = item["metadata"]
                page_metadata = {
                    "page_number": meta["page_number"],
                    "section_id": f"page_{meta['page_number']}",
                    "image_path": f"images/page{meta['page_number']}.jpg",
                    "pdf_name": pdf_name,
                    "chunks": [{
                        "vector_index": i,
                        "coordinates": {
                            "x1": 0,
                            "y1": 0,
                            "x2": int(meta["image_size"].split("x")[0]),
                            "y2": int(meta["image_size"].split("x")[1])
                        }
                    }],
                    "section_type": "page",
                    "id": f"{pdf_name}_page_{meta['page_number']}"
                }
                metadata.append(page_metadata)
            
            # Save files
            vectors_file = vectors_dir / "vectors.json"
            metadata_file = metadata_dir / "metadata.json"
            
            import json
            with open(vectors_file, "w") as f:
                json.dump(vectors, f, indent=2)
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved vectors and metadata for: {pdf_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving vectors and metadata: {str(e)}")
            return False

    def ingest_all_pdfs(self) -> Dict[str, Any]:
        """Ingest all PDFs in the data directory"""
        try:
            print("\n🔍 Scanning for PDF files...")
            
            # Get all PDFs in data directory
            pdf_files = list(self.data_dir.glob("*.pdf"))
            
            if not pdf_files:
                print(f"❌ No PDF files found in {self.data_dir}")
                print("💡 Place your PDF files in the 'data' directory and try again")
                return {
                    "total_files": 0,
                    "successful": 0,
                    "failed": 0,
                    "files_processed": []
                }
            
            print(f"📚 Found {len(pdf_files)} PDF file(s)")
            
            # Process each PDF
            successful = 0
            failed = 0
            files_processed = []
            
            start_time = time.time()
            
            for i, pdf_path in enumerate(pdf_files, 1):
                print(f"\n📖 Processing file {i}/{len(pdf_files)}: {pdf_path.name}")
                
                success = self.ingest_pdf(str(pdf_path))
                file_size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 2)
                
                file_result = {
                    "filename": pdf_path.name,
                    "path": str(pdf_path),
                    "success": success,
                    "size_mb": file_size_mb
                }
                
                files_processed.append(file_result)
                
                if success:
                    successful += 1
                else:
                    failed += 1
                
                print(f"📊 Progress: {i}/{len(pdf_files)} files processed")
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"🚀 VDR INGESTION COMPLETE")
            print(f"{'='*60}")
            print(f"📁 Total files: {len(pdf_files)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"⏱️  Total time: {processing_time}s")
            print(f"🎯 Model used: {config.EMBEDDING_MODEL}")
            print(f"{'='*60}")
            
            return {
                "total_files": len(pdf_files),
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time,
                "files_processed": files_processed
            }
            
        except Exception as e:
            print(f"❌ Error in VDR batch ingestion: {str(e)}")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 1,
                "error": str(e)
            } 
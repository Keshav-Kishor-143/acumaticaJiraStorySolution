"""
Local File Manager for Story Solutions Knowledge Base

Handles all local file operations for the knowledge_base/manuals directory structure:
- Image access from local directory
- Vector and metadata file management
- Document structure management
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from src.config.config import config
from src.utils.logger_utils import get_logger

class LocalManager:
    """Manages all local file operations for knowledge base"""
    
    def __init__(self):
        self.logger = get_logger("LOCAL_MANAGER")
        self.base_path = config.LOCAL_BASE_PATH
        
        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)
        
        # Enhanced initialization logging
        self.logger.info("Local Manager initializing", extra={
            "base_path": self.base_path,
            "base_path_absolute": os.path.abspath(self.base_path),
            "base_path_exists": os.path.exists(self.base_path),
            "base_path_is_dir": os.path.isdir(self.base_path) if os.path.exists(self.base_path) else False,
            "base_path_contents": os.listdir(self.base_path) if os.path.exists(self.base_path) else [],
            "current_working_dir": os.getcwd()
        })
    
    def get_document_paths(self, document_name: str) -> Dict[str, str]:
        """Get all paths for a specific document"""
        try:
            doc_base = os.path.join(self.base_path, document_name)
            doc_base_abs = os.path.abspath(doc_base)
            
            paths = {
                "base": doc_base,
                "data": os.path.join(doc_base, "data"),
                "images": os.path.join(doc_base, "images"),
                "vectors": os.path.join(doc_base, "vectors"),
                "metadata": os.path.join(doc_base, "metadata")
            }
            
            # Check if paths exist
            path_exists = {
                "base": os.path.exists(paths["base"]),
                "data": os.path.exists(paths["data"]),
                "images": os.path.exists(paths["images"]),
                "vectors": os.path.exists(paths["vectors"]),
                "metadata": os.path.exists(paths["metadata"])
            }
            
            self.logger.debug("Document paths", extra={
                "document": document_name,
                "base_path": self.base_path,
                "doc_base": doc_base,
                "doc_base_abs": doc_base_abs,
                "paths": paths,
                "path_exists": path_exists
            })
            
            return paths
            
        except Exception as e:
            self.logger.error("Failed to get document paths", extra={
                "document": document_name,
                "base_path": self.base_path,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def get_image_path(self, document_name: str, page_number: int, image_format: str = "jpg") -> Optional[str]:
        """Get local path for a specific image"""
        try:
            paths = self.get_document_paths(document_name)
            image_filename = f"page{page_number}.{image_format}"
            image_path = os.path.join(paths["images"], image_filename)
            
            if os.path.exists(image_path):
                return image_path
            
            self.logger.warning("Image not found", extra={
                "document_name": document_name,
                "page_number": page_number,
                "path": image_path
            })
            return None
            
        except Exception as e:
            self.logger.error("Failed to get image path", extra={
                "document_name": document_name,
                "page_number": page_number,
                "error": str(e)
            })
            return None
    
    def load_vectors(self, document_name: str) -> Optional[Dict[str, Any]]:
        """Load vector data from local storage"""
        try:
            paths = self.get_document_paths(document_name)
            vector_file = os.path.join(paths["vectors"], "vectors.json")
            
            if not os.path.exists(vector_file):
                self.logger.warning("Vector file not found", extra={
                    "document_name": document_name,
                    "path": vector_file
                })
                return None
            
            with open(vector_file, 'r', encoding='utf-8') as f:
                vectors_data = json.load(f)
            
            self.logger.debug("Loaded vectors from local storage", extra={
                "document_name": document_name,
                "path": vector_file
            })
            return vectors_data
            
        except Exception as e:
            self.logger.error("Failed to load vectors", extra={
                "document_name": document_name,
                "error": str(e)
            })
            return None
    
    def load_metadata(self, document_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata from local storage"""
        try:
            paths = self.get_document_paths(document_name)
            metadata_file = os.path.join(paths["metadata"], "metadata.json")
            
            if not os.path.exists(metadata_file):
                self.logger.warning("Metadata file not found", extra={
                    "document_name": document_name,
                    "path": metadata_file
                })
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.debug("Loaded metadata from local storage", extra={
                "document_name": document_name,
                "path": metadata_file
            })
            return metadata
            
        except Exception as e:
            self.logger.error("Failed to load metadata", extra={
                "document_name": document_name,
                "error": str(e)
            })
            return None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all available documents in local storage"""
        try:
            documents = []
            
            # Enhanced debug logging
            self.logger.debug("Enhanced base path info", extra={
                "base_path": self.base_path,
                "exists": os.path.exists(self.base_path),
                "absolute_path": os.path.abspath(self.base_path),
                "current_working_dir": os.getcwd(),
                "base_path_type": type(self.base_path).__name__,
                "base_path_normalized": os.path.normpath(self.base_path)
            })
            
            # List all subdirectories in base path
            if os.path.exists(self.base_path):
                try:
                    items = os.listdir(self.base_path)
                    self.logger.debug("Directory contents", extra={
                        "items_found": len(items),
                        "items": items
                    })
                    
                    for item in items:
                        doc_path = os.path.join(self.base_path, item)
                        if os.path.isdir(doc_path):
                            doc_info = {
                                "name": item
                            }
                            documents.append(doc_info)
                            self.logger.debug("Added document", extra={
                                "doc_name": item,
                                "doc_path": doc_path,
                                "is_directory": os.path.isdir(doc_path)
                            })
                    
                    self.logger.info("Listed documents from local storage", extra={
                        "document_count": len(documents),
                        "document_list": [d["name"] for d in documents],
                        "total_items": len(items)
                    })
                    
                except Exception as e:
                    self.logger.error("Failed to list base path", extra={
                        "base_path": self.base_path,
                        "error_message": str(e),
                        "error_type": type(e).__name__
                    })
                    raise
            else:
                self.logger.warning("Base path does not exist", extra={
                    "path": self.base_path,
                    "absolute_path": os.path.abspath(self.base_path),
                    "normalized_path": os.path.normpath(self.base_path)
                })
            
            return documents
            
        except Exception as e:
            self.logger.error("Failed to list documents", extra={
                "error_message": str(e),
                "base_path": self.base_path,
                "error_type": type(e).__name__
            })
            raise


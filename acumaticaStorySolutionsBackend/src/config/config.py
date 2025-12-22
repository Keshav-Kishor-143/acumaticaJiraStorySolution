import os
from typing import Dict
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for JIRA Story Solutions Service"""
    
    # Initialize logger first
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create a console handler if it doesn't exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Now initialize other class attributes
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Authentication Configuration
    API_KEY = os.getenv("API_KEY", "AcuBot2024!SecureAPIKey#Authentication")
    
    # Force local storage mode
    STORAGE_MODE = "local"
    
    # Local Storage Paths
    _current_file = os.path.abspath(__file__)  # /src/config/config.py
    _current_dir = os.path.dirname(_current_file)  # /src/config
    _src_dir = os.path.dirname(_current_dir)  # /src
    _app_dir = os.path.dirname(_src_dir)  # /acumaticaStorySolutions
    
    # Use absolute path from workspace root
    WORKSPACE_ROOT = os.path.dirname(_app_dir)  # Go up to workspace root
    LOCAL_BASE_PATH = os.path.join(_app_dir, "knowledge_base", "manuals")
    
    # Output directory for saving solution markdown files
    OUTPUT_DIR = os.path.join(_app_dir, "output")
    
    # Add direct path verification
    if not os.path.exists(LOCAL_BASE_PATH):
        logger.warning("LOCAL_BASE_PATH does not exist, will be created", extra={
            "path": LOCAL_BASE_PATH
        })
        os.makedirs(LOCAL_BASE_PATH, exist_ok=True)
    else:
        logger.info("LOCAL_BASE_PATH verified", extra={
            "path": LOCAL_BASE_PATH
        })
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        logger.info("Creating OUTPUT_DIR", extra={
            "path": OUTPUT_DIR
        })
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        logger.debug("OUTPUT_DIR verified", extra={
            "path": OUTPUT_DIR
        })
    
    logger.debug("Path resolution details", extra={
        "current_file": _current_file,
        "current_dir": _current_dir,
        "src_dir": _src_dir,
        "app_dir": _app_dir,
        "local_base_path": LOCAL_BASE_PATH,
        "path_exists": os.path.exists(LOCAL_BASE_PATH),
        "absolute_local_base": os.path.abspath(LOCAL_BASE_PATH),
        "cwd": os.getcwd()
    })

    @classmethod
    def get_document_paths(cls, document_name: str) -> Dict[str, str]:
        """Get all paths for a specific document"""
        doc_base = os.path.join(cls.LOCAL_BASE_PATH, document_name)
        return {
            "base": doc_base,
            "data": os.path.join(doc_base, "data"),
            "images": os.path.join(doc_base, "images"),
            "vectors": os.path.join(doc_base, "vectors"),
            "metadata": os.path.join(doc_base, "metadata")
        }
    
    # Directory Configuration - Local structure only
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(LOCAL_BASE_PATH, "default", "data"))
    IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(LOCAL_BASE_PATH, "default", "images"))
    VECTORS_DIR = os.getenv("VECTORS_DIR", os.path.join(LOCAL_BASE_PATH, "default", "vectors"))
    METADATA_DIR = os.getenv("METADATA_DIR", os.path.join(LOCAL_BASE_PATH, "default", "metadata"))
    
    # Vector Storage Configuration - Local only
    VECTOR_DB_TYPE = "local"
    
    # AI Model Configuration - Using FREE HuggingFace embeddings!
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
    
    # Image Processing Configuration (Cost Optimized)
    DPI = int(os.getenv("DPI", "200"))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
    
    # Model Parameters (Cost Optimized)
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Vector Database Configuration
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # Story Processing Configuration
    MAX_QUESTIONS_PER_STORY = int(os.getenv("MAX_QUESTIONS_PER_STORY", "5"))
    SOLUTION_MAX_TOKENS = int(os.getenv("SOLUTION_MAX_TOKENS", "3000"))
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.IMAGES_DIR, exist_ok=True)
        os.makedirs(cls.VECTORS_DIR, exist_ok=True)
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        return True
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        warnings = []
        
        # Check OpenAI API key but don't fail startup
        if not cls.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - LLM and Vision features will be disabled")
        
        # Guard against deprecated/retired vision models
        deprecated_vision_models = {"gpt-4-vision-preview", "gpt-4-turbo", "gpt-4-turbo-preview"}
        if cls.VISION_MODEL in deprecated_vision_models:
            warnings.append(f"VISION_MODEL '{cls.VISION_MODEL}' is deprecated - switching to 'gpt-4o-mini'")
            cls.VISION_MODEL = "gpt-4o-mini"

        # Guard against unsupported/unknown LLM models
        unsupported_llm_models = {"gpt-5-nano", "gpt-5-mini", "gpt-4-turbo", "gpt-4-turbo-preview"}
        if cls.LLM_MODEL in unsupported_llm_models:
            warnings.append(f"LLM_MODEL '{cls.LLM_MODEL}' is unsupported - switching to 'gpt-4o-mini'")
            cls.LLM_MODEL = "gpt-4o-mini"
        
        # Log warnings but don't fail
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(warning)
        
        return True
    
    @classmethod
    def calculate_cost_inr(cls, input_tokens=0, output_tokens=0, vision_tokens=0):
        """Calculate cost in Indian Rupees for OpenAI API usage"""
        # OpenAI pricing for GPT-4o-mini (as of 2024) in USD
        pricing = {
            'gpt-4o-mini': {
                'input': 0.15 / 1_000_000,   # $0.15 per 1M input tokens
                'output': 0.60 / 1_000_000   # $0.60 per 1M output tokens
            },
            'vision': {
                'input': 0.15 / 1_000_000    # Same as gpt-4o-mini input pricing
            }
        }
        
        # USD to INR conversion rate (approximate)
        usd_to_inr = 87.5
        
        # Calculate costs
        input_cost_usd = input_tokens * pricing['gpt-4o-mini']['input']
        output_cost_usd = output_tokens * pricing['gpt-4o-mini']['output']
        vision_cost_usd = vision_tokens * pricing['vision']['input']
        
        total_cost_usd = input_cost_usd + output_cost_usd + vision_cost_usd
        total_cost_inr = total_cost_usd * usd_to_inr
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'vision_tokens': vision_tokens,
            'input_cost_inr': input_cost_usd * usd_to_inr,
            'output_cost_inr': output_cost_usd * usd_to_inr,
            'vision_cost_inr': vision_cost_usd * usd_to_inr,
            'total_cost_inr': total_cost_inr,
            'total_cost_usd': total_cost_usd,
            'usd_to_inr_rate': usd_to_inr,
            'embedding_cost_saved': "FREE! (HuggingFace embeddings)"
        }

    # Optimized Configuration
    USE_PRECOMPUTED_VECTORS = os.getenv("USE_PRECOMPUTED_VECTORS", "true").lower() == "true"
    INCLUDE_VISION_ANALYSIS = os.getenv("INCLUDE_VISION_ANALYSIS", "true").lower() == "true"

# Create a global config instance
config = Config()


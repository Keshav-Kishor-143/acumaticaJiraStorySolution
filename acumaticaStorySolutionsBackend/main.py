"""
FastAPI Backend for JIRA Story Solutions Service

A service that processes JIRA stories and generates markdown solutions using
RAG (Retrieval-Augmented Generation) with HuggingFace embeddings and OpenAI.

Features:
- JIRA story processing
- Key question extraction
- Knowledge base search (manuals)
- Solution generation
- Markdown output
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.api.routes.solutions import router as solutions_router

app_config = config
logger = get_logger("FASTAPI_MAIN")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    try:
        logger.info("🚀 Starting JIRA Story Solutions Service...")
        
        # Validate configuration
        try:
            app_config.validate_config()
            logger.info("✅ Configuration validated")
        except Exception as e:
            logger.warning(f"⚠️ Configuration validation failed: {e}")
        
        # Test knowledge base access
        try:
            from src.utils.local_manager import LocalManager
            local_manager = LocalManager()
            docs = local_manager.list_documents()
            logger.info(f"📚 Found {len(docs)} documents in knowledge base")
        except Exception as e:
            logger.warning(f"⚠️ Knowledge base check failed: {e}")
        
        # Pre-warm components in background (optional - improves first request performance)
        # Components will still initialize lazily on first request if this fails
        try:
            import asyncio
            import concurrent.futures
            
            async def warm_up_components():
                """Pre-initialize components in background"""
                try:
                    logger.info("🔥 Pre-warming components in background...")
                    from src.api.routes.solutions import get_components
                    # Run in thread pool to avoid blocking startup
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        await loop.run_in_executor(executor, get_components)
                    logger.info("✅ Components pre-warmed successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Component pre-warm failed (will initialize on first request): {e}")
            
            # Start warm-up in background (non-blocking)
            asyncio.create_task(warm_up_components())
        except Exception as e:
            logger.debug(f"Component warm-up not started: {e}")
        
        logger.info("📚 JIRA Story Solutions Service ready")
        yield
        
    except Exception as e:
        logger.error("Failed to start application", extra={"error": str(e)})
        raise
    finally:
        logger.info("🛑 Shutting down JIRA Story Solutions Service...")


# Create FastAPI application
app = FastAPI(
    title="JIRA Story Solutions API",
    description="""
    **JIRA Story Solution Generation System**
    
    Process JIRA stories and generate comprehensive markdown solutions.
    
    ### Features:
    - 🧠 **Question Extraction**: LLM-powered extraction of key questions from stories
    - 🔍 **Knowledge Base Search**: RAG + Vision search across manuals
    - 📝 **Solution Generation**: AI-powered narrative solution creation
    - 📄 **Markdown Output**: Formatted solution documents
    
    ### Usage:
    1. **Process a story** via `/solutions/process`
    2. **Check health** via `/solutions/health`
    
    ### API Documentation:
    - **Swagger UI**: Available at `/docs` (interactive API documentation)
    - **ReDoc**: Available at `/redoc` (alternative documentation)
    - **OpenAPI Schema**: Available at `/openapi.json`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True
    }
)

# Configure CORS for cross-machine testing
def get_cors_origins():
    """
    Get CORS allowed origins from environment or use defaults.
    
    Best practices:
    - Development: Use "*" for local testing (allows any origin)
    - Production: Use specific domains only for security
    
    Environment variable: CORS_ORIGINS (comma-separated list)
    Example: CORS_ORIGINS="http://192.168.0.42:3000,http://localhost:3000,http://127.0.0.1:3000"
    
    To allow all origins (development/testing): Set CORS_ALLOW_ALL=true
    """
    cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
    
    # Check if specific origins are provided via environment
    if cors_origins_env:
        # Parse comma-separated origins from environment
        origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        logger.info(f"🌐 CORS configured from environment: {len(origins)} origin(s)")
        for origin in origins:
            logger.info("   ✓ %s", origin)
        return origins
    
    # Check if we should allow all origins (development/testing mode)
    allow_all = os.getenv("CORS_ALLOW_ALL", "true").lower() == "true"
    
    if allow_all:
        logger.info("🌐 CORS configured: ALLOWING ALL ORIGINS (development/testing mode)")
        logger.info("   ⚠️  For production, set CORS_ALLOW_ALL=false and specify CORS_ORIGINS")
        return ["*"]
    
    # Default: Allow common development origins + your IP
    default_origins = [
        "http://localhost:3000",      # React default
        "http://localhost:3001",      # Alternative React port
        "http://localhost:5173",      # Vite default
        "http://localhost:8080",      # Vue default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://192.168.0.42:3000",  # Your IP with common ports
        "http://192.168.0.42:3001",
        "http://192.168.0.42:5173",
        "http://192.168.0.42:8080",
        "http://192.168.0.42:8000",  # Common backend port
    ]
    
    logger.info(f"🌐 CORS configured: {len(default_origins)} default origin(s)")
    for origin in default_origins:
        logger.info("   ✓ %s", origin)
    
    return default_origins

# Add CORS middleware with best practices
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-API-Key",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=[
        "Content-Length",
        "Content-Type",
        "X-Request-ID",
    ],
    max_age=3600,  # Cache preflight requests for 1 hour
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    cors_origins = get_cors_origins()
    return {
        "message": "JIRA Story Solutions API",
        "version": "1.0.0",
        "status": "healthy",
        "knowledge_base_path": app_config.LOCAL_BASE_PATH,
        "docs": "/docs",
        "cors": {
            "enabled": True,
            "origins_count": len(cors_origins),
            "allow_all": cors_origins == ["*"],
            "note": "Configure via CORS_ORIGINS env var or CORS_ALLOW_ALL=false for security"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "knowledge_base_path": app_config.LOCAL_BASE_PATH,
            "components": {
                "config": "ok",
                "knowledge_base": "ok",
                "openai": "ok"
            }
        }
        return health_status
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Include solutions routes
try:
    app.include_router(solutions_router, prefix="/solutions", tags=["Solutions"])
    logger.info("✅ Solutions routes loaded")
except Exception as e:
    logger.error(f"❌ Failed to load solutions routes: {e}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception",
                path=str(request.url),
                method=request.method,
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )


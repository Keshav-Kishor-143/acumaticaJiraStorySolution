"""
Solutions API Routes - Process JIRA stories and generate solutions
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.api.models.requests import JIRAStoryRequest
from src.api.models.responses import SolutionResponse, SourceReference, HealthResponse
from src.core.story_processor import JIRAStoryProcessor
from src.core.rag_service import RAGService
from src.core.solution_generator import SolutionGenerator
from src.core.intent_layer import IntentUnderstandingLayer
from src.utils.markdown_formatter import MarkdownFormatter
from src.config.config import config
from src.utils.logger_utils import get_logger

router = APIRouter()
logger = get_logger("SOLUTIONS_API")


def save_solution_markdown(solution_markdown: str, story_id: Optional[str], title: str) -> str:
    """
    Save solution markdown to file in output directory.
    
    Args:
        solution_markdown: The markdown content to save
        story_id: Optional story ID for filename
        title: Title for filename sanitization
        
    Returns:
        Path to the saved file
    """
    from pathlib import Path
    import re
    from datetime import datetime
    
    # Ensure output directory exists
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from story_id or title
    if story_id:
        # Sanitize story_id for filename
        safe_filename = re.sub(r'[^\w\-_\.]', '_', story_id)
        filename = f"{safe_filename}.md"
    else:
        # Use title and timestamp
        safe_title = re.sub(r'[^\w\-_\.\s]', '', title)[:50]  # Limit length
        safe_title = re.sub(r'\s+', '_', safe_title.strip())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.md"
    
    # Full file path
    file_path = output_dir / filename
    
    # Write markdown content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(solution_markdown)
    
    logger.info("Solution markdown saved", extra={
        "file_path": str(file_path),
        "file_size": len(solution_markdown),
        "story_id": story_id
    })
    
    return str(file_path)


# Global components for lifecycle management
_story_processor: Optional[JIRAStoryProcessor] = None
_rag_service: Optional[RAGService] = None
_solution_generator: Optional[SolutionGenerator] = None
_markdown_formatter: Optional[MarkdownFormatter] = None
_intent_layer: Optional[IntentUnderstandingLayer] = None


def get_components() -> tuple:
    """Lazy initialization of system components"""
    global _story_processor, _rag_service, _solution_generator, _markdown_formatter, _intent_layer
    
    try:
        # Initialize Story Processor if needed
        if _story_processor is None:
            logger.debug("Creating new Story Processor instance")
            _story_processor = JIRAStoryProcessor()
            logger.debug("Story Processor instance created")
        
        # Initialize RAG Service if needed
        if _rag_service is None:
            logger.debug("Creating new RAG Service instance")
            _rag_service = RAGService()
            logger.debug("RAG Service instance created")
        
        # Initialize Solution Generator if needed
        if _solution_generator is None:
            logger.debug("Creating new Solution Generator instance")
            _solution_generator = SolutionGenerator()
            logger.debug("Solution Generator instance created")
        
        # Initialize Markdown Formatter if needed
        if _markdown_formatter is None:
            logger.debug("Creating new Markdown Formatter instance")
            _markdown_formatter = MarkdownFormatter()
            logger.debug("Markdown Formatter instance created")
        
        # Initialize Intent Understanding Layer if needed
        if _intent_layer is None:
            logger.debug("Creating new Intent Understanding Layer instance")
            _intent_layer = IntentUnderstandingLayer()
            logger.debug("Intent Understanding Layer instance created")
        
        return _story_processor, _rag_service, _solution_generator, _markdown_formatter, _intent_layer
        
    except Exception as e:
        logger.error("Failed to initialize components", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise


@router.post(
    "/process", 
    response_model=SolutionResponse,
    summary="Process JIRA Story and Generate Solution",
    description="""
    Process a JIRA story and generate a comprehensive markdown solution.
    
    **Processing Flow:**
    1. Extract key questions from the story using LLM (for internal guidance)
    2. Perform unified retrieval using story context + questions as guidance
    3. Generate single comprehensive solution answer using RAG + Vision
    4. Format output as structured markdown
    
    **Request Requirements:**
    - `description`: JIRA story description (required)
    - `acceptance_criteria`: List of acceptance criteria (required, min 1)
    - `story_id`: Optional JIRA story ID
    - `title`: Optional story title (will be generated if not provided)
    - `images`: Optional list of image URLs or base64 images
    
    **Response Includes:**
    - Complete markdown solution (THE answer to the JIRA story task)
    - Source references used in the solution
    - Processing time
    - Saved file path (if STORAGE_MODE is local)
    """,
    responses={
        200: {
            "description": "Successfully processed story",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "story_id": "STORY-001",
                        "solution_markdown": "# Solution for Story STORY-001\n\n## Solution\n\n[Complete comprehensive solution answer here]\n\n---\n\n## Acceptance Criteria\n\n- [ ] Criterion 1\n- [ ] Criterion 2\n\n---\n\n## References\n\n- **Document_Name** (Page 5, Confidence: 0.95)",
                        "sources": [
                            {
                                "document": "Sales_Returns_Manual",
                                "page": 5,
                                "confidence": 0.95,
                                "text_snippet": "Return processing steps..."
                            }
                        ],
                        "processing_time": 15.2,
                        "saved_file_path": "output/STORY-001.md"
                    }
                }
            }
        },
        400: {
            "description": "Bad request - Invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Failed to extract questions from story"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Internal processing error"
                    }
                }
            }
        }
    }
)
async def process_story(request: JIRAStoryRequest):
    """
    Process a JIRA story and generate markdown solution
    
    Flow:
    1. Extract key questions from story
    2. For each question: Search knowledge base and generate answer
    3. Generate narrative solution
    4. Format as markdown
    """
    start_time = time.time()
    
    try:
        logger.info("Processing JIRA story", extra={
            "story_id": request.story_id,
            "description_length": len(request.description),
            "criteria_count": len(request.acceptance_criteria)
        })
        
        # Get components
        story_processor, rag_service, solution_generator, markdown_formatter, intent_layer = get_components()
        
        # Step 1: Extract key questions
        story_json = {
            "description": request.description,
            "acceptance_criteria": request.acceptance_criteria,
            "images": request.images or [],
            "story_id": request.story_id
        }
        
        questions = story_processor.extract_key_questions(story_json)
        
        if not questions:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract questions from story"
            )
        
        logger.info("Questions extracted", extra={
            "question_count": len(questions),
            "questions": questions
        })
        
        # Step 2: Unified Retrieval Approach - Use all questions as context for comprehensive retrieval
        # This prevents hallucination by grounding everything in a single, comprehensive retrieval
        
        logger.info("Using unified retrieval approach", extra={
            "question_count": len(questions),
            "approach": "questions_as_context"
        })
        
        # Analyze unified story context (not individual questions)
        unified_search_params = None
        try:
            # Create unified query context for intent analysis
            unified_context = f"{request.description}\n\nKey aspects: {', '.join(questions)}"
            intent_analysis = await intent_layer.analyze_query(unified_context)
            unified_search_params = intent_layer.get_search_parameters(intent_analysis, original_query=unified_context)
            logger.info("Unified intent analysis completed", extra={
                "relevant_docs": len(unified_search_params.get("target_directories", [])),
                "primary_intent": unified_search_params.get("query_intent", {}).get("primary", "unknown")
            })
        except Exception as e:
            logger.warning("Unified intent analysis failed, using standard search", extra={
                "error": str(e)
            })
        
        # Step 3: Single comprehensive retrieval with all questions as context
        # Questions guide the retrieval but don't create separate queries
        comprehensive_result = await rag_service.generate_comprehensive_answer(
            story_context={
                "description": request.description,
                "acceptance_criteria": request.acceptance_criteria,
                "questions": questions  # Pass questions as context, not separate queries
            },
            top_k=config.TOP_K_RESULTS * 2,  # Get more context for comprehensive answer
            search_params=unified_search_params
        )
        
        # Extract sources from comprehensive result
        all_sources = comprehensive_result.get('sources', [])
        comprehensive_answer = comprehensive_result.get('answer', 'No answer found.')
        
        logger.info("Comprehensive retrieval completed", extra={
            "sources_found": len(all_sources),
            "answer_length": len(comprehensive_answer)
        })
        
        # Step 4: Generate focused narrative solution grounded in retrieved content
        # This is THE single comprehensive answer to the JIRA story task
        narrative = solution_generator.generate_focused_narrative(
            story_context=story_json,
            retrieved_content=comprehensive_result,
            questions=questions  # Questions guide narrative structure internally
        )
        
        logger.info("Focused narrative solution generated", extra={
            "narrative_length": len(narrative)
        })
        
        # Step 5: Format as markdown - this IS the solution answer
        title = request.title or f"Solution for Story {request.story_id or 'Unknown'}"
        
        # Format the single comprehensive solution as markdown
        solution_markdown = markdown_formatter.format_solution(
            title=title,
            story_id=request.story_id,
            questions=questions,  # Keep questions for reference in markdown
            answers=[],  # No separate Q&A pairs - narrative IS the answer
            narrative=narrative,  # This is the single comprehensive solution
            acceptance_criteria=request.acceptance_criteria,
            sources=all_sources,
            metadata={
                "processing_time": time.time() - start_time,
                "questions_extracted": len(questions),
                "solution_type": "unified_comprehensive"
            }
        )
        
        processing_time = time.time() - start_time
        
        # Save markdown file if STORAGE_MODE is local
        saved_file_path = None
        if config.STORAGE_MODE == "local":
            try:
                saved_file_path = save_solution_markdown(
                    solution_markdown=solution_markdown,
                    story_id=request.story_id,
                    title=title
                )
                logger.info("Solution markdown saved to file", extra={
                    "file_path": saved_file_path,
                    "story_id": request.story_id
                })
            except Exception as e:
                logger.warning("Failed to save solution markdown file", extra={
                    "error": str(e),
                    "story_id": request.story_id
                })
        
        # Single comprehensive solution - clean response with only essential fields
        # The solution_markdown IS the answer to the JIRA story task
        response = SolutionResponse(
            success=True,
            story_id=request.story_id,
            solution_markdown=solution_markdown,  # This IS the single comprehensive solution answer
            sources=[
                SourceReference(
                    document=s.get('document', 'Unknown'),
                    page=s.get('page', 0),
                    confidence=float(s.get('similarity_score', 0.0)),
                    text_snippet=s.get('content_preview', '')
                )
                for s in all_sources[:15]  # Top 15 sources for comprehensive solution
            ],
            processing_time=processing_time,
            saved_file_path=saved_file_path
        )
        
        logger.info("Story processing completed", extra={
            "story_id": request.story_id,
            "processing_time": processing_time,
            "success": True,
            "file_saved": saved_file_path is not None
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Story processing failed", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "story_id": request.story_id
        })
        
        return SolutionResponse(
            success=False,
            story_id=request.story_id,
            solution_markdown="",
            sources=[],
            processing_time=time.time() - start_time,
            saved_file_path=None,
            error=str(e)
        )


@router.get(
    "/health", 
    response_model=HealthResponse,
    summary="Solutions Service Health Check",
    description="""
    Check the health status of the solutions service and its components.
    
    **Checks:**
    - Story Processor initialization
    - RAG Service initialization and document availability
    - Solution Generator initialization
    
    **Response:**
    - Overall service status (healthy/degraded/unhealthy)
    - Individual component statuses
    - Document count in knowledge base
    """,
    responses={
        200: {
            "description": "Health check completed",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "message": "Solutions service healthy",
                        "timestamp": "2025-12-19T17:00:00",
                        "components": {
                            "story_processor": {
                                "status": "ok",
                                "message": "Story Processor ready"
                            },
                            "rag_service": {
                                "status": "ok",
                                "message": "RAG Service ready (5 documents available)"
                            },
                            "solution_generator": {
                                "status": "ok",
                                "message": "Solution Generator ready"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def health_check():
    """Health check for solutions service"""
    try:
        components = {}
        overall_status = "healthy"
        
        # Check Story Processor
        try:
            story_processor, _, _, _, _ = get_components()  # Fixed: unpack 5 values
            components["story_processor"] = {
                "status": "ok",
                "message": "Story Processor ready"
            }
        except Exception as e:
            components["story_processor"] = {
                "status": "error",
                "message": f"Story Processor failed: {str(e)}"
            }
            overall_status = "degraded"
            logger.warning("Story Processor health check failed", extra={"error": str(e)})
        
        # Check RAG Service
        try:
            _, rag_service, _, _, _ = get_components()  # Fixed: unpack 5 values
            docs = rag_service.list_available_documents()
            components["rag_service"] = {
                "status": "ok",
                "message": f"RAG Service ready ({len(docs)} documents available)"
            }
        except Exception as e:
            components["rag_service"] = {
                "status": "error",
                "message": f"RAG Service failed: {str(e)}"
            }
            overall_status = "degraded"
            logger.warning("RAG Service health check failed", extra={"error": str(e)})
        
        # Check Solution Generator
        try:
            _, _, solution_generator, _, _ = get_components()  # Fixed: unpack 5 values
            components["solution_generator"] = {
                "status": "ok",
                "message": "Solution Generator ready"
            }
        except Exception as e:
            components["solution_generator"] = {
                "status": "error",
                "message": f"Solution Generator failed: {str(e)}"
            }
            overall_status = "degraded"
            logger.warning("Solution Generator health check failed", extra={"error": str(e)})
        
        return HealthResponse(
            status=overall_status,
            message=f"Solutions service {overall_status}",
            timestamp=datetime.utcnow(),
            components=components
        )
        
    except Exception as e:
        logger.error("Health check failed", extra={
            "error": str(e)
        })
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.utcnow(),
            components={}
        )


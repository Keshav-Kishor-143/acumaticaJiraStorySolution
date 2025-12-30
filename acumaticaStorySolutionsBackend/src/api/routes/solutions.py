"""
Solutions API Routes - Process JIRA stories and generate solutions
"""

import time
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import json

from src.api.models.requests import JIRAStoryRequest, StoryNormalizationRequest, NormalizedStoryResponse
from src.api.models.responses import SolutionResponse, SourceReference, HealthResponse, ManualsListResponse, ManualInfo
from src.core.story_processor import JIRAStoryProcessor
from src.core.solution_generator import SolutionGenerator
from src.core.pipeline_controller import PipelineController
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
_solution_generator: Optional[SolutionGenerator] = None
_markdown_formatter: Optional[MarkdownFormatter] = None
_pipeline_controller: Optional[PipelineController] = None
_initialization_lock = None
_initializing = False

# Cancellation management
_cancellation_tokens: Dict[str, bool] = {}
_request_lock = None

def _get_request_lock():
    """Get asyncio lock for request management"""
    global _request_lock
    if _request_lock is None:
        _request_lock = asyncio.Lock()
    return _request_lock

async def _check_cancellation(request_id: str) -> None:
    """Check if request has been cancelled, raise exception if cancelled"""
    async with _get_request_lock():
        if _cancellation_tokens.get(request_id, False):
            logger.info("Request cancelled", extra={"request_id": request_id})
            raise asyncio.CancelledError(f"Request {request_id} was cancelled")

def _get_lock():
    """Get thread lock for initialization (lazy initialization)"""
    global _initialization_lock
    if _initialization_lock is None:
        import threading
        _initialization_lock = threading.Lock()
    return _initialization_lock


def get_components() -> tuple:
    """Lazy initialization of system components with thread safety"""
    global _story_processor, _solution_generator, _markdown_formatter, _pipeline_controller, _initializing
    
    # Use lock to prevent concurrent initialization
    lock = _get_lock()
    
    with lock:
        # Double-check pattern: verify components are still None after acquiring lock
        if _story_processor is not None and _pipeline_controller is not None:
            # Components already initialized, return immediately
            return _story_processor, _solution_generator, _markdown_formatter, _pipeline_controller
        
        # Prevent recursive initialization
        if _initializing:
            logger.warning("Initialization already in progress, waiting...")
            # Wait a bit and return existing components if available
            import time
            time.sleep(0.1)
            if _story_processor is not None and _pipeline_controller is not None:
                return _story_processor, _solution_generator, _markdown_formatter, _pipeline_controller
        
        _initializing = True
        
        try:
            # Initialize Story Processor if needed
            if _story_processor is None:
                logger.info("Initializing Story Processor...")
                _story_processor = JIRAStoryProcessor()
                logger.info("✅ Story Processor initialized")
            
            # Initialize Solution Generator if needed
            if _solution_generator is None:
                logger.info("Initializing Solution Generator...")
                _solution_generator = SolutionGenerator()
                logger.info("✅ Solution Generator initialized")
            
            # Initialize Markdown Formatter if needed
            if _markdown_formatter is None:
                logger.info("Initializing Markdown Formatter...")
                _markdown_formatter = MarkdownFormatter()
                logger.info("✅ Markdown Formatter initialized")
            
            # Initialize Pipeline Controller if needed
            if _pipeline_controller is None:
                logger.info("Initializing Pipeline Controller...")
                _pipeline_controller = PipelineController()
                logger.info("✅ Pipeline Controller initialized")
            
            logger.info("✅ All components initialized successfully")
            return _story_processor, _solution_generator, _markdown_formatter, _pipeline_controller
            
        except Exception as e:
            logger.error("Failed to initialize components", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Reset initialization flag on error
            _initializing = False
            raise
        finally:
            _initializing = False


@router.post(
    "/normalize",
    response_model=NormalizedStoryResponse,
    summary="Normalize JIRA Story Input",
    description="""
    Normalize raw JIRA story text into clean, canonical JSON structure.
    
    **Purpose:**
    Converts inconsistently formatted story text (with mixed indentation, bullets, subpoints)
    into a structured format with separated Description, Requirements, and Acceptance Criteria.
    
    **Input:**
    - Raw story text (may include Description, Requirements, AC sections in any format)
    
    **Output:**
    - Normalized JSON with:
      - story_title: Extracted title
      - description: Clean narrative only
      - requirements: List of business rules/constraints
      - acceptance_criteria: Hierarchical structure with subpoints
    
    **Handles:**
    - Email template stories (Story 1009)
    - Billing logic stories with Gherkin (Story 916)
    - Validation rule stories (Story 1171)
    - Any mixed-format story
    """,
    responses={
        200: {
            "description": "Story normalized successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "normalized_story": {
                            "story_title": "Email Quote Template",
                            "description": "As a sales user, I want the Email Quote template...",
                            "requirements": [],
                            "acceptance_criteria": [
                                {
                                    "id": "AC1",
                                    "text": "The email subject must follow the format...",
                                    "subpoints": []
                                }
                            ]
                        }
                    }
                }
            }
        }
    }
)
async def normalize_story(request: StoryNormalizationRequest):
    """
    Normalize raw JIRA story text into structured format
    """
    try:
        story_processor, _, _, _ = get_components()
        
        # Normalize the story
        normalized = story_processor.normalize_story_input(request.raw_story_text)
        
        return NormalizedStoryResponse(
            success=True,
            normalized_story=normalized
        )
    except Exception as e:
        logger.error("Story normalization failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        return NormalizedStoryResponse(
            success=False,
            normalized_story={},
            error=str(e)
        )


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
async def process_story(request: JIRAStoryRequest, http_request: Request):
    """
    Process a JIRA story and generate markdown solution
    
    Flow:
    1. Extract key questions from story
    2. For each question: Search knowledge base and generate answer
    3. Generate narrative solution
    4. Format as markdown
    
    Supports cancellation via /solutions/cancel/{request_id} endpoint
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Register request for cancellation support
    async with _get_request_lock():
        _cancellation_tokens[request_id] = False
    
    try:
        # Monitor client disconnect
        async def check_disconnect():
            while True:
                await asyncio.sleep(0.5)
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, cancelling request", extra={"request_id": request_id})
                    async with _get_request_lock():
                        _cancellation_tokens[request_id] = True
                    break
        
        # Start disconnect checker
        disconnect_task = asyncio.create_task(check_disconnect())
        
        logger.info("Processing JIRA story", extra={
            "request_id": request_id,
            "story_id": request.story_id,
            "description_length": len(request.description),
            "criteria_count": len(request.acceptance_criteria)
        })
        
        # Check cancellation before starting
        await _check_cancellation(request_id)
        
        # Get components (including pipeline controller)
        story_processor, solution_generator, markdown_formatter, pipeline_controller = get_components()
        
        # Use Pipeline Controller for complete multi-stage processing
        logger.info("Using Pipeline Controller for complete processing", extra={
            "request_id": request_id,
            "story_id": request.story_id
        })
        
        # Extract title and story_id from normalized_story if not provided in request
        title = request.title
        story_id = request.story_id
        if request.normalized_story:
            # Use normalized_story title/story_title if not provided in request
            if not title:
                title = request.normalized_story.get('story_title')
            if not story_id:
                story_id = request.normalized_story.get('story_title')
            # Use normalized_story requirements if not provided in request
            requirements = request.requirements or request.normalized_story.get('requirements', [])
        else:
            requirements = request.requirements or []
        
        # Prepare story request for pipeline
        story_request = {
            "description": request.description,
            "acceptance_criteria": request.acceptance_criteria,
            "story_id": story_id,  # Use extracted story_id (from normalized_story if available)
            "title": title,  # Use extracted title (from normalized_story if available)
            "images": request.images or [],
            "requirements": requirements,  # Use extracted requirements (from normalized_story if available)
            "normalized_story": request.normalized_story,
            "raw_story_text": getattr(request, 'raw_story_text', None)
        }
        
        # Process through complete pipeline
        await _check_cancellation(request_id)
        
        pipeline_result = await pipeline_controller.process_story(
            story_request=story_request,
            request_id=request_id
        )
        
        await _check_cancellation(request_id)
        
        # Extract results from pipeline
        normalized_story = pipeline_result.get('normalized_story', {})
        technical_entities = pipeline_result.get('technical_entities', {})
        validated_entities = pipeline_result.get('validated_entities', {})
        fact_table = pipeline_result.get('fact_table', {})
        ac_mappings = pipeline_result.get('ac_mappings', [])
        solution_narrative = pipeline_result.get('solution_narrative', '')
        all_sources = pipeline_result.get('sources', [])
        vision_analysis = pipeline_result.get('vision_analysis', [])
        
        logger.info("Pipeline processing completed", extra={
            "request_id": request_id,
            "entities_extracted": pipeline_result.get('metadata', {}).get('entities_extracted', 0),
            "entities_validated": pipeline_result.get('metadata', {}).get('entities_validated', 0),
            "sources_found": len(all_sources)
        })
        
        # Format as precision markdown with all new sections
        title = request.title or normalized_story.get('story_title') or f"Solution for Story {request.story_id or 'Unknown'}"
        
        # Get acceptance criteria for formatting
        ac_for_formatting = []
        for ac in normalized_story.get('acceptance_criteria', []):
            if isinstance(ac, dict):
                ac_for_formatting.append(ac.get('text', ''))
            else:
                ac_for_formatting.append(str(ac))
        
        if not ac_for_formatting:
            ac_for_formatting = request.acceptance_criteria
        
        # Extract confidence and classification from pipeline result
        confidence_score = pipeline_result.get('confidence_score')
        confidence_details = pipeline_result.get('confidence_details', {})
        story_classification = pipeline_result.get('story_classification')
        pipeline_metadata = pipeline_result.get('metadata', {})
        
        # Format markdown
        # Use validated entities for consistent confidence reporting in markdown
        # Default to FINAL (delivery) output; keep precision available by switching to format_precision_solution.
        output_style = getattr(config, "OUTPUT_STYLE", None) or "final"
        if str(output_style).lower() == "precision":
            solution_markdown = markdown_formatter.format_precision_solution(
                title=title,
                story_id=request.story_id,
                narrative=solution_narrative,
                fact_table=fact_table,
                ac_mappings=ac_mappings,
                technical_entities=validated_entities or technical_entities,
                acceptance_criteria=ac_for_formatting,
                sources=all_sources,
                metadata=pipeline_metadata,  # Use full pipeline metadata
                confidence_score=confidence_score,
                confidence_details=confidence_details,
                story_classification=story_classification
            )
        else:
            solution_markdown = markdown_formatter.format_final_solution(
                title=title,
                story_id=request.story_id,
                narrative=solution_narrative,
                acceptance_criteria=ac_for_formatting,
                sources=all_sources,
            )
        
        await _check_cancellation(request_id)
        
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
                    "request_id": request_id,
                    "file_path": saved_file_path,
                    "story_id": request.story_id
                })
            except Exception as e:
                logger.warning("Failed to save solution markdown file", extra={
                    "request_id": request_id,
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
            "request_id": request_id,
            "story_id": request.story_id,
            "processing_time": processing_time,
            "success": True,
            "file_saved": saved_file_path is not None
        })
        
        return response
        
    except asyncio.CancelledError as e:
        logger.warning("Story processing cancelled", extra={
            "request_id": request_id,
            "story_id": request.story_id,
            "error": str(e)
        })
        raise HTTPException(
            status_code=499,  # Client Closed Request
            detail=f"Processing cancelled: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Story processing failed", extra={
            "request_id": request_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "story_id": request.story_id
        })
        
        return SolutionResponse(
            success=False,
            request_id=request_id,
            story_id=request.story_id,
            solution_markdown="",
            sources=[],
            processing_time=time.time() - start_time,
            saved_file_path=None,
            error=str(e)
        )
    finally:
        # Cleanup: Remove request from tracking
        async with _get_request_lock():
            _cancellation_tokens.pop(request_id, None)
        if 'disconnect_task' in locals():
            disconnect_task.cancel()


@router.post(
    "/process-stream",
    summary="Process JIRA Story (Streaming)",
    description="Process a JIRA story and stream solution generation in real-time using Server-Sent Events (SSE)",
    response_class=StreamingResponse
)
async def process_story_stream(request: JIRAStoryRequest, http_request: Request):
    """
    Stream solution generation in real-time using Server-Sent Events (SSE).
    This endpoint provides line-by-line solution updates similar to ChatGPT.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    async def generate_stream():
        try:
            # Register request for cancellation
            async with _get_request_lock():
                _cancellation_tokens[request_id] = False
            
            # Step 1: Extract questions (send progress)
            yield f"data: {json.dumps({'type': 'progress', 'step': 'extracting_questions', 'message': 'Analyzing story...'})}\n\n"
            
            await _check_cancellation(request_id)
            
            story_processor, solution_generator, markdown_formatter, pipeline_controller = get_components()
            
            # Use normalized story if provided, otherwise use request fields
            if request.normalized_story:
                # Extract from normalized structure
                normalized = request.normalized_story
                description = normalized.get('description', request.description)
                
                # Extract title and story_id from normalized_story if not provided in request
                title = request.title or normalized.get('story_title') or None
                story_id = request.story_id or normalized.get('story_title') or None
                
                # Convert AC structure to flat list for question extraction
                parsed_criteria = []
                for ac in normalized.get('acceptance_criteria', []):
                    ac_text = ac.get('text', '')
                    if ac_text:
                        parsed_criteria.append(ac_text)
                    # Add subpoints as separate AC items
                    for subpoint in ac.get('subpoints', []):
                        if subpoint:
                            parsed_criteria.append(f"{ac_text} - {subpoint}")
                # Fallback to request acceptance_criteria if normalized AC is empty
                if not parsed_criteria:
                    parsed_criteria = request.acceptance_criteria
            else:
                # Use request fields directly
                description = request.description
                title = request.title
                story_id = request.story_id
                # Acceptance criteria is already parsed by frontend, use as-is
                parsed_criteria = request.acceptance_criteria
                if isinstance(parsed_criteria, str):
                    # If it's a string, split by newlines (fallback)
                    parsed_criteria = [c.strip() for c in parsed_criteria.split('\n') if c.strip()]
                elif not isinstance(parsed_criteria, list):
                    parsed_criteria = []
            
            # Validate parsed_criteria is a list
            if not isinstance(parsed_criteria, list):
                logger.warning("Acceptance criteria is not a list, converting", extra={
                    "type": type(parsed_criteria).__name__,
                    "request_id": request_id
                })
                parsed_criteria = []
            
            questions = story_processor.extract_key_questions({
                "description": description,
                "acceptance_criteria": parsed_criteria
            })
            
            yield f"data: {json.dumps({'type': 'progress', 'step': 'questions_extracted', 'count': len(questions)})}\n\n"
            
            await _check_cancellation(request_id)
            
            # Step 2: Use Pipeline Controller for processing
            yield f"data: {json.dumps({'type': 'progress', 'step': 'processing_story', 'message': 'Processing story through pipeline...'})}\n\n"
            
            await _check_cancellation(request_id)
            
            # Step 3: Use Pipeline Controller for complete processing
            yield f"data: {json.dumps({'type': 'progress', 'step': 'searching_knowledge_base', 'message': 'Searching knowledge base...'})}\n\n"
            
            # Build story request for pipeline
            story_request = {
                "description": description,
                "acceptance_criteria": parsed_criteria,
                "story_id": story_id,  # Use extracted story_id (from normalized_story if available)
                "title": title,  # Use extracted title (from normalized_story if available)
                "images": request.images if hasattr(request, 'images') else [],
                "normalized_story": request.normalized_story,  # Include normalized_story so pipeline can use it
                "requirements": request.requirements or (request.normalized_story.get('requirements', []) if request.normalized_story else [])
            }
            
            # Process through pipeline
            pipeline_result = await pipeline_controller.process_story(
                story_request=story_request,
                request_id=request_id
            )
            
            # Extract results from pipeline (same as non-streaming endpoint)
            comprehensive_result = {
                "answer": pipeline_result.get('solution_narrative', ''),
                "sources": pipeline_result.get('sources', []),
                "retrieved_chunks": pipeline_result.get('retrieved_chunks', []),
                "vision_analysis": pipeline_result.get('vision_analysis', [])
            }
            
            # Extract precision formatting data (same as non-streaming endpoint)
            fact_table = pipeline_result.get('fact_table', {})
            ac_mappings = pipeline_result.get('ac_mappings', [])
            technical_entities = pipeline_result.get('technical_entities', {})
            validated_entities = pipeline_result.get('validated_entities', {})
            solution_narrative = pipeline_result.get('solution_narrative', '')
            confidence_score = pipeline_result.get('confidence_score')
            confidence_details = pipeline_result.get('confidence_details', {})
            story_classification = pipeline_result.get('story_classification')
            pipeline_metadata = pipeline_result.get('metadata', {})
            
            await _check_cancellation(request_id)
            
            yield f"data: {json.dumps({'type': 'progress', 'step': 'retrieval_complete', 'sources': len(comprehensive_result.get('sources', []))})}\n\n"
            
            # Step 4: Stream solution generation
            yield f"data: {json.dumps({'type': 'progress', 'step': 'generating_solution', 'message': 'Generating solution...'})}\n\n"
            
            accumulated_text = ""
            async for chunk in solution_generator.generate_focused_narrative_stream(
                story_context={
                    "description": description,
                    "acceptance_criteria": parsed_criteria
                },
                retrieved_content=comprehensive_result,
                questions=questions
            ):
                await _check_cancellation(request_id)
                if chunk:
                    accumulated_text += chunk
                    yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
            
            # Step 5: Format final solution using same logic as non-streaming endpoint
            # Use extracted title (already extracted from normalized_story if available)
            final_title = title or request.title or f"Story {story_id or request.story_id or 'Solution'}"
            
            # Prepare acceptance criteria for formatting (same as non-streaming endpoint)
            ac_for_formatting = []
            if request.normalized_story and request.normalized_story.get('acceptance_criteria'):
                for ac in request.normalized_story.get('acceptance_criteria', []):
                    if isinstance(ac, dict):
                        ac_for_formatting.append(ac.get('text', ''))
                    else:
                        ac_for_formatting.append(str(ac))
            
            if not ac_for_formatting:
                ac_for_formatting = parsed_criteria
            
            # Use same OUTPUT_STYLE logic as non-streaming endpoint
            output_style = getattr(config, "OUTPUT_STYLE", None) or "final"
            if str(output_style).lower() == "precision":
                # Use precision formatting with all detailed data
                solution_markdown = markdown_formatter.format_precision_solution(
                    title=final_title,
                    story_id=story_id or request.story_id,
                    narrative=accumulated_text or solution_narrative,
                    fact_table=fact_table,
                    ac_mappings=ac_mappings,
                    technical_entities=validated_entities or technical_entities,
                    acceptance_criteria=ac_for_formatting,
                    sources=comprehensive_result.get('sources', []),
                    metadata=pipeline_metadata,
                    confidence_score=confidence_score,
                    confidence_details=confidence_details,
                    story_classification=story_classification
                )
            else:
                # Use final formatting (simpler output)
                solution_markdown = markdown_formatter.format_final_solution(
                    title=final_title,
                    story_id=story_id or request.story_id,
                    narrative=accumulated_text or solution_narrative,
                    acceptance_criteria=ac_for_formatting,
                    sources=comprehensive_result.get('sources', []),
                    generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            
            # Save solution
            saved_path = save_solution_markdown(solution_markdown, story_id or request.story_id, final_title)
            
            # Format sources for response (matching SolutionResponse format)
            formatted_sources = [
                {
                    "document": s.get('document', 'Unknown'),
                    "page": s.get('page', 0),
                    "confidence": float(s.get('similarity_score', 0.0)),
                    "text_snippet": s.get('content_preview', '')
                }
                for s in comprehensive_result.get('sources', [])[:15]  # Top 15 sources
            ]
            
            # Send complete solution with all fields matching SolutionResponse
            yield f"data: {json.dumps({
                'type': 'complete', 
                'solution': solution_markdown, 
                'solution_markdown': solution_markdown,  # Alias for consistency
                'story_id': story_id or request.story_id,  # Use extracted story_id
                'saved_file_path': saved_path, 
                'processing_time': time.time() - start_time,
                'sources': formatted_sources
            })}\n\n"
            
        except Exception as e:
            logger.error("Streaming error", extra={"error": str(e), "request_id": request_id})
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Cleanup
            async with _get_request_lock():
                _cancellation_tokens.pop(request_id, None)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post(
    "/cancel/{request_id}",
    summary="Cancel Story Processing",
    description="Cancel an ongoing story processing request",
    responses={
        200: {
            "description": "Cancellation request processed",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Request cancelled successfully",
                        "request_id": "uuid-here"
                    }
                }
            }
        },
        404: {
            "description": "Request not found or already completed",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Request not found or already completed"
                    }
                }
            }
        }
    }
)
async def cancel_story_processing(request_id: str):
    """Cancel an ongoing story processing request"""
    async with _get_request_lock():
        if request_id in _cancellation_tokens:
            _cancellation_tokens[request_id] = True
            logger.info("Cancellation requested", extra={"request_id": request_id})
            return {
                "success": True,
                "message": "Request cancelled successfully",
                "request_id": request_id
            }
        else:
            logger.warning("Cancellation requested for non-existent request", extra={"request_id": request_id})
            raise HTTPException(
                status_code=404,
                detail="Request not found or already completed"
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
            story_processor, _, _, _ = get_components()
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
        
        # Check RAG Service (now integrated into PipelineController)
        try:
            _, _, _, pipeline_controller = get_components()
            # RAG service functionality now handled by pipeline_controller
            doc_count = 0
            if hasattr(pipeline_controller, 'hybrid_retriever'):
                doc_count = len(pipeline_controller.hybrid_retriever.content_index)
            components["rag_service"] = {
                "status": "ok",
                "message": f"RAG functionality integrated into PipelineController ({doc_count} documents available)"
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
            _, solution_generator, _, _ = get_components()
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
        
        # Check Pipeline Controller
        try:
            _, _, _, pipeline_controller = get_components()
            components["pipeline_controller"] = {
                "status": "ok",
                "message": "Pipeline Controller ready"
            }
        except Exception as e:
            components["pipeline_controller"] = {
                "status": "error",
                "message": f"Pipeline Controller failed: {str(e)}"
            }
            overall_status = "degraded"
            logger.warning("Pipeline Controller health check failed", extra={"error": str(e)})
        
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


@router.get(
    "/manuals",
    response_model=ManualsListResponse,
    summary="List Available Manuals",
    description="""
    Get a list of all available manuals in the knowledge base.
    
    **Response:**
    - List of manual names
    - Total count of manuals
    """,
    responses={
        200: {
            "description": "Successfully retrieved manuals list",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "manuals": [
                            {"name": "EU_AccountingForProjects_2025R1", "display_name": "EU Accounting For Projects 2025R1"},
                            {"name": "C110_Case_Management_2025R1", "display_name": "C110 Case Management 2025R1"}
                        ],
                        "count": 2
                    }
                }
            }
        }
    }
)
async def list_manuals():
    """List all available manuals in the knowledge base"""
    try:
        logger.info("Listing available manuals")
        
        # Get pipeline controller to access document list
        _, _, _, pipeline_controller = get_components()
        
        # Get list of documents (via inferencer in pipeline controller)
        # Access hybrid_retriever's content_index for document list
        documents = []
        if hasattr(pipeline_controller, 'hybrid_retriever'):
            content_index = pipeline_controller.hybrid_retriever.content_index
            for doc_id in content_index.keys():
                documents.append({
                    "document_name": doc_id,
                    "document_id": doc_id
                })
        
        # Format manual names (remove underscores, add spaces, clean up)
        manuals = []
        for doc in documents:
            doc_name = doc.get("document_name", "")
            # Skip domain.json and other non-manual files
            if doc_name.endswith(".json") or not doc_name:
                continue
            
            # Format display name: replace underscores with spaces, remove version suffixes if desired
            display_name = doc_name.replace("_", " ").replace(" (1)", "").replace(" (2)", "")
            
            manuals.append(ManualInfo(
                name=doc_name,
                display_name=display_name
            ))
        
        # Sort manuals alphabetically by display name
        manuals.sort(key=lambda x: x.display_name or x.name)
        
        logger.info("Manuals list retrieved", extra={
            "manual_count": len(manuals)
        })
        
        return ManualsListResponse(
            success=True,
            manuals=manuals,
            count=len(manuals)
        )
        
    except Exception as e:
        logger.error("Failed to list manuals", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        return ManualsListResponse(
            success=False,
            manuals=[],
            count=0,
            error=str(e)
        )


"""
RAG Service - Wrapper around VDR logic for knowledge base search
"""

from typing import Dict, Any, List, Optional

from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation
from src.core.infer import VDRInferencer

class RAGService:
    """Wrapper service for RAG operations using VDRInferencer"""
    
    def __init__(self):
        self.logger = get_logger("RAG_SERVICE")
        
        try:
            self.logger.info("Initializing RAG Service with VDRInferencer", extra={
                "knowledge_base_path": config.LOCAL_BASE_PATH
            })
            
            # Initialize VDRInferencer (uses local config automatically)
            self.inferencer = VDRInferencer()
            
            self.logger.info("RAG Service initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize RAG Service", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    async def search_knowledge_base(self, question: str, top_k: int = 3, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search knowledge base for relevant information
        
        Args:
            question: Question to search for
            top_k: Number of top results to return
            search_params: Optional search parameters from intent analysis
            
        Returns:
            Dictionary with search results
        """
        try:
            with TimedOperation("rag_search", self.logger, question=question[:50]):
                self.logger.info("Searching knowledge base", extra={
                    "question": question[:100],
                    "top_k": top_k,
                    "has_search_params": bool(search_params)
                })
                
                # Use VDRInferencer to search with optional intent-based parameters
                result = await self.inferencer.ask_question(
                    question=question,
                    top_k=top_k,
                    max_retries=1,
                    search_params=search_params
                )
                
                self.logger.info("Knowledge base search completed", extra={
                    "sources_found": len(result.get('sources', [])),
                    "has_answer": bool(result.get('answer'))
                })
                
                return result
                
        except Exception as e:
            self.logger.error("Failed to search knowledge base", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "question": question[:100]
            })
            return {
                "answer": "Unable to search knowledge base at this time.",
                "sources": [],
                "error": str(e)
            }
    
    async def generate_answer(self, question: str, top_k: int = 3, search_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate accurate answer using RAG + Vision
        
        Args:
            question: Question to answer
            top_k: Number of top results to consider
            search_params: Optional search parameters from intent analysis
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        return await self.search_knowledge_base(question, top_k, search_params)
    
    def _create_unified_query(
        self, 
        description: str, 
        questions: List[str], 
        acceptance_criteria: List[str]
    ) -> str:
        """
        Create unified query that incorporates story context and all questions.
        Questions are used as context to guide retrieval, not as separate queries.
        """
        query_parts = [description]
        
        # Add questions as context (not separate queries)
        if questions:
            query_parts.append("\n\nKey aspects to address:")
            for i, q in enumerate(questions, 1):
                query_parts.append(f"{i}. {q}")
        
        # Add acceptance criteria as requirements
        if acceptance_criteria:
            query_parts.append("\n\nRequirements:")
            for criterion in acceptance_criteria:
                query_parts.append(f"- {criterion}")
        
        return "\n".join(query_parts)
    
    async def generate_comprehensive_answer(
        self, 
        story_context: Dict[str, Any],
        top_k: int = 16,  # Increased for comprehensive context
        search_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive answer using unified retrieval with all questions as context.
        
        This approach:
        1. Uses all questions together to guide retrieval (not as separate queries)
        2. Retrieves comprehensive context in a single pass
        3. Generates one focused answer grounded in retrieved content
        
        Args:
            story_context: Dictionary with 'description', 'questions', 'acceptance_criteria'
            top_k: Number of top results to retrieve (increased for comprehensive context)
            search_params: Optional search parameters from intent analysis
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            description = story_context.get("description", "")
            questions = story_context.get("questions", [])
            acceptance_criteria = story_context.get("acceptance_criteria", [])
            
            # Create unified query that incorporates all questions as context
            unified_query = self._create_unified_query(description, questions, acceptance_criteria)
            
            self.logger.info("Generating comprehensive answer with unified retrieval", extra={
                "description_length": len(description),
                "question_count": len(questions),
                "criteria_count": len(acceptance_criteria),
                "top_k": top_k
            })
            
            # Single comprehensive retrieval with unified context
            result = await self.inferencer.ask_question(
                question=unified_query,
                top_k=top_k,
                max_retries=1,
                search_params=search_params
            )
            
            self.logger.info("Comprehensive answer generated", extra={
                "sources_found": len(result.get('sources', [])),
                "has_answer": bool(result.get('answer'))
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to generate comprehensive answer", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "answer": "Unable to generate comprehensive answer at this time.",
                "sources": [],
                "error": str(e)
            }
    
    def list_available_documents(self) -> List[Dict[str, Any]]:
        """List all available documents in knowledge base"""
        try:
            return self.inferencer.list_available_documents()
        except Exception as e:
            self.logger.error("Failed to list documents", extra={
                "error": str(e)
            })
            return []


"""
Markdown formatting utilities for solution output
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger_utils import get_logger

class MarkdownFormatter:
    """Formats solution data into structured markdown"""
    
    def __init__(self):
        self.logger = get_logger("MARKDOWN_FORMATTER")
    
    def format_solution(
        self,
        title: str,
        story_id: Optional[str],
        questions: List[str],
        answers: List[Dict[str, Any]],
        narrative: str,
        acceptance_criteria: List[str],
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format complete solution as markdown
        
        Args:
            title: Solution title
            story_id: JIRA story ID (optional)
            questions: List of extracted questions
            answers: List of answer dictionaries with 'question', 'answer', 'sources'
            narrative: Generated narrative solution
            acceptance_criteria: Original acceptance criteria
            sources: List of source references
            metadata: Additional metadata
            
        Returns:
            Formatted markdown string
        """
        try:
            markdown_parts = []
            
            # Title and header
            markdown_parts.append(f"# {title}\n")
            if story_id:
                markdown_parts.append(f"**Story ID:** {story_id}\n")
            markdown_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            markdown_parts.append("\n---\n")
            
            # Single Comprehensive Solution (THE answer to the JIRA story task)
            # If answers list is empty, this means we're using unified retrieval approach
            # The narrative IS the complete solution answer
            if not answers:
                # Unified comprehensive solution format
                markdown_parts.append("## Solution\n\n")
                markdown_parts.append(narrative)
                markdown_parts.append("\n\n---\n")
                
                # Show questions that guided the solution (for reference only)
                if questions:
                    markdown_parts.append("## Key Questions Considered\n\n")
                    markdown_parts.append("*The following questions were used internally to guide the comprehensive solution:*\n\n")
                    for i, question in enumerate(questions, 1):
                        markdown_parts.append(f"{i}. {question}\n")
                    markdown_parts.append("\n---\n")
            else:
                # Legacy format with separate Q&A pairs (for backward compatibility)
                markdown_parts.append("## Overview\n")
                markdown_parts.append(narrative)
                markdown_parts.append("\n\n---\n")
                
                # Key Questions & Answers
                if questions and answers:
                    markdown_parts.append("## Key Questions & Answers\n\n")
                    for i, (question, answer_data) in enumerate(zip(questions, answers), 1):
                        markdown_parts.append(f"### Question {i}: {question}\n\n")
                        if isinstance(answer_data, dict):
                            answer_text = answer_data.get('answer', str(answer_data))
                        else:
                            answer_text = str(answer_data)
                        markdown_parts.append(f"{answer_text}\n\n")
                        
                        # Add sources for this answer if available
                        answer_sources = answer_data.get('sources', [])
                        if answer_sources:
                            markdown_parts.append("**Sources:**\n")
                            for source in answer_sources[:3]:  # Limit to top 3 sources
                                doc_name = source.get('document', 'Unknown')
                                page = source.get('page', 'N/A')
                                markdown_parts.append(f"- {doc_name} (Page {page})\n")
                            markdown_parts.append("\n")
                        markdown_parts.append("---\n\n")
                
                # Detailed Solution
                markdown_parts.append("## Detailed Solution\n\n")
                markdown_parts.append(narrative)
                markdown_parts.append("\n\n---\n")
            
            # Acceptance Criteria Checklist
            if acceptance_criteria:
                markdown_parts.append("## Acceptance Criteria\n\n")
                for i, criterion in enumerate(acceptance_criteria, 1):
                    markdown_parts.append(f"- [ ] {criterion}\n")
                markdown_parts.append("\n---\n")
            
            # Sources/References
            if sources:
                markdown_parts.append("## References\n\n")
                unique_sources = {}
                for source in sources:
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page', 'N/A')
                    key = f"{doc_name}_{page}"
                    if key not in unique_sources:
                        unique_sources[key] = source
                
                for source in list(unique_sources.values())[:10]:  # Limit to top 10 unique sources
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page', 'N/A')
                    confidence = source.get('confidence', 0.0)
                    markdown_parts.append(f"- **{doc_name}** (Page {page}, Confidence: {confidence:.2f})\n")
                markdown_parts.append("\n---\n")
            
            # Metadata section (if provided)
            if metadata:
                markdown_parts.append("## Processing Information\n\n")
                processing_time = metadata.get('processing_time', 0)
                markdown_parts.append(f"- **Processing Time:** {processing_time:.2f}s\n")
                
                if 'cost_metrics' in metadata:
                    cost = metadata['cost_metrics']
                    markdown_parts.append(f"- **Total Cost:** ₹{cost.get('total_cost_inr', 0):.4f}\n")
                
                if 'documents_analyzed' in metadata:
                    markdown_parts.append(f"- **Documents Analyzed:** {metadata['documents_analyzed']}\n")
            
            return "\n".join(markdown_parts)
            
        except Exception as e:
            self.logger.error("Failed to format markdown", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return basic fallback format
            return f"# {title}\n\n{narrative}\n\n## Questions\n\n" + "\n".join(f"- {q}" for q in questions)
    
    def format_simple_solution(
        self,
        title: str,
        narrative: str,
        questions: Optional[List[str]] = None,
        sources: Optional[List[Dict]] = None
    ) -> str:
        """Format a simple solution without all sections"""
        markdown_parts = [f"# {title}\n\n", narrative]
        
        if questions:
            markdown_parts.append("\n## Key Questions\n\n")
            for q in questions:
                markdown_parts.append(f"- {q}\n")
        
        if sources:
            markdown_parts.append("\n## Sources\n\n")
            for source in sources[:5]:
                doc = source.get('document', 'Unknown')
                page = source.get('page', 'N/A')
                markdown_parts.append(f"- {doc} (Page {page})\n")
        
        return "\n".join(markdown_parts)


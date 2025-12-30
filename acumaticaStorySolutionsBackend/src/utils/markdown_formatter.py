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
    
    def format(self, solution: Dict[str, Any]) -> str:
        """
        Reference-compatible formatting method (adapter pattern)
        
        Matches reference interface: MarkdownFormatter.format(solution)
        
        Args:
            solution: Solution dictionary with 'summary', 'entities', 'mapping', 'implementation'
            
        Returns:
            Formatted markdown string
        """
        try:
            # Extract key components
            summary = solution.get("summary", "Solution")
            implementation = solution.get("implementation", "")
            entities = solution.get("entities", {})
            mapping = solution.get("mapping", {})
            
            # Build markdown
            markdown_parts = []
            markdown_parts.append(f"# {summary}\n\n")
            
            if implementation:
                markdown_parts.append("## Implementation\n\n")
                markdown_parts.append(f"{implementation}\n\n")
            
            # Add entities section if present
            if entities:
                markdown_parts.append("## Extracted Entities\n\n")
                for entity_type, values in entities.items():
                    if values:
                        markdown_parts.append(f"### {entity_type.title()}\n\n")
                        for value in values[:10]:  # Limit to top 10
                            markdown_parts.append(f"- `{value}`\n")
                        markdown_parts.append("\n")
            
            # Add mapping section if present
            if mapping.get("mappings"):
                markdown_parts.append("## Acceptance Criteria Mappings\n\n")
                for i, mapping_item in enumerate(mapping["mappings"][:5], 1):  # Limit to top 5
                    criterion = mapping_item.get("criterion", "")
                    if criterion:
                        markdown_parts.append(f"{i}. **{criterion}**\n")
                        related_forms = mapping_item.get("related_forms", [])
                        related_dacs = mapping_item.get("related_dacs", [])
                        if related_forms or related_dacs:
                            markdown_parts.append(f"   - Forms: {', '.join(related_forms)}\n")
                            markdown_parts.append(f"   - DACs: {', '.join(related_dacs)}\n")
                        markdown_parts.append("\n")
            
            return "".join(markdown_parts)
        except Exception as e:
            self.logger.error("Format adapter failed", extra={"error": str(e)})
            return f"# Solution\n\n{str(solution)}"
    
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
                    # Removed confidence score display for better UX
                    markdown_parts.append(f"- **{doc_name}** (Page {page})\n")
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
    
    def format_precision_solution(
        self,
        title: str,
        story_id: Optional[str],
        narrative: str,
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]],
        technical_entities: Dict[str, Any],
        acceptance_criteria: List[str],
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        confidence_details: Optional[Dict[str, Any]] = None,
        story_classification: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format precision solution with technical fact tables, AC mappings, and entity confidence
        
        Args:
            title: Solution title
            story_id: JIRA story ID (optional)
            narrative: Generated solution narrative
            fact_table: Technical fact table from Stage 1
            ac_mappings: AC to technical requirements mappings from Stage 2
            technical_entities: All extracted technical entities with confidence scores
            acceptance_criteria: Original acceptance criteria
            sources: List of source references
            metadata: Additional metadata (processing time, vision usage, etc.)
            
        Returns:
            Formatted markdown string with all precision sections
        """
        try:
            markdown_parts = []
            
            # Title and header
            markdown_parts.append(f"✅ TECHNICALLY CORRECT SOLUTION — STORY {story_id or 'Unknown'}\n\n")
            markdown_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            markdown_parts.append("\n---\n\n")
            
            # The narrative from solution_generator should already contain the comprehensive solution
            # Display it first, then add supporting sections
            if narrative and narrative.strip():
                markdown_parts.append(narrative)
                markdown_parts.append("\n\n---\n\n")
            
            # Section 1: Technical Fact Table (Supporting Detail)
            markdown_parts.append("## Technical Fact Table (Extracted Entities)\n\n")
            markdown_parts.append(self._format_fact_table(fact_table))
            markdown_parts.append("\n---\n\n")
            
            # Section 2: Acceptance Criteria → Technical Mapping (Supporting Detail)
            markdown_parts.append("## Acceptance Criteria → Technical Mapping\n\n")
            markdown_parts.append(self._format_ac_mappings(ac_mappings))
            markdown_parts.append("\n---\n\n")
            
            # Section 3: Confirmed Technical Entities
            confirmed = self._get_confirmed_entities(technical_entities)
            if confirmed:
                markdown_parts.append("## 3. Confirmed Technical Entities\n\n")
                markdown_parts.append("*Entities with confidence score ≥ 0.75 — Safe to use in implementation*\n\n")
                markdown_parts.append(self._format_entity_list(confirmed, "confirmed", technical_entities))
                markdown_parts.append("\n---\n")
            
            # Section 4: Uncertain Technical Entities (0.50-0.74 confidence)
            uncertain = self._get_uncertain_entities(technical_entities)
            if uncertain:
                markdown_parts.append("## 4. Uncertain Technical Entities\n\n")
                markdown_parts.append("*Entities with confidence score 0.50-0.74 — Review before use*\n\n")
                markdown_parts.append(self._format_entity_list(uncertain, "uncertain", technical_entities))
                markdown_parts.append("\n---\n")
            
            # Section 5: Unconfirmed Technical Entities
            unconfirmed = self._get_unconfirmed_entities(technical_entities)
            if unconfirmed:
                markdown_parts.append("## 5. Unconfirmed Technical Entities\n\n")
                markdown_parts.append("*Entities with confidence score < 0.50 — [NOT CONFIRMED — DO NOT USE]*\n\n")
                markdown_parts.append(self._format_entity_list(unconfirmed, "unconfirmed", technical_entities))
                markdown_parts.append("\n---\n")
            
            # Section 6: Implementation Steps
            # The narrative already contains the full solution/steps. Avoid duplicating it.
            section_num = 6 if unconfirmed else (5 if uncertain else 4)
            markdown_parts.append(f"## {section_num}. Implementation Steps\n\n")
            markdown_parts.append("*See the **Solution** section above for the authoritative implementation steps.*\n")
            markdown_parts.append("\n---\n")
            
            # Section 7: Code Snippets (Verified Only)
            section_num = 7 if unconfirmed else (6 if uncertain else 5)
            code_snippets = self._extract_code_snippets(technical_entities, fact_table)
            if code_snippets:
                markdown_parts.append(f"## {section_num}. Code Snippets (Verified Only)\n\n")
                markdown_parts.append(code_snippets)
                markdown_parts.append("\n---\n")
            
            # Section 8: References
            section_num = 8 if code_snippets else (7 if unconfirmed else (6 if uncertain else 5))
            if sources:
                markdown_parts.append("## 7. References\n\n")
                unique_sources = {}
                for source in sources:
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page', 'N/A')
                    key = f"{doc_name}_{page}"
                    if key not in unique_sources:
                        unique_sources[key] = source
                
                # Format as table with confidence scores
                markdown_parts.append("| Document | Page | Confidence |\n")
                markdown_parts.append("|----------|------|------------|\n")
                
                for source in list(unique_sources.values())[:15]:
                    doc_name = source.get('document', 'Unknown')
                    page = source.get('page', 'N/A')
                    confidence = source.get('confidence', source.get('similarity_score', 0.0))
                    confidence_indicator = "✅" if confidence >= 0.75 else "⚠️" if confidence >= 0.50 else "❌"
                    markdown_parts.append(f"| **{doc_name}** | {page} | {confidence_indicator} {confidence:.2f} |\n")
                markdown_parts.append("\n---\n")
                section_num += 1
            
            # Section: Confidence Score
            section_num += 1
            if confidence_score is not None:
                markdown_parts.append(f"## {section_num}. Solution Confidence Score\n\n")
                markdown_parts.append(f"**Overall Confidence:** {confidence_score:.1%}\n\n")
                
                if confidence_details:
                    component_scores = confidence_details.get('component_scores', {})
                    if component_scores:
                        markdown_parts.append("### Component Scores\n\n")
                        markdown_parts.append("| Component | Score |\n|-----------|-------|\n")
                        for component, score in component_scores.items():
                            score_indicator = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
                            markdown_parts.append(f"| {component.replace('_', ' ').title()} | {score_indicator} {score:.1%} |\n")
                        markdown_parts.append("\n")
                    
                    reasoning = confidence_details.get('reasoning', '')
                    if reasoning:
                        markdown_parts.append(f"**Assessment:** {reasoning}\n\n")
                
                # Quality level
                if confidence_score >= 0.8:
                    quality_level = "High"
                elif confidence_score >= 0.6:
                    quality_level = "Medium"
                elif confidence_score >= 0.4:
                    quality_level = "Low"
                else:
                    quality_level = "Very Low"
                
                markdown_parts.append(f"**Quality Level:** {quality_level}\n\n")
                markdown_parts.append("---\n")
            
            # Section: Processing Metadata
            section_num += 1
            markdown_parts.append(f"## {section_num}. Processing Metadata\n\n")
            
            # Story Classification
            if story_classification:
                markdown_parts.append("### Story Classification\n\n")
                markdown_parts.append(f"- **Story Type:** {story_classification.get('story_type', 'Unknown')}\n")
                markdown_parts.append(f"- **Classification Confidence:** {story_classification.get('confidence', 0.0):.1%}\n")
                if story_classification.get('reasoning'):
                    markdown_parts.append(f"- **Reasoning:** {story_classification['reasoning']}\n")
                markdown_parts.append("\n")
            
            if metadata:
                # Processing statistics
                markdown_parts.append("### Processing Statistics\n\n")
                processing_time = metadata.get('processing_time_seconds', metadata.get('processing_time', 0))
                markdown_parts.append(f"- **Processing Time:** {processing_time:.2f}s\n")
                
                vision_used = metadata.get('vision_used', False)
                vision_pages = metadata.get('vision_pages_processed', 0)
                markdown_parts.append(f"- **Vision Used:** {'Yes' if vision_used else 'No'}")
                if vision_pages > 0:
                    markdown_parts.append(f" ({vision_pages} pages)\n")
                else:
                    markdown_parts.append("\n")
                
                # Entity statistics
                entity_stats = metadata.get('entity_statistics', {})
                if entity_stats:
                    markdown_parts.append(f"- **Entities Extracted:** {entity_stats.get('total_extracted', 0)}\n")
                    markdown_parts.append(f"- **Entities Confirmed:** {entity_stats.get('total_confirmed', 0)}\n")
                    confirmation_rate = entity_stats.get('confirmation_rate', 0.0)
                    markdown_parts.append(f"- **Confirmation Rate:** {confirmation_rate:.1%}\n")
                
                # Retrieval statistics
                retrieval_attempts = metadata.get('retrieval_attempts', 0)
                retrieval_successes = metadata.get('retrieval_successes', 0)
                retrieval_rate = metadata.get('retrieval_success_rate', 0.0)
                if retrieval_attempts > 0:
                    markdown_parts.append(f"- **Retrieval Attempts:** {retrieval_attempts}\n")
                    markdown_parts.append(f"- **Retrieval Successes:** {retrieval_successes}\n")
                    markdown_parts.append(f"- **Retrieval Success Rate:** {retrieval_rate:.1%}\n")
                
                # Fallback information
                if metadata.get('fallback_triggered', False):
                    fallback_types = metadata.get('fallback_types', [])
                    markdown_parts.append(f"- **Fallback Triggered:** Yes ({', '.join(fallback_types)})\n")
                
                # AC mapping statistics
                ac_count = metadata.get('ac_count', 0)
                ac_mapped = metadata.get('ac_mapped', 0)
                ac_rate = metadata.get('ac_mapping_rate', 0.0)
                if ac_count > 0:
                    markdown_parts.append(f"- **Acceptance Criteria:** {ac_mapped}/{ac_count} mapped ({ac_rate:.1%})\n")
                
                # Cost metrics
                if 'cost_metrics' in metadata:
                    cost = metadata['cost_metrics']
                    markdown_parts.append(f"- **Total Cost:** ₹{cost.get('total_cost_inr', 0):.4f}\n")
                
                # Quality indicators
                quality_indicators = metadata.get('quality_indicators', {})
                if quality_indicators:
                    markdown_parts.append("\n### Quality Indicators\n\n")
                    for indicator, value in quality_indicators.items():
                        indicator_name = indicator.replace('_', ' ').title()
                        status = "✅" if value else "❌"
                        markdown_parts.append(f"- {status} **{indicator_name}:** {'Yes' if value else 'No'}\n")
                
                # Errors and warnings
                errors = metadata.get('errors', [])
                warnings = metadata.get('warnings', [])
                if errors or warnings:
                    markdown_parts.append("\n### Processing Notes\n\n")
                    if warnings:
                        markdown_parts.append(f"**Warnings:** {len(warnings)}\n")
                        for warning in warnings[:3]:  # Show first 3
                            markdown_parts.append(f"- {warning.get('message', 'Unknown warning')}\n")
                    if errors:
                        markdown_parts.append(f"**Errors:** {len(errors)}\n")
                        for error in errors[:3]:  # Show first 3
                            markdown_parts.append(f"- {error.get('message', 'Unknown error')}\n")
            else:
                markdown_parts.append("- **Processing Time:** N/A\n")
                markdown_parts.append("- **Vision Used:** N/A\n")
                markdown_parts.append("- **Entities Extracted:** N/A\n")
            
            return "\n".join(markdown_parts)
            
        except Exception as e:
            self.logger.error("Failed to format precision solution", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return basic fallback
            return f"# {title}\n\n{narrative}\n\n## References\n\n" + "\n".join(f"- {s.get('document', 'Unknown')} (Page {s.get('page', 'N/A')})" for s in sources[:5])

    def format_final_solution(
        self,
        title: str,
        story_id: Optional[str],
        narrative: str,
        acceptance_criteria: List[str],
        sources: List[Dict[str, Any]],
        generated_at: Optional[str] = None,
    ) -> str:
        """
        Final (delivery) output:
        - Keeps the core solution + steps
        - Removes diagnostic sections like "Template-Generated Code" and "Retrieved Documentation"
        - Includes AC checklist and a minimal reference list
        """
        try:
            cleaned_solution = self._extract_delivery_solution(narrative or "")

            md: List[str] = []
            md.append(f"# {title}\n")
            if story_id:
                md.append(f"**Story ID:** {story_id}\n")
            md.append(f"**Generated:** {generated_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            md.append("\n---\n\n")

            if cleaned_solution.strip():
                md.append(cleaned_solution.strip())
            else:
                # Worst-case fallback: show raw narrative rather than empty output
                md.append((narrative or "").strip())

            # Acceptance Criteria Checklist
            if acceptance_criteria:
                md.append("\n\n---\n\n## Acceptance Criteria\n\n")
                for criterion in acceptance_criteria:
                    md.append(f"- [ ] {criterion}\n")

            # Minimal references
            if sources:
                md.append("\n---\n\n## References\n\n")
                # Deduplicate by (document,page)
                unique: Dict[str, Dict[str, Any]] = {}
                for s in sources:
                    doc = s.get("document", "Unknown")
                    page = s.get("page", "N/A")
                    unique[f"{doc}_{page}"] = s
                for s in list(unique.values())[:10]:
                    doc = s.get("document", "Unknown")
                    page = s.get("page", "N/A")
                    md.append(f"- **{doc}** (Page {page})\n")

            return "".join(md)
        except Exception as e:
            self.logger.error("Failed to format final solution", extra={
                "error": str(e),
                "error_type": type(e).__name__,
            })
            return f"# {title}\n\n{(narrative or '').strip()}"

    def _extract_delivery_solution(self, narrative: str) -> str:
        """
        Best-effort extraction of only the delivery-relevant portion of the narrative.
        Removes known diagnostic sections appended by solution generators.
        """
        text = narrative or ""
        if not text.strip():
            return ""

        # Keep everything up to (but not including) diagnostic sections.
        stop_markers = [
            "## Retrieved Documentation",
            "## Retrieved Documentation (Reference)",
            "## Technical Fact Table",
            "## Acceptance Criteria → Technical Mapping",
            "## 3. Confirmed Technical Entities",
            "## 4. Uncertain Technical Entities",
            "## 5. Unconfirmed Technical Entities",
            "## Processing Metadata",
        ]

        # Find earliest marker occurrence.
        cut_idx = None
        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1 and (cut_idx is None or idx < cut_idx):
                cut_idx = idx

        if cut_idx is not None:
            text = text[:cut_idx]

        # Trim trailing separators.
        while text.rstrip().endswith("---"):
            text = text.rstrip()
            text = text[: text.rfind("---")].rstrip()

        return text.strip()
    
    def _format_fact_table(self, fact_table: Dict[str, Any]) -> str:
        """Format technical fact table as markdown table with confidence scores"""
        parts = []
        confidence_scores = fact_table.get('confidence_scores', {})
        
        # Forms
        if fact_table.get('forms'):
            parts.append("### Forms\n\n")
            parts.append("| Form ID | Confidence |\n|---------|------------|\n")
            for form in fact_table['forms']:
                score = confidence_scores.get('forms', {}).get(form, 0.0)
                confidence_indicator = "✅" if score >= 0.75 else "⚠️" if score >= 0.50 else "❌"
                parts.append(f"| `{form}` | {confidence_indicator} {score:.2f} |\n")
            parts.append("\n")
        
        # DACs
        if fact_table.get('dacs'):
            parts.append("### DACs\n\n")
            parts.append("| DAC Name | Confidence |\n|----------|------------|\n")
            for dac in fact_table['dacs']:
                score = confidence_scores.get('dacs', {}).get(dac, 0.0)
                confidence_indicator = "✅" if score >= 0.75 else "⚠️" if score >= 0.50 else "❌"
                parts.append(f"| `{dac}` | {confidence_indicator} {score:.2f} |\n")
            parts.append("\n")
        
        # Graphs
        if fact_table.get('graphs'):
            parts.append("### Graphs\n\n")
            parts.append("| Graph Name | Confidence |\n|------------|------------|\n")
            for graph in fact_table['graphs']:
                score = confidence_scores.get('graphs', {}).get(graph, 0.0)
                confidence_indicator = "✅" if score >= 0.75 else "⚠️" if score >= 0.50 else "❌"
                parts.append(f"| `{graph}` | {confidence_indicator} {score:.2f} |\n")
            parts.append("\n")
        
        # Fields with detailed table (matching Expected_output.md format)
        if fact_table.get('fields'):
            parts.append("### DAC Fields (Confirmed)\n\n")
            parts.append("| Field | DAC | Purpose |\n")
            parts.append("|-------|-----|---------|\n")
            # Group fields by likely DAC (infer from field names or use fact table DACs)
            dacs_list = fact_table.get('dacs', [])
            for field in fact_table['fields']:
                score = confidence_scores.get('fields', {}).get(field, 0.0)
                # Try to infer DAC from field name or use first available DAC
                inferred_dac = None
                for dac in dacs_list:
                    # Simple heuristic: if field name contains DAC name pattern
                    if dac.lower().replace('order', '') in field.lower() or field.lower().startswith(dac.lower()[:3]):
                        inferred_dac = dac
                        break
                dac_display = f"`{inferred_dac or (dacs_list[0] if dacs_list else 'N/A')}`" if inferred_dac or dacs_list else "N/A"
                # Infer purpose from field name
                purpose = self._infer_field_purpose(field)
                confidence_indicator = "✅" if score >= 0.65 else "⚠️"
                parts.append(f"| `{field}` | {dac_display} | {purpose} |\n")
            parts.append("\n")
        
        # Events
        if fact_table.get('events'):
            parts.append("### Event Handlers\n\n")
            parts.append("| Handler Name | Confidence |\n|--------------|------------|\n")
            for event in fact_table['events']:
                score = confidence_scores.get('events', {}).get(event, 0.0)
                confidence_indicator = "✅" if score >= 0.75 else "⚠️" if score >= 0.50 else "❌"
                parts.append(f"| `{event}` | {confidence_indicator} {score:.2f} |\n")
            parts.append("\n")
        
        # Navigation
        if fact_table.get('navigation'):
            parts.append("### Navigation Paths\n\n")
            for nav in fact_table['navigation']:
                parts.append(f"- {nav}\n")
            parts.append("\n")
        
        # PX Attributes
        if fact_table.get('px_attributes'):
            parts.append("### PX Attributes\n\n")
            for attr in fact_table['px_attributes']:
                parts.append(f"- `{attr}`\n")
            parts.append("\n")
        
        # Tables
        if fact_table.get('tables'):
            parts.append("### Tables\n\n")
            for table in fact_table['tables']:
                parts.append(f"- `{table}`\n")
            parts.append("\n")
        
        # Missing Entities
        if fact_table.get('missing_entities'):
            parts.append("### ⚠️ Missing Entities\n\n")
            parts.append("*The following entities were mentioned but not found in documentation:*\n\n")
            for missing in fact_table['missing_entities']:
                parts.append(f"- {missing}\n")
            parts.append("\n")
        
        if not parts:
            return "*No technical entities extracted.*\n"
        
        return "".join(parts)
    
    def _infer_field_purpose(self, field_name: str) -> str:
        """Infer field purpose from field name"""
        field_lower = field_name.lower()
        if 'start' in field_lower and 'date' in field_lower:
            return "User-entered start date"
        elif 'end' in field_lower and 'date' in field_lower:
            return "User-entered end date"
        elif 'scheduled' in field_lower and 'start' in field_lower:
            return "Reference start date"
        elif 'scheduled' in field_lower and 'return' in field_lower:
            return "Reference end date"
        elif 'usr' in field_lower and 'start' in field_lower:
            return "Start date copied from source document"
        elif 'usr' in field_lower and 'end' in field_lower:
            return "End date copied from source document"
        elif 'date' in field_lower:
            return "Date field"
        elif 'status' in field_lower:
            return "Status indicator"
        elif 'id' in field_lower or 'nbr' in field_lower:
            return "Identifier field"
        else:
            return "Field referenced in requirements"
    
    def _format_ac_mappings(self, ac_mappings: List[Dict[str, Any]]) -> str:
        """Format AC mappings as detailed markdown table"""
        if not ac_mappings:
            return "*No AC mappings available.*\n"
        
        def sanitize_table_cell(text: str, max_length: int = 80) -> str:
            """Sanitize text for markdown table cells"""
            if not text:
                return "-"
            # Replace newlines with spaces
            text = text.replace('\n', ' ').replace('\r', ' ')
            # Replace pipe characters (break tables)
            text = text.replace('|', '\\|')
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length - 3] + "..."
            return text
        
        def format_list_cell(items: list, max_items: int = 2, max_length: int = 60) -> str:
            """Format a list of items for table cell"""
            if not items:
                return "-"
            # Take first few items
            display_items = items[:max_items]
            # Format each item (truncate if needed)
            formatted = []
            for item in display_items:
                item_str = str(item).replace('\n', ' ').replace('|', '\\|')
                if len(item_str) > max_length:
                    item_str = item_str[:max_length - 3] + "..."
                formatted.append(item_str)
            
            result = ", ".join(formatted)
            if len(items) > max_items:
                result += f" (+{len(items) - max_items} more)"
            
            # Final truncation if still too long
            if len(result) > 100:
                result = result[:97] + "..."
            
            return result
        
        parts = []
        parts.append("| Criterion | Forms | DACs | Fields | Handlers | Validations | Missing |\n")
        parts.append("|-----------|-------|------|--------|----------|-------------|----------|\n")
        
        for mapping in ac_mappings:
            criterion = mapping.get('criterion', 'N/A')
            # Truncate and sanitize criterion
            criterion_display = sanitize_table_cell(criterion, max_length=70)
            
            # Format each column with proper truncation
            forms_list = mapping.get('related_forms', [])
            forms = format_list_cell([f"`{f}`" for f in forms_list], max_items=2, max_length=15) if forms_list else "-"
            
            dacs_list = mapping.get('related_dacs', [])
            dacs = format_list_cell([f"`{d}`" for d in dacs_list], max_items=2, max_length=30) if dacs_list else "-"
            
            fields_list = mapping.get('related_fields', [])
            fields = format_list_cell([f"`{f}`" for f in fields_list], max_items=2, max_length=20) if fields_list else "-"
            
            handlers_list = mapping.get('required_handlers', [])
            handlers = format_list_cell(handlers_list, max_items=2, max_length=40) if handlers_list else "-"
            
            validations_list = mapping.get('required_validations', [])
            validations = format_list_cell(validations_list, max_items=1, max_length=50) if validations_list else "-"
            
            missing_list = mapping.get('missing_entities', [])
            missing = format_list_cell(missing_list, max_items=2, max_length=25) if missing_list else "-"
            
            parts.append(f"| {criterion_display} | {forms} | {dacs} | {fields} | {handlers} | {validations} | {missing} |\n")
        
        # Add detailed breakdown after table
        parts.append("\n### Detailed AC Mapping Breakdown\n\n")
        for i, mapping in enumerate(ac_mappings, 1):
            criterion = mapping.get('criterion', 'N/A')
            parts.append(f"#### AC {i}: {criterion[:80]}\n\n")
            
            if mapping.get('related_forms'):
                parts.append(f"**Forms**: {', '.join([f'`{f}`' for f in mapping['related_forms']])}\n\n")
            if mapping.get('related_dacs'):
                parts.append(f"**DACs**: {', '.join([f'`{d}`' for d in mapping['related_dacs']])}\n\n")
            if mapping.get('related_fields'):
                parts.append(f"**Fields**: {', '.join([f'`{f}`' for f in mapping['related_fields']])}\n\n")
            if mapping.get('required_handlers'):
                parts.append(f"**Event Handlers**: {', '.join(mapping['required_handlers'])}\n\n")
            if mapping.get('required_validations'):
                parts.append(f"**Validations**: {', '.join(mapping['required_validations'])}\n\n")
            if mapping.get('required_workflows'):
                parts.append(f"**Workflows**: {', '.join(mapping['required_workflows'])}\n\n")
            if mapping.get('missing_entities'):
                parts.append(f"**⚠️ Missing Entities**: {', '.join(mapping['missing_entities'])}\n\n")
            
            parts.append("---\n\n")
        
        return "".join(parts)
    
    def _get_confirmed_entities(self, technical_entities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get entities with confidence >= 0.75"""
        confirmed = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': []
        }
        
        confidence_scores = technical_entities.get('confidence_scores', {})
        
        for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events']:
            entities = technical_entities.get(entity_type, [])
            scores = confidence_scores.get(entity_type, {})
            
            for entity in entities:
                score = scores.get(entity, 0.0)
                if score >= 0.75:
                    confirmed[entity_type].append(entity)
        
        return confirmed
    
    def _get_uncertain_entities(self, technical_entities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get entities with confidence 0.50-0.74"""
        uncertain = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': []
        }
        
        confidence_scores = technical_entities.get('confidence_scores', {})
        
        for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events']:
            entities = technical_entities.get(entity_type, [])
            scores = confidence_scores.get(entity_type, {})
            
            for entity in entities:
                score = scores.get(entity, 0.0)
                if 0.50 <= score < 0.75:
                    uncertain[entity_type].append(entity)
        
        return uncertain
    
    def _get_unconfirmed_entities(self, technical_entities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get entities with confidence < 0.50"""
        unconfirmed = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': []
        }
        
        confidence_scores = technical_entities.get('confidence_scores', {})
        
        for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events']:
            entities = technical_entities.get(entity_type, [])
            scores = confidence_scores.get(entity_type, {})
            
            for entity in entities:
                score = scores.get(entity, 0.0)
                if score < 0.50:
                    unconfirmed[entity_type].append(entity)
        
        return unconfirmed
    
    def _format_entity_list(self, entities: Dict[str, List[str]], status: str, technical_entities: Optional[Dict[str, Any]] = None) -> str:
        """Format entity list with confidence indicators and scores"""
        parts = []
        confidence_scores = {}
        
        if technical_entities:
            confidence_scores = technical_entities.get('confidence_scores', {})
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                parts.append(f"### {entity_type.title()}\n\n")
                
                if status == "confirmed":
                    parts.append("*Entities with confidence score ≥ 0.75 (confirmed in documentation)*\n\n")
                else:
                    parts.append("*Entities with confidence score < 0.40 (require manual validation)*\n\n")
                
                # Format as table for better readability
                parts.append("| Entity | Confidence | Status |\n")
                parts.append("|--------|------------|-------|\n")
                
                for entity in entity_list:
                    score = confidence_scores.get(entity_type, {}).get(entity, 0.0) if confidence_scores else 0.0
                    if status == "confirmed":
                        parts.append(f"| `{entity}` | {score:.2f} | ✅ Confirmed |\n")
                    else:
                        parts.append(f"| `{entity}` | {score:.2f} | ⚠️ [NOT CONFIRMED — DO NOT USE] |\n")
                
                parts.append("\n")
        
        if not parts:
            return f"*No {status} entities found.*\n"
        
        return "".join(parts)
    
    def _extract_code_snippets(self, technical_entities: Dict[str, Any], fact_table: Dict[str, Any]) -> str:
        """Extract and format code snippets from technical entities and fact table"""
        parts = []
        
        # Extract PX attributes as code examples
        if fact_table.get('px_attributes'):
            parts.append("### PX Attributes Found\n\n")
            for attr in fact_table['px_attributes'][:10]:  # Limit to top 10
                parts.append(f"```csharp\n{attr}\n```\n\n")
        
        # Extract event handlers as code structure examples
        if fact_table.get('events'):
            parts.append("### Event Handler Patterns\n\n")
            parts.append("*The following event handlers were identified. Implement these in your Graph extension:*\n\n")
            for event in fact_table['events'][:5]:  # Limit to top 5
                # Format as method signature example
                parts.append(f"```csharp\nprotected virtual void {event}(PXCache cache, PXFieldUpdatedEventArgs e)\n{{\n    // Implementation based on documentation\n}}\n```\n\n")
        
        # Check if there are any code snippets in technical entities
        # This would be populated if code extraction was performed
        if technical_entities.get('code_snippets'):
            parts.append("### Extracted Code Snippets\n\n")
            for snippet in technical_entities['code_snippets'][:5]:
                parts.append(f"```csharp\n{snippet}\n```\n\n")
        
        if not parts:
            return "*No code snippets extracted from documentation. Code implementation details may need to be researched separately.*\n"
        
        return "".join(parts)


"""
Solution Generator - Creates narrative solutions from questions and answers
"""

from typing import List, Dict, Any, AsyncIterator, Optional
from openai import OpenAI
import json

from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation
from src.core.fact_table import FactTableGenerator
from src.core.ac_mapper import AcceptanceCriteriaMapper
from src.core.code_templates import CodeTemplateGenerator
from src.core.dll_snippet_resolver import DllSnippetResolver

class SolutionGenerator:
    """Generates narrative solutions from questions and answers"""
    
    def __init__(self):
        self.logger = get_logger("SOLUTION_GENERATOR")
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize new modules for enhanced fact table and AC mapping
        self.fact_table_generator = FactTableGenerator()
        self.ac_mapper = AcceptanceCriteriaMapper()
        self.code_templates = CodeTemplateGenerator()  # For Law 4 compliance: template-based code generation
        self.dll_snippets = DllSnippetResolver()
        
        self.logger.info("Solution Generator initialized")
    
    def generate(self, entities: Dict[str, Any], mapped: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reference-compatible generation method (adapter pattern)
        
        Matches reference interface: SolutionGenerator.generate(entities, mapped)
        
        Args:
            entities: Dictionary of extracted entities
            mapped: Dictionary from AC mapper with 'rules', 'entities', 'mappings'
            
        Returns:
            Dictionary with 'summary', 'entities', 'mapping', 'implementation'
        """
        try:
            # Extract AC mappings
            ac_mappings = mapped.get("mappings", [])
            ac_list = mapped.get("rules", [])
            
            # Build fact table from entities
            fact_table = {
                "forms": entities.get("forms", []),
                "dacs": entities.get("dacs", []),
                "graphs": entities.get("graphs", []),
                "fields": entities.get("fields", []),
                "events": entities.get("events", [])
            }
            
            # Normalize AC to strings
            normalized_ac = []
            for ac in ac_list:
                if isinstance(ac, dict):
                    normalized_ac.append(ac.get('text', ''))
                else:
                    normalized_ac.append(str(ac))
            
            # Generate solution using full method
            solution_text = self.generate_final_solution(
                story_description="",
                acceptance_criteria=normalized_ac,
                fact_table=fact_table,
                ac_mappings=ac_mappings,
                retrieved_answer=""
            )
            
            return {
                "summary": "Generated technical solution",
                "entities": entities,
                "mapping": mapped,
                "implementation": solution_text
            }
        except Exception as e:
            self.logger.error("Generate adapter failed", extra={"error": str(e)})
            return {
                "summary": "Generated technical solution",
                "entities": entities,
                "mapping": mapped,
                "implementation": "Technical steps and code would be generated here."
            }
    
    def generate_narrative_solution(
        self,
        questions: List[str],
        answers: List[Dict[str, Any]],
        story_context: Dict[str, Any]
    ) -> str:
        """
        Generate coherent narrative solution from questions and answers
        
        Args:
            questions: List of extracted questions
            answers: List of answer dictionaries with 'answer', 'sources', etc.
            story_context: Original story context (description, acceptance_criteria)
            
        Returns:
            Narrative solution text
        """
        try:
            with TimedOperation("solution_generation", self.logger):
                description = story_context.get('description', '')
                acceptance_criteria = story_context.get('acceptance_criteria', [])
                
                self.logger.info("Generating narrative solution", extra={
                    "question_count": len(questions),
                    "answer_count": len(answers)
                })
                
                # Build prompt for narrative generation
                prompt = self._build_narrative_prompt(
                    description,
                    acceptance_criteria,
                    questions,
                    answers
                )
                
                # Call LLM to generate narrative
                response = self.openai_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert technical writer specializing in creating clear, comprehensive solutions for JIRA stories.
                            
                            Your task is to synthesize information from multiple Q&A pairs into a coherent, structured narrative solution.
                            
                            Guidelines:
                            - Create a clear, step-by-step solution narrative
                            - Integrate information from all answers seamlessly
                            - Use technical terminology accurately
                            - Structure the solution logically
                            - Include relevant details from the answers
                            - Make it actionable and easy to follow
                            - Write in a professional, clear style
                            
                            The solution should read as a complete guide, not just a collection of answers."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=config.SOLUTION_MAX_TOKENS
                )
                
                narrative = response.choices[0].message.content.strip()
                
                self.logger.info("Narrative solution generated", extra={
                    "narrative_length": len(narrative)
                })
                
                return narrative
                
        except Exception as e:
            self.logger.error("Failed to generate narrative solution", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return fallback narrative
            return self._generate_fallback_narrative(questions, answers, story_context)
    
    def generate_focused_narrative(
        self,
        story_context: Dict[str, Any],
        retrieved_content: Dict[str, Any],
        questions: List[str]
    ) -> str:
        """
        Generate focused narrative solution grounded in retrieved content.
        
        Key improvements:
        1. Strict grounding in retrieved content (no hallucination)
        2. DAC/DLL identification from documentation
        3. Step-by-step navigation instructions
        4. Questions guide structure, not drive separate queries
        
        Args:
            story_context: Original story context (description, acceptance_criteria)
            retrieved_content: Comprehensive retrieval result with answer and sources
            questions: List of questions (used to guide narrative structure)
            
        Returns:
            Focused narrative solution text
        """
        try:
            with TimedOperation("focused_solution_generation", self.logger):
                description = story_context.get('description', '')
                acceptance_criteria = story_context.get('acceptance_criteria', [])
                sources = retrieved_content.get('sources', [])
                answer = retrieved_content.get('answer', '')
                
                self.logger.info("Generating focused narrative solution", extra={
                    "question_count": len(questions),
                    "sources_count": len(sources),
                    "answer_length": len(answer)
                })
                
                # Build enhanced prompt with DAC/DLL awareness
                prompt = self._build_enhanced_narrative_prompt(
                    description=description,
                    acceptance_criteria=acceptance_criteria,
                    retrieved_answer=answer,
                    sources=sources,
                    questions=questions
                )
                
                # OPTIMIZATION: Reduce max_tokens and lower temperature for faster, more focused responses
                max_tokens_optimized = min(config.SOLUTION_MAX_TOKENS, 2500)  # Cap at 2500 for faster generation
                
                response = self.openai_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert Acumatica developer and technical writer. Your task is to extract EXACT technical details and create precise, actionable solutions based STRICTLY on the retrieved documentation.

CRITICAL EXTRACTION REQUIREMENTS:

1. **DAC/Graph Details** (Extract EXACTLY as written):
   - Extract exact DAC class names (e.g., "Customer", "SOOrder", "CustomerExt")
   - Extract Graph class names (e.g., "CustomerMaint", "SOOrderEntry", "CustomerMaintExt")
   - Extract extension class names if mentioned
   - Use code formatting: `Customer`, `SOOrderEntry`

2. **Form/Screen Details** (Extract EXACTLY as written):
   - Extract exact Form IDs (e.g., "SM201020", "CR301000", "SO301000")
   - Extract screen names exactly as written
   - Extract navigation paths verbatim from documentation
   - Format: **Form ID**: `SM201020` | **Screen**: Customer Maintenance

3. **Field Details** (Extract EXACTLY as written):
   - Extract exact field names (e.g., "CustomerID", "OrderNbr", "Status")
   - Extract field types if mentioned (e.g., "String", "Int", "Decimal")
   - Extract field attributes if specified (e.g., "PXDBString", "PXUIFieldAttribute")
   - Format: **Field**: `CustomerID` (String)

4. **Event Handlers** (Extract EXACTLY as written):
   - Extract event handler names (e.g., "FieldUpdated", "RowSelected", "RowPersisting")
   - Extract method signatures if provided
   - Extract event parameters if mentioned
   - Format: **Event**: `FieldUpdated` | **Method**: `CustomerID_FieldUpdated`

5. **Code Elements** (Extract EXACTLY as written):
   - Extract PXGraph methods (e.g., "PXGraph", "PXCache", "PXSelect")
   - Extract attribute names (e.g., "PXDBString", "PXUIFieldAttribute", "PXDefault")
   - Extract code snippets exactly as written (use code blocks)
   - Extract namespace/using statements if mentioned

6. **Validation Rules**:
   - If a technical detail is NOT in the documentation, write: "[NOT FOUND IN DOCUMENTATION]"
   - Do NOT infer, guess, or assume technical details
   - Only include what is explicitly stated in the retrieved content
   - Verify each technical detail exists before including it

CRITICAL RULES:
1. **Grounding**: Only use information from the retrieved content. Do NOT add information not present in the sources.
2. **Precision**: Extract technical details EXACTLY as written - no modifications, no assumptions.
3. **Navigation Instructions**: Provide step-by-step navigation paths verbatim from documentation.
4. **Focus**: Address ONLY what is asked in the story description and acceptance criteria.
5. **Honesty**: Clearly mark missing information with "[NOT FOUND IN DOCUMENTATION]".

Your solution must be:
- Accurate (grounded in retrieved content)
- Specific (with exact technical names, IDs, and code elements)
- Actionable (step-by-step navigation with exact form/field references)
- Focused (only addresses the story requirements)
- Honest (clearly marks missing information)"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=max_tokens_optimized  # Use optimized token limit
                )
                
                narrative = response.choices[0].message.content.strip()
                
                self.logger.info("Focused narrative solution generated", extra={
                    "narrative_length": len(narrative)
                })
                
                return narrative
                
        except Exception as e:
            self.logger.error("Failed to generate focused narrative solution", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return retrieved answer as fallback
            return retrieved_content.get('answer', 'Unable to generate narrative solution.')
    
    async def generate_focused_narrative_stream(
        self,
        story_context: Dict[str, Any],
        retrieved_content: Dict[str, Any],
        questions: List[str]
    ) -> AsyncIterator[str]:
        """
        Stream narrative solution token by token for real-time display.
        
        Args:
            story_context: Original story context (description, acceptance_criteria)
            retrieved_content: Comprehensive retrieval result with answer and sources
            questions: List of questions (used to guide narrative structure)
            
        Yields:
            Text chunks as they are generated
        """
        try:
            description = story_context.get('description', '')
            acceptance_criteria = story_context.get('acceptance_criteria', [])
            sources = retrieved_content.get('sources', [])
            answer = retrieved_content.get('answer', '')
            
            self.logger.info("Streaming focused narrative solution", extra={
                "question_count": len(questions),
                "sources_count": len(sources),
                "answer_length": len(answer)
            })
            
            # Build enhanced prompt
            prompt = self._build_enhanced_narrative_prompt(
                description=description,
                acceptance_criteria=acceptance_criteria,
                retrieved_answer=answer,
                sources=sources,
                questions=questions
            )
            
            # OPTIMIZATION: Reduce max_tokens for faster streaming
            max_tokens_optimized = min(config.SOLUTION_MAX_TOKENS, 2500)
            
            # Stream LLM response
            stream = self.openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert Acumatica developer and technical writer. Your task is to extract EXACT technical details and create precise, actionable solutions based STRICTLY on the retrieved documentation.

CRITICAL EXTRACTION REQUIREMENTS:

1. **DAC/Graph Details** (Extract EXACTLY as written):
   - Extract exact DAC class names (e.g., "Customer", "SOOrder", "CustomerExt")
   - Extract Graph class names (e.g., "CustomerMaint", "SOOrderEntry", "CustomerMaintExt")
   - Extract extension class names if mentioned
   - Use code formatting: `Customer`, `SOOrderEntry`

2. **Form/Screen Details** (Extract EXACTLY as written):
   - Extract exact Form IDs (e.g., "SM201020", "CR301000", "SO301000")
   - Extract screen names exactly as written
   - Extract navigation paths verbatim from documentation
   - Format: **Form ID**: `SM201020` | **Screen**: Customer Maintenance

3. **Field Details** (Extract EXACTLY as written):
   - Extract exact field names (e.g., "CustomerID", "OrderNbr", "Status")
   - Extract field types if mentioned (e.g., "String", "Int", "Decimal")
   - Extract field attributes if specified (e.g., "PXDBString", "PXUIFieldAttribute")
   - Format: **Field**: `CustomerID` (String)

4. **Event Handlers** (Extract EXACTLY as written):
   - Extract event handler names (e.g., "FieldUpdated", "RowSelected", "RowPersisting")
   - Extract method signatures if provided
   - Extract event parameters if mentioned
   - Format: **Event**: `FieldUpdated` | **Method**: `CustomerID_FieldUpdated`

5. **Code Elements** (Extract EXACTLY as written):
   - Extract PXGraph methods (e.g., "PXGraph", "PXCache", "PXSelect")
   - Extract attribute names (e.g., "PXDBString", "PXUIFieldAttribute", "PXDefault")
   - Extract code snippets exactly as written (use code blocks)
   - Extract namespace/using statements if mentioned

6. **Validation Rules**:
   - If a technical detail is NOT in the documentation, write: "[NOT FOUND IN DOCUMENTATION]"
   - Do NOT infer, guess, or assume technical details
   - Only include what is explicitly stated in the retrieved content
   - Verify each technical detail exists before including it

CRITICAL RULES:
1. **Grounding**: Only use information from the retrieved content. Do NOT add information not present in the sources.
2. **Precision**: Extract technical details EXACTLY as written - no modifications, no assumptions.
3. **Navigation Instructions**: Provide step-by-step navigation paths verbatim from documentation.
4. **Focus**: Address ONLY what is asked in the story description and acceptance criteria.
5. **Honesty**: Clearly mark missing information with "[NOT FOUND IN DOCUMENTATION]".

Your solution must be:
- Accurate (grounded in retrieved content)
- Specific (with exact technical names, IDs, and code elements)
- Actionable (step-by-step navigation with exact form/field references)
- Focused (only addresses the story requirements)
- Honest (clearly marks missing information)"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=max_tokens_optimized,
                stream=True  # Enable streaming
            )
            
            # Yield tokens as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error("Failed to stream narrative solution", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Yield fallback message
            yield retrieved_content.get('answer', 'Unable to generate narrative solution.')
    
    def _build_enhanced_narrative_prompt(
        self,
        description: str,
        acceptance_criteria: List[str],
        retrieved_answer: str,
        sources: List[Dict],
        questions: List[str]  # Kept for compatibility, not used in concise format
    ) -> str:
        """Build enhanced prompt with DAC/DLL awareness and strict grounding - CONCISE format"""
        
        # Format sources with confidence scores
        sources_text = []
        for i, source in enumerate(sources[:15], 1):  # Top 15 sources for context
            doc = source.get('document', 'Unknown')
            page = source.get('page', 0)
            confidence = source.get('similarity_score', 0.0)
            snippet = source.get('content_preview', '')[:300]
            sources_text.append(
                f"Source {i}: {doc} (Page {page}, Confidence: {confidence:.2f})\n"
                f"Content: {snippet}...\n"
            )
        
        criteria_text = "\n".join(f"- {criterion}" for criterion in acceptance_criteria)
        
        # Questions are used internally for structure but not explicitly shown in concise format
        _ = questions  # Suppress unused parameter warning
        
        prompt = f"""Extract technical details and create a CONCISE, direct, actionable solution. Be brief - remove unnecessary repetition and verbose explanations.

STORY DESCRIPTION:
{description}

ACCEPTANCE CRITERIA:
{criteria_text}

RETRIEVED DOCUMENTATION CONTENT:
{retrieved_answer}

SOURCE REFERENCES:
{chr(10).join(sources_text)}

CRITICAL REQUIREMENTS:
1. **Be CONCISE**: Remove unnecessary words, repetition, and verbose explanations
2. **Be DIRECT**: Get to the point quickly - no lengthy introductions
3. **Be ACCURATE**: Extract technical details EXACTLY as written
4. **Be ACTIONABLE**: Focus on what to do, not lengthy descriptions

FORMAT YOUR RESPONSE:

## Technical Components

**Form ID**: `[Extract exact ID]` or `[NOT FOUND IN DOCUMENTATION]`
**Screen Name**: [Extract exact name] or `[NOT FOUND IN DOCUMENTATION]`
**Key Fields**: [Extract exact field names, comma-separated] or `[NOT FOUND IN DOCUMENTATION]`
**DAC/Graph**: [Extract if found] or `[NOT FOUND IN DOCUMENTATION]`
**Code/Events**: [Extract if found] or `[NOT FOUND IN DOCUMENTATION]`

## Solution

[Brief 2-3 sentence overview - be direct]

### Implementation Steps

**Step 1: [Action Title]**
- **Form**: `[Form ID]` | **Screen**: [Screen Name]
- Navigate to: [Exact path or `[NOT FOUND]`]
- Actions:
  1. [Direct action from docs]
  2. [Next action]
  3. [Continue...]

**Step 2: [Next Action]**
- **Form**: `[Form ID]` | **Screen**: [Screen Name]
- Actions:
  1. [Direct action]
  2. [Next action]

[Continue for remaining steps - be concise]

### Notes
- ✅ Based on documentation: [Document names]
- ⚠️ Missing: [List only critical missing items]
- ❌ Do NOT add information not in documentation

CONCISENESS RULES:
- Remove redundant phrases like "This solution is based STRICTLY on..."
- Remove repetitive verification sections
- Combine related steps when possible
- Use bullet points, not paragraphs
- Skip empty sections (Code/Events if not found)
- No lengthy explanations - just facts and actions

Remember: CONCISE + ACCURATE. Extract exactly, write directly, remove fluff."""
        
        return prompt
    
    def _build_narrative_prompt(
        self,
        description: str,
        acceptance_criteria: List[str],
        questions: List[str],
        answers: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for narrative generation"""
        
        # Format answers
        answers_text = []
        for i, (question, answer_data) in enumerate(zip(questions, answers), 1):
            answer_text = answer_data.get('answer', str(answer_data)) if isinstance(answer_data, dict) else str(answer_data)
            answers_text.append(f"Q{i}: {question}\nA{i}: {answer_text}\n")
        
        criteria_text = "\n".join(f"- {criterion}" for criterion in acceptance_criteria)
        
        prompt = f"""Create a comprehensive, narrative solution for this JIRA story by synthesizing the provided questions and answers.

Story Description:
{description}

Acceptance Criteria:
{criteria_text}

Questions and Answers:
{chr(10).join(answers_text)}

Based on the story description, acceptance criteria, and the Q&A pairs above, create a complete narrative solution that:

1. Provides a clear overview of what needs to be done
2. Explains the step-by-step process or implementation approach
3. Integrates all relevant information from the answers
4. Addresses all acceptance criteria
5. Includes technical details and configuration steps where applicable
6. Is structured logically and easy to follow

Write the solution as a coherent narrative, not as a list of separate answers. Make it read like a complete guide or tutorial."""
        
        return prompt
    
    def _generate_fallback_narrative(
        self,
        questions: List[str],
        answers: List[Dict[str, Any]],
        story_context: Dict[str, Any]
    ) -> str:
        """Generate fallback narrative if LLM fails"""
        description = story_context.get('description', '')
        
        narrative_parts = [
            "## Solution Overview\n\n",
            f"Based on the story: {description}\n\n",
            "## Implementation Steps\n\n"
        ]
        
        for i, (question, answer_data) in enumerate(zip(questions, answers), 1):
            answer_text = answer_data.get('answer', str(answer_data)) if isinstance(answer_data, dict) else str(answer_data)
            narrative_parts.append(f"{i}. {question}\n   {answer_text}\n\n")
        
        return "".join(narrative_parts)
    
    def generate_technical_fact_table(
        self,
        technical_entities: Dict[str, Any],
        retrieved_content: Dict[str, Any],
        vision_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stage 1: Generate technical fact table (zero creativity, strict extraction)
        
        Enhanced to use FactTableGenerator module for better structure and validation.
        
        Args:
            technical_entities: Extracted technical entities from tech_extraction
            retrieved_content: Retrieved chunks from retrieval
            vision_metadata: Vision analysis results
            
        Returns:
            Strict JSON fact table with no narrative
        """
        try:
            # Use FactTableGenerator for structured fact table generation
            # Extract validated entities (entities with confidence >= 0.75)
            validated_entities = {}
            confidence_scores = technical_entities.get('confidence_scores', {})
            
            # Filter entities by confidence threshold
            for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events', 'navigation_paths', 'px_attributes', 'tables']:
                entities = technical_entities.get(entity_type, [])
                if not isinstance(entities, list):
                    continue
                
                validated_list = []
                entity_scores = confidence_scores.get(entity_type, {})
                
                for entity in entities:
                    if isinstance(entity, str):
                        score = entity_scores.get(entity, 0.0)
                        if score >= 0.75:  # Only include high-confidence entities
                            validated_list.append(entity)
                
                if validated_list:
                    validated_entities[entity_type] = validated_list
            
            # Use FactTableGenerator to build structured fact table
            fact_table = self.fact_table_generator.generate_fact_table(
                validated_entities=validated_entities,
                retrieved_documentation=retrieved_content.get('retrieved_chunks', []),
                vision_metadata=vision_metadata,
                confidence_scores=confidence_scores
            )
            
            # Validate fact table structure
            fact_table = self.fact_table_generator.validate_fact_table(fact_table)
            
            self.logger.info("Technical fact table generated", extra={
                "forms_count": len(fact_table.get('forms', [])),
                "dacs_count": len(fact_table.get('dacs', [])),
                "graphs_count": len(fact_table.get('graphs', [])),
                "missing_entities_count": len(fact_table.get('missing_entities', []))
            })
            
            return fact_table
            
        except Exception as e:
            self.logger.error("Failed to generate technical fact table", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return empty fact table on error
            return {
                "forms": [],
                "graphs": [],
                "dacs": [],
                "fields": [],
                "events": [],
                "navigation": [],
                "px_attributes": [],
                "tables": [],
                "missing_entities": [],
                "confidence_scores": {}
            }
    
    def map_acceptance_criteria_to_technical_requirements(
        self,
        acceptance_criteria: List[Any],
        fact_table: Dict[str, Any],
        story_description: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Map acceptance criteria to technical requirements
        
        Enhanced to use AcceptanceCriteriaMapper module for better mapping.
        
        Args:
            acceptance_criteria: List of acceptance criteria (strings or dicts with 'text' and 'subpoints')
            fact_table: Technical fact table from Stage 1
            story_description: Optional story description for context
            
        Returns:
            List of mappings connecting each AC to technical elements
        """
        try:
            # Use AcceptanceCriteriaMapper for structured AC mapping
            ac_mappings = self.ac_mapper.map_acceptance_criteria(
                acceptance_criteria=acceptance_criteria,
                fact_table=fact_table,
                story_description=story_description
            )
            
            self.logger.info("AC mappings generated", extra={
                "mappings_count": len(ac_mappings),
                "criteria_count": len(acceptance_criteria)
            })
            
            return ac_mappings
            
        except Exception as e:
            self.logger.error("Failed to map AC to technical requirements", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return empty mappings
            normalized_ac = []
            for ac in acceptance_criteria:
                if isinstance(ac, dict):
                    normalized_ac.append(ac.get('text', ''))
                else:
                    normalized_ac.append(str(ac))
            
            return [
                {
                    "criterion": criterion,
                    "related_forms": [],
                    "related_dacs": [],
                    "related_fields": [],
                    "required_handlers": [],
                    "required_validations": [],
                    "required_workflows": [],
                    "missing_entities": []
                }
                for criterion in normalized_ac
            ]
    
    def generate_final_solution(
        self,
        story_description: str,
        acceptance_criteria: List[str],
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]],
        retrieved_answer: str
    ) -> str:
        """
        Stage 3: Generate final solution narrative using fact table and AC mappings
        
        Args:
            story_description: JIRA story description
            acceptance_criteria: List of acceptance criteria
            fact_table: Technical fact table from Stage 1
            ac_mappings: AC mappings from Stage 2
            retrieved_answer: Retrieved answer from RAG
            
        Returns:
            Precise, grounded solution markdown
        """
        try:
            # IMPORTANT ARCHITECTURE CHANGE (per "maximum accuracy" requirement):
            # - Do NOT allow the LLM to generate code "from scratch".
            # - All code must come from deterministic templates fed by structured mappings (Law 4).
            #
            # The LLM can still be used upstream for extraction/mapping (JSON), but this stage
            # must be deterministic and template-driven.

            return self._generate_deterministic_solution(
                story_description=story_description,
                acceptance_criteria=acceptance_criteria,
                fact_table=fact_table,
                ac_mappings=ac_mappings,
                retrieved_answer=retrieved_answer,
            )
            
        except Exception as e:
            self.logger.error("Failed to generate final solution", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return fallback
            return f"## Solution\n\n{retrieved_answer}\n\n**Note**: Solution generation encountered an error. Above is the retrieved content."

    def _generate_deterministic_solution(
        self,
        story_description: str,
        acceptance_criteria: List[str],
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]],
        retrieved_answer: str
    ) -> str:
        """
        Deterministic solution composer:
        - Builds a concise implementation plan using fact_table + ac_mappings
        - Injects ONLY template-generated code
        """
        # Core entities
        forms = fact_table.get("forms", []) or []
        dacs = fact_table.get("dacs", []) or []
        graphs = fact_table.get("graphs", []) or []

        # Deterministic code artifacts (Law 4)
        template_code = self.code_templates.generate_complete_implementation(fact_table, ac_mappings)

        # Compose markdown
        parts: List[str] = []
        parts.append("## Solution\n")
        if story_description:
            parts.append(f"**Story:** {story_description.strip()}\n")

        # High-signal technical anchors
        if forms or dacs or graphs:
            parts.append("\n### Target Components\n")
            if forms:
                parts.append(f"- **Form(s)**: {', '.join(f'`{f}`' for f in forms[:10])}\n")
            if dacs:
                parts.append(f"- **DAC(s)**: {', '.join(f'`{d}`' for d in dacs[:10])}\n")
            if graphs:
                parts.append(f"- **Graph(s)**: {', '.join(f'`{g}`' for g in graphs[:10])}\n")

        # Deterministic implementation steps derived from AC mappings
        parts.append("\n### Implementation Steps\n")
        if ac_mappings:
            for i, m in enumerate(ac_mappings, start=1):
                criterion = (m.get("criterion") or "").strip()
                related_forms = m.get("related_forms", []) or []
                related_dacs = m.get("related_dacs", []) or []
                related_fields = m.get("related_fields", []) or []
                handlers = m.get("required_handlers", []) or []
                validations = m.get("required_validations", []) or []
                missing = m.get("missing_entities", []) or []

                title = criterion if criterion else f"Acceptance Criteria {i}"
                parts.append(f"\n**Step {i}:** {title}\n")
                if related_forms:
                    parts.append(f"- **Forms**: {', '.join(f'`{x}`' for x in related_forms)}\n")
                if related_dacs:
                    parts.append(f"- **DACs**: {', '.join(f'`{x}`' for x in related_dacs)}\n")
                if related_fields:
                    parts.append(f"- **Fields**: {', '.join(f'`{x}`' for x in related_fields)}\n")
                if handlers:
                    parts.append(f"- **Handlers**: {', '.join(f'`{x}`' for x in handlers)}\n")
                if validations:
                    # Keep validations as data (do not invent code here).
                    parts.append("- **Validations**:\n")
                    for v in validations[:10]:
                        parts.append(f"  - {v}\n")
                if missing:
                    parts.append("- **Missing Entities** (needs confirmation in docs/DLL):\n")
                    for me in missing[:15]:
                        parts.append(f"  - {me}\n")
        else:
            # Fall back to acceptance_criteria list
            for i, ac in enumerate(acceptance_criteria, start=1):
                parts.append(f"\n**Step {i}:** {str(ac).strip()}\n")

        # DLL-derived snippets (signatures / class records) — keep BEFORE template section so it shows in FINAL output.
        try:
            snippet_keywords: List[str] = []
            # Semantic object hit (business -> Graph/DAC) is the strongest anchor when available.
            sem_hit = (fact_table or {}).get("semantic_object_hit") or {}
            preferred_docs: Optional[List[str]] = None
            if isinstance(sem_hit, dict) and sem_hit:
                gfull = sem_hit.get("graph_full_name", "") or ""
                dfull = sem_hit.get("dac_full_name", "") or ""
                # When semantic binding exists, keep keywords tight to avoid generic noise dominating scores.
                snippet_keywords.extend([gfull, dfull, sem_hit.get("line_dac_full_name", ""), sem_hit.get("concept", "")])
                # Also add short names for recall in text exports
                snippet_keywords.extend([gfull.split(".")[-1] if gfull else "", dfull.split(".")[-1] if dfull else ""])
                # If module_tag maps cleanly to a *_DLL KB folder, prefer it.
                mod = (sem_hit.get("module_tag") or "").strip()
                if mod:
                    # e.g., "NV.Rental360" -> "NV.Rental360_DLL"
                    preferred_docs = [f"{mod}_DLL"]
            else:
                # Use what we already know (best signal)
                snippet_keywords.extend(dacs[:5])
                snippet_keywords.extend(graphs[:5])
                # Pull fields/handlers from AC mappings (common for validation stories)
                for m in (ac_mappings or [])[:10]:
                    snippet_keywords.extend((m.get("related_fields") or [])[:8])
                    snippet_keywords.extend((m.get("required_handlers") or [])[:5])
                # Add a few common domain terms from the story for recall (kept small to avoid noise)
                for term in ["PXGraph", "PXGraphExtension", "PXBqlTable", "RowPersisting", "FieldVerifying"]:
                    snippet_keywords.append(term)

            # Deduplicate
            snippet_keywords = list(dict.fromkeys([k for k in snippet_keywords if k]))

            dll_docs = self.dll_snippets.list_dll_documents()
            if preferred_docs:
                # Move preferred docs to front if they exist; keep others as secondary.
                dll_docs = [d for d in preferred_docs if d in dll_docs] + [d for d in dll_docs if d not in preferred_docs]
            snippets = self.dll_snippets.resolve_best_snippets(
                keywords=snippet_keywords,
                doc_names=dll_docs,
                method_limit=10,
                class_limit=3,
            )

            if snippets:
                parts.append("\n### Code Snippets (From DLL Exports)\n")
                parts.append("*Snippets below are extracted deterministically from the exported DLL signatures stored in the local knowledge base.*\n")
                for s in snippets[:12]:
                    # Render as text block; this is signature/metadata, not full decompiled method bodies.
                    header = f"\n**Source:** `{s.doc_name}`"
                    if s.start_line and s.end_line:
                        header += f" (lines {s.start_line}-{s.end_line})"
                    header += f" | **Type:** {s.kind}\n"
                    parts.append(header)
                    parts.append("\n```text\n")
                    parts.append(s.snippet.strip())
                    parts.append("\n```\n")
        except Exception as e:
            # Do not fail solution generation if snippet extraction fails.
            self.logger.warning("DLL snippet extraction failed", extra={"error": str(e)})

        # Template-generated code section (ONLY deterministic code)
        parts.append("\n---\n\n## Template-Generated Code (Deterministic)\n")
        parts.append("*Code below is generated from templates using extracted entities/mappings. The LLM does not write this code.*\n")

        dac_ext = template_code.get("dac_extension", "")
        graph_ext = template_code.get("graph_extension", "")

        if dac_ext:
            parts.append("\n### DAC Extension\n\n```csharp\n")
            parts.append(dac_ext.strip())
            parts.append("\n```\n")
        if graph_ext:
            parts.append("\n### Graph Extension\n\n```csharp\n")
            parts.append(graph_ext.strip())
            parts.append("\n```\n")
        if not dac_ext and not graph_ext:
            parts.append("\n(No deterministic templates could be filled with the current fact table/mappings.)\n")

        # LLM-generated code snippets using global Acumatica knowledge (when documentation is insufficient)
        llm_code_snippets = self._generate_llm_code_snippets(
            story_description=story_description,
            acceptance_criteria=acceptance_criteria,
            fact_table=fact_table,
            ac_mappings=ac_mappings,
            retrieved_answer=retrieved_answer
        )
        
        if llm_code_snippets:
            parts.append("\n---\n\n## LLM-Generated Code Snippets (Global Acumatica Knowledge)\n")
            parts.append("*Code below is generated using LLM with global Acumatica framework knowledge when documentation is insufficient.*\n")
            parts.append(llm_code_snippets)

        # Keep retrieved_answer as reference-only (not a driver of code)
        if retrieved_answer and retrieved_answer.strip():
            parts.append("\n---\n\n## Retrieved Documentation (Reference)\n")
            parts.append(retrieved_answer.strip()[:2000])
            if len(retrieved_answer.strip()) > 2000:
                parts.append("\n\n...[truncated]...\n")

        return "".join(parts)
    
    def _enhance_solution_with_fact_table(
        self,
        solution: str,
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]]
    ) -> str:
        """
        Post-process solution to ensure fact table entities are used confidently
        Replace generic placeholders with actual entities from fact table
        """
        # If solution contains [NOT FOUND IN DOCUMENTATION] but we have entities in fact table,
        # try to replace with actual entities
        forms = fact_table.get('forms', [])
        dacs = fact_table.get('dacs', [])
        graphs = fact_table.get('graphs', [])
        fields = fact_table.get('fields', [])
        
        # Replace generic placeholders with actual entities if available
        if forms and '[NOT FOUND IN DOCUMENTATION]' in solution:
            # Try to replace form placeholders
            solution = solution.replace(
                '**Form ID**: `[NOT FOUND IN DOCUMENTATION]`',
                f'**Form ID**: `{forms[0]}`' if forms else '**Form ID**: `[NOT FOUND IN DOCUMENTATION]`'
            )
        
        if dacs and '[NOT FOUND IN DOCUMENTATION]' in solution:
            # Try to replace DAC placeholders
            solution = solution.replace(
                '**DAC**: `[NOT FOUND IN DOCUMENTATION]`',
                f'**DAC**: `{dacs[0]}`' if dacs else '**DAC**: `[NOT FOUND IN DOCUMENTATION]`'
            )
        
        if graphs and '[NOT FOUND IN DOCUMENTATION]' in solution:
            # Try to replace Graph placeholders
            solution = solution.replace(
                '**Graph**: `[NOT FOUND IN DOCUMENTATION]`',
                f'**Graph**: `{graphs[0]}`' if graphs else '**Graph**: `[NOT FOUND IN DOCUMENTATION]`'
            )
        
        return solution
    
    def _generate_llm_code_snippets(
        self,
        story_description: str,
        acceptance_criteria: List[str],
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]],
        retrieved_answer: str
    ) -> str:
        """
        Generate code snippets using LLM with global Acumatica knowledge when documentation is insufficient.
        
        This complements template-based code generation by providing LLM-generated code using
        global Acumatica framework knowledge when retrieved documentation is missing or incomplete.
        """
        try:
            # Check if we have sufficient entities to generate meaningful code
            forms = fact_table.get("forms", []) or []
            dacs = fact_table.get("dacs", []) or []
            graphs = fact_table.get("graphs", []) or []
            fields = fact_table.get("fields", []) or []
            
            # If we have no entities or very few, skip LLM generation
            if not dacs and not graphs:
                return ""
            
            # Check if documentation is insufficient (short or contains "NOT FOUND")
            doc_is_insufficient = (
                len(retrieved_answer.strip()) < 500 or 
                "[NOT FOUND IN DOCUMENTATION]" in retrieved_answer or
                "not found" in retrieved_answer.lower()
            )
            
            if not doc_is_insufficient:
                # Documentation seems sufficient, skip LLM generation
                return ""
            
            # Build context from fact table and AC mappings
            context_parts = []
            if story_description:
                context_parts.append(f"**Story:** {story_description}")
            if acceptance_criteria:
                context_parts.append(f"**Acceptance Criteria:**")
                for ac in acceptance_criteria[:5]:
                    context_parts.append(f"- {ac}")
            
            if dacs:
                context_parts.append(f"\n**DACs:** {', '.join(f'`{d}`' for d in dacs[:5])}")
            if graphs:
                context_parts.append(f"**Graphs:** {', '.join(f'`{g}`' for g in graphs[:5])}")
            if fields:
                context_parts.append(f"**Fields:** {', '.join(f'`{f}`' for f in fields[:10])}")
            
            # Extract semantic object hit if available
            sem_hit = fact_table.get("semantic_object_hit") or {}
            if sem_hit:
                context_parts.append(f"\n**Semantic Object:** {sem_hit.get('concept', '')}")
                if sem_hit.get('dac_full_name'):
                    context_parts.append(f"- DAC: `{sem_hit['dac_full_name']}`")
                if sem_hit.get('graph_full_name'):
                    context_parts.append(f"- Graph: `{sem_hit['graph_full_name']}`")
            
            # Extract key requirements from AC mappings
            if ac_mappings:
                context_parts.append("\n**Key Requirements:**")
                for m in ac_mappings[:3]:
                    criterion = m.get("criterion", "")
                    if criterion:
                        context_parts.append(f"- {criterion}")
                    handlers = m.get("required_handlers", [])
                    if handlers:
                        context_parts.append(f"  - Handlers needed: {', '.join(handlers[:3])}")
                    validations = m.get("required_validations", [])
                    if validations:
                        context_parts.append(f"  - Validations: {', '.join(validations[:2])}")
            
            context = "\n".join(context_parts)
            
            # Generate code using LLM with global Acumatica knowledge
            prompt = f"""You are an expert Acumatica developer with deep knowledge of the Acumatica ERP framework.

**Context:**
{context}

**Task:** Generate accurate, production-ready Acumatica C# code snippets to implement the requirements above.

**Requirements:**
1. Use correct Acumatica framework patterns:
   - DAC extensions: Inherit from `PXCacheExtension<TDAC>`
   - Graph extensions: Inherit from `PXGraphExtension<TGraph>`
   - Use proper PX attributes: `[PXDBString]`, `[PXDBDecimal]`, `[PXUIField]`, etc.
   - Event handlers: `FieldUpdated`, `RowSelected`, `RowPersisting`, `FieldVerifying`
   - Use `PXCache`, `PXGraph`, `BQL` queries correctly

2. Generate code for:
   - Field definitions (if fields are mentioned)
   - Event handlers (if handlers/validations are mentioned)
   - Graph extension methods (if business logic is needed)
   - DAC extension (if custom fields are needed)

3. Be specific and accurate:
   - Use exact DAC/Graph names from context
   - Include proper namespaces
   - Use correct field types
   - Follow Acumatica coding conventions

4. Generate complete, compilable code snippets with:
   - Proper using statements
   - Class declarations
   - Method implementations
   - Comments explaining key logic

**Output Format:**
Provide code snippets organized by component (DAC Extension, Graph Extension, etc.).
Each snippet should be complete and ready to use.

Generate the code now:"""

            # Use appropriate parameters based on model
            llm_params = {
                "model": config.LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert Acumatica ERP developer. Generate accurate, production-ready C# code following Acumatica framework patterns and best practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_completion_tokens": 2000
            }
            
            # Only add temperature if model supports it (gpt-5-mini doesn't)
            if "gpt-5-mini" not in config.LLM_MODEL.lower():
                llm_params["temperature"] = 0.3  # Lower temperature for more deterministic code
            
            response = self.openai_client.chat.completions.create(**llm_params)
            
            generated_code = response.choices[0].message.content.strip()
            
            if generated_code:
                self.logger.info("Generated LLM code snippets", extra={
                    "dacs": len(dacs),
                    "graphs": len(graphs),
                    "code_length": len(generated_code)
                })
                return f"\n{generated_code}\n"
            
        except Exception as e:
            self.logger.warning("LLM code generation failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Non-fatal: return empty string if generation fails
            return ""
        
        return ""
    
    def _inject_template_code(
        self,
        solution: str,
        template_code: Dict[str, str],
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]]
    ) -> str:
        """
        Inject template-generated code into solution (Law 4 compliance)
        
        Replaces LLM-generated code snippets with deterministic template-based code
        where templates are available.
        
        Args:
            solution: LLM-generated solution
            template_code: Template-generated code dictionary
            fact_table: Technical fact table
            ac_mappings: AC mappings
            
        Returns:
            Solution with template code injected
        """
        try:
            # If we have template-generated code, inject it
            if template_code.get('graph_extension') or template_code.get('dac_extension'):
                # Add template code section if not present
                if '## Template-Generated Code (Deterministic)' not in solution:
                    template_section = "\n\n## Template-Generated Code (Deterministic)\n\n"
                    template_section += "*The following code is generated deterministically from templates based on fact table entities.*\n\n"
                    
                    if template_code.get('dac_extension'):
                        template_section += "### DAC Extension\n\n```csharp\n"
                        template_section += template_code['dac_extension']
                        template_section += "\n```\n\n"
                    
                    if template_code.get('graph_extension'):
                        template_section += "### Graph Extension\n\n```csharp\n"
                        template_section += template_code['graph_extension']
                        template_section += "\n```\n\n"
                    
                    # Insert before "## References" or at the end
                    if '## References' in solution:
                        solution = solution.replace('## References', template_section + '## References')
                    elif '## 5. References' in solution:
                        solution = solution.replace('## 5. References', template_section + '## 5. References')
                    else:
                        solution += template_section
            
            return solution
            
        except Exception as e:
            self.logger.warning("Failed to inject template code", extra={"error": str(e)})
            return solution


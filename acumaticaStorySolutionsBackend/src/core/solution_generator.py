"""
Solution Generator - Creates narrative solutions from questions and answers
"""

from typing import List, Dict, Any, AsyncIterator
from openai import OpenAI

from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation

class SolutionGenerator:
    """Generates narrative solutions from questions and answers"""
    
    def __init__(self):
        self.logger = get_logger("SOLUTION_GENERATOR")
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.logger.info("Solution Generator initialized")
    
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
                    temperature=0.4,
                    max_tokens=config.SOLUTION_MAX_TOKENS
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
                    temperature=0.2,  # Lower temperature for more focused, less hallucinatory output
                    max_tokens=max_tokens_optimized  # Use optimized token limit
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
                temperature=0.2,
                max_tokens=max_tokens_optimized,
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


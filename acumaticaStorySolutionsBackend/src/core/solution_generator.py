"""
Solution Generator - Creates narrative solutions from questions and answers
"""

from typing import List, Dict, Any
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
                
                response = self.openai_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert Acumatica developer and technical writer. Your task is to create precise, actionable solutions based STRICTLY on the retrieved documentation.

CRITICAL RULES:
1. **Grounding**: Only use information from the retrieved content. Do NOT add information not present in the sources.
2. **DAC/DLL Identification**: Identify specific DACs (Data Access Classes), screens, forms, and extension points mentioned in the documentation.
3. **Navigation Instructions**: Provide step-by-step navigation paths (e.g., "Navigate to: System > Customization > Customization Projects > [Project Name] > [Screen ID]").
4. **Precision**: If the documentation doesn't mention a specific screen/DAC, state that clearly rather than guessing.
5. **Focus**: Address ONLY what is asked in the story description and acceptance criteria. Do not add unrelated solutions.
6. **Step-by-Step**: Break down implementation into numbered steps with specific navigation paths and form names.

Your solution must be:
- Accurate (grounded in retrieved content)
- Specific (with exact screen names, form IDs, DAC names when available)
- Actionable (step-by-step navigation and instructions)
- Focused (only addresses the story requirements)
- Honest (clearly state when information is missing from documentation)"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,  # Lower temperature for more focused, less hallucinatory output
                    max_tokens=config.SOLUTION_MAX_TOKENS
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
    
    def _build_enhanced_narrative_prompt(
        self,
        description: str,
        acceptance_criteria: List[str],
        retrieved_answer: str,
        sources: List[Dict],
        questions: List[str]
    ) -> str:
        """Build enhanced prompt with DAC/DLL awareness and strict grounding"""
        
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
        questions_text = "\n".join(f"- {q}" for q in questions) if questions else "None specified"
        
        prompt = f"""Create a precise, actionable solution for this Acumatica development story.

STORY DESCRIPTION:
{description}

ACCEPTANCE CRITERIA:
{criteria_text}

KEY QUESTIONS TO ADDRESS (for narrative structure):
{questions_text}

RETRIEVED DOCUMENTATION CONTENT:
{retrieved_answer}

SOURCE REFERENCES:
{chr(10).join(sources_text)}

TASK: Create a solution that:

1. **Identifies Specific Components**:
   - Which DAC (Data Access Class) needs to be extended or created? (if mentioned in docs)
   - Which screen/form (with exact form ID) needs to be modified? (if mentioned in docs)
   - Where exactly should the extension be applied? (if mentioned in docs)
   - If not mentioned, clearly state: "Specific DAC/Screen not identified in available documentation"

2. **Provides Step-by-Step Navigation**:
   - Exact navigation path (e.g., "System > Customization > Customization Projects")
   - Specific form IDs and screen names (only if present in documentation)
   - Where to find each setting or option (based on retrieved content)

3. **Gives Implementation Steps**:
   - Numbered steps with specific actions from documentation
   - Exact field names, menu options, and form locations (only if in docs)
   - Code extension points if mentioned in documentation

4. **Stays Focused**:
   - Address ONLY what is in the story description
   - Do NOT add solutions for unrelated features
   - If documentation doesn't cover something, state: "This aspect is not covered in the available documentation"

5. **Uses Retrieved Content Only**:
   - Base your solution STRICTLY on the retrieved documentation above
   - Do NOT invent or assume information not present in sources
   - If information is missing, indicate what additional documentation might be needed

FORMAT YOUR RESPONSE AS:

## Solution Overview
[Brief overview based STRICTLY on retrieved content]

## Required Components
- **DAC**: [Specific DAC name if mentioned in documentation, or "Not specified in available documentation"]
- **Screen/Form**: [Exact form ID and screen name if mentioned, or "Not specified in available documentation"]
- **Extension Point**: [Where to apply the extension if mentioned, or "Not specified in available documentation"]

## Step-by-Step Implementation

### Step 1: Navigate to [Specific Location]
**Navigation Path**: [Exact menu path from documentation, or "Navigation path not specified in documentation"]
**Screen/Form**: [Form ID and name if mentioned]

**Action**: [Specific action based on documentation]
[Detailed instructions from documentation]

### Step 2: [Next Step]
[Continue with specific steps ONLY from documentation]

## Important Notes
- This solution is based STRICTLY on the provided documentation
- If specific details are missing, additional documentation may be required
- Follow the exact steps and locations mentioned in the documentation
- Do not add information not present in the retrieved sources

Remember: Be precise, grounded, and focused. No hallucination. Only use information from the retrieved documentation."""
        
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
            f"## Solution Overview\n\n",
            f"Based on the story: {description}\n\n",
            "## Implementation Steps\n\n"
        ]
        
        for i, (question, answer_data) in enumerate(zip(questions, answers), 1):
            answer_text = answer_data.get('answer', str(answer_data)) if isinstance(answer_data, dict) else str(answer_data)
            narrative_parts.append(f"{i}. {question}\n   {answer_text}\n\n")
        
        return "".join(narrative_parts)


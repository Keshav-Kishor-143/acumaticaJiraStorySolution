"""
JIRA Story Processor - Extracts key questions from JIRA stories
"""

import json
from typing import List, Dict, Any
from openai import OpenAI

from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation

class JIRAStoryProcessor:
    """Processes JIRA stories and extracts key questions using LLM"""
    
    def __init__(self):
        self.logger = get_logger("STORY_PROCESSOR")
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.logger.info("JIRA Story Processor initialized")
    
    def extract_key_questions(self, story_json: Dict[str, Any]) -> List[str]:
        """
        Extract or generate key questions from JIRA story using LLM
        
        Args:
            story_json: Dictionary with 'description', 'acceptance_criteria', 'images', etc.
            
        Returns:
            List of focused questions that need to be answered
        """
        try:
            with TimedOperation("question_extraction", self.logger):
                description = story_json.get('description', '')
                acceptance_criteria = story_json.get('acceptance_criteria', [])
                story_id = story_json.get('story_id', 'Unknown')
                
                self.logger.info("Extracting key questions from JIRA story", extra={
                    "story_id": story_id,
                    "description_length": len(description),
                    "criteria_count": len(acceptance_criteria)
                })
                
                # Build prompt for question extraction
                prompt = self._build_question_extraction_prompt(description, acceptance_criteria)
                
                # Call LLM to extract questions
                response = self.openai_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at analyzing JIRA stories and identifying the key questions that need to be answered to solve the story. 
                            Your task is to extract or generate focused, specific questions that will guide the search for solutions in technical documentation.
                            
                            Focus on:
                            - What needs to be implemented or configured?
                            - What are the specific steps or procedures required?
                            - What technical details are needed?
                            - What are the dependencies or prerequisites?
                            
                            Return ONLY a JSON array of questions, no other text."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Extract questions from response
                questions = result.get('questions', [])
                if isinstance(questions, str):
                    # If single string, try to split by newlines
                    questions = [q.strip() for q in questions.split('\n') if q.strip()]
                elif not isinstance(questions, list):
                    questions = [str(questions)]
                
                # Limit number of questions
                questions = questions[:config.MAX_QUESTIONS_PER_STORY]
                
                # Clean and validate questions
                cleaned_questions = []
                for q in questions:
                    q_clean = q.strip()
                    if q_clean and len(q_clean) > 10:  # Minimum question length
                        # Remove question numbers if present
                        if q_clean[0].isdigit() and ('.' in q_clean[:5] or ')' in q_clean[:5]):
                            q_clean = q_clean.split('.', 1)[-1].split(')', 1)[-1].strip()
                        cleaned_questions.append(q_clean)
                
                self.logger.info("Key questions extracted", extra={
                    "story_id": story_id,
                    "question_count": len(cleaned_questions),
                    "questions": cleaned_questions
                })
                
                return cleaned_questions
                
        except Exception as e:
            self.logger.error("Failed to extract key questions", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "story_id": story_json.get('story_id', 'Unknown')
            })
            # Return fallback questions based on description
            return self._generate_fallback_questions(story_json)
    
    def _build_question_extraction_prompt(self, description: str, acceptance_criteria: List[str]) -> str:
        """Build prompt for question extraction"""
        criteria_text = "\n".join(f"- {criterion}" for criterion in acceptance_criteria)
        
        prompt = f"""Analyze this JIRA story and extract or generate the key questions that need to be answered to solve it.

Story Description:
{description}

Acceptance Criteria:
{criteria_text}

Based on the story description and acceptance criteria, identify the specific questions that need to be answered. These questions should:
1. Be focused on what needs to be implemented, configured, or understood
2. Guide the search for solutions in technical documentation
3. Cover the technical aspects, procedures, and requirements
4. Be specific enough to find relevant documentation

Return a JSON object with a "questions" array containing the extracted/generated questions.

Example format:
{{
  "questions": [
    "How do I configure sales return processing in Acumatica?",
    "What are the prerequisites for processing returns?",
    "What fields need to be set up for return orders?"
  ]
}}"""
        
        return prompt
    
    def _generate_fallback_questions(self, story_json: Dict[str, Any]) -> List[str]:
        """Generate fallback questions if LLM extraction fails"""
        description = story_json.get('description', '')
        acceptance_criteria = story_json.get('acceptance_criteria', [])
        
        questions = []
        
        # Extract key terms from description
        if description:
            # Simple keyword-based question generation
            if 'return' in description.lower():
                questions.append("How do I process returns in Acumatica?")
            if 'order' in description.lower():
                questions.append("How do I create and process orders?")
            if 'configure' in description.lower() or 'setup' in description.lower():
                questions.append("How do I configure this feature?")
        
        # Generate questions from acceptance criteria
        for criterion in acceptance_criteria[:3]:  # Limit to first 3
            if len(criterion) > 20:
                questions.append(f"How do I {criterion.lower()}?")
        
        # Default question if none generated
        if not questions:
            questions.append("What are the steps to implement this feature?")
        
        return questions[:config.MAX_QUESTIONS_PER_STORY]


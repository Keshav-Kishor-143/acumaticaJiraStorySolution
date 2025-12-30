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
                    max_completion_tokens=500,
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
    
    def normalize_story_input(self, raw_story_text: str) -> Dict[str, Any]:
        """
        Normalize raw JIRA story text into clean, canonical JSON structure
        
        Uses LLM to parse inconsistent formatting and extract:
        - story_title
        - description (clean narrative only)
        - requirements (business rules, constraints)
        - acceptance_criteria (hierarchical structure with subpoints)
        
        Args:
            raw_story_text: Raw story text from user (may include mixed formatting)
            
        Returns:
            Normalized JSON structure with separated fields
        """
        try:
            with TimedOperation("story_normalization", self.logger):
                self.logger.info("Normalizing story input", extra={
                    "text_length": len(raw_story_text)
                })
                
                # Validate input
                if not raw_story_text or not raw_story_text.strip():
                    self.logger.warning("Empty story text provided for normalization")
                    return {
                        "story_title": "",
                        "description": "",
                        "requirements": [],
                        "acceptance_criteria": []
                    }
                
                # Build normalization prompt
                prompt = self._build_normalization_prompt(raw_story_text)
                
                # Call LLM for normalization (using gpt-4o-mini for cost efficiency)
                self.logger.debug("Calling LLM for story normalization", extra={
                    "model": config.LLM_MODEL,
                    "prompt_length": len(prompt)
                })
                
                response = self.openai_client.chat.completions.create(
                    model=config.LLM_MODEL,  # gpt-4o-mini (cheap and efficient)
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an intelligent JIRA story parser. Your task is to extract structured data from raw story text into: story_title, description, requirements, and acceptance_criteria.

CRITICAL EXTRACTION RULES:

1. STORY_TITLE:
   - Extract from user story format: "As a [role] I want [feature]..." → title = the feature name
   - Example: "As a sales user, I want the 'Email Quote' template..." → story_title = "Email Quote Template"
   - If no clear title in user story, extract from first meaningful phrase or leave empty
   - Title should be SHORT (3-8 words max)

2. DESCRIPTION:
   - Extract ONLY the narrative/user story statement
   - Format: "As a [role] I want [goal] So that [benefit]"
   - OR: Problem statement like "When X happens, Y occurs"
   - STOP at the first specific requirement or field specification
   - Do NOT include detailed requirements, field lists, or format specifications
   - Description is the HIGH-LEVEL goal, not implementation details

3. REQUIREMENTS:
   - Only extract if explicitly labeled: "Requirements:", "Business Rules:", "Core Rules:", etc.
   - These are business-level constraints or rules
   - If not explicitly labeled, leave as empty array []

4. ACCEPTANCE_CRITERIA (MOST IMPORTANT):
   - Extract ALL specific, testable requirements as acceptance criteria
   - These include:
     * Format specifications ("The email subject must follow the format...")
     * Field requirements ("The email body must include the following fields...")
     * Behavior specifications ("The greeting line must correctly populate...")
     * Validation rules ("Event Start Date must be greater than Scheduled Start Date")
     * Any "must", "should", "will" statements that specify what the system must do
   - Each distinct requirement becomes a separate AC entry
   - If requirements are grouped (like "The email body must include:" followed by a list), create ONE AC with subpoints
   - Given/When/Then statements → group as subpoints under ONE parent AC
   - Numbered rules → each becomes a separate AC entry

5. SEPARATION LOGIC:
   - If text starts with "Description" header → everything after until first requirement is description
   - After description, if you see specific requirements (format, fields, validations) → these are AC
   - User story statement = description
   - Specific "must/should/will" statements = acceptance criteria

EXAMPLE INPUT:
"Description
As a sales user, I want the Email Quote template to use standardized format.
The email subject must follow: USS Fence Quote #Opp-XXXXXXX
The email body must include: Document Number, Customer, Location
The greeting must populate: Dear ((Billing_Contact.Attention))"

EXAMPLE OUTPUT:
{
  "story_title": "Email Quote Template",
  "description": "As a sales user, I want the Email Quote template to use standardized format.",
  "requirements": [],
  "acceptance_criteria": [
    {"id": "AC1", "text": "The email subject must follow the format: USS Fence Quote #Opp-XXXXXXX", "subpoints": []},
    {"id": "AC2", "text": "The email body must include required fields", "subpoints": ["Document Number", "Customer", "Location"]},
    {"id": "AC3", "text": "The greeting must populate: Dear ((Billing_Contact.Attention))", "subpoints": []}
  ]
}

Return ONLY valid JSON - no explanation, no markdown formatting."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_completion_tokens=3000,  # Increased to handle complex stories with multiple AC entries
                    response_format={"type": "json_object"}
                )
                
                # Validate response
                if not response or not response.choices or len(response.choices) == 0:
                    raise ValueError("Empty response from LLM")
                
                # Parse response
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Empty content in LLM response")
                
                self.logger.debug("LLM response received", extra={
                    "content_length": len(content),
                    "content_preview": content[:200] if len(content) > 200 else content
                })
                
                normalized = json.loads(content)
                
                # Validate parsed JSON structure
                if not isinstance(normalized, dict):
                    raise ValueError(f"LLM returned non-dict response: {type(normalized)}")
                
                # Validate and clean structure
                normalized = self._validate_normalized_structure(normalized)
                
                # Final validation - ensure we got meaningful data
                has_data = (
                    bool(normalized.get('story_title', '').strip()) or
                    bool(normalized.get('description', '').strip()) or
                    len(normalized.get('requirements', [])) > 0 or
                    len(normalized.get('acceptance_criteria', [])) > 0
                )
                
                if not has_data:
                    self.logger.warning("LLM returned empty normalization, falling back to rule-based extraction", extra={
                        "normalized": normalized
                    })
                    # Try fallback if LLM returned empty data
                    fallback_result = self._generate_fallback_normalization(raw_story_text)
                    # Only use fallback if it has more data than LLM result
                    fallback_has_data = (
                        bool(fallback_result.get('story_title', '').strip()) or
                        bool(fallback_result.get('description', '').strip()) or
                        len(fallback_result.get('requirements', [])) > 0 or
                        len(fallback_result.get('acceptance_criteria', [])) > 0
                    )
                    if fallback_has_data:
                        self.logger.info("Using fallback normalization result")
                        return fallback_result
                
                self.logger.info("Story normalized successfully", extra={
                    "has_title": bool(normalized.get('story_title')),
                    "has_description": bool(normalized.get('description')),
                    "requirements_count": len(normalized.get('requirements', [])),
                    "ac_count": len(normalized.get('acceptance_criteria', []))
                })
                
                return normalized
                
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse LLM JSON response", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "content_preview": str(e)[:500] if hasattr(e, 'msg') else str(e)
            })
            # Try fallback
            return self._generate_fallback_normalization(raw_story_text)
        except Exception as e:
            self.logger.error("Story normalization failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "error_traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
            })
            # Return fallback structure
            return self._generate_fallback_normalization(raw_story_text)
    
    def _build_normalization_prompt(self, raw_story_text: str) -> str:
        """Build dynamic normalization prompt that adapts to any story format"""
        return f"""Extract structured data from this JIRA story text. Be precise and separate the narrative description from specific requirements.

RAW STORY TEXT:
{raw_story_text}

EXTRACTION INSTRUCTIONS:

1. STORY_TITLE:
   - Extract from user story: "As a [role] I want [feature]..." → extract the feature name
   - Example: "I want the 'Email Quote' template..." → story_title = "Email Quote Template"
   - If no clear feature name, extract from first meaningful phrase
   - Keep it short (3-8 words)

2. DESCRIPTION:
   - Extract ONLY the narrative/user story statement
   - This is the HIGH-LEVEL goal: "As a [role] I want [goal] So that [benefit]"
   - OR problem statement: "When X happens, Y occurs"
   - STOP when you encounter specific requirements, field lists, or format specifications
   - Description should be 1-3 sentences maximum
   - Do NOT include detailed requirements in description

3. REQUIREMENTS:
   - Only if explicitly labeled: "Requirements:", "Business Rules:", etc.
   - If not labeled, return empty array []

4. ACCEPTANCE_CRITERIA (CRITICAL):
   - Extract ALL specific, testable requirements
   - Look for statements with: "must", "should", "will", "must include", "must follow", "must populate"
   - Each distinct requirement = separate AC entry
   - If requirements are grouped under a header (like "The email body must include:"), create ONE AC with subpoints
   - Format specifications, field requirements, validation rules → all are AC
   - Given/When/Then → group as subpoints under ONE parent AC
   - Numbered rules → each becomes separate AC

SEPARATION EXAMPLE:
Input: "Description
As a sales user, I want Email Quote template standardized.
The email subject must follow: USS Fence Quote #Opp-XXXXXXX
The email body must include: Document Number, Customer"

Output:
- story_title: "Email Quote Template"
- description: "As a sales user, I want Email Quote template standardized."
- acceptance_criteria: [
    {{"id": "AC1", "text": "The email subject must follow the format: USS Fence Quote #Opp-XXXXXXX", "subpoints": []}},
    {{"id": "AC2", "text": "The email body must include required fields", "subpoints": ["Document Number", "Customer"]}}
  ]

OUTPUT FORMAT:
{{
  "story_title": "extracted title or empty string",
  "description": "narrative description only (user story or problem statement)",
  "requirements": [] or ["requirement1", "requirement2"],
  "acceptance_criteria": [
    {{"id": "AC1", "text": "requirement text", "subpoints": []}},
    {{"id": "AC2", "text": "grouped requirement", "subpoints": ["detail1", "detail2"]}}
  ]
}}

CRITICAL: 
- Description = narrative only (the "what" and "why")
- Acceptance Criteria = specific requirements (the "how" and "what must happen")
- Preserve exact text including special characters like ((NVRTJobSite))
- Extract EXACTLY what appears - zero invention

Return ONLY valid JSON (no explanation, no markdown):"""
    
    def _validate_normalized_structure(self, normalized: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean normalized structure"""
        # Ensure required fields exist
        result = {
            "story_title": normalized.get("story_title", ""),
            "description": normalized.get("description", ""),
            "requirements": normalized.get("requirements", []),
            "acceptance_criteria": normalized.get("acceptance_criteria", [])
        }
        
        # Clean description - remove markdown prefixes and headers
        description = result["description"]
        if not description or not description.strip():
            # If description is empty, try to infer from raw text
            # This handles cases where LLM didn't extract it properly
            description = ""
        else:
            import re
            # Remove markdown bold formatting
            description = re.sub(r'\*\*description\*\*', '', description, flags=re.IGNORECASE)
            description = re.sub(r'\*\*Description\*\*', '', description)
            # Remove "description" prefix if present (with or without colon)
            if description.lower().startswith("description"):
                # Remove "description" or "Description:" prefix
                description = description.split(":", 1)[-1].strip()
            # Remove any Requirements text that might have leaked into description
            if "requirements:" in description.lower():
                description = description.split("requirements:", 1)[0].split("Requirements:", 1)[0].strip()
            # Remove any Acceptance Criteria text that might have leaked
            if "acceptance criteria" in description.lower():
                description = description.split("acceptance criteria", 1)[0].split("Acceptance Criteria", 1)[0].strip()
            # Remove any remaining markdown formatting
            description = re.sub(r'^\*\*', '', description).strip()
            description = re.sub(r'\*\*$', '', description).strip()
            description = description.strip()
        
        result["description"] = description
        
        # Ensure requirements is a list
        if not isinstance(result["requirements"], list):
            if isinstance(result["requirements"], str):
                # Split by newlines if it's a string
                result["requirements"] = [r.strip() for r in result["requirements"].split('\n') if r.strip()]
            else:
                result["requirements"] = []
        
        # Filter out empty requirements
        result["requirements"] = [r for r in result["requirements"] if r and r.strip()]
        
        # Ensure acceptance_criteria is a list
        if not isinstance(result["acceptance_criteria"], list):
            result["acceptance_criteria"] = []
        
        # Validate AC structure and merge hierarchical items
        # CRITICAL: Given/When/Then statements MUST be grouped as subpoints under ONE parent AC
        validated_ac = []
        current_ac = None
        given_when_then_group = []  # Track consecutive Given/When/Then statements
        
        for i, ac in enumerate(result["acceptance_criteria"], 1):
            if isinstance(ac, dict):
                ac_text = ac.get("text", str(ac.get("criterion", "")) if "criterion" in ac else "").strip()
                ac_subpoints = ac.get("subpoints", []) if isinstance(ac.get("subpoints"), list) else []
                
                # Check if this is a top-level AC (Rule, Section header, etc.)
                ac_lower = ac_text.lower()
                is_top_level = (
                    ac_lower.startswith("rule") or
                    ac_lower.startswith("core date rules") or
                    ac_lower.startswith("cross-document") or
                    ac_lower.startswith("validation error") or
                    (not any(keyword in ac_lower for keyword in ["given", "when", "then"]) and 
                     len(ac_text.split()) > 3)  # Longer text likely a header
                )
                
                # Check if it's a subpoint (Given/When/Then)
                is_subpoint = (
                    ac_lower.startswith("given") or
                    ac_lower.startswith("when") or
                    ac_lower.startswith("then") or
                    ac_lower.startswith("an event") or
                    ac_lower.startswith("the user") or
                    (len(ac_text.split()) <= 5 and not is_top_level)
                )
                
                if is_top_level:
                    # Save any pending Given/When/Then group first
                    if given_when_then_group:
                        if current_ac:
                            current_ac["subpoints"].extend(given_when_then_group)
                        else:
                            # Create parent AC for the Given/When/Then group
                            current_ac = {
                                "id": f"AC{len(validated_ac) + 1}",
                                "text": "Acceptance Criteria",  # Generic title for Given/When/Then group
                                "subpoints": given_when_then_group.copy()
                            }
                        given_when_then_group = []
                    
                    # Save previous AC if exists
                    if current_ac:
                        validated_ac.append(current_ac)
                    # Start new AC
                    current_ac = {
                        "id": ac.get("id", f"AC{len(validated_ac) + 1}"),
                        "text": ac_text,
                        "subpoints": ac_subpoints.copy() if ac_subpoints else []
                    }
                elif is_subpoint:
                    # This is a Given/When/Then statement - add to group
                    given_when_then_group.append(ac_text)
                    if ac_subpoints:
                        given_when_then_group.extend(ac_subpoints)
                else:
                    # Ambiguous - check if we have pending Given/When/Then group
                    if given_when_then_group:
                        # Save the group first
                        if current_ac:
                            current_ac["subpoints"].extend(given_when_then_group)
                        else:
                            current_ac = {
                                "id": f"AC{len(validated_ac) + 1}",
                                "text": "Acceptance Criteria",
                                "subpoints": given_when_then_group.copy()
                            }
                        given_when_then_group = []
                    
                    # Treat as top-level if no current AC, otherwise as subpoint
                    if current_ac:
                        current_ac["subpoints"].append(ac_text)
                        if ac_subpoints:
                            current_ac["subpoints"].extend(ac_subpoints)
                    else:
                        validated_ac.append({
                            "id": ac.get("id", f"AC{len(validated_ac) + 1}"),
                            "text": ac_text,
                            "subpoints": ac_subpoints.copy() if ac_subpoints else []
                        })
            elif isinstance(ac, str):
                ac_text = ac.strip()
                ac_lower = ac_text.lower()
                
                # Check if top-level
                is_top_level = (
                    ac_lower.startswith("rule") or
                    ac_lower.startswith("core date rules") or
                    ac_lower.startswith("cross-document") or
                    ac_lower.startswith("validation error") or
                    (not ac_lower.startswith(("given", "when", "then")) and len(ac_text.split()) > 3)
                )
                
                is_subpoint = (
                    ac_lower.startswith("given") or
                    ac_lower.startswith("when") or
                    ac_lower.startswith("then") or
                    ac_lower.startswith("an event") or
                    ac_lower.startswith("the user")
                )
                
                if is_top_level:
                    # Save any pending Given/When/Then group first
                    if given_when_then_group:
                        if current_ac:
                            current_ac["subpoints"].extend(given_when_then_group)
                        else:
                            current_ac = {
                                "id": f"AC{len(validated_ac) + 1}",
                                "text": "Acceptance Criteria",
                                "subpoints": given_when_then_group.copy()
                            }
                        given_when_then_group = []
                    
                    if current_ac:
                        validated_ac.append(current_ac)
                    current_ac = {
                        "id": f"AC{len(validated_ac) + 1}",
                        "text": ac_text,
                        "subpoints": []
                    }
                elif is_subpoint:
                    # Add to Given/When/Then group
                    given_when_then_group.append(ac_text)
                else:
                    # Save any pending Given/When/Then group first
                    if given_when_then_group:
                        if current_ac:
                            current_ac["subpoints"].extend(given_when_then_group)
                        else:
                            current_ac = {
                                "id": f"AC{len(validated_ac) + 1}",
                                "text": "Acceptance Criteria",
                                "subpoints": given_when_then_group.copy()
                            }
                        given_when_then_group = []
                    
                    # Ambiguous
                    if current_ac:
                        current_ac["subpoints"].append(ac_text)
                    else:
                        validated_ac.append({
                            "id": f"AC{len(validated_ac) + 1}",
                            "text": ac_text,
                            "subpoints": []
                        })
        
        # Handle any remaining Given/When/Then group at the end
        if given_when_then_group:
            if current_ac:
                current_ac["subpoints"].extend(given_when_then_group)
            else:
                current_ac = {
                    "id": f"AC{len(validated_ac) + 1}",
                    "text": "Acceptance Criteria",
                    "subpoints": given_when_then_group.copy()
                }
        
        # Handle any remaining Given/When/Then group at the end
        if given_when_then_group:
            if current_ac:
                current_ac["subpoints"].extend(given_when_then_group)
            else:
                # Create parent AC for the Given/When/Then group
                current_ac = {
                    "id": f"AC{len(validated_ac) + 1}",
                    "text": "Acceptance Criteria",
                    "subpoints": given_when_then_group.copy()
                }
        
        # Add final AC if exists
        if current_ac:
            validated_ac.append(current_ac)
        
        # Remove duplicate subpoints
        for ac in validated_ac:
            if ac.get("subpoints"):
                # Remove duplicates while preserving order
                seen = set()
                unique_subpoints = []
                for sp in ac["subpoints"]:
                    if sp and sp not in seen:
                        seen.add(sp)
                        unique_subpoints.append(sp)
                ac["subpoints"] = unique_subpoints
        
        result["acceptance_criteria"] = validated_ac
        
        return result
    
    def _generate_fallback_normalization(self, raw_story_text: str) -> Dict[str, Any]:
        """Generate fallback normalization if LLM fails - improved extraction"""
        import re
        # Simple fallback: try to extract basic structure intelligently
        lines = raw_story_text.split('\n')
        
        description_parts = []
        ac_parts = []
        in_ac_section = False
        in_description_section = False
        found_description_header = False
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            line_lower = line_stripped.lower()
            
            # Detect description section
            if any(keyword in line_lower for keyword in ['**description**', 'description:', 'description']):
                if '**description**' in line_lower or 'description:' in line_lower:
                    found_description_header = True
                    in_description_section = True
                    in_ac_section = False
                    continue  # Skip the header line itself
            
            # Detect AC section
            if any(keyword in line_lower for keyword in ['acceptance criteria', '**acceptance criteria**', 'ac:', 'ac ']):
                in_ac_section = True
                in_description_section = False
                continue  # Skip the header line itself
            
            # Detect Requirements section
            if any(keyword in line_lower for keyword in ['requirements:', 'business rules:', 'core rules:']):
                in_description_section = False
                in_ac_section = False
                continue  # Skip requirements for now in fallback
            
            if in_ac_section:
                # Remove bullets/numbers/markdown
                cleaned = re.sub(r'^[-*•]\s+', '', line_stripped)
                cleaned = re.sub(r'^\d+[.)]\s+', '', cleaned).strip()
                cleaned = re.sub(r'^\*\*', '', cleaned).strip()  # Remove markdown bold
                cleaned = re.sub(r'\*\*$', '', cleaned).strip()
                if cleaned and len(cleaned) > 3:  # Minimum length
                    ac_parts.append(cleaned)
            elif in_description_section or (not in_ac_section and not found_description_header):
                # Extract description - either explicitly marked or before AC section
                cleaned = re.sub(r'^\*\*', '', line_stripped).strip()
                cleaned = re.sub(r'\*\*$', '', cleaned).strip()
                if cleaned and len(cleaned) > 3:
                    description_parts.append(cleaned)
        
        # Process AC parts to group Given/When/Then
        processed_ac = []
        current_ac = None
        
        for ac_text in ac_parts:
            ac_lower = ac_text.lower()
            if ac_lower.startswith(('given', 'when', 'then')):
                if current_ac:
                    current_ac["subpoints"].append(ac_text)
                else:
                    # Create parent AC from first few words
                    words = ac_text.split()[:5]
                    parent_title = " ".join(words)
                    if len(parent_title) > 50:
                        parent_title = parent_title[:47] + "..."
                    current_ac = {
                        "id": f"AC{len(processed_ac) + 1}",
                        "text": parent_title,
                        "subpoints": [ac_text]
                    }
            else:
                # Save current AC if exists
                if current_ac:
                    processed_ac.append(current_ac)
                    current_ac = None
                # Create new AC entry
                processed_ac.append({
                    "id": f"AC{len(processed_ac) + 1}",
                    "text": ac_text,
                    "subpoints": []
                })
        
        # Add final AC if exists
        if current_ac:
            processed_ac.append(current_ac)
        
        return {
            "story_title": "",
            "description": "\n".join(description_parts).strip(),
            "requirements": [],
            "acceptance_criteria": processed_ac if processed_ac else [
                {"id": f"AC{i+1}", "text": ac, "subpoints": []}
                for i, ac in enumerate(ac_parts[:5])  # Limit to 5 in fallback
            ]
        }


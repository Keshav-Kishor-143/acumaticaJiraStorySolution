#!/usr/bin/env python3
"""
Story Classifier - Automatically detect story type to determine processing intensity

Classifies stories into types:
- validation_rule
- ui_story
- email_template
- billing_logic
- workflow
- generic

Classification based on:
- Keywords
- AC patterns
- Presence of screenshots
- Presence of DAC/Graph references
"""

from typing import Dict, Any, List, Optional
import re

from src.utils.logger_utils import get_logger


class StoryClassifier:
    """Classifies JIRA stories by type to optimize processing"""
    
    def __init__(self):
        self.logger = get_logger("STORY_CLASSIFIER")
        
        # Keyword patterns for each story type
        self.type_patterns = {
            'validation_rule': {
                'keywords': [
                    'validation', 'validate', 'valid', 'invalid', 'error message',
                    'field verifying', 'row persisting', 'exception', 'pxexception',
                    'required field', 'mandatory', 'check', 'verify', 'constraint',
                    'business rule', 'validation rule', 'field level validation'
                ],
                'ac_patterns': [
                    r'validation',
                    r'error.*message',
                    r'field.*required',
                    r'cannot.*empty',
                    r'must.*valid'
                ],
                'dac_patterns': ['FieldVerifying', 'RowPersisting', 'PXException']
            },
            'ui_story': {
                'keywords': [
                    'ui', 'user interface', 'screen', 'form', 'field', 'button',
                    'label', 'display', 'show', 'hide', 'visible', 'readonly',
                    'enable', 'disable', 'layout', 'design', 'appearance',
                    'customization', 'custom field', 'user field', 'screen editor'
                ],
                'ac_patterns': [
                    r'screen',
                    r'form.*\d{6}',  # Form ID pattern
                    r'field.*display',
                    r'button.*visible',
                    r'ui.*element'
                ],
                'dac_patterns': ['PXUIFieldAttribute', 'PXDefault', 'PXUIEnabled']
            },
            'email_template': {
                'keywords': [
                    'email', 'template', 'notification', 'send', 'mail',
                    'smtp', 'email template', 'notification template',
                    'email notification', 'email body', 'email subject'
                ],
                'ac_patterns': [
                    r'email.*template',
                    r'send.*email',
                    r'notification',
                    r'email.*body',
                    r'email.*subject'
                ],
                'dac_patterns': ['EmailTemplate', 'Notification']
            },
            'billing_logic': {
                'keywords': [
                    'billing', 'invoice', 'bill', 'charge', 'payment',
                    'billing cycle', 'billing period', 'recurring',
                    'subscription', 'invoice generation', 'billing rule'
                ],
                'ac_patterns': [
                    r'billing',
                    r'invoice.*generat',
                    r'billing.*cycle',
                    r'charge.*customer',
                    r'payment.*process'
                ],
                'dac_patterns': ['ARInvoice', 'ARRegister', 'Billing']
            },
            'workflow': {
                'keywords': [
                    'workflow', 'process', 'approval', 'approve', 'reject',
                    'status', 'state', 'transition', 'workflow step',
                    'approval workflow', 'business process', 'automation'
                ],
                'ac_patterns': [
                    r'workflow',
                    r'approval.*process',
                    r'status.*change',
                    r'state.*transition',
                    r'automation'
                ],
                'dac_patterns': ['Workflow', 'Approval', 'Status']
            }
        }
        
        self.logger.info("Story Classifier initialized")
    
    def classify_story(
        self,
        description: str,
        acceptance_criteria: List[Any],
        has_images: bool = False,
        technical_entities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a story into a specific type
        
        Args:
            description: Story description text
            acceptance_criteria: List of acceptance criteria (strings or dicts)
            has_images: Whether story has screenshots/images
            technical_entities: Optional extracted technical entities
            
        Returns:
            Dictionary with:
                - story_type: Primary type (validation_rule, ui_story, etc.)
                - confidence: Confidence score (0.0-1.0)
                - secondary_types: List of other possible types
                - reasoning: Explanation of classification
        """
        try:
            # Normalize AC to strings
            ac_texts = []
            for ac in acceptance_criteria:
                if isinstance(ac, dict):
                    ac_texts.append(ac.get('text', ''))
                else:
                    ac_texts.append(str(ac))
            
            combined_text = f"{description} {' '.join(ac_texts)}".lower()
            
            # Score each type
            type_scores = {}
            
            for story_type, patterns in self.type_patterns.items():
                score = 0.0
                matches = []
                
                # Keyword matching
                keyword_matches = sum(1 for kw in patterns['keywords'] if kw.lower() in combined_text)
                if keyword_matches > 0:
                    score += min(keyword_matches * 0.15, 0.5)  # Max 0.5 from keywords
                    matches.extend([kw for kw in patterns['keywords'] if kw.lower() in combined_text][:3])
                
                # AC pattern matching
                ac_pattern_matches = sum(1 for pattern in patterns['ac_patterns'] 
                                       if re.search(pattern, combined_text, re.IGNORECASE))
                if ac_pattern_matches > 0:
                    score += min(ac_pattern_matches * 0.2, 0.4)  # Max 0.4 from patterns
                
                # DAC/Graph pattern matching (if entities provided)
                if technical_entities:
                    dac_matches = sum(1 for dac_pattern in patterns['dac_patterns']
                                    for dac in technical_entities.get('dacs', [])
                                    if dac_pattern.lower() in dac.lower())
                    if dac_matches > 0:
                        score += min(dac_matches * 0.1, 0.2)  # Max 0.2 from DACs
                
                # Image bonus for UI stories
                if story_type == 'ui_story' and has_images:
                    score += 0.1
                
                type_scores[story_type] = {
                    'score': min(score, 1.0),
                    'matches': matches[:5]  # Top 5 matches
                }
            
            # Find primary type
            primary_type = max(type_scores.items(), key=lambda x: x[1]['score'])
            
            # If no strong match, classify as generic
            if primary_type[1]['score'] < 0.3:
                primary_type = ('generic', {'score': 0.3, 'matches': []})
            
            # Get secondary types (score > 0.2 but not primary)
            secondary_types = [
                (stype, data['score'])
                for stype, data in type_scores.items()
                if stype != primary_type[0] and data['score'] > 0.2
            ]
            secondary_types.sort(key=lambda x: x[1], reverse=True)
            
            # Build reasoning
            reasoning = f"Classified as '{primary_type[0]}' based on "
            if primary_type[1]['matches']:
                reasoning += f"keywords: {', '.join(primary_type[1]['matches'][:3])}"
            else:
                reasoning += "general patterns"
            
            if has_images and primary_type[0] == 'ui_story':
                reasoning += " and presence of screenshots"
            
            result = {
                'story_type': primary_type[0],
                'confidence': primary_type[1]['score'],
                'secondary_types': [stype for stype, _ in secondary_types[:2]],
                'reasoning': reasoning,
                'all_scores': {stype: data['score'] for stype, data in type_scores.items()}
            }
            
            self.logger.info("Story classified", extra={
                "story_type": result['story_type'],
                "confidence": result['confidence'],
                "secondary_types": result['secondary_types']
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Story classification failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return generic as fallback
            return {
                'story_type': 'generic',
                'confidence': 0.3,
                'secondary_types': [],
                'reasoning': 'Classification failed, defaulting to generic',
                'all_scores': {}
            }
    
    def requires_technical_processing(self, story_type: str) -> bool:
        """
        Determine if story type requires intensive technical processing
        
        Args:
            story_type: Classified story type
            
        Returns:
            True if requires technical processing, False for simple UI/email stories
        """
        # High-tech stories that need full processing
        high_tech_types = ['validation_rule', 'billing_logic', 'workflow']
        
        # Low-tech stories that can skip some processing
        low_tech_types = ['ui_story', 'email_template']
        
        if story_type in high_tech_types:
            return True
        elif story_type in low_tech_types:
            return False
        else:  # generic
            return True  # Default to full processing for safety


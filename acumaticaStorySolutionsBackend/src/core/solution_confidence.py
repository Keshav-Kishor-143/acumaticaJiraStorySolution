#!/usr/bin/env python3
"""
Solution Confidence - Compute final confidence score for the solution

Scoring Inputs:
- % of AC items successfully mapped
- % of entities validated
- Presence of required code handlers
- Retrieval coverage (%)
- Missing entities count
"""

from typing import Dict, Any, List, Optional

from src.utils.logger_utils import get_logger


class SolutionConfidenceScorer:
    """Computes final confidence score for generated solutions"""
    
    def __init__(self):
        self.logger = get_logger("SOLUTION_CONFIDENCE")
        
        # Weight configuration for different factors
        self.weights = {
            'ac_mapping': 0.25,      # 25% - AC mapping completeness
            'entity_validation': 0.25,  # 25% - Entity validation rate
            'retrieval_coverage': 0.20,  # 20% - Retrieval success
            'code_handlers': 0.15,   # 15% - Presence of code handlers
            'missing_entities': 0.15  # 15% - Missing entities penalty
        }
        
        self.logger.info("Solution Confidence Scorer initialized")
    
    def calculate_confidence(
        self,
        ac_mappings: List[Dict[str, Any]],
        validated_entities: Dict[str, Any],
        fact_table: Dict[str, Any],
        retrieval_metadata: Dict[str, Any],
        total_ac_count: int
    ) -> Dict[str, Any]:
        """
        Calculate final confidence score for solution
        
        Args:
            ac_mappings: List of AC mappings
            validated_entities: Validated entities with confidence scores
            fact_table: Technical fact table
            retrieval_metadata: Retrieval metadata
            total_ac_count: Total number of acceptance criteria
            
        Returns:
            Dictionary with:
                - confidence_score: Final confidence (0.0-1.0)
                - component_scores: Individual component scores
                - reasoning: Explanation of score
        """
        try:
            component_scores = {}
            
            # 1. AC Mapping Score (0.0-1.0)
            ac_score = self._calculate_ac_mapping_score(ac_mappings, total_ac_count)
            component_scores['ac_mapping'] = ac_score
            
            # 2. Entity Validation Score (0.0-1.0)
            entity_score = self._calculate_entity_validation_score(validated_entities)
            component_scores['entity_validation'] = entity_score
            
            # 3. Retrieval Coverage Score (0.0-1.0)
            retrieval_score = self._calculate_retrieval_coverage_score(retrieval_metadata)
            component_scores['retrieval_coverage'] = retrieval_score
            
            # 4. Code Handlers Score (0.0-1.0)
            handlers_score = self._calculate_code_handlers_score(fact_table, ac_mappings)
            component_scores['code_handlers'] = handlers_score
            
            # 5. Missing Entities Penalty (0.0-1.0, where 1.0 = no penalty)
            missing_score = self._calculate_missing_entities_score(fact_table)
            component_scores['missing_entities'] = missing_score
            
            # Calculate weighted final score
            final_score = (
                ac_score * self.weights['ac_mapping'] +
                entity_score * self.weights['entity_validation'] +
                retrieval_score * self.weights['retrieval_coverage'] +
                handlers_score * self.weights['code_handlers'] +
                missing_score * self.weights['missing_entities']
            )
            
            # Ensure score is in [0.0, 1.0] range
            final_score = max(0.0, min(1.0, final_score))
            
            # Generate reasoning
            reasoning = self._generate_reasoning(component_scores, final_score)
            
            result = {
                'confidence_score': round(final_score, 3),
                'component_scores': {k: round(v, 3) for k, v in component_scores.items()},
                'reasoning': reasoning,
                'weights': self.weights
            }
            
            self.logger.info("Solution confidence calculated", extra={
                "final_score": result['confidence_score'],
                "component_scores": result['component_scores']
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to calculate solution confidence", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return low confidence on error
            return {
                'confidence_score': 0.3,
                'component_scores': {},
                'reasoning': f'Confidence calculation failed: {str(e)}',
                'weights': self.weights
            }
    
    def _calculate_ac_mapping_score(self, ac_mappings: List[Dict[str, Any]], total_ac_count: int) -> float:
        """Calculate AC mapping completeness score"""
        if total_ac_count == 0:
            return 1.0  # No ACs to map, perfect score
        
        mapped_count = len(ac_mappings)
        
        # Check how many ACs have technical mappings
        acs_with_mappings = sum(1 for mapping in ac_mappings 
                              if mapping.get('related_forms') or 
                                 mapping.get('related_dacs') or 
                                 mapping.get('related_fields'))
        
        # Score based on both count and quality
        count_score = mapped_count / total_ac_count
        quality_score = acs_with_mappings / max(mapped_count, 1)
        
        # Combined score (weighted towards quality)
        return (count_score * 0.6 + quality_score * 0.4)
    
    def _calculate_entity_validation_score(self, validated_entities: Dict[str, Any]) -> float:
        """Calculate entity validation rate score"""
        confidence_scores = validated_entities.get('confidence_scores', {})
        
        if not confidence_scores:
            return 0.5  # No validation data, neutral score
        
        total_entities = 0
        validated_count = 0
        
        for entity_type, entity_scores in confidence_scores.items():
            for entity, score in entity_scores.items():
                total_entities += 1
                if score >= 0.75:  # Confirmed threshold
                    validated_count += 1
        
        if total_entities == 0:
            return 0.5
        
        return validated_count / total_entities
    
    def _calculate_retrieval_coverage_score(self, retrieval_metadata: Dict[str, Any]) -> float:
        """Calculate retrieval coverage score"""
        attempts = retrieval_metadata.get('retrieval_attempts', 0)
        successes = retrieval_metadata.get('retrieval_successes', 0)
        
        if attempts == 0:
            return 0.5  # No retrieval attempts, neutral score
        
        success_rate = successes / attempts
        
        # Penalize fallback usage
        fallback_penalty = 0.1 if retrieval_metadata.get('fallback_triggered', False) else 0.0
        
        return max(0.0, success_rate - fallback_penalty)
    
    def _calculate_code_handlers_score(self, fact_table: Dict[str, Any], ac_mappings: List[Dict[str, Any]]) -> float:
        """Calculate code handlers presence score"""
        # Check if fact table has event handlers
        events = fact_table.get('events', [])
        has_handlers = len(events) > 0
        
        # Check if AC mappings require handlers
        required_handlers = []
        for mapping in ac_mappings:
            handlers = mapping.get('required_handlers', [])
            required_handlers.extend(handlers)
        
        if not required_handlers:
            # No handlers required, perfect score
            return 1.0
        
        # Check if required handlers are present
        required_set = set(required_handlers)
        present_set = set(events)
        
        if required_set.issubset(present_set):
            return 1.0  # All required handlers present
        elif len(present_set & required_set) > 0:
            # Some handlers present
            return len(present_set & required_set) / len(required_set)
        else:
            return 0.3  # No required handlers present
    
    def _calculate_missing_entities_score(self, fact_table: Dict[str, Any]) -> float:
        """Calculate missing entities penalty score"""
        missing_entities = fact_table.get('missing_entities', [])
        
        if not missing_entities:
            return 1.0  # No missing entities, perfect score
        
        # Penalty increases with number of missing entities
        # Cap at 10 missing entities for scoring
        missing_count = min(len(missing_entities), 10)
        
        # Score decreases linearly: 1.0 (0 missing) -> 0.0 (10+ missing)
        return max(0.0, 1.0 - (missing_count / 10.0))
    
    def _generate_reasoning(self, component_scores: Dict[str, float], final_score: float) -> str:
        """Generate human-readable reasoning for confidence score"""
        reasons = []
        
        if final_score >= 0.8:
            reasons.append("High confidence solution")
        elif final_score >= 0.6:
            reasons.append("Moderate confidence solution")
        else:
            reasons.append("Low confidence solution - review recommended")
        
        # Add component-specific feedback
        if component_scores.get('ac_mapping', 0) < 0.7:
            reasons.append("Some acceptance criteria were not fully mapped")
        
        if component_scores.get('entity_validation', 0) < 0.7:
            reasons.append("Some entities could not be validated")
        
        if component_scores.get('retrieval_coverage', 0) < 0.7:
            reasons.append("Retrieval coverage was limited")
        
        if component_scores.get('code_handlers', 0) < 0.7:
            reasons.append("Some required code handlers may be missing")
        
        if component_scores.get('missing_entities', 0) < 0.7:
            reasons.append("Some entities were not found in documentation")
        
        return ". ".join(reasons) + "."
    
    def should_reject_solution(self, confidence_score: float, threshold: float = 0.5) -> bool:
        """
        Determine if solution should be rejected based on confidence
        
        Args:
            confidence_score: Final confidence score
            threshold: Rejection threshold
            
        Returns:
            True if solution should be rejected
        """
        return confidence_score < threshold
    
    def get_quality_level(self, confidence_score: float) -> str:
        """
        Get quality level label for confidence score
        
        Args:
            confidence_score: Final confidence score
            
        Returns:
            Quality level string
        """
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"


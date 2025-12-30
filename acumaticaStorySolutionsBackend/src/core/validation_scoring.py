#!/usr/bin/env python3
"""
Entity Validation & Confidence Scoring Module

Validates extracted entities against retrieved documentation and assigns
confidence scores. Performs negative retrieval to eliminate hallucinations.
"""

from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, Counter
import re

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.tech_extraction import TechnicalEntityExtractor
from src.core.acumatica_patterns import AcumaticaPatterns


class EntityValidationScorer:
    """Validates entities and assigns confidence scores"""
    
    def __init__(self):
        self.logger = get_logger("VALIDATION_SCORER")
        self.entity_extractor = TechnicalEntityExtractor()
        self.patterns = AcumaticaPatterns()
        self.logger.info("Entity Validation Scorer initialized")
    
    def validate_and_score_entities(
        self,
        entities: Dict[str, List[str]],
        retrieved_texts: List[str],
        vision_texts: List[str],
        retriever_scores: Optional[Dict[str, float]] = None,
        initial_confidence_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Validate entities and assign confidence scores
        
        Args:
            entities: Dictionary of entity types to lists of entity values
            retrieved_texts: List of retrieved text content
            vision_texts: List of vision-extracted text content
            retriever_scores: Optional scores from retriever
            
        Returns:
            Dictionary with:
            - validated_entities: Entities with confidence >= 0.75
            - uncertain_entities: Entities with confidence 0.50-0.74
            - unconfirmed_entities: Entities with confidence < 0.50
            - confidence_scores: Full confidence score dictionary
        """
        try:
            self.logger.info("Validating and scoring entities", extra={
                "entity_types": list(entities.keys()),
                "total_entities": sum(len(v) for v in entities.values() if isinstance(v, list)),
                "retrieved_texts_count": len(retrieved_texts),
                "vision_texts_count": len(vision_texts)
            })
            
            # Combine all text for validation
            all_texts = retrieved_texts + vision_texts
            combined_text = ' '.join(all_texts).lower()
            
            # Calculate confidence scores
            confidence_scores = {}
            validated_entities = defaultdict(list)
            uncertain_entities = defaultdict(list)
            unconfirmed_entities = defaultdict(list)
            
            for entity_type, entity_list in entities.items():
                if not isinstance(entity_list, list):
                    continue
                
                confidence_scores[entity_type] = {}
                
                for entity in entity_list:
                    if not isinstance(entity, str) or not entity.strip():
                        continue
                    
                    # Check for initial confidence from LLM extraction (pattern-resolved entities)
                    initial_confidence = None
                    if initial_confidence_scores:
                        type_scores = initial_confidence_scores.get(entity_type, {})
                        initial_confidence = type_scores.get(entity, None)
                    
                    # Check if entity is resolved by AcumaticaPatterns (known valid entity)
                    is_pattern_resolved = self._is_pattern_resolved_entity(entity_type, entity)
                    
                    # If entity has high initial confidence (from LLM + pattern resolution), preserve it
                    if initial_confidence and initial_confidence >= 0.65:
                        # This is a pattern-resolved entity from LLM extraction - preserve high confidence
                        confidence = initial_confidence
                        self.logger.debug("Preserving initial confidence for pattern-resolved entity", extra={
                            "entity": entity,
                            "type": entity_type,
                            "confidence": confidence
                        })
                    elif is_pattern_resolved:
                        # Entity matches known Acumatica pattern - boost confidence significantly
                        # Pattern-resolved entities are valid Acumatica entities even if not in docs
                        base_confidence = self._calculate_confidence(
                            entity=entity,
                            entity_type=entity_type,
                            retrieved_texts=retrieved_texts,
                            vision_texts=vision_texts,
                            combined_text=combined_text,
                            retriever_score=retriever_scores.get(f"{entity_type}:{entity}", 0.0) if retriever_scores else 0.0
                        )
                        # Strong boost for pattern-resolved entities - they're valid even if not in retrieved docs
                        # This ensures entities extracted from story text make it into fact table
                        confidence = max(0.75, min(0.90, base_confidence + 0.50))
                        self.logger.debug("Boosting confidence for pattern-resolved entity", extra={
                            "entity": entity,
                            "type": entity_type,
                            "base_confidence": base_confidence,
                            "final_confidence": confidence
                        })
                    else:
                        # Standard confidence calculation
                        base_confidence = self._calculate_confidence(
                            entity=entity,
                            entity_type=entity_type,
                            retrieved_texts=retrieved_texts,
                            vision_texts=vision_texts,
                            combined_text=combined_text,
                            retriever_score=retriever_scores.get(f"{entity_type}:{entity}", 0.0) if retriever_scores else 0.0
                        )
                        confidence = base_confidence
                    
                    confidence_scores[entity_type][entity] = confidence
                    
                    # Categorize by confidence (lowered threshold to accept pattern-resolved entities)
                    if confidence >= 0.65:  # Lowered from 0.75 to accept pattern-resolved entities
                        validated_entities[entity_type].append(entity)
                    elif confidence >= 0.40:  # Lowered from 0.50
                        uncertain_entities[entity_type].append(entity)
                    else:
                        unconfirmed_entities[entity_type].append(entity)
            
            result = {
                "validated_entities": dict(validated_entities),
                "uncertain_entities": dict(uncertain_entities),
                "unconfirmed_entities": dict(unconfirmed_entities),
                "confidence_scores": confidence_scores
            }
            
            self.logger.info("Entity validation completed", extra={
                "validated_count": sum(len(v) for v in validated_entities.values()),
                "uncertain_count": sum(len(v) for v in uncertain_entities.values()),
                "unconfirmed_count": sum(len(v) for v in unconfirmed_entities.values())
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to validate and score entities", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return empty result on error
            return {
                "validated_entities": {},
                "uncertain_entities": {},
                "unconfirmed_entities": entities,
                "confidence_scores": {}
            }
    
    def _calculate_confidence(
        self,
        entity: str,
        entity_type: str,
        retrieved_texts: List[str],
        vision_texts: List[str],
        combined_text: str,
        retriever_score: float = 0.0
    ) -> float:
        """
        Calculate confidence score for an entity
        
        Confidence formula:
        - Occurrence count in retrieved text: 0-0.5 (max 10 occurrences = 0.5)
        - Occurrence count in vision text: 0-0.2 (max 5 occurrences = 0.2)
        - Retriever score: 0-0.3 (already 0.0-1.0, scaled to 0.3)
        
        Total: 0.0-1.0
        
        Args:
            entity: Entity value
            entity_type: Type of entity
            retrieved_texts: List of retrieved text content
            vision_texts: List of vision-extracted text content
            combined_text: Combined text (lowercase)
            retriever_score: Score from retriever (0.0-1.0)
            
        Returns:
            Confidence score (0.0-1.0)
        """
        entity_lower = entity.lower().strip()
        
        # Count occurrences in retrieved text
        retrieved_count = 0
        for text in retrieved_texts:
            if text:
                text_lower = text.lower()
                # Exact match with word boundaries
                pattern = r'\b' + re.escape(entity_lower) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                retrieved_count += matches
        
        # Normalize occurrence count (max 10 = 0.5)
        occurrence_score = min(retrieved_count / 10.0, 1.0) * 0.5
        
        # Count occurrences in vision text
        vision_count = 0
        for text in vision_texts:
            if text:
                text_lower = text.lower()
                pattern = r'\b' + re.escape(entity_lower) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                vision_count += matches
        
        # Normalize vision count (max 5 = 0.2)
        vision_score = min(vision_count / 5.0, 1.0) * 0.2
        
        # Retriever score (already 0.0-1.0, scale to 0.3)
        retriever_weighted = retriever_score * 0.3
        
        # Calculate final confidence
        confidence = occurrence_score + vision_score + retriever_weighted
        
        return min(confidence, 1.0)
    
    def perform_negative_retrieval(
        self,
        entity: str,
        entity_type: str,
        documentation_texts: List[str]
    ) -> bool:
        """
        Perform negative retrieval: search for entity in documentation
        
        Args:
            entity: Entity value to search for
            entity_type: Type of entity
            documentation_texts: List of documentation text content
            
        Returns:
            True if entity found, False otherwise
        """
        if not documentation_texts:
            return False
        
        entity_lower = entity.lower().strip()
        
        # Search in all documentation texts
        for text in documentation_texts:
            if not text:
                continue
            
            text_lower = text.lower()
            
            # Exact match with word boundaries
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            if re.search(pattern, text_lower):
                return True
            
            # Also check for partial matches (e.g., "SOOrder" might appear as "SO Order")
            entity_words = entity_lower.replace('_', ' ').split()
            if len(entity_words) > 1:
                # Check if all words appear together
                words_pattern = r'\b' + r'\b.*\b'.join(re.escape(w) for w in entity_words) + r'\b'
                if re.search(words_pattern, text_lower):
                    return True
        
        return False
    
    def _is_pattern_resolved_entity(self, entity_type: str, entity_value: str) -> bool:
        """
        Check if entity matches known Acumatica patterns (valid Acumatica entity)
        
        Args:
            entity_type: Type of entity ('forms', 'dacs', 'graphs', etc.)
            entity_value: Entity value to check
            
        Returns:
            True if entity matches known patterns, False otherwise
        """
        try:
            if entity_type == 'dacs':
                # Check if DAC is in known patterns
                resolved = self.patterns.resolve_dac_alias(entity_value)
                return resolved is not None
            elif entity_type == 'forms':
                # Check if Form ID matches known pattern format (AA######)
                # First check if it's in known patterns
                normalized = self.patterns.resolve_form_id(entity_value)
                if normalized is not None:
                    return True
                # Also check if it matches standard Acumatica form ID pattern
                if re.match(r'^[A-Z]{2}\d{6}$', entity_value.upper()):
                    return True
            elif entity_type == 'graphs':
                # Check if Graph name matches known patterns
                # Get graphs for known DACs
                for dac_name, dac_info in self.patterns.dac_form_graph_map.items():
                    if entity_value in dac_info.get('graphs', []):
                        return True
                # Also check if it's a standard graph name pattern (ends with Entry/Maint/Setup)
                if re.match(r'^[A-Z][a-zA-Z0-9]*(Entry|Maint|Setup)$', entity_value):
                    return True
            elif entity_type == 'fields':
                # Field names are harder to validate, but check common patterns
                # Fields are usually PascalCase
                if re.match(r'^[A-Z][a-zA-Z0-9]*$', entity_value):
                    return True
            elif entity_type == 'events':
                # Event handlers follow standard patterns
                if entity_value in ['FieldVerifying', 'FieldUpdated', 'RowPersisting', 'RowSelecting', 'RowInserting', 'RowUpdating', 'RowDeleting']:
                    return True
            
            return False
        except Exception as e:
            self.logger.debug("Error checking pattern resolution", extra={
                "entity": entity_value,
                "type": entity_type,
                "error": str(e)
            })
            return False
    
    def validate_all_entities_negative_retrieval(
        self,
        entities: Dict[str, List[str]],
        documentation_texts: List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate all entities using negative retrieval
        
        Args:
            entities: Dictionary of entity types to lists of entity values
            documentation_texts: List of documentation text content
            
        Returns:
            Dictionary of validated entities (only those found in documentation)
        """
        validated = defaultdict(list)
        
        for entity_type, entity_list in entities.items():
            if not isinstance(entity_list, list):
                continue
            
            for entity in entity_list:
                if not isinstance(entity, str) or not entity.strip():
                    continue
                
                if self.perform_negative_retrieval(entity, entity_type, documentation_texts):
                    validated[entity_type].append(entity)
                else:
                    self.logger.debug(f"Entity not found in documentation: {entity_type}:{entity}")
        
        return dict(validated)
    
    def mark_unconfirmed_entities(
        self,
        entities: Dict[str, List[str]],
        confidence_scores: Dict[str, Dict[str, float]],
        threshold: float = 0.75
    ) -> Dict[str, List[str]]:
        """
        Mark entities below confidence threshold as unconfirmed
        
        Args:
            entities: Dictionary of entity types to lists of entity values
            confidence_scores: Confidence scores dictionary
            threshold: Confidence threshold (default 0.75)
            
        Returns:
            Dictionary of unconfirmed entities
        """
        unconfirmed = defaultdict(list)
        
        for entity_type, entity_list in entities.items():
            if not isinstance(entity_list, list):
                continue
            
            scores = confidence_scores.get(entity_type, {})
            
            for entity in entity_list:
                if not isinstance(entity, str) or not entity.strip():
                    continue
                
                confidence = scores.get(entity, 0.0)
                if confidence < threshold:
                    unconfirmed[entity_type].append(f"{entity} [NOT CONFIRMED — DO NOT USE]")
        
        return dict(unconfirmed)


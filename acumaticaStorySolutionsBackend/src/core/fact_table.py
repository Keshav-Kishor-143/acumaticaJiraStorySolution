#!/usr/bin/env python3
"""
Fact Table Generation Module

Creates structured fact tables from validated technical entities.
This module ensures all technical entities are properly organized and validated
before being used in solution generation.
"""

from typing import Dict, Any, List, Optional
import json

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.acumatica_patterns import AcumaticaPatterns


class FactTableGenerator:
    """Generates structured fact tables from validated technical entities"""
    
    def __init__(self):
        self.logger = get_logger("FACT_TABLE_GENERATOR")
        self.patterns = AcumaticaPatterns()
        self.logger.info("Fact Table Generator initialized")
    
    def generate_fact_table(
        self,
        validated_entities: Dict[str, Any],
        retrieved_documentation: List[Dict[str, Any]],
        vision_metadata: List[Dict[str, Any]],
        confidence_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured fact table from validated entities
        
        Args:
            validated_entities: Dictionary of validated technical entities
            retrieved_documentation: List of retrieved document chunks
            vision_metadata: Vision-extracted metadata
            confidence_scores: Optional confidence scores for entities
            
        Returns:
            Structured fact table dictionary
        """
        try:
            self.logger.info("Generating fact table", extra={
                "entities_count": sum(len(v) if isinstance(v, list) else 0 for v in validated_entities.values() if isinstance(v, list)),
                "docs_count": len(retrieved_documentation),
                "vision_count": len(vision_metadata)
            })
            
            # Filter entities by confidence threshold
            # Use lower threshold (0.65) for pattern-resolved entities from LLM extraction
            # These are reliable even if not explicitly in documentation
            confidence_threshold = 0.65  # Lowered from 0.75 to accept pattern-resolved entities
            filtered_entities = {}
            
            for entity_type in ['forms', 'graphs', 'dacs', 'fields', 'events', 'navigation_paths', 'px_attributes', 'tables']:
                entities = validated_entities.get(entity_type, [])
                if not isinstance(entities, list):
                    filtered_entities[entity_type] = []
                    continue
                
                filtered_list = []
                type_scores = confidence_scores.get(entity_type, {}) if confidence_scores else {}
                
                for entity in entities:
                    if isinstance(entity, str):
                        score = type_scores.get(entity, 0.85)  # Default to high confidence for pattern-resolved entities
                        # Accept entities with score >= threshold OR if they're pattern-resolved (score >= 0.65)
                        if score >= confidence_threshold:
                            filtered_list.append(entity)
                        else:
                            self.logger.debug("Filtered out low-confidence entity", extra={
                                "entity": entity,
                                "type": entity_type,
                                "score": score,
                                "threshold": confidence_threshold
                            })
                
                filtered_entities[entity_type] = filtered_list
            
            # Resolve aliases using Acumatica patterns
            resolved_entities = self._resolve_entity_aliases(filtered_entities)
            
            # Extract entities by type
            fact_table = {
                "forms": resolved_entities.get('forms', []),
                "graphs": resolved_entities.get('graphs', []),
                "dacs": resolved_entities.get('dacs', []),
                "fields": resolved_entities.get('fields', []),
                "events": resolved_entities.get('events', []),
                "navigation": resolved_entities.get('navigation_paths', []),
                "px_attributes": resolved_entities.get('px_attributes', []),
                "tables": resolved_entities.get('tables', []),
                "missing_entities": [],
                "confidence_scores": confidence_scores or {}
            }
            
            # Deduplicate all lists
            for key in ['forms', 'graphs', 'dacs', 'fields', 'events', 'navigation', 'px_attributes', 'tables']:
                if isinstance(fact_table[key], list):
                    fact_table[key] = list(dict.fromkeys(fact_table[key]))  # Preserve order
            
            # Extract missing entities (entities mentioned but not found)
            missing_entities = self._identify_missing_entities(
                validated_entities,
                retrieved_documentation,
                vision_metadata
            )
            fact_table['missing_entities'] = missing_entities
            
            self.logger.info("Fact table generated", extra={
                "forms_count": len(fact_table['forms']),
                "dacs_count": len(fact_table['dacs']),
                "graphs_count": len(fact_table['graphs']),
                "fields_count": len(fact_table['fields']),
                "missing_count": len(missing_entities)
            })
            
            return fact_table
            
        except Exception as e:
            self.logger.error("Failed to generate fact table", extra={
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
    
    def _identify_missing_entities(
        self,
        validated_entities: Dict[str, Any],
        retrieved_documentation: List[Dict[str, Any]],
        vision_metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify entities that were mentioned but not found in documentation
        
        Args:
            validated_entities: Validated entities
            retrieved_documentation: Retrieved documentation chunks
            vision_metadata: Vision metadata
            
        Returns:
            List of missing entity names
        """
        missing = []
        
        # Combine all documentation text
        all_text = []
        for doc in retrieved_documentation:
            text = doc.get('text_content', '') or doc.get('content', '')
            if text:
                all_text.append(text.lower())
        
        for vision in vision_metadata:
            vision_text = vision.get('vision_content', '') or vision.get('text', '')
            if vision_text:
                all_text.append(vision_text.lower())
        
        combined_text = ' '.join(all_text)
        
        # Check each entity type for missing items
        # This is a simplified check - in production, you'd want more sophisticated matching
        for entity_type, entities in validated_entities.items():
            if not isinstance(entities, list):
                continue
            
            for entity in entities:
                if not isinstance(entity, str):
                    continue
                
                # Check if entity appears in documentation
                entity_lower = entity.lower()
                if entity_lower not in combined_text:
                    # Check for partial matches (e.g., "SOOrder" might appear as "SO Order")
                    entity_words = entity_lower.replace('_', ' ').split()
                    found = False
                    for word in entity_words:
                        if len(word) > 3 and word in combined_text:
                            found = True
                            break
                    
                    if not found:
                        missing.append(f"{entity_type}:{entity}")
        
        return missing
    
    def _resolve_entity_aliases(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Resolve entity aliases to canonical names using Acumatica patterns
        
        Args:
            entities: Dictionary of entities by type
            
        Returns:
            Dictionary with resolved canonical names
        """
        resolved = {}
        
        for entity_type, entity_list in entities.items():
            resolved_list = []
            
            for entity in entity_list:
                if entity_type == 'dacs':
                    # Resolve DAC aliases
                    canonical = self.patterns.resolve_dac_alias(entity)
                    if canonical:
                        resolved_list.append(canonical)
                        # Also add related forms and graphs
                        forms = self.patterns.get_forms_for_dac(canonical)
                        graphs = self.patterns.get_graphs_for_dac(canonical)
                        # Add to respective lists if not already present
                        if 'forms' not in resolved:
                            resolved['forms'] = []
                        resolved['forms'].extend(forms)
                        if 'graphs' not in resolved:
                            resolved['graphs'] = []
                        resolved['graphs'].extend(graphs)
                    else:
                        resolved_list.append(entity)
                
                elif entity_type == 'forms':
                    # Normalize form IDs
                    normalized = self.patterns.resolve_form_id(entity)
                    if normalized:
                        resolved_list.append(normalized)
                    else:
                        resolved_list.append(entity)
                
                elif entity_type == 'fields':
                    # Resolve field aliases
                    canonical = self.patterns.resolve_field_alias(entity)
                    if canonical:
                        resolved_list.append(canonical)
                    else:
                        resolved_list.append(entity)
                
                else:
                    # Keep as-is for other types
                    resolved_list.append(entity)
            
            resolved[entity_type] = list(dict.fromkeys(resolved_list))  # Deduplicate
        
        return resolved
    
    def validate_fact_table(self, fact_table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fact table structure and content
        
        Args:
            fact_table: Fact table to validate
            
        Returns:
            Validated fact table with any corrections
        """
        validated = fact_table.copy()
        
        # Ensure all required keys exist
        required_keys = [
            'forms', 'graphs', 'dacs', 'fields', 'events',
            'navigation', 'px_attributes', 'tables', 'missing_entities'
        ]
        
        for key in required_keys:
            if key not in validated:
                validated[key] = []
        
        # Ensure confidence_scores exists
        if 'confidence_scores' not in validated:
            validated['confidence_scores'] = {}
        
        # Validate list types
        for key in required_keys:
            if not isinstance(validated[key], list):
                validated[key] = []
        
        # Remove empty strings and None values
        for key in required_keys:
            validated[key] = [
                item for item in validated[key]
                if item and isinstance(item, str) and item.strip()
            ]
        
        return validated
    
    def merge_fact_tables(
        self,
        fact_table1: Dict[str, Any],
        fact_table2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two fact tables, combining entities and confidence scores
        
        Args:
            fact_table1: First fact table
            fact_table2: Second fact table
            
        Returns:
            Merged fact table
        """
        merged = {
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
        
        # Merge entity lists
        entity_keys = ['forms', 'graphs', 'dacs', 'fields', 'events', 'navigation', 'px_attributes', 'tables']
        for key in entity_keys:
            list1 = fact_table1.get(key, []) or []
            list2 = fact_table2.get(key, []) or []
            # Combine and deduplicate
            merged[key] = list(dict.fromkeys(list1 + list2))
        
        # Merge missing entities
        missing1 = fact_table1.get('missing_entities', []) or []
        missing2 = fact_table2.get('missing_entities', []) or []
        merged['missing_entities'] = list(dict.fromkeys(missing1 + missing2))
        
        # Merge confidence scores (take maximum)
        scores1 = fact_table1.get('confidence_scores', {}) or {}
        scores2 = fact_table2.get('confidence_scores', {}) or {}
        
        for entity_type in scores1:
            if entity_type not in merged['confidence_scores']:
                merged['confidence_scores'][entity_type] = {}
            for entity, score in scores1[entity_type].items():
                existing_score = merged['confidence_scores'][entity_type].get(entity, 0.0)
                merged['confidence_scores'][entity_type][entity] = max(existing_score, score)
        
        for entity_type in scores2:
            if entity_type not in merged['confidence_scores']:
                merged['confidence_scores'][entity_type] = {}
            for entity, score in scores2[entity_type].items():
                existing_score = merged['confidence_scores'][entity_type].get(entity, 0.0)
                merged['confidence_scores'][entity_type][entity] = max(existing_score, score)
        
        return merged


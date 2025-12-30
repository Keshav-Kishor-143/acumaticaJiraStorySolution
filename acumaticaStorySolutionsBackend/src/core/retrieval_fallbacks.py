#!/usr/bin/env python3
"""
Retrieval Fallbacks - Fallback retrieval modes when main hybrid retriever fails

Provides:
- Synonym expansion
- Fuzzy matching
- Table → DAC resolution
- Graph → DAC resolution
- Loosening semantic constraints
- Fallback scoring rules
"""

from typing import Dict, Any, List, Optional
import re
from difflib import SequenceMatcher

from src.utils.logger_utils import get_logger
from src.core.acumatica_patterns import AcumaticaPatterns


class RetrievalFallbacks:
    """Fallback retrieval strategies when primary retrieval fails"""
    
    def __init__(self):
        self.logger = get_logger("RETRIEVAL_FALLBACKS")
        self.patterns = AcumaticaPatterns()
        
        self.logger.info("Retrieval Fallbacks initialized")
    
    def expand_synonyms(self, query: str, entities: Dict[str, List[str]]) -> List[str]:
        """
        Expand query with synonyms and aliases
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            List of expanded query variations
        """
        expanded_queries = [query]
        
        # Expand DAC aliases
        for dac in entities.get('dacs', []):
            if not isinstance(dac, str) or not dac:
                continue
            canonical = self.patterns.resolve_dac_alias(dac)
            if canonical and canonical != dac:
                # Replace alias with canonical name
                expanded_query = query.replace(dac, canonical)
                if expanded_query != query:
                    expanded_queries.append(expanded_query)
        
        # Expand form IDs
        for form in entities.get('forms', []):
            if not isinstance(form, str) or not form:
                continue
            normalized = self.patterns.resolve_form_id(form)
            if normalized and normalized != form:
                expanded_query = query.replace(form, normalized)
                if expanded_query != query:
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def fuzzy_match_entity(self, entity: str, entity_type: str, threshold: float = 0.7) -> Optional[str]:
        """
        Fuzzy match entity against known patterns
        
        Args:
            entity: Entity to match
            entity_type: Type ('dac', 'form', 'graph', 'field')
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Matched entity name or None
        """
        entity_lower = entity.lower().strip()
        
        if entity_type == 'dac':
            candidates = self.patterns.get_all_dacs()
        elif entity_type == 'form':
            candidates = self.patterns.get_all_forms()
        elif entity_type == 'graph':
            candidates = self.patterns.get_all_graphs()
        else:
            return None
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = SequenceMatcher(None, entity_lower, candidate.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        
        if best_match:
            self.logger.debug("Fuzzy matched entity", extra={
                "original": entity,
                "matched": best_match,
                "score": best_score,
                "type": entity_type
            })
        
        return best_match
    
    def resolve_table_to_dac(self, table_name: str) -> Optional[str]:
        """
        Resolve table name to DAC
        
        Args:
            table_name: Database table name
            
        Returns:
            DAC name or None
        """
        # Common table naming patterns
        # Tables often match DAC names or have prefixes
        
        # Remove common prefixes
        clean_name = table_name
        prefixes = ['dbo.', 'SO', 'AR', 'PO', 'IN', 'EP', 'PM']
        for prefix in prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # Try to match against known DACs
        dac_match = self.patterns.resolve_dac_alias(clean_name)
        if dac_match:
            return dac_match
        
        # Try fuzzy match
        return self.fuzzy_match_entity(clean_name, 'dac', threshold=0.6)
    
    def resolve_graph_to_dac(self, graph_name: str) -> Optional[str]:
        """
        Resolve graph name to DAC
        
        Args:
            graph_name: Graph class name
            
        Returns:
            DAC name or None
        """
        # Graph names often follow patterns: DACNameEntry, DACNameMaint
        # Extract DAC name from graph
        
        # Remove common suffixes
        graph_clean = graph_name
        suffixes = ['Entry', 'Maint', 'Setup', 'Inquiry', 'EntryBase']
        for suffix in suffixes:
            if graph_clean.endswith(suffix):
                graph_clean = graph_clean[:-len(suffix)]
                break
        
        # Try to match against known DACs
        dac_match = self.patterns.resolve_dac_alias(graph_clean)
        if dac_match:
            return dac_match
        
        # Try fuzzy match
        return self.fuzzy_match_entity(graph_clean, 'dac', threshold=0.6)
    
    def loosen_semantic_constraints(self, query: str) -> List[str]:
        """
        Generate looser query variations for broader search
        
        Args:
            query: Original query
            
        Returns:
            List of loosened query variations
        """
        variations = [query]
        
        # Remove stop words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = query.split()
        filtered_words = [w for w in words if w.lower() not in stop_words]
        if len(filtered_words) < len(words):
            variations.append(' '.join(filtered_words))
        
        # Extract key technical terms
        technical_terms = []
        # Form IDs
        form_ids = re.findall(r'[A-Z]{2}\d{6}', query)
        technical_terms.extend(form_ids)
        # DAC names (CamelCase)
        dac_names = re.findall(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b', query)
        technical_terms.extend(dac_names)
        
        if technical_terms:
            variations.append(' '.join(technical_terms))
        
        # Remove punctuation and special characters
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        clean_query = ' '.join(clean_query.split())
        if clean_query != query:
            variations.append(clean_query)
        
        return list(set(variations))
    
    def apply_fallback_scoring(self, results: List[Dict[str, Any]], fallback_type: str) -> List[Dict[str, Any]]:
        """
        Apply fallback-specific scoring rules
        
        Args:
            results: Search results
            fallback_type: Type of fallback used ('synonym', 'fuzzy', 'loosened')
            
        Returns:
            Results with adjusted scores
        """
        score_adjustments = {
            'synonym': 0.9,  # Slight penalty for synonym expansion
            'fuzzy': 0.8,    # More penalty for fuzzy matching
            'loosened': 0.7  # Most penalty for loosened constraints
        }
        
        adjustment = score_adjustments.get(fallback_type, 1.0)
        
        for result in results:
            # Adjust score based on fallback type
            if 'score' in result:
                result['score'] = result['score'] * adjustment
            if 'combined_score' in result:
                result['combined_score'] = result['combined_score'] * adjustment
            
            # Add fallback metadata
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['fallback_type'] = fallback_type
            result['metadata']['fallback_adjustment'] = adjustment
        
        return results
    
    def should_trigger_fallback(self, results: List[Dict[str, Any]], min_results: int = 3, min_score: float = 0.3) -> bool:
        """
        Determine if fallback retrieval should be triggered
        
        Args:
            results: Current search results
            min_results: Minimum number of results required
            min_score: Minimum score threshold
            
        Returns:
            True if fallback should be triggered
        """
        if len(results) < min_results:
            return True
        
        if results and results[0].get('score', 0) < min_score:
            return True
        
        return False
    
    def get_fallback_strategy(self, query: str, entities: Dict[str, List[str]], 
                             failed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine best fallback strategy
        
        Args:
            query: Original query
            entities: Extracted entities
            failed_results: Results from failed primary retrieval
            
        Returns:
            Dictionary with fallback strategy details
        """
        strategy = {
            'type': 'none',
            'queries': [query],
            'reason': 'No fallback needed'
        }
        
        # Check if we have entities to expand
        has_entities = any(entities.get(key, []) for key in ['dacs', 'forms', 'graphs'])
        
        if has_entities:
            # Try synonym expansion first
            expanded = self.expand_synonyms(query, entities)
            if len(expanded) > 1:
                strategy = {
                    'type': 'synonym',
                    'queries': expanded,
                    'reason': f'Expanded {len(expanded)} query variations using entity synonyms'
                }
            else:
                # Try loosening constraints
                loosened = self.loosen_semantic_constraints(query)
                strategy = {
                    'type': 'loosened',
                    'queries': loosened,
                    'reason': f'Generated {len(loosened)} loosened query variations'
                }
        else:
            # No entities, try loosening constraints
            loosened = self.loosen_semantic_constraints(query)
            strategy = {
                'type': 'loosened',
                'queries': loosened,
                'reason': f'Generated {len(loosened)} loosened query variations (no entities to expand)'
            }
        
        self.logger.info("Fallback strategy determined", extra={
            "strategy_type": strategy['type'],
            "query_count": len(strategy['queries']),
            "reason": strategy['reason']
        })
        
        return strategy


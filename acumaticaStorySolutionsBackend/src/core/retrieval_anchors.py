#!/usr/bin/env python3
"""
Entity-Anchored Retrieval Module

Performs secondary hard-keyword retrieval for extracted entities.
This ensures that technical entities found during extraction are validated
through targeted exact-match searches in the documentation.
"""

from typing import Dict, Any, List, Optional, Set, Union
from collections import defaultdict

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.hybrid_retriever import HybridRetriever, SearchResult


class EntityAnchoredRetriever:
    """Performs entity-anchored retrieval using hard keyword searches"""
    
    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None):
        self.logger = get_logger("ENTITY_ANCHORED_RETRIEVER")
        self.hybrid_retriever = hybrid_retriever
        self.logger.info("Entity-Anchored Retriever initialized")
    
    def perform_entity_anchored_retrieval(
        self,
        entities: Dict[str, List[str]],
        base_query: Optional[str] = None,
        top_k_per_entity: int = 3
    ) -> List[SearchResult]:
        """
        Perform secondary retrieval pass for each extracted entity
        
        Args:
            entities: Dictionary of entity types to lists of entity values
            base_query: Optional base query for context
            top_k_per_entity: Number of results to retrieve per entity
            
        Returns:
            List of SearchResult objects from entity-anchored searches
        """
        if not self.hybrid_retriever:
            self.logger.warning("HybridRetriever not provided, skipping entity-anchored retrieval")
            return []
        
        try:
            self.logger.info("Performing entity-anchored retrieval", extra={
                "entity_types": list(entities.keys()),
                "total_entities": sum(len(v) for v in entities.values() if isinstance(v, list)),
                "top_k_per_entity": top_k_per_entity
            })
            
            all_results = []
            seen_results = set()  # Track by (pdf_name, page_number) to avoid duplicates
            
            # Priority order: forms, DACs, graphs, fields, events
            priority_order = ['forms', 'dacs', 'graphs', 'fields', 'events']
            
            for entity_type in priority_order:
                if entity_type not in entities:
                    continue
                
                entity_list = entities[entity_type]
                if not isinstance(entity_list, list):
                    continue
                
                for entity in entity_list[:10]:  # Limit to top 10 entities per type
                    if not isinstance(entity, str) or not entity.strip():
                        continue
                    
                    # Create entity-specific query
                    entity_query = self._build_entity_query(entity, entity_type, base_query)
                    
                    # Perform hard keyword search
                    try:
                        results = self._force_keyword_search(
                            entity_query,
                            entity,
                            top_k=top_k_per_entity
                        )
                        
                        # Add results if not already seen
                        for result in results:
                            result_key = (result.pdf_name, result.page_number)
                            if result_key not in seen_results:
                                seen_results.add(result_key)
                                # Boost score for entity matches
                                result.combined_score = min(result.combined_score * 1.2, 1.0)
                                all_results.append(result)
                                
                    except Exception as e:
                        self.logger.warning(f"Entity-anchored search failed for {entity_type}:{entity}", extra={
                            "error": str(e)
                        })
                        continue
            
            # Sort by combined score (entity matches get boosted)
            all_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            self.logger.info("Entity-anchored retrieval completed", extra={
                "results_found": len(all_results),
                "unique_pages": len(seen_results)
            })
            
            return all_results
            
        except Exception as e:
            self.logger.error("Failed to perform entity-anchored retrieval", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return []
    
    def _build_entity_query(
        self,
        entity: str,
        entity_type: str,
        base_query: Optional[str]
    ) -> str:
        """Build query string for entity search"""
        
        # Entity-specific query patterns
        query_parts = []
        
        if entity_type == 'forms':
            query_parts.append(f'Form ID {entity}')
            query_parts.append(f'Screen {entity}')
        elif entity_type == 'dacs':
            query_parts.append(f'DAC {entity}')
            query_parts.append(f'class {entity}')
            query_parts.append(f'{entity} DAC')
        elif entity_type == 'graphs':
            query_parts.append(f'Graph {entity}')
            query_parts.append(f'class {entity}')
            query_parts.append(f'{entity} Graph')
        elif entity_type == 'fields':
            query_parts.append(f'Field {entity}')
            query_parts.append(f'property {entity}')
            query_parts.append(f'{entity} field')
        elif entity_type == 'events':
            query_parts.append(f'Event {entity}')
            query_parts.append(f'Handler {entity}')
            query_parts.append(f'{entity} event')
        
        # Add base query context if provided
        if base_query:
            query_parts.insert(0, base_query)
        
        return ' '.join(query_parts)
    
    def _force_keyword_search(
        self,
        query: str,
        entity: str,
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Force exact keyword search for entity
        
        Args:
            query: Search query
            entity: Entity value to search for
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.hybrid_retriever:
            return []
        
        try:
            # Use hybrid retriever's search method with entity-specific query
            # The query already contains entity-specific patterns from _build_entity_query
            # Use include_strategies to prioritize keyword matching
            
            # Create search parameters that prioritize exact keyword matching
            search_params = {
                'force_keyword_match': True,
                'entity_exact_match': entity,
                'boost_exact_match': True
            }
            
            # Call hybrid retriever's search method
            # Use include_strategies to prioritize keyword and domain-specific search
            results = self.hybrid_retriever.search(
                query=query,
                document_id=None,  # Search all documents
                top_k=top_k,
                include_strategies=['enhanced_keyword', 'domain_specific'],  # Prioritize keyword matching
                search_params=search_params
            )
            
            return results if isinstance(results, list) else []
            
        except Exception as e:
            self.logger.warning(f"Keyword search failed for entity {entity}", extra={
                "error": str(e),
                "query": query
            })
            return []
    
    def _dict_to_search_result(self, chunk_dict: Dict[str, Any]) -> SearchResult:
        """Convert dictionary chunk to SearchResult object"""
        try:
            return SearchResult(
                pdf_name=chunk_dict.get('pdf_name', ''),
                page_number=chunk_dict.get('page_number', 0),
                score=chunk_dict.get('score', 0.0),
                text_content=chunk_dict.get('text_content', '') or chunk_dict.get('content', ''),
                metadata=chunk_dict.get('metadata', {}),
                relevance_signals=chunk_dict.get('relevance_signals', {}),
                search_strategy=chunk_dict.get('search_strategy', 'unknown'),
                combined_score=chunk_dict.get('combined_score', chunk_dict.get('score', 0.0)),
                content=chunk_dict.get('content', ''),
                image_path=chunk_dict.get('image_path', ''),
                vision_extracted_text=chunk_dict.get('vision_extracted_text', ''),
                section_type=chunk_dict.get('section_type', ''),
                image_type=chunk_dict.get('image_type', ''),
                dll_name=chunk_dict.get('dll_name', '')
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert dict to SearchResult: {e}", extra={
                "chunk_keys": list(chunk_dict.keys())
            })
            # Return minimal SearchResult
            return SearchResult(
                pdf_name=chunk_dict.get('pdf_name', ''),
                page_number=chunk_dict.get('page_number', 0),
                score=chunk_dict.get('score', 0.0),
                text_content=chunk_dict.get('text_content', '') or chunk_dict.get('content', ''),
                metadata={},
                relevance_signals={},
                search_strategy='unknown',
                combined_score=chunk_dict.get('combined_score', chunk_dict.get('score', 0.0))
            )
    
    def _search_result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult object to dictionary"""
        return {
            'pdf_name': result.pdf_name,
            'page_number': result.page_number,
            'score': result.score,
            'text_content': result.text_content or result.content,
            'content': result.content or result.text_content,
            'metadata': result.metadata,
            'relevance_signals': result.relevance_signals,
            'search_strategy': result.search_strategy,
            'combined_score': result.combined_score,
            'image_path': result.image_path or getattr(result, 'get_image_path', lambda: '')(),
            'vision_extracted_text': result.vision_extracted_text,
            'section_type': result.section_type,
            'image_type': result.image_type,
            'dll_name': result.dll_name,
            'document_id': result.metadata.get('document_id', ''),
            'image_exists': bool(result.image_path)
        }
    
    def merge_with_base_results(
        self,
        base_results: Union[List[SearchResult], List[Dict[str, Any]]],
        entity_results: List[SearchResult],
        entity_weight: float = 0.6
    ) -> Union[List[SearchResult], List[Dict[str, Any]]]:
        """
        Merge entity-anchored results with base retrieval results
        
        Args:
            base_results: Results from initial retrieval (can be SearchResult objects or dicts)
            entity_results: Results from entity-anchored retrieval (SearchResult objects)
            entity_weight: Weight for entity matches (default 0.6 = 60%)
            
        Returns:
            Merged and re-ranked results (same format as base_results)
        """
        try:
            # Handle empty base_results
            if not base_results:
                return []
            
            # Detect format: check if base_results are dicts or SearchResult objects
            base_is_dict = isinstance(base_results[0], dict)
            
            # Convert base results to SearchResult objects if needed
            base_search_results = []
            for result in base_results:
                if isinstance(result, dict):
                    base_search_results.append(self._dict_to_search_result(result))
                else:
                    base_search_results.append(result)
            
            # Create result map by (pdf_name, page_number)
            result_map = {}
            
            # Add base results
            for result in base_search_results:
                key = (result.pdf_name, result.page_number)
                if key not in result_map:
                    result_map[key] = result
            
            # Merge entity results with boosted scores
            for result in entity_results:
                key = (result.pdf_name, result.page_number)
                if key in result_map:
                    # Boost existing result score
                    existing_score = result_map[key].combined_score
                    entity_score = result.combined_score
                    # Weighted combination: entity_weight * entity_score + (1 - entity_weight) * base_score
                    result_map[key].combined_score = (
                        entity_weight * entity_score +
                        (1 - entity_weight) * existing_score
                    )
                    # Also update base score for consistency
                    result_map[key].score = result_map[key].combined_score
                else:
                    # Add new result
                    result_map[key] = result
            
            # Convert back to list and sort
            merged_results = list(result_map.values())
            merged_results.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Convert back to dicts if original format was dicts
            if base_is_dict:
                merged_results = [self._search_result_to_dict(r) for r in merged_results]
            
            self.logger.info("Results merged", extra={
                "base_count": len(base_results),
                "entity_count": len(entity_results),
                "merged_count": len(merged_results),
                "format": "dict" if base_is_dict else "SearchResult"
            })
            
            return merged_results
            
        except Exception as e:
            self.logger.error("Failed to merge results", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return base results on error (preserve format)
            return base_results


#!/usr/bin/env python3
"""
Pipeline Metadata - Diagnostic metadata for solutions (debug + quality assurance)

Metadata Includes:
- retrieval_attempts
- entities_extracted
- entities_confirmed
- vision_used
- fallback_triggered
- story_type
- final_confidence_score
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logger_utils import get_logger


class PipelineMetadata:
    """Collects and manages pipeline processing metadata"""
    
    def __init__(self):
        self.logger = get_logger("PIPELINE_METADATA")
        self.metadata = {}
        self.logger.info("Pipeline Metadata initialized")
    
    def reset(self):
        """Reset metadata for new processing"""
        self.metadata = {
            'processing_start_time': datetime.now().isoformat(),
            'retrieval_attempts': 0,
            'retrieval_successes': 0,
            'retrieval_failures': 0,
            'entities_extracted': {},
            'entities_confirmed': {},
            'entities_unconfirmed': {},
            'vision_used': False,
            'vision_pages_processed': 0,
            'fallback_triggered': False,
            'fallback_types': [],
            'story_type': None,
            'story_type_confidence': 0.0,
            'ac_count': 0,
            'ac_mapped': 0,
            'fact_table_entities': {},
            'missing_entities': [],
            'final_confidence_score': 0.0,
            'processing_stages_completed': [],
            'errors': [],
            'warnings': []
        }
    
    def record_retrieval_attempt(self, success: bool, method: str = "hybrid"):
        """Record a retrieval attempt"""
        self.metadata['retrieval_attempts'] += 1
        if success:
            self.metadata['retrieval_successes'] += 1
        else:
            self.metadata['retrieval_failures'] += 1
        
        if 'retrieval_methods' not in self.metadata:
            self.metadata['retrieval_methods'] = []
        self.metadata['retrieval_methods'].append(method)
    
    def record_entities_extracted(self, entities: Dict[str, List[str]]):
        """Record extracted entities"""
        for entity_type, entity_list in entities.items():
            if entity_type not in self.metadata['entities_extracted']:
                self.metadata['entities_extracted'][entity_type] = []
            self.metadata['entities_extracted'][entity_type].extend(entity_list)
    
    def record_entities_confirmed(self, entities: Dict[str, List[str]], confidence_scores: Dict[str, Dict[str, float]]):
        """Record confirmed entities with confidence scores"""
        confirmed = {}
        unconfirmed = {}
        
        for entity_type, entity_list in entities.items():
            confirmed[entity_type] = []
            unconfirmed[entity_type] = []
            
            type_scores = confidence_scores.get(entity_type, {})
            
            for entity in entity_list:
                score = type_scores.get(entity, 0.0)
                if score >= 0.75:
                    confirmed[entity_type].append(entity)
                else:
                    unconfirmed[entity_type].append(entity)
        
        self.metadata['entities_confirmed'] = confirmed
        self.metadata['entities_unconfirmed'] = unconfirmed
    
    def record_vision_usage(self, pages_processed: int = 1):
        """Record vision API usage"""
        self.metadata['vision_used'] = True
        self.metadata['vision_pages_processed'] += pages_processed
    
    def record_fallback(self, fallback_type: str):
        """Record fallback retrieval trigger"""
        self.metadata['fallback_triggered'] = True
        if 'fallback_types' not in self.metadata:
            self.metadata['fallback_types'] = []
        self.metadata['fallback_types'].append(fallback_type)
    
    def record_story_type(self, story_type: str, confidence: float):
        """Record story classification"""
        self.metadata['story_type'] = story_type
        self.metadata['story_type_confidence'] = confidence
    
    def record_ac_mapping(self, total_ac: int, mapped_ac: int):
        """Record AC mapping statistics"""
        self.metadata['ac_count'] = total_ac
        self.metadata['ac_mapped'] = mapped_ac
    
    def record_fact_table(self, fact_table: Dict[str, Any]):
        """Record fact table statistics"""
        self.metadata['fact_table_entities'] = {
            'forms': len(fact_table.get('forms', [])),
            'dacs': len(fact_table.get('dacs', [])),
            'graphs': len(fact_table.get('graphs', [])),
            'fields': len(fact_table.get('fields', [])),
            'events': len(fact_table.get('events', []))
        }
        self.metadata['missing_entities'] = fact_table.get('missing_entities', [])
    
    def record_stage_completion(self, stage_name: str):
        """Record completion of a processing stage"""
        if 'processing_stages_completed' not in self.metadata:
            self.metadata['processing_stages_completed'] = []
        self.metadata['processing_stages_completed'].append(stage_name)
    
    def record_error(self, error: str, stage: Optional[str] = None):
        """Record an error"""
        error_entry = {
            'message': error,
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        }
        self.metadata['errors'].append(error_entry)
    
    def record_warning(self, warning: str, stage: Optional[str] = None):
        """Record a warning"""
        warning_entry = {
            'message': warning,
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        }
        self.metadata['warnings'].append(warning_entry)
    
    def record_final_confidence(self, confidence_score: float):
        """Record final solution confidence score"""
        self.metadata['final_confidence_score'] = confidence_score
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize metadata and add summary statistics
        
        Returns:
            Complete metadata dictionary
        """
        self.metadata['processing_end_time'] = datetime.now().isoformat()
        
        # Calculate processing time
        if 'processing_start_time' in self.metadata:
            start = datetime.fromisoformat(self.metadata['processing_start_time'])
            end = datetime.fromisoformat(self.metadata['processing_end_time'])
            self.metadata['processing_time_seconds'] = (end - start).total_seconds()
        
        # Calculate entity statistics
        total_extracted = sum(len(v) if isinstance(v, list) else 0 
                            for v in self.metadata.get('entities_extracted', {}).values())
        total_confirmed = sum(len(v) if isinstance(v, list) else 0 
                            for v in self.metadata.get('entities_confirmed', {}).values())
        total_unconfirmed = sum(len(v) if isinstance(v, list) else 0 
                              for v in self.metadata.get('entities_unconfirmed', {}).values())
        
        self.metadata['entity_statistics'] = {
            'total_extracted': total_extracted,
            'total_confirmed': total_confirmed,
            'total_unconfirmed': total_unconfirmed,
            'confirmation_rate': total_confirmed / total_extracted if total_extracted > 0 else 0.0
        }
        
        # Calculate retrieval success rate
        total_retrievals = self.metadata.get('retrieval_attempts', 0)
        successful_retrievals = self.metadata.get('retrieval_successes', 0)
        self.metadata['retrieval_success_rate'] = (
            successful_retrievals / total_retrievals if total_retrievals > 0 else 0.0
        )
        
        # Calculate AC mapping rate
        ac_count = self.metadata.get('ac_count', 0)
        ac_mapped = self.metadata.get('ac_mapped', 0)
        self.metadata['ac_mapping_rate'] = (
            ac_mapped / ac_count if ac_count > 0 else 0.0
        )
        
        # Add quality indicators
        self.metadata['quality_indicators'] = {
            'high_confidence': self.metadata.get('final_confidence_score', 0.0) >= 0.75,
            'good_retrieval': self.metadata.get('retrieval_success_rate', 0.0) >= 0.7,
            'good_entity_confirmation': self.metadata.get('entity_statistics', {}).get('confirmation_rate', 0.0) >= 0.7,
            'complete_ac_mapping': self.metadata.get('ac_mapping_rate', 0.0) >= 0.8,
            'no_fallbacks': not self.metadata.get('fallback_triggered', False),
            'vision_used': self.metadata.get('vision_used', False)
        }
        
        self.logger.info("Pipeline metadata finalized", extra={
            "processing_time": self.metadata.get('processing_time_seconds', 0),
            "final_confidence": self.metadata.get('final_confidence_score', 0.0),
            "entities_confirmed": total_confirmed,
            "retrieval_success_rate": self.metadata.get('retrieval_success_rate', 0.0)
        })
        
        return self.metadata
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get current metadata"""
        return self.metadata.copy()
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of metadata
        
        Returns:
            Summary string
        """
        lines = [
            "=== Pipeline Processing Summary ===",
            f"Story Type: {self.metadata.get('story_type', 'Unknown')} (confidence: {self.metadata.get('story_type_confidence', 0.0):.2f})",
            f"Processing Time: {self.metadata.get('processing_time_seconds', 0):.2f}s",
            "",
            "Retrieval:",
            f"  Attempts: {self.metadata.get('retrieval_attempts', 0)}",
            f"  Successes: {self.metadata.get('retrieval_successes', 0)}",
            f"  Success Rate: {self.metadata.get('retrieval_success_rate', 0.0):.1%}",
            f"  Fallback Triggered: {self.metadata.get('fallback_triggered', False)}",
            "",
            "Entities:",
            f"  Extracted: {self.metadata.get('entity_statistics', {}).get('total_extracted', 0)}",
            f"  Confirmed: {self.metadata.get('entity_statistics', {}).get('total_confirmed', 0)}",
            f"  Confirmation Rate: {self.metadata.get('entity_statistics', {}).get('confirmation_rate', 0.0):.1%}",
            "",
            "Acceptance Criteria:",
            f"  Total: {self.metadata.get('ac_count', 0)}",
            f"  Mapped: {self.metadata.get('ac_mapped', 0)}",
            f"  Mapping Rate: {self.metadata.get('ac_mapping_rate', 0.0):.1%}",
            "",
            "Vision:",
            f"  Used: {self.metadata.get('vision_used', False)}",
            f"  Pages Processed: {self.metadata.get('vision_pages_processed', 0)}",
            "",
            f"Final Confidence Score: {self.metadata.get('final_confidence_score', 0.0):.2f}",
        ]
        
        if self.metadata.get('errors'):
            lines.append(f"\nErrors: {len(self.metadata['errors'])}")
        
        if self.metadata.get('warnings'):
            lines.append(f"Warnings: {len(self.metadata['warnings'])}")
        
        return "\n".join(lines)


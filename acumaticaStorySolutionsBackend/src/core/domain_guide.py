#!/usr/bin/env python3
"""
Domain Guide - Uses domain.json to guide initial intent understanding and search

This module provides domain-guided query enhancement by:
1. Matching story keywords to domain topics in domain.json
2. Identifying relevant documents BEFORE search
3. Enhancing queries with domain-specific terms
4. Guiding retrieval to the right documents
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter

from src.config.config import config
from src.utils.logger_utils import get_logger


class DomainGuide:
    """Uses domain.json to guide intent understanding and search"""
    
    def __init__(self):
        self.logger = get_logger("DOMAIN_GUIDE")
        self.domain_metadata = self._load_domain_metadata()
        self.logger.info("Domain Guide initialized", extra={
            "documents_loaded": len(self.domain_metadata.get("document_summaries", {}))
        })
    
    def _load_domain_metadata(self) -> Dict[str, Any]:
        """Load domain.json metadata"""
        try:
            domain_path = Path(config.LOCAL_BASE_PATH) / "domain.json"
            if domain_path.exists():
                with open(domain_path, encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning("Failed to load domain metadata", extra={"error": str(e)})
            return {}
    
    def identify_relevant_documents(
        self,
        story_text: str,
        acceptance_criteria: List[str]
    ) -> Dict[str, float]:
        """
        Identify relevant documents based on story content and domain.json
        
        Args:
            story_text: Story description
            acceptance_criteria: List of acceptance criteria texts
            
        Returns:
            Dictionary mapping document names to relevance scores (0.0-1.0)
        """
        try:
            combined_text = f"{story_text} {' '.join(acceptance_criteria)}".lower()
            
            # Extract keywords from story
            story_keywords = self._extract_keywords(combined_text)
            
            # Score each document
            document_scores = {}
            summaries = self.domain_metadata.get("document_summaries", {})
            
            for doc_name, doc_info in summaries.items():
                score = self._calculate_document_relevance(
                    story_keywords,
                    doc_info,
                    combined_text
                )
                if score > 0.1:  # Only include documents with meaningful relevance
                    document_scores[doc_name] = score
            
            # Sort by score (descending)
            sorted_docs = dict(sorted(document_scores.items(), key=lambda x: x[1], reverse=True))
            
            self.logger.info("Relevant documents identified", extra={
                "total_documents": len(summaries),
                "relevant_documents": len(sorted_docs),
                "top_documents": list(sorted_docs.keys())[:5]
            })
            
            return sorted_docs
            
        except Exception as e:
            self.logger.error("Failed to identify relevant documents", extra={"error": str(e)})
            return {}
    
    def enhance_query_with_domain(
        self,
        base_query: str,
        story_text: str,
        acceptance_criteria: List[str]
    ) -> str:
        """
        Enhance query with domain-specific terms from domain.json
        
        Args:
            base_query: Original query
            story_text: Story description
            acceptance_criteria: List of acceptance criteria
            
        Returns:
            Enhanced query with domain-specific terms
        """
        try:
            combined_text = f"{story_text} {' '.join(acceptance_criteria)}".lower()
            story_keywords = self._extract_keywords(combined_text)
            
            # Find matching domain topics
            domain_terms = self._extract_domain_terms(story_keywords)
            
            # Build enhanced query
            enhanced_parts = [base_query]
            
            # Add high-relevance domain terms
            for term in domain_terms[:5]:  # Top 5 domain terms
                if term.lower() not in base_query.lower():
                    enhanced_parts.append(term)
            
            enhanced_query = " ".join(enhanced_parts)
            
            self.logger.debug("Query enhanced with domain terms", extra={
                "original_length": len(base_query),
                "enhanced_length": len(enhanced_query),
                "domain_terms_added": len(domain_terms[:5])
            })
            
            return enhanced_query
            
        except Exception as e:
            self.logger.warning("Failed to enhance query with domain", extra={"error": str(e)})
            return base_query
    
    def get_document_priority_list(
        self,
        story_text: str,
        acceptance_criteria: List[str]
    ) -> List[str]:
        """
        Get prioritized list of documents to search based on domain matching
        
        Args:
            story_text: Story description
            acceptance_criteria: List of acceptance criteria
            
        Returns:
            List of document names ordered by relevance (most relevant first)
        """
        relevant_docs = self.identify_relevant_documents(story_text, acceptance_criteria)
        return list(relevant_docs.keys())
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # Extract words (3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and return unique set
        keywords = {word for word in words if word not in stop_words}
        
        return keywords
    
    def _calculate_document_relevance(
        self,
        story_keywords: Set[str],
        doc_info: Dict[str, Any],
        story_text: str
    ) -> float:
        """Calculate relevance score for a document based on story keywords"""
        score = 0.0
        
        # Check core topics
        core_topics = doc_info.get("core_topics", [])
        topic_matches = 0
        for topic in core_topics:
            topic_lower = topic.lower()
            # Check if any story keyword matches topic
            for keyword in story_keywords:
                if keyword in topic_lower or topic_lower in keyword:
                    topic_matches += 1
                    break
            # Also check if topic appears in story text
            if topic_lower in story_text:
                topic_matches += 1
        
        if core_topics:
            score += (topic_matches / len(core_topics)) * 0.5
        
        # Check key operations
        key_operations = doc_info.get("key_operations", [])
        operation_matches = 0
        for operation in key_operations:
            operation_lower = operation.lower()
            for keyword in story_keywords:
                if keyword in operation_lower or operation_lower in keyword:
                    operation_matches += 1
                    break
            if operation_lower in story_text:
                operation_matches += 1
        
        if key_operations:
            score += (operation_matches / len(key_operations)) * 0.3
        
        # Check summary
        summary = doc_info.get("summary", "").lower()
        summary_matches = sum(1 for keyword in story_keywords if keyword in summary)
        if story_keywords:
            score += (summary_matches / len(story_keywords)) * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_domain_terms(
        self,
        story_keywords: Set[str]
    ) -> List[str]:
        """Extract domain-specific terms that match story keywords"""
        domain_terms = []
        summaries = self.domain_metadata.get("document_summaries", {})
        
        # Collect all matching domain terms
        for doc_info in summaries.values():
            # Check core topics
            for topic in doc_info.get("core_topics", []):
                topic_lower = topic.lower()
                for keyword in story_keywords:
                    if keyword in topic_lower or topic_lower in keyword:
                        if topic not in domain_terms:
                            domain_terms.append(topic)
            
            # Check key operations
            for operation in doc_info.get("key_operations", []):
                operation_lower = operation.lower()
                for keyword in story_keywords:
                    if keyword in operation_lower or operation_lower in keyword:
                        if operation not in domain_terms:
                            domain_terms.append(operation)
        
        return domain_terms


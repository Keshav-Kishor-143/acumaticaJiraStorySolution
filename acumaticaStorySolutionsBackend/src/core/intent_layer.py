#!/usr/bin/env python3
"""
Intent Understanding Layer for Acumatica Documentation Search
Uses domain knowledge and LLM to direct search to relevant sections
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI

from src.config.config import config
from src.utils.logger_utils import get_logger

class IntentUnderstandingLayer:
    """Intelligent layer for query analysis and search direction"""
    
    def __init__(self):
        self.logger = get_logger("INTENT_LAYER")
        
        # Load domain knowledge
        domain_path = os.path.join(config.LOCAL_BASE_PATH, "domain.json")
        try:
            with open(domain_path, 'r', encoding='utf-8') as f:
                self.domain_knowledge = json.load(f)
            self.logger.info("Domain knowledge loaded", extra={
                "docs_available": len(self.domain_knowledge.get("document_summaries", {})),
                "path": domain_path
            })
        except Exception as e:
            self.logger.error("Failed to load domain knowledge", extra={
                "error": str(e),
                "path": domain_path
            })
            # Initialize with empty domain knowledge to allow graceful degradation
            self.domain_knowledge = {"document_summaries": {}}
            self.logger.warning("Continuing with empty domain knowledge")
            
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Cache for intent analysis results (simple in-memory cache)
        self._intent_cache = {}
        self._cache_max_size = 100  # Limit cache size to prevent memory issues
        
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query (normalized)"""
        # Normalize query for caching (lowercase, strip whitespace)
        return query.lower().strip()
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and determine relevant document sections with caching"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self._intent_cache:
                self.logger.info("Using cached intent analysis", extra={"query": query[:50]})
                return self._intent_cache[cache_key]
            
            self.logger.info("Analyzing query intent", extra={"query": query})
            
            # Check if domain knowledge is available
            if not self.domain_knowledge.get("document_summaries"):
                self.logger.warning("No domain knowledge available, returning default analysis")
                return {
                    "relevant_docs": [],
                    "query_intent": {"primary": "unknown", "secondary": [], "technical_level": "intermediate"},
                    "key_concepts": {"primary": [], "related": [], "forms": [], "operations": []},
                    "search_focus": {"priority_sections": [], "exclude_sections": [], "context_requirements": []}
                }
            
            # Prepare domain context (only relevant summaries)
            domain_context = {
                "document_summaries": self.domain_knowledge["document_summaries"]
            }
            
            prompt = f"""You are an expert semantic intelligence system for Acumatica documentation. Your task is to understand the MEANING and CONTEXT of queries, not match keywords.

            Query: "{query}"

            Available Documentation Sections (with semantic summaries):
            {json.dumps(domain_context, indent=2)}

            YOUR INTELLIGENCE TASK:
            
            **SEMANTIC UNDERSTANDING PRINCIPLES:**
            
            1. **UNDERSTAND MEANING, NOT KEYWORDS**:
               - "Mobile App" semantically means: mobile application, mobile screens, mobile customization, mobile forms
               - "Work Order in Mobile App" = mobile application customization for work orders
               - "Prevent unapproval" = validation logic, business rules, customization
               - Match documents based on SEMANTIC ALIGNMENT with query meaning, not exact word matches
            
            2. **CONTEXTUAL INFERENCE**:
               - If query mentions a module (Sales Order, Work Order, Ticket, etc.) + customization/implementation
                 → Understand this needs technical documentation for that module
               - If query mentions "Mobile App" + any customization task
                 → Mobile App customization guide is semantically most relevant (score >= 0.9)
               - If query involves changing system behavior, adding validations, preventing actions
                 → This is semantically a customization/implementation task
            
            3. **SEMANTIC RELATIONSHIP MATCHING**:
               - Read each document's summary and core_topics SEMANTICALLY
               - Match based on conceptual alignment, not keyword overlap
               - Example: Query "Mobile App field customization" semantically aligns with:
                 * "mobile applications" guide (high score 0.9+)
                 * "screen customization" guide (medium score 0.7+)
                 * NOT just "customization" guide (lower score 0.6)
            
            4. **INTELLIGENT PRIORITIZATION**:
               - Most specific match gets highest score (e.g., Mobile App guide for mobile queries)
               - General guides get lower scores unless query is truly general
               - Multiple relevant docs should be included with appropriate scores
               - Scores should reflect semantic relevance: 0.9+ = perfect semantic match, 0.7-0.9 = strong match, 0.5-0.7 = relevant, <0.5 = weak match

            **ANALYSIS PROCESS:**
            
            Step 1: Understand the semantic meaning of the query
            - What is the user really trying to accomplish?
            - What type of information do they need (implementation, usage, configuration)?
            - What domain/module does this relate to?
            
            Step 2: Semantically match to document summaries
            - Read each document summary and understand its semantic scope
            - Determine semantic alignment: Does this document's meaning match the query's meaning?
            - Consider: core_topics, key_operations, forms - do they semantically relate?
            
            Step 3: Score based on semantic relevance
            - Perfect semantic match (e.g., "Mobile App customization" → Mobile App guide): 0.9-1.0
            - Strong semantic match (e.g., "customization" → Customization guide): 0.7-0.9
            - Relevant match (e.g., "field modification" → Customization guide): 0.5-0.7
            - Weak match: <0.5
            
            Step 4: Provide intelligent reasoning
            - Explain WHY each document is relevant semantically
            - Show your understanding of the relationship between query and document

            Return a JSON object with:
            - relevant_docs: List of document IDs ordered by semantic relevance score (0.0-1.0)
              Format: [{{"id": "doc_id", "score": 0.95, "reasoning": "Semantic explanation: why this document's meaning aligns with query meaning"}}]
              IMPORTANT: Use semantic understanding, not keyword matching. Scores should reflect true semantic relevance.
            - query_intent: {{
                "primary": "semantic understanding of what user needs",
                "secondary": ["additional semantic intents"],
                "technical_level": "basic|intermediate|advanced",
                "is_customization_task": true/false (based on semantic meaning, not keywords)
              }}
            - key_concepts: {{
                "primary": ["main concepts understood semantically"],
                "related": ["related concepts inferred"],
                "forms": ["relevant form IDs if mentioned"],
                "operations": ["relevant operations semantically"]
              }}
            - search_focus: {{
                "priority_sections": ["specific sections semantically relevant"],
                "exclude_sections": ["sections semantically irrelevant"],
                "context_requirements": ["semantic context needed"]
              }}
            """

            response = await self.openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{
                    "role": "system",
                    "content": """You are a semantic intelligence system for Acumatica documentation. Your core capability is UNDERSTANDING MEANING, not matching keywords.

YOUR SEMANTIC INTELLIGENCE:

1. **SEMANTIC UNDERSTANDING**:
   - Understand queries at a conceptual level, not keyword level
   - "Mobile App customization" semantically means: mobile application development, mobile screen modification, mobile field customization
   - "Prevent unapproval" semantically means: validation logic, business rules, custom validation, preventing state changes
   - Match documents based on MEANING ALIGNMENT, not word overlap

2. **CONTEXTUAL INFERENCE**:
   - Infer what type of documentation is needed from query context
   - If query involves modifying/restricting/changing behavior → semantically an implementation task
   - If query mentions a specific platform (Mobile App, Web, API) → semantically match platform-specific guides
   - Understand relationships: "Work Order in Mobile App" = mobile application customization for work orders

3. **INTELLIGENT DOCUMENT MATCHING**:
   - Read document summaries semantically, understand their scope and meaning
   - Match based on conceptual relevance, not keyword frequency
   - Most specific semantic match gets highest score (e.g., Mobile App guide for mobile queries)
   - General guides score lower unless query is truly general

4. **SEMANTIC SCORING**:
   - Score reflects semantic relevance: How well does document's meaning match query's meaning?
   - 0.9-1.0: Perfect semantic match (e.g., Mobile App query → Mobile App guide)
   - 0.7-0.9: Strong semantic match (e.g., customization query → customization guide)
   - 0.5-0.7: Relevant match
   - <0.5: Weak semantic match

5. **NO KEYWORD DEPENDENCY**:
   - Do NOT rely on specific keywords
   - Understand meaning even if words differ
   - "Mobile App" = "mobile application" = "mobile screens" = semantically the same
   - Infer intent from context, not keyword presence

6. **DOMAIN KNOWLEDGE UTILIZATION**:
   - Use document summaries, core_topics, key_operations to understand semantic scope
   - Match queries to documents based on conceptual alignment
   - Understand that "mobile app customization" semantically aligns with mobile-specific guides

7. **ACCURACY FOR ANY QUERY QUALITY**:
   - Infer semantic meaning even from vague queries
   - Understand implicit requirements from context
   - Provide intelligent reasoning for document selection

Your goal: Match queries to documents based on TRUE SEMANTIC RELEVANCE, demonstrating deep understanding of both query intent and document content."""
                }, {
                    "role": "user",
                    "content": prompt
                }],
                response_format={ "type": "json_object" }
            )

            analysis = json.loads(response.choices[0].message.content)
            
            # Cache the result (with size limit)
            if len(self._intent_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._intent_cache))
                del self._intent_cache[oldest_key]
            
            self._intent_cache[cache_key] = analysis
            
            self.logger.info("Query analysis complete", extra={
                "relevant_docs": analysis.get("relevant_docs", []),
                "query_intent": analysis.get("query_intent", "unknown"),
                "cache_size": len(self._intent_cache)
            })
            
            return analysis
            
        except Exception as e:
            self.logger.error("Query analysis failed", extra={
                "error": str(e),
                "query": query
            })
            # Return safe default that won't restrict search
            return {
                "relevant_docs": [],
                "query_intent": {"primary": "unknown", "secondary": [], "technical_level": "intermediate"},
                "key_concepts": {"primary": [], "related": [], "forms": [], "operations": []},
                "search_focus": {"priority_sections": [], "exclude_sections": [], "context_requirements": []}
            }
            
    def _trust_llm_semantic_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Trust LLM's semantic analysis for customization detection.
        No static keywords - rely purely on LLM intelligence.
        """
        query_intent = analysis.get("query_intent", {})
        if isinstance(query_intent, dict):
            # Trust LLM's semantic understanding
            return query_intent.get("is_customization_task", False)
        return False
    
    def _normalize_document_id(self, doc_id: str) -> str:
        """Normalize document ID for matching (handles spaces, parentheses, etc.)"""
        # Remove extra spaces and normalize
        normalized = doc_id.strip()
        # Keep as-is but prepare for fuzzy matching
        return normalized
    
    def _fuzzy_match_document_id(self, target_id: str, available_ids: List[str]) -> Optional[str]:
        """Fuzzy match document ID against available IDs"""
        target_normalized = self._normalize_document_id(target_id).lower()
        
        # Exact match first
        for doc_id in available_ids:
            if self._normalize_document_id(doc_id).lower() == target_normalized:
                return doc_id
        
        # Partial match (contains)
        for doc_id in available_ids:
            doc_normalized = self._normalize_document_id(doc_id).lower()
            if target_normalized in doc_normalized or doc_normalized in target_normalized:
                return doc_id
        
        # Word-based match (handles "AcumaticaERP_CustomizationGuide (1)" vs variations)
        target_words = set(target_normalized.replace("(", "").replace(")", "").split())
        for doc_id in available_ids:
            doc_normalized = self._normalize_document_id(doc_id).lower()
            doc_words = set(doc_normalized.replace("(", "").replace(")", "").split())
            # If most words match, consider it a match
            if len(target_words & doc_words) >= max(2, len(target_words) * 0.7):
                return doc_id
        
        return None
    
    def get_search_parameters(self, analysis: Dict[str, Any], original_query: str = "") -> Dict[str, Any]:
        """Convert intent analysis to search parameters with enhanced precision and fallback logic"""
        try:
            # Get available document IDs from domain knowledge for matching
            available_doc_ids = list(self.domain_knowledge.get("document_summaries", {}).keys())
            
            # Extract relevant docs with scores
            relevant_docs = analysis.get("relevant_docs", [])
            target_directories = []
            
            # Trust LLM's semantic analysis - no keyword fallback
            # The LLM understands meaning, not keywords
            is_customization = self._trust_llm_semantic_analysis(analysis)
            
            if isinstance(relevant_docs, list) and relevant_docs and isinstance(relevant_docs[0], dict):
                # New format with scores - use fuzzy matching for document IDs
                for doc in relevant_docs:
                    doc_id = doc["id"]
                    score = doc.get("score", 0.0)
                    
                    # Use fuzzy matching to find actual document ID
                    matched_id = self._fuzzy_match_document_id(doc_id, available_doc_ids)
                    if matched_id:
                        target_directories.append({
                            "path": os.path.join(config.LOCAL_BASE_PATH, matched_id),
                            "score": score,
                            "reasoning": doc.get("reasoning", "")
                        })
                    else:
                        self.logger.warning("Document ID not found, attempting direct match", extra={
                            "requested_id": doc_id,
                            "available_ids": available_doc_ids[:5]
                        })
                        # Try direct path match as fallback
                        target_directories.append({
                            "path": os.path.join(config.LOCAL_BASE_PATH, doc_id),
                            "score": score,
                            "reasoning": doc.get("reasoning", "")
                        })
            else:
                # Legacy format fallback with fuzzy matching
                for doc_id in relevant_docs:
                    matched_id = self._fuzzy_match_document_id(doc_id, available_doc_ids)
                    if matched_id:
                        target_directories.append({
                            "path": os.path.join(config.LOCAL_BASE_PATH, matched_id),
                            "score": 1.0,
                            "reasoning": ""
                        })
            
            # SEMANTIC FALLBACK: Trust LLM's semantic understanding
            # If LLM detected customization task but didn't include relevant docs, 
            # it means LLM determined they're not semantically relevant
            # We trust the LLM's semantic intelligence - no forced additions
            
            # However, if NO documents were selected (empty list), that's a problem
            # In that case, we should include general customization guides as semantic fallback
            if not target_directories and is_customization:
                self.logger.warning("No documents selected by LLM for customization task, adding semantic fallback", extra={
                    "query_preview": original_query[:100],
                    "note": "LLM should have selected documents - this may indicate prompt issue"
                })
                # Only add if LLM completely failed to select anything
                customization_doc_ids = [
                    "AcumaticaERP_CustomizationGuide (1)",
                    "S200_SysAdmAdvanced_2017R2"
                ]
                
                for cust_doc_id in customization_doc_ids:
                    matched_id = self._fuzzy_match_document_id(cust_doc_id, available_doc_ids)
                    if matched_id:
                        target_directories.append({
                            "path": os.path.join(config.LOCAL_BASE_PATH, matched_id),
                            "score": 0.6,  # Moderate score - LLM didn't select it, so lower confidence
                            "reasoning": "Semantic fallback: LLM selected no documents for customization task"
                        })
            
            # Filter by minimum score (lowered threshold to 0.3 for better recall)
            target_directories = [
                d for d in target_directories 
                if d.get("score", 0.0) >= 0.3
            ]

            # Extract enhanced query intent
            query_intent = analysis.get("query_intent", {})
            if isinstance(query_intent, dict):
                intent_info = {
                    "primary": query_intent.get("primary", "unknown"),
                    "secondary": query_intent.get("secondary", []),
                    "technical_level": query_intent.get("technical_level", "intermediate")
                }
            else:
                intent_info = {
                    "primary": query_intent if isinstance(query_intent, str) else "unknown",
                    "secondary": [],
                    "technical_level": "intermediate"
                }

            # Extract enhanced key concepts
            key_concepts = analysis.get("key_concepts", {})
            if isinstance(key_concepts, dict):
                concepts_info = {
                    "primary": key_concepts.get("primary", []),
                    "related": key_concepts.get("related", []),
                    "forms": key_concepts.get("forms", []),
                    "operations": key_concepts.get("operations", [])
                }
            else:
                concepts_info = {
                    "primary": key_concepts if isinstance(key_concepts, list) else [],
                    "related": [],
                    "forms": [],
                    "operations": []
                }

            # Extract enhanced search focus
            search_focus = analysis.get("search_focus", {})
            if isinstance(search_focus, dict):
                focus_info = {
                    "priority_sections": search_focus.get("priority_sections", []),
                    "exclude_sections": search_focus.get("exclude_sections", []),
                    "context_requirements": search_focus.get("context_requirements", [])
                }
            else:
                focus_info = {
                    "priority_sections": [],
                    "exclude_sections": [],
                    "context_requirements": []
                }

            search_params = {
                "target_directories": target_directories,
                "query_intent": intent_info,
                "key_concepts": concepts_info,
                "search_focus": focus_info
            }
            
            self.logger.debug("Enhanced search parameters prepared", extra={
                "num_target_dirs": len(target_directories),
                "primary_intent": intent_info["primary"],
                "technical_level": intent_info["technical_level"],
                "num_primary_concepts": len(concepts_info["primary"]),
                "num_forms": len(concepts_info["forms"])
            })
            
            return search_params
            
        except Exception as e:
            self.logger.error("Failed to prepare enhanced search parameters", extra={
                "error": str(e),
                "analysis": analysis
            })
            return {
                "target_directories": [],
                "query_intent": {"primary": "unknown", "secondary": [], "technical_level": "intermediate"},
                "key_concepts": {"primary": [], "related": [], "forms": [], "operations": []},
                "search_focus": {"priority_sections": [], "exclude_sections": [], "context_requirements": []}
            }


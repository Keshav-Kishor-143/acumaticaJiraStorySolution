#!/usr/bin/env python3
"""
Technical Entity Extraction Module

Extracts and validates Acumatica technical entities from retrieved content:
- Form IDs, DAC classes, Graph classes, DAC fields
- Event handlers, Navigation paths, PX Attributes
- Table names, Vision metadata

Provides confidence scoring and negative retrieval validation to eliminate hallucinations.
"""

import re
import json
from typing import List, Dict, Any, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.acumatica_patterns import AcumaticaPatterns
from openai import OpenAI


@dataclass
class TechnicalEntity:
    """Represents a single technical entity with metadata"""
    entity_type: str  # 'form', 'dac', 'graph', 'field', 'event', 'navigation', 'px_attribute', 'table'
    value: str  # The actual entity value (e.g., "SO301000", "ARInvoice")
    occurrences: int = 1  # Number of times found
    sources: List[str] = field(default_factory=list)  # Source documents/pages
    confidence: float = 0.0  # Confidence score
    vision_confirmed: bool = False  # Whether confirmed by Vision API
    retriever_score: float = 0.0  # Score from retriever


class TechnicalEntityExtractor:
    """Extracts and validates Acumatica technical entities from text and vision content"""
    
    def __init__(self):
        self.logger = get_logger("TECH_EXTRACTION")
        self.patterns_resolver = AcumaticaPatterns()
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Regex patterns for entity extraction
        self.patterns = {
            'form_id': re.compile(r'\b[A-Z]{2}\d{6}\b'),
            'dac_class': re.compile(r'class\s+(\w+)\s*:\s*PX\.Data\.IBqlTable'),
            'graph_class': re.compile(r'class\s+(\w+)\s*:\s*PXGraph'),
            'dac_field': re.compile(r'\[PXDB.*?\][\s\S]*?public\s+[\w\<\>]+\s+(\w+)\s*{'),
            'event_handler': re.compile(
                r'protected\s+void\s+(\w+?)_(?:Field|Row)\w+?\(.*?\)',
                re.MULTILINE | re.DOTALL
            ),
            'px_attribute': re.compile(r'\[PX\w+?\([^)]*\)\]'),
            'navigation_keywords': [
                r'Navigate\s+to[:\s]+([^\n]+)',
                r'Go\s+to[:\s]+([^\n]+)',
                r'Menu[:\s]+([^\n]+)',
                r'Path[:\s]+([^\n]+)',
                r'Screen[:\s]+([^\n]+)',
                r'Workspace[:\s]+([^\n]+)',
            ],
            'table_name': re.compile(r'Table\s+Name[:\s]+(\w+)', re.IGNORECASE),
        }
        
        # Navigation path pattern compilation
        self.nav_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns['navigation_keywords']]

        # DLL KB text format patterns (from KB `metadata.json` entries with `section_type: dll_text`)
        self._dll_class_re = re.compile(r"\*\*Class:\s*(?P<class>.+?)\*\*", re.IGNORECASE)
        self._dll_full_re = re.compile(r"Full Name:\s*`(?P<full>[^`]+)`", re.IGNORECASE)
        self._dll_inherits_re = re.compile(r"Inherits from:\s*`(?P<inh>[^`]+)`", re.IGNORECASE)
        
        # Entity storage
        self._extracted_entities: Dict[str, List[TechnicalEntity]] = defaultdict(list)
    
    def extract_from_natural_language(
        self,
        story_text: str,
        acceptance_criteria: List[str],
        source: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract technical entities from natural language story text
        
        Law 1 Compliance: Extract Before Understanding
        - First extracts ALL possible entities via pure patterns
        - Then uses LLM only for disambiguation and additional extraction
        - This prevents hallucination and ensures surface-level grounding
        
        Args:
            story_text: Story description text
            acceptance_criteria: List of acceptance criteria texts
            source: Source identifier
            
        Returns:
            Dictionary of entity types to lists of extracted values
        """
        try:
            # Combine story text and AC
            combined_text = story_text
            if acceptance_criteria:
                combined_text += "\n\nAcceptance Criteria:\n" + "\n".join(acceptance_criteria)
            
            # LAW 1 COMPLIANCE: Extract ALL possible entities via patterns FIRST
            # This ensures we extract everything before any "understanding"
            pattern_entities = self.extract_from_text(combined_text, source=f"{source}_pattern", retriever_score=0.9)
            
            # Extract conditions, errors, constraints via patterns
            pattern_entities.update(self._extract_conditions_and_errors(combined_text))
            
            self.logger.info("Pattern-based extraction completed (Law 1)", extra={
                "forms_found": len(pattern_entities.get('forms', [])),
                "dacs_found": len(pattern_entities.get('dacs', [])),
                "fields_found": len(pattern_entities.get('fields', []))
            })
            
            # Build extraction prompt
            extraction_prompt = f"""Extract all Acumatica technical entities from the following story text.

Story Text:
{combined_text}

Extract the following entity types:
1. **Document Types**: Event Order, Sales Order, Purchase Order, Quote, Invoice, Customer, etc.
2. **Form IDs**: Screen identifiers like EV301000, SO301000, CR306015 (format: [A-Z]{{2}}\\d{{6}})
3. **DAC Names**: Data Access Classes like EVOrder, SOOrder, Customer (format: PascalCase)
4. **Graph Names**: Business Logic classes like EVOrderEntry, SOOrderEntry, QuoteMaint (format: PascalCase ending in Entry/Maint/Setup)
5. **Field Names**: Field identifiers mentioned like EventStartDate, ScheduledStartDate, EventEndDate, ScheduledReturnDate
6. **Event Handlers**: Event types needed like FieldVerifying, RowPersisting, FieldUpdated
7. **Navigation Paths**: Menu paths or screen navigation mentioned

Return ONLY a JSON object with this structure:
{{
    "document_types": ["Event Order", "EX Order", "EX Quote"],
    "forms": ["EV301000", "SO301000"],
    "dacs": ["EVOrder", "SOOrder"],
    "graphs": ["EVOrderEntry", "SOOrderEntry", "QuoteMaint"],
    "fields": ["EventStartDate", "EventEndDate", "ScheduledStartDate", "ScheduledReturnDate"],
    "events": ["FieldVerifying", "RowPersisting"],
    "navigation_paths": []
}}

Rules:
- Extract entities mentioned explicitly or strongly implied
- Use exact technical names (e.g., "EVOrder" not "event order")
- For document types, include both business names and technical names if mentioned
- For fields, extract exact field names as they would appear in code
- If an entity is not mentioned, use empty array []
- Do NOT invent or guess entities not present in the text"""

            # Call LLM for extraction
            response = self.openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Acumatica developer. Extract technical entities from business language accurately. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                max_completion_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            llm_entities = json.loads(response.choices[0].message.content)
            
            # Resolve aliases using AcumaticaPatterns
            resolved_entities = {
                'forms': [],
                'dacs': [],
                'graphs': [],
                'fields': [],
                'events': [],
                'navigation_paths': [],
                'px_attributes': [],
                'tables': []
            }
            
            # Resolve document types to DACs, Forms, and Graphs
            document_types = llm_entities.get('document_types', [])
            for doc_type in document_types:
                resolved_dac = self.patterns_resolver.resolve_dac_alias(doc_type)
                if resolved_dac:
                    if resolved_dac not in resolved_entities['dacs']:
                        resolved_entities['dacs'].append(resolved_dac)
                    # Add related forms and graphs
                    forms = self.patterns_resolver.get_forms_for_dac(resolved_dac)
                    graphs = self.patterns_resolver.get_graphs_for_dac(resolved_dac)
                    resolved_entities['forms'].extend([f for f in forms if f not in resolved_entities['forms']])
                    resolved_entities['graphs'].extend([g for g in graphs if g not in resolved_entities['graphs']])
            
            # Add directly extracted forms
            for form_id in llm_entities.get('forms', []):
                normalized = self.patterns_resolver.resolve_form_id(form_id)
                if normalized and normalized not in resolved_entities['forms']:
                    resolved_entities['forms'].append(normalized)
            
            # Add directly extracted DACs
            for dac in llm_entities.get('dacs', []):
                resolved = self.patterns_resolver.resolve_dac_alias(dac) or dac
                if resolved not in resolved_entities['dacs']:
                    resolved_entities['dacs'].append(resolved)
                # Expand with forms and graphs
                forms = self.patterns_resolver.get_forms_for_dac(resolved)
                graphs = self.patterns_resolver.get_graphs_for_dac(resolved)
                resolved_entities['forms'].extend([f for f in forms if f not in resolved_entities['forms']])
                resolved_entities['graphs'].extend([g for g in graphs if g not in resolved_entities['graphs']])
            
            # Add directly extracted graphs
            for graph in llm_entities.get('graphs', []):
                if graph not in resolved_entities['graphs']:
                    resolved_entities['graphs'].append(graph)
            
            # Add fields (no resolution needed, use as-is)
            resolved_entities['fields'] = list(set(llm_entities.get('fields', [])))
            
            # Add events
            resolved_entities['events'] = list(set(llm_entities.get('events', [])))
            
            # Add navigation paths
            resolved_entities['navigation_paths'] = list(set(llm_entities.get('navigation_paths', [])))
            
            # LAW 1 COMPLIANCE: Merge pattern-extracted entities with LLM-extracted entities
            # Pattern entities take precedence (higher confidence)
            for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events', 'navigation_paths']:
                pattern_values = pattern_entities.get(entity_type, [])
                llm_values = resolved_entities.get(entity_type, [])
                
                # Combine, with pattern values first (they're more reliable)
                merged_values = list(dict.fromkeys(pattern_values + llm_values))
                resolved_entities[entity_type] = merged_values
            
            # Store entities with confidence based on source (Law 1 compliance)
            if source:
                for entity_type, values in resolved_entities.items():
                    for value in values:
                        if value:
                            # Higher confidence for pattern-extracted entities (Law 1 compliance)
                            is_pattern_extracted = value in pattern_entities.get(entity_type, [])
                            retriever_score = 0.95 if is_pattern_extracted else 0.85
                            confidence = 0.90 if is_pattern_extracted else 0.85
                            
                            entity = TechnicalEntity(
                                entity_type=entity_type,
                                value=value,
                                occurrences=1,
                                sources=[source],
                                retriever_score=retriever_score,
                                confidence=confidence
                            )
                            self._extracted_entities[entity_type].append(entity)
            
            self.logger.info("Natural language entity extraction completed (Law 1 compliant)", extra={
                "forms": len(resolved_entities['forms']),
                "dacs": len(resolved_entities['dacs']),
                "graphs": len(resolved_entities['graphs']),
                "fields": len(resolved_entities['fields']),
                "events": len(resolved_entities['events']),
                "pattern_extracted": sum(len(pattern_entities.get(et, [])) for et in ['forms', 'dacs', 'graphs', 'fields'])
            })
            
            return resolved_entities
            
        except Exception as e:
            self.logger.error("Natural language entity extraction failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Fallback to regex-only extraction
            return self.extract_from_text(combined_text, source=source, retriever_score=0.0)
    
    def extract_from_text(
        self,
        text: str,
        source: Optional[str] = None,
        retriever_score: float = 0.0
    ) -> Dict[str, List[str]]:
        """
        Extract all technical entities from text content
        
        Args:
            text: Text content to extract from
            source: Source identifier (document/page)
            retriever_score: Score from retriever (0.0-1.0)
            
        Returns:
            Dictionary of entity types to lists of extracted values
        """
        if not text:
            return {}
        
        entities = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': [],
            'navigation_paths': [],
            'px_attributes': [],
            'tables': []
        }
        
        # Extract Form IDs
        form_ids = self.patterns['form_id'].findall(text)
        entities['forms'] = list(set(form_ids))  # Deduplicate

        # Extract DLL-derived entities (graphs/dacs/bql fields) when text comes from KB DLL dumps
        dll_entities = self._extract_from_dll_text(text)
        for k, vals in dll_entities.items():
            for v in vals:
                if v and v not in entities.get(k, []):
                    entities.setdefault(k, []).append(v)
        
        # Extract DAC classes
        dac_matches = self.patterns['dac_class'].findall(text)
        entities['dacs'] = list(set(dac_matches))
        
        # Extract Graph classes
        graph_matches = self.patterns['graph_class'].findall(text)
        entities['graphs'] = list(set(graph_matches))
        
        # Extract DAC fields
        field_matches = self.patterns['dac_field'].findall(text)
        entities['fields'] = list(set(field_matches))
        
        # Extract Event handlers
        event_matches = self.patterns['event_handler'].findall(text)
        entities['events'] = list(set(event_matches))
        
        # Extract PX Attributes
        px_matches = self.patterns['px_attribute'].findall(text)
        entities['px_attributes'] = list(set(px_matches))
        
        # Extract Navigation paths
        nav_paths = []
        for pattern in self.nav_patterns:
            matches = pattern.findall(text)
            nav_paths.extend([match.strip() for match in matches if match.strip()])
        entities['navigation_paths'] = list(set(nav_paths))
        
        # Extract Table names
        table_matches = self.patterns['table_name'].findall(text)
        entities['tables'] = list(set(table_matches))

        # Store entities with metadata
        if source:
            for entity_type, values in entities.items():
                for value in values:
                    if value:  # Only store non-empty values
                        entity = TechnicalEntity(
                            entity_type=entity_type,
                            value=value,
                            occurrences=1,
                            sources=[source],
                            retriever_score=retriever_score
                        )
                        self._extracted_entities[entity_type].append(entity)

        # Return only non-empty keys (keeps pipeline cleaner)
        return {k: v for k, v in entities.items() if v}

    def _extract_from_dll_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from KB DLL metadata text blocks.

        Example block:
        **Class: NVRTCycleBillProcessing**
        - Full Name: `NV.Rental360.RentalCycle.NVRTCycleBillProcessing`
        - Inherits from: `PX.Data.PXGraph`1[[...]]`
        """
        entities: Dict[str, List[str]] = {}
        if "Full Name:" not in text or "**Class:" not in text:
            return entities

        full_m = self._dll_full_re.search(text)
        inh_m = self._dll_inherits_re.search(text)
        if not full_m:
            return entities

        full_name = full_m.group("full").strip()
        inherits = (inh_m.group("inh").strip() if inh_m else "")

        # Prefer fully-qualified names to avoid ambiguity across DLLs.
        if "PX.Data.PXGraphExtension" in inherits or "PX.Data.PXGraph" in inherits:
            entities.setdefault("graphs", []).append(full_name)
        elif "PX.Data.PXBqlTable" in inherits:
            entities.setdefault("dacs", []).append(full_name)
        elif "BqlType" in inherits and "+Field" in inherits:
            entities.setdefault("fields", []).append(full_name)

        return entities
    
    def extract_from_vision_metadata(
        self,
        vision_text: str,
        source: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extract technical entities from Vision API metadata extraction
        
        Args:
            vision_text: Text extracted from Vision API
            source: Source identifier
            
        Returns:
            Dictionary of entity types to lists of extracted values
        """
        if not vision_text:
            return {}
        
        entities = self.extract_from_text(vision_text, source=source, retriever_score=0.0)
        
        # Mark vision-confirmed entities
        for entity_type, entity_list in self._extracted_entities.items():
            for entity in entity_list:
                if source in entity.sources:
                    entity.vision_confirmed = True
        
        return entities
    
    def merge_entities(
        self,
        new_entities: Dict[str, List[TechnicalEntity]]
    ) -> None:
        """
        Merge new entities into existing collection, updating occurrences and sources
        
        Args:
            new_entities: Dictionary of entity types to lists of TechnicalEntity objects
        """
        for entity_type, entity_list in new_entities.items():
            for new_entity in entity_list:
                # Check if entity already exists
                existing = None
                for existing_entity in self._extracted_entities[entity_type]:
                    if existing_entity.value.lower() == new_entity.value.lower():
                        existing = existing_entity
                        break
                
                if existing:
                    # Update existing entity
                    existing.occurrences += new_entity.occurrences
                    existing.sources.extend(new_entity.sources)
                    existing.sources = list(set(existing.sources))  # Deduplicate sources
                    existing.retriever_score = max(existing.retriever_score, new_entity.retriever_score)
                    if new_entity.vision_confirmed:
                        existing.vision_confirmed = True
                else:
                    # Add new entity
                    self._extracted_entities[entity_type].append(new_entity)
    
    def calculate_confidence_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence scores for all extracted entities
        
        Confidence formula:
        confidence = (occurrence_count * 0.5) + (retriever_score * 0.3) + (vision_hits * 0.2)
        
        Returns:
            Dictionary mapping entity_type -> entity_value -> confidence_score
        """
        confidence_scores = {}
        
        for entity_type, entity_list in self._extracted_entities.items():
            confidence_scores[entity_type] = {}
            
            for entity in entity_list:
                # Normalize occurrence count (max 10 occurrences = 1.0)
                occurrence_score = min(entity.occurrences / 10.0, 1.0) * 0.5
                
                # Retriever score (already 0.0-1.0)
                retriever_score = entity.retriever_score * 0.3
                
                # Vision confirmation (binary: 0 or 0.2)
                vision_score = 0.2 if entity.vision_confirmed else 0.0
                
                # Calculate final confidence
                confidence = occurrence_score + retriever_score + vision_score
                entity.confidence = confidence
                confidence_scores[entity_type][entity.value] = confidence
        
        return confidence_scores
    
    def get_confirmed_entities(self, threshold: float = 0.75) -> Dict[str, List[str]]:
        """
        Get entities with confidence >= threshold
        
        Args:
            threshold: Minimum confidence score (default 0.75)
            
        Returns:
            Dictionary of entity types to lists of confirmed entity values
        """
        self.calculate_confidence_scores()
        
        confirmed = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': [],
            'navigation_paths': [],
            'px_attributes': [],
            'tables': []
        }
        
        for entity_type, entity_list in self._extracted_entities.items():
            for entity in entity_list:
                if entity.confidence >= threshold:
                    confirmed[entity_type].append(entity.value)
        
        return confirmed
    
    def get_unconfirmed_entities(self, threshold: float = 0.40) -> Dict[str, List[str]]:
        """
        Get entities with confidence < threshold
        
        Args:
            threshold: Maximum confidence score for unconfirmed (default 0.40)
            
        Returns:
            Dictionary of entity types to lists of unconfirmed entity values
        """
        self.calculate_confidence_scores()
        
        unconfirmed = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': [],
            'navigation_paths': [],
            'px_attributes': [],
            'tables': []
        }
        
        for entity_type, entity_list in self._extracted_entities.items():
            for entity in entity_list:
                if entity.confidence < threshold:
                    unconfirmed[entity_type].append(entity.value)
        
        return unconfirmed
    
    def get_all_entities(self) -> Dict[str, Any]:
        """
        Get all extracted entities with full metadata
        
        Returns:
            Dictionary with entity types, values, and confidence scores
        """
        self.calculate_confidence_scores()
        
        result = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': [],
            'navigation_paths': [],
            'px_attributes': [],
            'tables': [],
            'vision_metadata': [],
            'confidence_scores': {}
        }
        
        # Build entity lists
        for entity_type, entity_list in self._extracted_entities.items():
            result[entity_type] = [entity.value for entity in entity_list]
        
        # Build confidence scores dictionary
        confidence_dict = {}
        for entity_type, entity_list in self._extracted_entities.items():
            confidence_dict[entity_type] = {
                entity.value: entity.confidence
                for entity in entity_list
            }
        result['confidence_scores'] = confidence_dict
        
        return result
    
    def validate_entity_exists(
        self,
        entity_type: str,
        entity_value: str,
        search_texts: List[str]
    ) -> bool:
        """
        Validate that an entity exists in the provided search texts (negative retrieval)
        
        Args:
            entity_type: Type of entity ('form', 'dac', etc.)
            entity_value: Value to search for
            search_texts: List of text content to search in
            
        Returns:
            True if entity found in search texts, False otherwise
        """
        if not search_texts:
            return False
        
        # Normalize entity value for search
        search_value = entity_value.lower().strip()
        
        # Search in all provided texts
        for text in search_texts:
            if not text:
                continue
            
            text_lower = text.lower()
            
            # Exact match or word boundary match
            if search_value in text_lower:
                # Check for word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(search_value) + r'\b'
                if re.search(pattern, text_lower):
                    return True
        
        return False
    
    def validate_all_entities(
        self,
        search_texts: List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate all extracted entities against search texts
        
        Args:
            search_texts: List of text content to validate against
            
        Returns:
            Dictionary of entity_type -> list of validated entity values
        """
        validated = {
            'forms': [],
            'dacs': [],
            'graphs': [],
            'fields': [],
            'events': [],
            'navigation_paths': [],
            'px_attributes': [],
            'tables': []
        }
        
        for entity_type, entity_list in self._extracted_entities.items():
            for entity in entity_list:
                if self.validate_entity_exists(entity_type, entity.value, search_texts):
                    validated[entity_type].append(entity.value)
                else:
                    # Mark as unconfirmed if not found
                    entity.confidence = min(entity.confidence, 0.39)
        
        return validated
    
    def reset(self):
        """Reset all extracted entities"""
        self._extracted_entities = defaultdict(list)
    
    def extract(self, story: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reference-compatible extraction method (adapter pattern)
        
        Matches reference interface: TechExtractor.extract(story)
        
        Args:
            story: Story dictionary with 'description' and optionally 'acceptance_criteria'
            
        Returns:
            Dictionary with 'entities' (entity dict) and 'raw' (text)
        """
        try:
            description = story.get("description", "")
            ac_list = story.get("acceptance_criteria", [])
            
            # Normalize AC to list of strings
            if ac_list:
                normalized_ac = []
                for ac in ac_list:
                    if isinstance(ac, dict):
                        normalized_ac.append(ac.get('text', ''))
                        normalized_ac.extend(ac.get('subpoints', []))
                    else:
                        normalized_ac.append(str(ac))
            else:
                normalized_ac = []
            
            # Extract entities using natural language extraction
            entities = self.extract_from_natural_language(
                story_text=description,
                acceptance_criteria=normalized_ac,
                source="story_extract_adapter"
            )
            
            return {
                "entities": entities,
                "raw": description
            }
        except Exception as e:
            self.logger.error("Extract adapter failed", extra={"error": str(e)})
            return {"entities": {}, "raw": story.get("description", "")}
    
    def extract_from_search_results(
        self,
        search_results: List[Any],
        retriever_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from a list of search results
        
        Args:
            search_results: List of SearchResult objects or dictionaries
            retriever_scores: Optional dictionary mapping result IDs to scores
            
        Returns:
            Dictionary of all extracted entities
        """
        self.reset()
        
        for result in search_results:
            # Extract source identifier
            if hasattr(result, 'pdf_name') and hasattr(result, 'page_number'):
                source = f"{result.pdf_name}_page_{result.page_number}"
                text_content = getattr(result, 'text_content', '') or getattr(result, 'content', '')
                vision_text = getattr(result, 'vision_extracted_text', '')
                score = getattr(result, 'combined_score', 0.0) or getattr(result, 'score', 0.0)
            elif isinstance(result, dict):
                source = f"{result.get('pdf_name', 'unknown')}_page_{result.get('page_number', 0)}"
                text_content = result.get('text_content', '') or result.get('content', '')
                vision_text = result.get('vision_extracted_text', '')
                score = result.get('combined_score', 0.0) or result.get('score', 0.0)
            else:
                continue
            
            # Extract from text content
            if text_content:
                self.extract_from_text(text_content, source=source, retriever_score=score)
            
            # Extract from vision text
            if vision_text:
                self.extract_from_vision_metadata(vision_text, source=source)
        
        # Calculate confidence scores
        self.calculate_confidence_scores()
        
        return self.get_all_entities()


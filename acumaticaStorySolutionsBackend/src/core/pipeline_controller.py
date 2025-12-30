#!/usr/bin/env python3
"""
Pipeline Controller - Orchestrates the Complete Multi-Stage Processing Pipeline

This module ensures NO stage is skipped and enforces the strict processing order:
1. Normalize story input
2. Retrieve documents
3. Extract entities
4. Validate entities
5. Build fact table
6. Map AC to technical requirements
7. Generate final solution
"""

from typing import Dict, Any, List, Optional
import asyncio

from src.config.config import config
from src.utils.logger_utils import get_logger, TimedOperation
from src.core.story_processor import JIRAStoryProcessor
from src.core.tech_extraction import TechnicalEntityExtractor
from src.core.validation_scoring import EntityValidationScorer
from src.core.fact_table import FactTableGenerator
from src.core.ac_mapper import AcceptanceCriteriaMapper
from src.core.solution_generator import SolutionGenerator
from src.core.retrieval_anchors import EntityAnchoredRetriever
from src.core.hybrid_retriever import HybridRetriever
from src.core.infer import VDRInferencer
from src.core.story_classifier import StoryClassifier
from src.core.acumatica_patterns import AcumaticaPatterns
from src.core.retrieval_fallbacks import RetrievalFallbacks
from src.core.pipeline_metadata import PipelineMetadata
from src.core.solution_confidence import SolutionConfidenceScorer
from src.core.dll_validator import DLLEntityValidator
from src.core.domain_guide import DomainGuide
from src.core.dll_symbol_index import DllSymbolIndexBuilder
from src.core.semantic_object_index import SemanticObjectIndexBuilder


class PipelineController:
    """Orchestrates the complete multi-stage processing pipeline"""
    
    def __init__(self):
        self.logger = get_logger("PIPELINE_CONTROLLER")
        
        # Initialize all components
        self.story_processor = JIRAStoryProcessor()
        self.entity_extractor = TechnicalEntityExtractor()
        self.validation_scorer = EntityValidationScorer()
        self.fact_table_generator = FactTableGenerator()
        self.ac_mapper = AcceptanceCriteriaMapper()
        self.solution_generator = SolutionGenerator()
        
        # Initialize hybrid retriever for entity-anchored retrieval
        self.hybrid_retriever = HybridRetriever()
        self.entity_anchored_retriever = EntityAnchoredRetriever(self.hybrid_retriever)
        self.inferencer = VDRInferencer()
        
        # Initialize new modules
        self.story_classifier = StoryClassifier()
        self.acumatica_patterns = AcumaticaPatterns()
        self.retrieval_fallbacks = RetrievalFallbacks()
        self.pipeline_metadata = PipelineMetadata()
        self.confidence_scorer = SolutionConfidenceScorer()
        self.dll_validator = DLLEntityValidator()  # For Law 2 compliance: DLL validation
        self.domain_guide = DomainGuide()  # For domain-guided intent understanding
        self.dll_symbol_index_builder = DllSymbolIndexBuilder()
        
        self.logger.info("Pipeline Controller initialized")
    
    async def process_story(
        self,
        story_request: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a JIRA story through the complete multi-stage pipeline
        
        Args:
            story_request: Story request dictionary with:
                - description: Story description
                - acceptance_criteria: List of acceptance criteria
                - story_id: Optional story ID
                - title: Optional title
                - images: Optional list of images
                - normalized_story: Optional pre-normalized story
            request_id: Optional request ID for tracking
            
        Returns:
            Complete solution dictionary with:
                - normalized_story: Normalized story structure
                - technical_entities: Extracted entities
                - validated_entities: Validated entities with confidence scores
                - fact_table: Technical fact table
                - ac_mappings: AC to technical mappings
                - solution_narrative: Final solution narrative
                - sources: Source references
                - metadata: Processing metadata
        """
        try:
            # Reset metadata for new processing
            self.pipeline_metadata.reset()
            
            with TimedOperation("complete_pipeline", self.logger, request_id=request_id):
                self.logger.info("Starting complete pipeline processing", extra={
                    "request_id": request_id,
                    "story_id": story_request.get('story_id')
                })
                
                # STAGE 1: Parse & Normalize Input
                normalized_story = await self._stage1_normalize_input(story_request)
                self.pipeline_metadata.record_stage_completion("normalize_input")
                
                # Classify story type
                has_images = bool(story_request.get('images'))
                classification = self.story_classifier.classify_story(
                    description=normalized_story.get('description', ''),
                    acceptance_criteria=normalized_story.get('acceptance_criteria', []),
                    has_images=has_images
                )
                self.pipeline_metadata.record_story_type(
                    classification['story_type'],
                    classification['confidence']
                )
                
                # STAGE 2: Internal Entity Extraction (from story text)
                story_entities = await self._stage2_extract_story_entities(normalized_story)
                self.pipeline_metadata.record_entities_extracted(story_entities)
                self.pipeline_metadata.record_stage_completion("extract_story_entities")
                
                # STAGE 3: Retrieve Required Documentation (BEFORE AC mapping)
                retrieval_result = await self._stage3_retrieve_documentation(
                    normalized_story,
                    story_entities,
                    request_id
                )
                self.pipeline_metadata.record_stage_completion("retrieve_documentation")
                
                # STAGE 4: Extract Technical Entities from Retrieved Content
                technical_entities = await self._stage4_extract_technical_entities(
                    retrieval_result,
                    story_entities
                )
                self.pipeline_metadata.record_stage_completion("extract_technical_entities")
                
                # STAGE 5: Entity Validation & Confidence Scoring (with DLL validation)
                validation_result = await self._stage5_validate_entities(
                    technical_entities,
                    retrieval_result
                )
                
                # Law 2 Compliance: DLL Validation - Validate entities against actual DLL symbols
                dll_validation_results = self.dll_validator.validate_entities(technical_entities)
                self.logger.info("DLL validation completed", extra={
                    "dacs_validated": len(dll_validation_results.get('dacs', {})),
                    "graphs_validated": len(dll_validation_results.get('graphs', {})),
                    "fields_validated": len(dll_validation_results.get('fields', {}))
                })
                self.pipeline_metadata.record_entities_confirmed(
                    validation_result.get("validated_entities", {}),
                    validation_result.get("confidence_scores", {})
                )
                self.pipeline_metadata.record_stage_completion("validate_entities")
                
                # STAGE 6: Build Fact Table
                fact_table = await self._stage6_build_fact_table(
                    validation_result,
                    retrieval_result
                )
                # Attach semantic object hit (if any) so downstream stages can anchor on it deterministically
                sem_hit = self.pipeline_metadata.metadata.get("semantic_object_hit")
                if isinstance(sem_hit, dict) and sem_hit:
                    fact_table["semantic_object_hit"] = sem_hit
                self.pipeline_metadata.record_fact_table(fact_table)
                self.pipeline_metadata.record_stage_completion("build_fact_table")
                
                # STAGE 7: Map Acceptance Criteria to Technical Requirements
                ac_mappings = await self._stage7_map_acceptance_criteria(
                    normalized_story,
                    fact_table
                )
                total_ac = len(normalized_story.get('acceptance_criteria', []))
                mapped_ac = len([m for m in ac_mappings if m.get('related_forms') or m.get('related_dacs')])
                self.pipeline_metadata.record_ac_mapping(total_ac, mapped_ac)
                self.pipeline_metadata.record_stage_completion("map_acceptance_criteria")
                
                # STAGE 8: Generate Final Solution
                solution_narrative = await self._stage8_generate_solution(
                    normalized_story,
                    fact_table,
                    ac_mappings,
                    retrieval_result,
                    classification
                )
                
                # Calculate final confidence score
                confidence_result = self.confidence_scorer.calculate_confidence(
                    ac_mappings=ac_mappings,
                    validated_entities=validation_result,
                    fact_table=fact_table,
                    retrieval_metadata=self.pipeline_metadata.get_metadata(),
                    total_ac_count=total_ac
                )
                self.pipeline_metadata.record_final_confidence(confidence_result['confidence_score'])
                
                # Finalize metadata
                final_metadata = self.pipeline_metadata.finalize()
                
                # Compile final result
                result = {
                    "normalized_story": normalized_story,
                    "technical_entities": technical_entities,
                    "validated_entities": validation_result.get("validated_entities", {}),
                    "uncertain_entities": validation_result.get("uncertain_entities", {}),
                    "unconfirmed_entities": validation_result.get("unconfirmed_entities", {}),
                    "fact_table": fact_table,
                    "ac_mappings": ac_mappings,
                    "solution_narrative": solution_narrative,
                    "sources": retrieval_result.get("sources", []),
                    "retrieved_chunks": retrieval_result.get("retrieved_chunks", []),
                    "vision_analysis": retrieval_result.get("vision_analysis", []),
                    "story_classification": classification,
                    "confidence_score": confidence_result['confidence_score'],
                    "confidence_details": confidence_result,
                    "metadata": final_metadata
                }
                
                self.logger.info("Pipeline processing completed", extra={
                    "request_id": request_id,
                    "story_id": story_request.get('story_id'),
                    "entities_extracted": result["metadata"].get("entity_statistics", {}).get("total_extracted", 0),
                    "entities_confirmed": result["metadata"].get("entity_statistics", {}).get("total_confirmed", 0)
                })
                
                return result
                
        except Exception as e:
            self.logger.error("Pipeline processing failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "request_id": request_id,
                "story_id": story_request.get('story_id')
            })
            raise
    
    async def _stage1_normalize_input(self, story_request: Dict[str, Any]) -> Dict[str, Any]:
        """STAGE 1: Parse & Normalize Input"""
        try:
            # Check if already normalized
            if story_request.get('normalized_story'):
                self.logger.debug("Story already normalized, using provided structure")
                return story_request['normalized_story']
            
            # Normalize raw story text if provided
            raw_story_text = story_request.get('raw_story_text')
            if raw_story_text:
                normalized = self.story_processor.normalize_story_input(raw_story_text)
                return normalized
            
            # Build normalized structure from request fields
            normalized = {
                "story_title": story_request.get('title', ''),
                "description": story_request.get('description', ''),
                "requirements": story_request.get('requirements', []),
                "acceptance_criteria": []
            }
            
            # Normalize acceptance criteria
            ac_list = story_request.get('acceptance_criteria', [])
            for i, ac in enumerate(ac_list):
                if isinstance(ac, dict):
                    normalized["acceptance_criteria"].append(ac)
                else:
                    normalized["acceptance_criteria"].append({
                        "id": f"AC{i+1}",
                        "text": str(ac),
                        "subpoints": []
                    })
            
            return normalized
            
        except Exception as e:
            self.logger.error("Stage 1 (normalize) failed", extra={"error": str(e)})
            raise
    
    async def _stage2_extract_story_entities(self, normalized_story: Dict[str, Any]) -> Dict[str, List[str]]:
        """STAGE 2: Extract entities from story text using LLM + Pattern Resolution"""
        try:
            # Extract entities from story description and AC
            story_text = normalized_story.get('description', '')
            ac_texts = []
            for ac in normalized_story.get('acceptance_criteria', []):
                if isinstance(ac, dict):
                    ac_texts.append(ac.get('text', ''))
                    ac_texts.extend(ac.get('subpoints', []))
                else:
                    ac_texts.append(str(ac))
            
            # Use LLM-based natural language extraction with pattern resolution
            entities = self.entity_extractor.extract_from_natural_language(
                story_text=story_text,
                acceptance_criteria=ac_texts,
                source="story_text"
            )
            
            # Also do regex extraction as fallback/complement
            combined_text = story_text + ' ' + ' '.join(ac_texts)
            regex_entities = self.entity_extractor.extract_from_text(combined_text, source="story_text_regex", retriever_score=0.5)
            
            # Enrich entities using semantic object index (e.g., "EX Order" -> NAWEXOrderEntry)
            try:
                from src.core.semantic_object_index import SemanticObjectIndexBuilder
                sem_index = SemanticObjectIndexBuilder().load_or_build(force_rebuild=False)
                sem_hit = sem_index.resolve(
                    f"{story_text} {' '.join(ac_texts[:3])}",
                    field_names=entities.get("fields", [])[:5]
                )
                if sem_hit:
                    obj, score = sem_hit
                    if score >= 0.40:
                        # Add semantic object entities to extracted entities
                        if obj.dac_full_name and obj.dac_full_name not in entities.get("dacs", []):
                            entities.setdefault("dacs", []).append(obj.dac_full_name)
                        if obj.graph_full_name and obj.graph_full_name not in entities.get("graphs", []):
                            entities.setdefault("graphs", []).append(obj.graph_full_name)
                        if obj.line_dac_full_name and obj.line_dac_full_name not in entities.get("dacs", []):
                            entities.setdefault("dacs", []).append(obj.line_dac_full_name)
                        # Add screen ID if available
                        if hasattr(obj, 'primary_screen_id') and obj.primary_screen_id:
                            if obj.primary_screen_id not in entities.get("forms", []):
                                entities.setdefault("forms", []).append(obj.primary_screen_id)
                        self.logger.info("Enriched entities with semantic object", extra={
                            "concept": obj.concept,
                            "dac": obj.dac_full_name,
                            "graph": obj.graph_full_name,
                            "score": score
                        })
            except Exception as e:
                self.logger.warning("Semantic object enrichment failed", extra={"error": str(e)})
            
            # Merge both results (LLM takes precedence)
            for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events', 'navigation_paths']:
                if entity_type in regex_entities:
                    for value in regex_entities[entity_type]:
                        if value and value not in entities.get(entity_type, []):
                            entities.setdefault(entity_type, []).append(value)
            
            self.logger.info("Story entity extraction completed", extra={
                "forms": len(entities.get('forms', [])),
                "dacs": len(entities.get('dacs', [])),
                "graphs": len(entities.get('graphs', [])),
                "fields": len(entities.get('fields', []))
            })
            
            return entities
            
        except Exception as e:
            self.logger.error("Stage 2 (extract story entities) failed", extra={"error": str(e)})
            # Fallback to regex-only extraction
            try:
                combined_text = normalized_story.get('description', '') + ' ' + ' '.join([
                    ac.get('text', '') if isinstance(ac, dict) else str(ac)
                    for ac in normalized_story.get('acceptance_criteria', [])
                ])
                return self.entity_extractor.extract_from_text(combined_text, source="story_text_fallback")
            except:
                return {}
    
    async def _stage3_retrieve_documentation(
        self,
        normalized_story: Dict[str, Any],
        story_entities: Dict[str, List[str]],
        request_id: Optional[str]
    ) -> Dict[str, Any]:
        """STAGE 3: Retrieve Required Documentation"""
        try:
            # Build query from story
            description = normalized_story.get('description', '')
            ac_list = []
            for ac in normalized_story.get('acceptance_criteria', []):
                if isinstance(ac, dict):
                    ac_list.append(ac.get('text', ''))
                else:
                    ac_list.append(str(ac))
            
            # Build enhanced query with resolved entities
            base_query = f"{description} {' '.join(ac_list[:3])}"
            
            # STEP 1: Enhance query with domain-guided terms (intent layer)
            # This uses domain.json to understand the story domain BEFORE search
            enhanced_query = self.domain_guide.enhance_query_with_domain(
                base_query=base_query,
                story_text=description,
                acceptance_criteria=ac_list
            )
            
            # STEP 2: Identify relevant documents based on domain matching
            relevant_documents = self.domain_guide.identify_relevant_documents(
                story_text=description,
                acceptance_criteria=ac_list
            )
            
            if relevant_documents:
                self.logger.info("Domain-guided document prioritization", extra={
                    "top_documents": list(relevant_documents.keys())[:5],
                    "top_scores": list(relevant_documents.values())[:5]
                })
            
            # STEP 3: Enhance query with resolved technical entities
            entity_terms = []
            if story_entities:
                # Add DAC names
                for dac in story_entities.get('dacs', [])[:3]:
                    entity_terms.append(dac)
                # Add Form IDs
                for form_id in story_entities.get('forms', [])[:2]:
                    entity_terms.append(form_id)
                # Add Graph names
                for graph in story_entities.get('graphs', [])[:2]:
                    entity_terms.append(graph)
                # Add field names
                for field in story_entities.get('fields', [])[:3]:
                    entity_terms.append(field)

            # STEP 3b: Semantic object index enrichment (business -> Graph/DAC)
            # If we can resolve "Rental Return"/etc to a concrete object, add it as an anchor for retrieval.
            try:
                sem = SemanticObjectIndexBuilder().load_or_build(force_rebuild=False)
                sem_hit = sem.resolve(
                    f"{description} {' '.join(ac_list)}",
                    field_names=(story_entities.get("fields", []) if story_entities else []),
                )
                if sem_hit:
                    obj, score = sem_hit
                    # Conservative threshold: only use as an anchor when meaningfully confident
                    if score >= 0.40:
                        entity_terms.extend([obj.graph_full_name, obj.dac_full_name])
                        # Record for downstream (fact table / solution generator) without forcing a guess
                        self.pipeline_metadata.metadata["semantic_object_hit"] = {
                            "concept": obj.concept,
                            "phrases": getattr(obj, "phrases", []) or [],
                            "module_tag": obj.module_tag,
                            "graph_full_name": obj.graph_full_name,
                            "dac_full_name": obj.dac_full_name,
                            "line_dac_full_name": obj.line_dac_full_name,
                            "score": score,
                        }
            except Exception:
                pass
            
            if entity_terms:
                enhanced_query = f"{enhanced_query} {' '.join(entity_terms)}"
            
            # Perform initial retrieval using VDRInferencer
            self.pipeline_metadata.record_retrieval_attempt(True, "hybrid")

            # Build intent-directed doc shortlist:
            # - Start with domain.json suggestions
            # - Add custom DLLs for billing/customization keywords (DLL-first for custom billing)
            target_docs: List[str] = list(relevant_documents.keys())[:6] if relevant_documents else []
            combined_story = f"{description} {' '.join(ac_list)}".lower()
            custom_signals = any(k in combined_story for k in ["ex order", "extension", "e&c", "surcharge", "billing", "invoice"])
            rental_signals = any(k in combined_story for k in ["rental", "return", "equipment", "damage fee", "damaged", "return ticket"])
            if custom_signals:
                # Prefer your custom DLL KB folders when the story looks like custom billing logic.
                for dll_doc in ["NV.Rental360_DLL", "NAWUnitedSiteServices_DLL"]:
                    if dll_doc not in target_docs:
                        target_docs.insert(0, dll_doc)
                # Also keep core patterns/customization guide for scaffolding
                for core_doc in ["PX.Objects_DLL", "AcumaticaERP_CustomizationGuide (1)"]:
                    if core_doc not in target_docs:
                        target_docs.append(core_doc)

            if rental_signals:
                # Prioritize Rental360 + related custom DLL KB folders for rental/return stories.
                for dll_doc in ["NV.Rental360_DLL", "NAWUnitedSiteServices_DLL"]:
                    if dll_doc not in target_docs:
                        target_docs.insert(0, dll_doc)
                for core_doc in ["PX.Objects_DLL", "PX.Objects.FS_DLL", "AcumaticaERP_CustomizationGuide (1)"]:
                    if core_doc not in target_docs:
                        target_docs.append(core_doc)

            search_params = {"target_directories": target_docs, "search_focus": "dll_first_billing"} if target_docs else None

            retrieval_result = await self.inferencer.ask_question(
                question=enhanced_query,
                top_k=config.TOP_K_RESULTS,
                search_params=search_params,
                request_id=request_id
            )
            
            # Check if vision was used
            if retrieval_result.get('vision_analysis'):
                vision_pages = len(retrieval_result['vision_analysis'])
                self.pipeline_metadata.record_vision_usage(vision_pages)
            
            # Perform entity-anchored retrieval if entities found
            if story_entities and any(v for v in story_entities.values() if v):
                entity_results = self.entity_anchored_retriever.perform_entity_anchored_retrieval(
                    entities=story_entities,
                    base_query=description,
                    top_k_per_entity=2
                )
                
                # Merge with base results
                base_results = retrieval_result.get('retrieved_chunks', [])
                if entity_results:
                    merged_results = self.entity_anchored_retriever.merge_with_base_results(
                        base_results=base_results,
                        entity_results=entity_results,
                        entity_weight=0.6
                    )
                    retrieval_result['retrieved_chunks'] = merged_results
            
            # Check if fallback retrieval is needed
            if self.retrieval_fallbacks.should_trigger_fallback(
                retrieval_result.get('retrieved_chunks', []),
                min_results=3,
                min_score=0.3
            ):
                self.logger.warning("Primary retrieval weak, triggering fallback")
                self.pipeline_metadata.record_fallback("primary_weak")
                
                # Get fallback strategy
                fallback_strategy = self.retrieval_fallbacks.get_fallback_strategy(
                    query=enhanced_query,  # Use enhanced_query instead of undefined query
                    entities=story_entities,
                    failed_results=retrieval_result.get('retrieved_chunks', [])
                )
                
                # Try fallback queries
                for fallback_query in fallback_strategy['queries'][:2]:  # Try top 2
                    try:
                        fallback_result = await self.inferencer.ask_question(
                            question=fallback_query,
                            top_k=config.TOP_K_RESULTS,
                            request_id=request_id
                        )
                        
                        # Merge fallback results
                        fallback_chunks = fallback_result.get('retrieved_chunks', [])
                        if fallback_chunks:
                            # Apply fallback scoring
                            fallback_chunks = self.retrieval_fallbacks.apply_fallback_scoring(
                                fallback_chunks,
                                fallback_strategy['type']
                            )
                            
                            # Merge with existing results
                            existing_chunks = retrieval_result.get('retrieved_chunks', [])
                            all_chunks = existing_chunks + fallback_chunks
                            # Deduplicate by page key
                            seen = set()
                            unique_chunks = []
                            for chunk in all_chunks:
                                key = f"{chunk.get('pdf_name', '')}_{chunk.get('page_number', '')}"
                                if key not in seen:
                                    seen.add(key)
                                    unique_chunks.append(chunk)
                            
                            retrieval_result['retrieved_chunks'] = unique_chunks[:config.TOP_K_RESULTS]
                            break  # Stop after first successful fallback
                    except Exception as e:
                        self.logger.warning("Fallback retrieval failed", extra={"error": str(e)})
                        continue
            
            return retrieval_result
            
        except Exception as e:
            self.logger.error("Stage 3 (retrieve documentation) failed", extra={"error": str(e)})
            self.pipeline_metadata.record_retrieval_attempt(False, "hybrid")
            self.pipeline_metadata.record_error(str(e), "retrieve_documentation")
            raise
    
    async def _stage4_extract_technical_entities(
        self,
        retrieval_result: Dict[str, Any],
        story_entities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """STAGE 4: Extract Technical Entities from Retrieved Content"""
        try:
            # Extract from retrieved chunks
            retrieved_chunks = retrieval_result.get('retrieved_chunks', [])
            vision_analysis = retrieval_result.get('vision_analysis', [])
            
            # Extract entities from retrieved content
            all_entities = story_entities.copy() if story_entities else {}
            
            for chunk in retrieved_chunks:
                text_content = chunk.get('text_content', '') or chunk.get('content', '')
                if text_content:
                    source = f"{chunk.get('pdf_name', 'unknown')}_page_{chunk.get('page_number', 0)}"
                    score = chunk.get('combined_score', 0.0) or chunk.get('score', 0.0)
                    chunk_entities = self.entity_extractor.extract_from_text(
                        text_content,
                        source=source,
                        retriever_score=score
                    )
                    # Merge entities
                    for entity_type, values in chunk_entities.items():
                        if entity_type not in all_entities:
                            all_entities[entity_type] = []
                        all_entities[entity_type].extend(values)
            
            # Extract from vision analysis
            for vision in vision_analysis:
                vision_text = vision.get('vision_content', '') or vision.get('text', '')
                if vision_text:
                    source = f"vision_{vision.get('pdf_name', 'unknown')}_page_{vision.get('page_number', 0)}"
                    vision_entities = self.entity_extractor.extract_from_vision_metadata(
                        vision_text,
                        source=source
                    )
                    # Merge entities
                    for entity_type, values in vision_entities.items():
                        if entity_type not in all_entities:
                            all_entities[entity_type] = []
                        all_entities[entity_type].extend(values)
            
            # Get all entities with metadata
            all_entities_result = self.entity_extractor.get_all_entities()
            
            return all_entities_result
            
        except Exception as e:
            self.logger.error("Stage 4 (extract technical entities) failed", extra={"error": str(e)})
            return {}
    
    async def _stage5_validate_entities(
        self,
        technical_entities: Dict[str, Any],
        retrieval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """STAGE 5: Entity Validation & Confidence Scoring"""
        try:
            # Prepare texts for validation
            retrieved_texts = []
            for chunk in retrieval_result.get('retrieved_chunks', []):
                text = chunk.get('text_content', '') or chunk.get('content', '')
                if text:
                    retrieved_texts.append(text)
            
            vision_texts = []
            for vision in retrieval_result.get('vision_analysis', []):
                vision_text = vision.get('vision_content', '') or vision.get('text', '')
                if vision_text:
                    vision_texts.append(vision_text)
            
            # Extract entity lists (with initial confidence scores from extraction)
            entities_dict = {}
            initial_confidence_scores = technical_entities.get('confidence_scores', {})
            
            for entity_type in ['forms', 'dacs', 'graphs', 'fields', 'events', 'navigation_paths', 'px_attributes', 'tables']:
                entities_dict[entity_type] = technical_entities.get(entity_type, [])
            
            # Validate and score (pass initial confidence scores to preserve LLM-extracted confidence)
            validation_result = self.validation_scorer.validate_and_score_entities(
                entities=entities_dict,
                retrieved_texts=retrieved_texts,
                vision_texts=vision_texts,
                initial_confidence_scores=initial_confidence_scores
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error("Stage 5 (validate entities) failed", extra={"error": str(e)})
            return {
                "validated_entities": {},
                "uncertain_entities": {},
                "unconfirmed_entities": technical_entities,
                "confidence_scores": {}
            }
    
    async def _stage6_build_fact_table(
        self,
        validation_result: Dict[str, Any],
        retrieval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """STAGE 6: Build Fact Table"""
        try:
            # Use validated entities
            validated_entities = validation_result.get("validated_entities", {})
            confidence_scores = validation_result.get("confidence_scores", {})
            
            # Build fact table
            fact_table = self.fact_table_generator.generate_fact_table(
                validated_entities=validated_entities,
                retrieved_documentation=retrieval_result.get('retrieved_chunks', []),
                vision_metadata=retrieval_result.get('vision_analysis', []),
                confidence_scores=confidence_scores
            )
            
            # Add semantic object hit to fact table if available
            sem_hit = self.pipeline_metadata.metadata.get("semantic_object_hit")
            if sem_hit:
                fact_table["semantic_object_hit"] = sem_hit
                # Also add semantic object DAC/Graph to fact table entities if not already present
                if sem_hit.get("dac_full_name") and sem_hit["dac_full_name"] not in fact_table.get("dacs", []):
                    fact_table.setdefault("dacs", []).append(sem_hit["dac_full_name"])
                if sem_hit.get("graph_full_name") and sem_hit["graph_full_name"] not in fact_table.get("graphs", []):
                    fact_table.setdefault("graphs", []).append(sem_hit["graph_full_name"])
                if sem_hit.get("line_dac_full_name") and sem_hit["line_dac_full_name"] not in fact_table.get("dacs", []):
                    fact_table.setdefault("dacs", []).append(sem_hit["line_dac_full_name"])
            
            # Validate fact table structure
            fact_table = self.fact_table_generator.validate_fact_table(fact_table)
            
            return fact_table
            
        except Exception as e:
            self.logger.error("Stage 6 (build fact table) failed", extra={"error": str(e)})
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
    
    async def _stage7_map_acceptance_criteria(
        self,
        normalized_story: Dict[str, Any],
        fact_table: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """STAGE 7: Map Acceptance Criteria to Technical Requirements"""
        try:
            acceptance_criteria = normalized_story.get('acceptance_criteria', [])
            description = normalized_story.get('description', '')
            
            # Map AC to technical requirements
            ac_mappings = self.ac_mapper.map_acceptance_criteria(
                acceptance_criteria=acceptance_criteria,
                fact_table=fact_table,
                story_description=description
            )
            
            return ac_mappings
            
        except Exception as e:
            self.logger.error("Stage 7 (map AC) failed", extra={"error": str(e)})
            return []
    
    async def _stage8_generate_solution(
        self,
        normalized_story: Dict[str, Any],
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]],
        retrieval_result: Dict[str, Any],
        classification: Optional[Dict[str, Any]] = None
    ) -> str:
        """STAGE 8: Generate Final Solution"""
        try:
            description = normalized_story.get('description', '')
            ac_list = []
            for ac in normalized_story.get('acceptance_criteria', []):
                if isinstance(ac, dict):
                    ac_list.append(ac.get('text', ''))
                else:
                    ac_list.append(str(ac))
            
            retrieved_answer = retrieval_result.get('answer', '')
            
            # Determine processing depth based on story type
            story_type = classification.get('story_type', 'generic') if classification else 'generic'
            requires_tech_processing = self.story_classifier.requires_technical_processing(story_type)
            
            # Generate final solution
            solution_narrative = self.solution_generator.generate_final_solution(
                story_description=description,
                acceptance_criteria=ac_list,
                fact_table=fact_table,
                ac_mappings=ac_mappings,
                retrieved_answer=retrieved_answer
            )
            
            return solution_narrative
            
        except Exception as e:
            self.logger.error("Stage 8 (generate solution) failed", extra={"error": str(e)})
            return retrieval_result.get('answer', 'Unable to generate solution.')


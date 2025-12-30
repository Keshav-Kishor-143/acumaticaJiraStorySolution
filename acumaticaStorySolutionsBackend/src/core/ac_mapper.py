#!/usr/bin/env python3
"""
Acceptance Criteria → Technical Mapping Module

Maps each acceptance criterion to specific technical requirements:
- Related DAC fields
- Related forms/screens
- Required event handlers
- Required validations
- Required workflows
- Required code components
"""

from typing import Dict, Any, List, Optional
import json
import re
from openai import OpenAI

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.dll_reflection_index import DllReflectionIndexBuilder


class AcceptanceCriteriaMapper:
    """Maps acceptance criteria to technical requirements"""
    
    def __init__(self):
        self.logger = get_logger("AC_MAPPER")
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.logger.info("Acceptance Criteria Mapper initialized")
    
    def map_acceptance_criteria(
        self,
        acceptance_criteria: List[Any],
        fact_table: Dict[str, Any],
        story_description: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Map each acceptance criterion to technical requirements
        
        Args:
            acceptance_criteria: List of acceptance criteria (strings or dicts with 'text' and 'subpoints')
            fact_table: Technical fact table from entity extraction
            story_description: Optional story description for context
            
        Returns:
            List of mappings, each containing:
            - criterion: The AC text
            - related_forms: List of form IDs
            - related_dacs: List of DAC names
            - related_fields: List of field names
            - required_handlers: List of event handlers
            - required_validations: List of validation rules
            - required_workflows: List of workflows
            - missing_entities: List of entities needed but not found
        """
        try:
            self.logger.info("Mapping acceptance criteria to technical requirements", extra={
                "criteria_count": len(acceptance_criteria),
                "forms_available": len(fact_table.get('forms', [])),
                "dacs_available": len(fact_table.get('dacs', []))
            })
            
            # Normalize acceptance criteria to list of strings
            normalized_ac = self._normalize_acceptance_criteria(acceptance_criteria)
            
            # Build mapping prompt
            prompt = self._build_mapping_prompt(normalized_ac, fact_table, story_description)
            
            # Call LLM for mapping
            response = self.openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an Acumatica technical requirements mapper. 
                        Your task is to map each acceptance criterion to specific technical components.

CRITICAL RULES:
1. Only map to entities that exist in the fact table
2. If an entity is needed but not in fact table, add it to missing_entities
3. Be specific: map to exact form IDs, DAC names, field names
4. Identify required event handlers (FieldVerifying, RowPersisting, etc.)
5. Identify validation rules needed
6. If AC is non-technical, map only to business rules

Return ONLY valid JSON - no explanation."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract mappings
            mappings = result.get('mappings', [])
            if not isinstance(mappings, list):
                mappings = []
            
            # Validate and enrich mappings
            validated_mappings = []
            for i, mapping in enumerate(mappings):
                validated = self._validate_mapping(mapping, fact_table, normalized_ac[i] if i < len(normalized_ac) else "")
                validated_mappings.append(validated)
            
            self.logger.info("AC mappings generated", extra={
                "mappings_count": len(validated_mappings)
            })
            
            return validated_mappings
            
        except Exception as e:
            self.logger.error("Failed to map acceptance criteria", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Return fallback mappings
            return self._generate_fallback_mappings(acceptance_criteria, fact_table)
    
    def _normalize_acceptance_criteria(self, acceptance_criteria: List[Any]) -> List[str]:
        """Normalize acceptance criteria to list of strings"""
        normalized = []
        
        for ac in acceptance_criteria:
            if isinstance(ac, dict):
                # Extract main text
                ac_text = ac.get('text', '') or ac.get('criterion', '')
                if ac_text:
                    normalized.append(ac_text)
                # Include subpoints as context (not separate AC items)
                subpoints = ac.get('subpoints', [])
                if subpoints:
                    # Combine subpoints with main text for better context
                    subpoints_text = " | ".join([sp.strip() for sp in subpoints if sp and sp.strip()])
                    if subpoints_text:
                        normalized.append(f"{ac_text}: {subpoints_text}")
            elif isinstance(ac, str):
                normalized.append(ac.strip())
        
        return [ac for ac in normalized if ac]
    
    def _build_mapping_prompt(
        self,
        acceptance_criteria: List[str],
        fact_table: Dict[str, Any],
        story_description: Optional[str]
    ) -> str:
        """Build prompt for AC mapping"""
        
        criteria_text = "\n".join([f"{i+1}. {ac}" for i, ac in enumerate(acceptance_criteria)])
        
        fact_table_summary = {
            "forms": fact_table.get('forms', [])[:20],  # Limit for token efficiency
            "dacs": fact_table.get('dacs', [])[:20],
            "graphs": fact_table.get('graphs', [])[:20],
            "fields": fact_table.get('fields', [])[:30],
            "events": fact_table.get('events', [])[:20]
        }
        
        prompt = f"""Map each acceptance criterion to technical requirements.

STORY DESCRIPTION:
{story_description or "Not provided"}

ACCEPTANCE CRITERIA:
{criteria_text}

AVAILABLE TECHNICAL ENTITIES (from fact table):
{json.dumps(fact_table_summary, indent=2)}

TASK:
For each acceptance criterion, dynamically identify:
1. Which forms/screens are involved (extract from AC text and map to fact table)
2. Which DACs are involved (extract document types from AC and resolve to DAC names)
3. Which fields are involved (extract field names mentioned in AC, e.g., "Event Start Date" → EventStartDate)
4. Which event handlers are required (infer from AC: validation needs → FieldVerifying, save blocking → RowPersisting)
5. What validation rules are needed (extract exact validation logic from AC text)
6. What workflows are needed (identify cross-document operations mentioned in AC)
7. What entities are missing (needed but not in fact table)

FIELD EXTRACTION RULES:
- Extract field names from AC text dynamically
- "Event Start Date" → EventStartDate
- "Event End Date" → EventEndDate
- "Scheduled Start Date" → ScheduledStartDate
- "Scheduled Return Date" → ScheduledReturnDate
- Use PascalCase for field names
- If field name contains spaces, remove spaces and use PascalCase

OUTPUT JSON FORMAT:
{{
  "mappings": [
    {{
      "criterion": "<AC text>",
      "related_forms": ["FormID1", "FormID2"],
      "related_dacs": ["DAC1", "DAC2"],
      "related_fields": ["Field1", "Field2"],
      "required_handlers": ["FieldVerifying", "RowPersisting"],
      "required_validations": ["Validation rule 1", "Validation rule 2"],
      "required_workflows": ["Workflow 1"],
      "missing_entities": ["Entity1", "Entity2"]
    }}
  ]
}}

RULES:
- Use entities from fact table confidently - if they exist, use them
- Extract field names dynamically from AC text (e.g., "Event Start Date" → EventStartDate)
- Infer event handlers from AC requirements:
  * Date validation → FieldVerifying + RowPersisting
  * Save blocking → RowPersisting
  * Field updates → FieldUpdated
- Map validation rules exactly as stated in AC
- Identify cross-document operations (EX Quote/Order creation, etc.)
- If entity needed but not in fact table, add to missing_entities
- Be specific: use exact Form IDs, DAC names, field names from fact table
- If AC is non-technical, map only to business rules"""
        
        return prompt
    
    def _validate_mapping(
        self,
        mapping: Dict[str, Any],
        fact_table: Dict[str, Any],
        criterion_text: str
    ) -> Dict[str, Any]:
        """Validate and enrich a single mapping"""
        
        validated = {
            "criterion": mapping.get('criterion', criterion_text),
            "related_forms": [],
            "related_dacs": [],
            "related_fields": [],
            "required_handlers": [],
            "required_validations": [],
            "required_workflows": [],
            "missing_entities": []
        }
        
        # Validate forms (must exist in fact table)
        forms = mapping.get('related_forms', [])
        available_forms = fact_table.get('forms', [])
        validated['related_forms'] = [f for f in forms if f in available_forms]
        validated['missing_entities'].extend([f"Form:{f}" for f in forms if f not in available_forms])
        
        # Validate DACs
        dacs = mapping.get('related_dacs', [])
        available_dacs = fact_table.get('dacs', [])
        validated['related_dacs'] = [d for d in dacs if d in available_dacs]
        validated['missing_entities'].extend([f"DAC:{d}" for d in dacs if d not in available_dacs])
        
        # Validate fields
        fields = mapping.get('related_fields', [])
        available_fields = fact_table.get('fields', [])
        validated['related_fields'] = [f for f in fields if f in available_fields]
        validated['missing_entities'].extend([f"Field:{f}" for f in fields if f not in available_fields])
        
        # Handlers, validations, workflows are inferred, so keep as-is
        validated['required_handlers'] = mapping.get('required_handlers', [])
        validated['required_validations'] = mapping.get('required_validations', [])
        validated['required_workflows'] = mapping.get('required_workflows', [])
        
        # Deduplicate missing entities
        validated['missing_entities'] = list(dict.fromkeys(validated['missing_entities']))
        
        return validated
    
    def _generate_fallback_mappings(
        self,
        acceptance_criteria: List[Any],
        fact_table: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate fallback mappings if LLM mapping fails.

        This is intentionally heuristic and conservative: it does NOT guess DAC/Graph/Form IDs,
        but it can still infer required handlers/validations and surface missing entities.
        """
        normalized_ac = self._normalize_acceptance_criteria(acceptance_criteria)
        
        # Optional: load DLL reflection index (cached) for suggestions.
        # This helps us say "not found; closest match is X" instead of guessing.
        idx = None
        try:
            idx = DllReflectionIndexBuilder().load_or_build(force_rebuild=False)
        except Exception:
            idx = None

        def _suggest_similar_fields(desired: str, *, limit: int = 5) -> List[str]:
            if not idx:
                return []
            want = (desired or "").strip().lower()
            if not want:
                return []
            tokens = re.split(r"[^a-z0-9]+", want)
            tokens = [t for t in tokens if t]
            if not tokens:
                return []
            scored: List[tuple[int, str]] = []
            for r in idx.records:
                # Bias toward NV.Rental360 records for rental stories, but keep general.
                hay_parts = []
                hay_parts.extend(r.properties or [])
                hay_parts.extend(r.nested_types or [])
                hay = " ".join(hay_parts).lower()
                score = sum(1 for t in tokens if t in hay)
                if score:
                    # Return "TypeFullName.FieldName"
                    for f in hay_parts:
                        if all(t in f.lower() for t in tokens if len(t) > 2):
                            scored.append((score, f"{r.full_name}.{f}"))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Deduplicate
            out = []
            seen = set()
            for _, val in scored:
                if val in seen:
                    continue
                seen.add(val)
                out.append(val)
                if len(out) >= limit:
                    break
            return out

        mappings = []
        for ac in normalized_ac:
            ac_lower = ac.lower()

            # Helper: infer canonical PascalCase field name from a human-friendly label.
            def _to_pascal_case(label: str) -> str:
                parts = re.split(r"[^A-Za-z0-9]+", label.strip())
                return "".join(p[:1].upper() + p[1:] for p in parts if p)

            # Heuristic field extraction for date-validation stories.
            common_date_labels = [
                "Event Start Date",
                "Event End Date",
                "Scheduled Start Date",
                "Scheduled Return Date",
                "Service Date",
                "Extension Date",
            ]
            inferred_fields: List[str] = []
            for lbl in common_date_labels:
                if lbl.lower() in ac_lower:
                    inferred_fields.append(_to_pascal_case(lbl))

            # Handler inference:
            # - Field-level validation typically uses FieldVerifying/FieldUpdated
            # - Save blocking typically uses RowPersisting
            required_handlers: List[str] = []
            if "date" in ac_lower or "enters" in ac_lower:
                required_handlers.append("FieldVerifying")
            if any(k in ac_lower for k in ["save", "saving", "persist", "record is not saved"]):
                required_handlers.append("RowPersisting")
            # Default for validation rules if nothing matched.
            if not required_handlers and any(k in ac_lower for k in ["must be", "greater than", "less than", "earlier than", "later than"]):
                required_handlers = ["FieldVerifying", "RowPersisting"]

            # Validation inference (keep as plain-language rules; do not invent code).
            required_validations: List[str] = []
            if "event start date" in ac_lower:
                required_validations.append(
                    "EventStartDate must be >= ScheduledStartDate and <= ScheduledReturnDate"
                )
            if "event end date" in ac_lower:
                required_validations.append(
                    "EventEndDate must be <= ScheduledReturnDate and >= max(ScheduledStartDate, EventStartDate)"
                )
            if "invalid event date entry" in ac_lower:
                required_validations.append('Show error message on save: "Invalid Event Date Entry"')
            if "ex quote" in ac_lower or "ex order" in ac_lower:
                required_validations.append(
                    "When EX Quote/Order is created from Event Order, apply the same EventStartDate/EventEndDate date range rules"
                )

            # Workflows inference (cross-document behaviors).
            required_workflows: List[str] = []
            if "ex quote" in ac_lower or "ex order" in ac_lower:
                required_workflows.append("EX Quote/Order creation from Event Order must retain/validate event dates")

            # Missing entities: we do not assume they exist in fact_table.
            missing_entities: List[str] = []
            available_fields = set(fact_table.get("fields", []) or [])
            for f in inferred_fields:
                if f not in available_fields:
                    missing_entities.append(f"Field:{f}")

            # Damage fee / rental return heuristics (dynamic, but deterministic)
            damage_story = any(k in ac_lower for k in ["damage fee", "severely damaged", "missing damage fee", "rental return"])
            if damage_story:
                # Field anchors from story text
                dmg_fields = [
                    "Condition",
                    "Qty",
                    "UsrDamageFeeAdded",
                    "UsrDamageFeeTimestamp",
                    "UsrDamageFeeTechnician",
                    "UsrDamageFee",
                    "DamageAmount",
                    "CuryUnitPrice",
                    "CuryLineAmt",
                ]
                for f in dmg_fields:
                    if f not in inferred_fields:
                        inferred_fields.append(f)

                # Handlers for this pattern
                for h in ["FieldUpdated", "RowPersisting"]:
                    if h not in required_handlers:
                        required_handlers.append(h)

                # Validations / formulas
                if "severe" in ac_lower or "severely damaged" in ac_lower:
                    required_validations.append("SeverityMultiplier = 1.5 when Condition = Severely Damaged; otherwise 1.0")
                if "qty" in ac_lower or "quantity" in ac_lower:
                    required_validations.append("DamageAmount = Qty * BaseFee * SeverityMultiplier")
                if "missing damage fee item" in ac_lower or "cannot be saved" in ac_lower:
                    required_validations.append('Block save and show error: "Missing Damage Fee Item – Rental Return cannot be saved."')
                if "audit" in ac_lower or "technician" in ac_lower:
                    required_validations.append("Populate UsrDamageFeeAdded=true, UsrDamageFeeTimestamp=Now, UsrDamageFeeTechnician=CurrentUser")

                # Missing entities + suggestions from DLLs
                for f in dmg_fields:
                    if f not in available_fields:
                        suggestions = _suggest_similar_fields(f, limit=3)
                        if suggestions:
                            missing_entities.append(f"Field:{f} (closest: {', '.join(suggestions)})")
                        else:
                            missing_entities.append(f"Field:{f}")

            # Deduplicate while preserving order.
            def _dedupe(seq: List[str]) -> List[str]:
                return list(dict.fromkeys([x for x in seq if x]))

            mappings.append({
                "criterion": ac,
                "related_forms": [],
                "related_dacs": [],
                "related_fields": _dedupe(inferred_fields),
                "required_handlers": _dedupe(required_handlers),
                "required_validations": _dedupe(required_validations),
                "required_workflows": _dedupe(required_workflows),
                "missing_entities": _dedupe(missing_entities)
            })
        
        return mappings
    
    def map(self, ac_list: List[Any], entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reference-compatible mapping method (adapter pattern)
        
        Matches reference interface: ACMapper.map(ac_list, entities)
        
        Args:
            ac_list: List of acceptance criteria
            entities: Dictionary of extracted entities
            
        Returns:
            Dictionary with 'rules' (AC list) and 'entities' (entity dict)
        """
        try:
            # Build a minimal fact table from entities
            fact_table = {
                "forms": entities.get("forms", []),
                "dacs": entities.get("dacs", []),
                "graphs": entities.get("graphs", []),
                "fields": entities.get("fields", []),
                "events": entities.get("events", []),
                "navigation_paths": entities.get("navigation_paths", [])
            }
            
            # Use full mapping method
            mappings = self.map_acceptance_criteria(
                acceptance_criteria=ac_list,
                fact_table=fact_table
            )
            
            return {
                "rules": ac_list,
                "entities": entities,
                "mappings": mappings
            }
        except Exception as e:
            self.logger.error("Map adapter failed", extra={"error": str(e)})
            return {
                "rules": ac_list,
                "entities": entities,
                "mappings": []
            }


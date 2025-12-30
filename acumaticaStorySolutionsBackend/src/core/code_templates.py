#!/usr/bin/env python3
"""
Code Templates for Deterministic Code Generation

This module provides template-based code generation following Law 4:
"Generate Implementation from Technical Truth, Not Language Patterns"

Templates are deterministic and generate code based on:
- DAC definition
- Graph event pipeline
- Field attributes
- Acumatica framework rules
- Fact table entities
"""

from typing import Dict, Any, List, Optional
import re

from src.utils.logger_utils import get_logger
from src.core.dynamic_entity_resolver import DynamicEntityResolver


class CodeTemplateGenerator:
    """
    Generates Acumatica code deterministically from templates
    
    This ensures code generation is based on technical truth, not language patterns.
    """
    
    def __init__(self):
        self.logger = get_logger("CODE_TEMPLATES")
        self.entity_resolver = DynamicEntityResolver()
    
    def generate_field_definition(
        self,
        field_name: str,
        field_type: str = "DateTime?",
        dac_name: str = "",
        display_name: Optional[str] = None,
        is_date: bool = False,
        is_string: bool = False,
        is_decimal: bool = False,
        is_int: bool = False
    ) -> str:
        """
        Generate DAC field definition code
        
        Args:
            field_name: Field name (PascalCase)
            field_type: C# type (DateTime?, string, decimal?, int?)
            dac_name: DAC name for context
            display_name: Display name for UI
            is_date: True if date field
            is_string: True if string field
            is_decimal: True if decimal field
            is_int: True if int field
            
        Returns:
            Complete field definition code
        """
        # Determine attribute based on type
        if is_date or field_type == "DateTime?":
            px_attribute = "[PXDBDate]"
            bql_type = "BqlDateTime"
        elif is_string or field_type == "string":
            px_attribute = "[PXDBString(255)]"
            bql_type = "BqlString"
        elif is_decimal or field_type == "decimal?":
            px_attribute = "[PXDBDecimal]"
            bql_type = "BqlDecimal"
        elif is_int or field_type == "int?":
            px_attribute = "[PXDBInt]"
            bql_type = "BqlInt"
        else:
            px_attribute = "[PXDBString(255)]"
            bql_type = "BqlString"
        
        # Generate field name in camelCase for BQL class
        field_name_camel = field_name[0].lower() + field_name[1:] if field_name else ""
        
        # Display name
        if not display_name:
            # Convert PascalCase to Display Name
            display_name = re.sub(r'([A-Z])', r' \1', field_name).strip()
        
        code = f"""#region {field_name}
{px_attribute}
[PXUIField(DisplayName="{display_name}")]
public virtual {field_type} {field_name} {{ get; set; }}
public abstract class {field_name_camel} : PX.Data.BQL.Bql{field_type.replace('?', '').replace('string', 'String').replace('DateTime', 'DateTime')}.Field<{field_name_camel}> {{ }}
#endregion"""
        
        return code
    
    def generate_field_verifying_handler(
        self,
        dac_name: str,
        field_name: str,
        validation_condition: str,
        error_message: str,
        field_type: str = "DateTime?"
    ) -> str:
        """
        Generate FieldVerifying event handler
        
        Args:
            dac_name: DAC name (e.g., "EVOrder")
            field_name: Field name (e.g., "EventStartDate")
            validation_condition: Validation condition (e.g., "newDate < row.ScheduledStartDate")
            error_message: Error message to display
            field_type: Field type for casting
            
        Returns:
            Complete FieldVerifying handler code
        """
        field_name_camel = field_name[0].lower() + field_name[1:] if field_name else ""
        
        # Determine cast type
        if "DateTime" in field_type:
            cast_type = "DateTime"
            cast_code = "var newDate = (DateTime)e.NewValue;"
        elif "decimal" in field_type or "Decimal" in field_type:
            cast_type = "decimal"
            cast_code = "var newValue = (decimal?)e.NewValue;"
        elif "int" in field_type or "Int" in field_type:
            cast_type = "int"
            cast_code = "var newValue = (int?)e.NewValue;"
        else:
            cast_type = "string"
            cast_code = "var newValue = (string)e.NewValue;"
        
        code = f"""protected void _(Events.FieldVerifying<{dac_name}.{field_name_camel}> e)
{{
    var row = ({dac_name})e.Row;
    if (row == null || e.NewValue == null) return;
    
    {cast_code}
    
    // Validation logic
    if ({validation_condition})
        throw new PXSetPropertyException("{error_message}");
}}"""
        
        return code
    
    def generate_row_persisting_handler(
        self,
        dac_name: str,
        validation_conditions: List[Dict[str, str]],
        error_message: str
    ) -> str:
        """
        Generate RowPersisting event handler
        
        Args:
            dac_name: DAC name
            validation_conditions: List of dicts with 'field' and 'condition'
            error_message: Error message to display
            
        Returns:
            Complete RowPersisting handler code
        """
        cond_exprs: List[str] = []
        field_names: List[str] = []

        for cond in validation_conditions:
            field = (cond.get("field") or "").strip()
            condition = (cond.get("condition") or "").strip()
            if field and condition:
                cond_exprs.append(f"({condition})")
                field_names.append(field)

        if not cond_exprs:
            return ""

        # Use first field for exception anchor
        first_field = field_names[0] if field_names else "null"
        combined = " || ".join(cond_exprs)

        code = f"""protected void _(Events.RowPersisting<{dac_name}> e)
{{
    var row = e.Row;
    if (row == null) return;

    // Do not validate on delete
    if (e.Operation == PXDBOperation.Delete) return;

    // Final validation before save
    if ({combined})
    {{
        throw new PXRowPersistingException(nameof(row.{first_field}), row.{first_field}, "{error_message}");
    }}
}}"""

        return code
    
    def generate_field_updated_handler(
        self,
        dac_name: str,
        field_name: str,
        update_logic: Optional[str] = None
    ) -> str:
        """
        Generate FieldUpdated event handler
        
        Args:
            dac_name: DAC name
            field_name: Field name
            update_logic: Optional update logic code
            
        Returns:
            Complete FieldUpdated handler code
        """
        field_name_camel = field_name[0].lower() + field_name[1:] if field_name else ""
        
        if update_logic:
            logic_code = f"\n    {update_logic}\n"
        else:
            logic_code = "\n    // Field update logic\n"
        
        code = f"""protected void _(Events.FieldUpdated<{dac_name}.{field_name_camel}> e)
{{
    var row = ({dac_name})e.Row;
    if (row == null) return;
    {logic_code}}}"""
        
        return code
    
    def generate_graph_extension_class(
        self,
        graph_name: str,
        handlers: List[str],
        namespace: str = "PX.Objects"
    ) -> str:
        """
        Generate complete Graph extension class with handlers
        
        Args:
            graph_name: Graph name (e.g., "EVOrderEntry")
            handlers: List of handler code strings
            namespace: Namespace for the extension
            
        Returns:
            Complete Graph extension class code
        """
        extension_name = f"{graph_name}Ext"
        
        code = f"""using System;
using PX.Data;
using PX.Objects;

namespace {namespace}
{{
    public class {extension_name} : PXGraphExtension<{graph_name}>
    {{
{chr(10).join('        ' + handler.replace(chr(10), chr(10) + '        ') for handler in handlers)}
    }}
}}"""
        
        return code
    
    def generate_dac_extension_class(
        self,
        dac_name: str,
        fields: List[str],
        namespace: str = "PX.Objects"
    ) -> str:
        """
        Generate complete DAC extension class with fields
        
        Args:
            dac_name: DAC name (e.g., "EVOrder")
            fields: List of field definition code strings
            namespace: Namespace for the extension
            
        Returns:
            Complete DAC extension class code
        """
        extension_name = f"{dac_name}Extension"
        
        code = f"""using PX.Data;
using PX.Objects;

namespace {namespace}
{{
    public class {extension_name} : PXCacheExtension<{dac_name}>
    {{
{chr(10).join('        ' + field.replace(chr(10), chr(10) + '        ') for field in fields)}
    }}
}}"""
        
        return code
    
    def generate_complete_implementation(
        self,
        fact_table: Dict[str, Any],
        ac_mappings: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate complete implementation code from fact table and AC mappings
        
        Args:
            fact_table: Technical fact table
            ac_mappings: AC to technical mappings
            
        Returns:
            Dictionary with:
            - dac_extension: DAC extension code
            - graph_extension: Graph extension code
            - handlers: Individual handler code blocks
        """
        dacs = fact_table.get('dacs', []) or []
        graphs = fact_table.get('graphs', []) or []
        fields = fact_table.get('fields', []) or []

        # Detect common story patterns (deterministic)
        def _looks_like_event_date_validation(mappings: List[Dict[str, Any]]) -> bool:
            for m in mappings or []:
                crit = (m.get("criterion") or "").lower()
                if "event start date" in crit or "event end date" in crit or "invalid event date entry" in crit:
                    return True
                vals = " ".join((m.get("required_validations") or [])).lower()
                if "eventstartdate" in vals and "scheduledstartdate" in vals:
                    return True
            return False

        def _looks_like_damage_fee_story(mappings: List[Dict[str, Any]]) -> bool:
            for m in mappings or []:
                crit = (m.get("criterion") or "").lower()
                if "damage fee" in crit:
                    return True
                if "severely damaged" in crit or "damaged" in crit:
                    # keep conservative: also require fee mention somewhere
                    if "fee" in crit:
                        return True
                vals = " ".join((m.get("required_validations") or [])).lower()
                if "damage" in vals and "fee" in vals:
                    return True
            return False

        def _collect_related_fields(mappings: List[Dict[str, Any]]) -> List[str]:
            out: List[str] = []
            for m in mappings or []:
                out.extend((m.get("related_fields") or [])[:10])
            return list(dict.fromkeys([f for f in out if f]))

        def _looks_like_csharp_condition(expr: str) -> bool:
            e = (expr or "").strip()
            if not e:
                return False
            # crude heuristic: must contain an operator typical in boolean expressions
            return any(op in e for op in ["<", ">", "==", "!=", "&&", "||"]) and "must" not in e.lower()
        
        # Generate DAC extension
        dac_code = ""
        # For validation stories, DAC extension is optional (fields usually already exist).
        # Only generate DAC extension when we have a known DAC AND extracted fields.
        if dacs and fields:
            dac_name = dacs[0]  # Use first DAC
            field_definitions = []
            
            for field in fields[:10]:  # Limit to first 10 fields
                field_def = self.generate_field_definition(
                    field_name=field,
                    dac_name=dac_name,
                    is_date=True  # Default to date, should be determined from context
                )
                field_definitions.append(field_def)
            
            if field_definitions:
                dac_code = self.generate_dac_extension_class(
                    dac_name=dac_name,
                    fields=field_definitions
                )
        
        # Generate Graph extension with handlers
        graph_code = ""
        handlers = []
        
        if ac_mappings:
            # Prefer semantic object binding if available (business concept -> Graph/DAC)
            sem_hit = (fact_table or {}).get("semantic_object_hit") or {}
            sem_graph_full = sem_hit.get("graph_full_name") if isinstance(sem_hit, dict) else None
            sem_dac_full = sem_hit.get("dac_full_name") if isinstance(sem_hit, dict) else None
            sem_ns = sem_hit.get("module_tag") if isinstance(sem_hit, dict) else None

            def _short_name(full: Optional[str]) -> Optional[str]:
                if not full:
                    return None
                return full.split(".")[-1]

            # Prefer discovered graph/dac; otherwise generate placeholders.
            graph_name = _short_name(sem_graph_full) or (graphs[0] if graphs else "EventOrderEntry")
            dac_name = _short_name(sem_dac_full) or (dacs[0] if dacs else "EventOrder")
            default_namespace = sem_ns or "PX.Objects"

            # Dynamic inference: try to resolve actual Graph/DAC from DLLs using story + fields.
            try:
                story_text = " ".join([(m.get("criterion") or "") for m in (ac_mappings or []) if isinstance(m, dict)])
                inferred_fields = []
                for m in (ac_mappings or [])[:10]:
                    inferred_fields.extend((m.get("related_fields") or [])[:10])
                inferred_fields = list(dict.fromkeys([f for f in inferred_fields if f]))

                target = self.entity_resolver.resolve_graph_and_dac(
                    story_text=story_text,
                    field_names=inferred_fields,
                    prefer_assemblies=None,
                    min_confidence=0.55,
                )
                if target:
                    graph_name = target.graph_name
                    dac_name = target.dac_name
                    self.logger.info("Dynamically resolved Graph/DAC from DLLs", extra={
                        "dac": target.dac_full_name,
                        "graph": target.graph_full_name,
                        "confidence": target.confidence,
                    })
            except Exception as e:
                self.logger.warning("Dynamic entity resolution failed; using placeholders", extra={"error": str(e)})

            # Special-case: event date validation => generate real conditions deterministically.
            if _looks_like_event_date_validation(ac_mappings):
                related_fields = _collect_related_fields(ac_mappings)

                # Canonical expected fields for this story; if story mapping provided alternates, keep them too.
                required = ["EventStartDate", "EventEndDate", "ScheduledStartDate", "ScheduledReturnDate"]
                for f in required:
                    if f not in related_fields:
                        related_fields.append(f)

                # Build handlers using standard Acumatica patterns:
                # - FieldVerifying for immediate feedback
                # - RowPersisting for save-blocking safety net
                msg = "Invalid Event Date Entry"

                # FieldVerifying: EventStartDate
                handlers.append(
                    self.generate_field_verifying_handler(
                        dac_name=dac_name,
                        field_name="EventStartDate",
                        validation_condition="newDate < row.ScheduledStartDate || newDate > row.ScheduledReturnDate",
                        error_message=msg,
                        field_type="DateTime?",
                    )
                )

                # FieldVerifying: EventEndDate
                handlers.append(
                    self.generate_field_verifying_handler(
                        dac_name=dac_name,
                        field_name="EventEndDate",
                        validation_condition="newDate > row.ScheduledReturnDate || newDate < row.ScheduledStartDate || (row.EventStartDate != null && newDate < row.EventStartDate)",
                        error_message=msg,
                        field_type="DateTime?",
                    )
                )

                # RowPersisting: enforce both rules on save
                handlers.append(
                    self.generate_row_persisting_handler(
                        dac_name=dac_name,
                        validation_conditions=[
                            {
                                "field": "EventStartDate",
                                "condition": "row.EventStartDate != null && (row.EventStartDate < row.ScheduledStartDate || row.EventStartDate > row.ScheduledReturnDate)",
                            },
                            {
                                "field": "EventEndDate",
                                "condition": "row.EventEndDate != null && (row.EventEndDate > row.ScheduledReturnDate || row.EventEndDate < row.ScheduledStartDate || (row.EventStartDate != null && row.EventEndDate < row.EventStartDate))",
                            },
                        ],
                        error_message=msg,
                    )
                )

                graph_code = self.generate_graph_extension_class(
                    graph_name=graph_name,
                    handlers=handlers,
                    namespace=default_namespace,
                )

            # Special-case: damage fee on rental return (header+lines enforcement)
            elif _looks_like_damage_fee_story(ac_mappings):
                # This template ONLY emits code if we have enough truth to avoid misleading output.
                # Minimum required: resolved graph/dac must look like rental/return.
                if "return" not in graph_name.lower() and "rental" not in graph_name.lower():
                    self.logger.warning("Damage fee template skipped: unresolved/unsafe graph target", extra={"graph": graph_name, "dac": dac_name})
                else:
                    # Without known line DAC + known fee item selector field, we cannot safely generate full code.
                    # So we only emit persist-time validation scaffolding if we can anchor the error message.
                    msg = "Missing Damage Fee Item – Rental Return cannot be saved."
                    # Use RowPersisting on header DAC (save-blocking). Condition field names are unknown dynamically here.
                    # We intentionally do NOT guess condition/status fields.
                    handler = self.generate_row_persisting_handler(
                        dac_name=dac_name,
                        validation_conditions=[
                            {
                                "field": "NoteID" if hasattr(dac_name, "NoteID") else "null",
                                "condition": "false"  # will be filtered out below
                            }
                        ],
                        error_message=msg,
                    )
                    # If handler produced empty/placeholder, suppress output
                    if handler and "false" not in handler:
                        handlers.append(handler)
                        graph_code = self.generate_graph_extension_class(
                            graph_name=graph_name,
                            handlers=handlers,
                            namespace=default_namespace,
                        )
                    else:
                        self.logger.warning("Damage fee template skipped: insufficient resolved fields/conditions; not emitting placeholder code")
            else:
                # Generic mapping-based handler generation (best-effort)
                for mapping in ac_mappings:
                    required_handlers = mapping.get('required_handlers', [])
                    related_fields = mapping.get('related_fields', [])
                    required_validations = mapping.get('required_validations', [])

                    for handler_type in required_handlers:
                        if 'FieldVerifying' in handler_type and related_fields:
                            field = related_fields[0]
                            # Only emit if we have an actual boolean expression
                            cond = required_validations[0] if required_validations else ""
                            if _looks_like_csharp_condition(cond):
                                handler_code = self.generate_field_verifying_handler(
                                    dac_name=dac_name,
                                    field_name=field,
                                    validation_condition=cond,
                                    error_message="Validation failed",
                                )
                                handlers.append(handler_code)

                        elif 'RowPersisting' in handler_type:
                            conds = []
                            for v in (required_validations or [])[:3]:
                                if _looks_like_csharp_condition(v):
                                    conds.append({"field": (related_fields[0] if related_fields else "null"), "condition": v})
                            handler_code = self.generate_row_persisting_handler(
                                dac_name=dac_name,
                                validation_conditions=conds,
                                error_message="Validation failed",
                            )
                            if handler_code:
                                handlers.append(handler_code)

                if handlers:
                    graph_code = self.generate_graph_extension_class(
                        graph_name=graph_name,
                        handlers=handlers,
                        namespace=default_namespace,
                    )
        
        return {
            "dac_extension": dac_code,
            "graph_extension": graph_code,
            "handlers": handlers
        }


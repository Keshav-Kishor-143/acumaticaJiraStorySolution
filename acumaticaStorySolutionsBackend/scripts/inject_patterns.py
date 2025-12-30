#!/usr/bin/env python3
"""
Pattern Injection Script (CRUCIAL for high accuracy).

Creates a first-class retrievable "manual" in the knowledge base:
  knowledge_base/manuals/Acumatica_Pattern_Library/

These pattern documents are designed to be retrieved alongside manuals/DLL text,
and then used by the deterministic template engine (Law 4).
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure `src/` is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# We need PROJECT_ROOT on sys.path so `import src...` resolves to {PROJECT_ROOT}/src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.knowledge_ingestor import KnowledgeIngestor


def build_patterns() -> list[tuple[str, str]]:
    """
    Return list of (section_id, text).
    Keep these short, structured, and highly retrievable.
    """
    patterns: list[tuple[str, str]] = []

    patterns.append(
        (
            "date_validation",
            """
Pattern: Date Validation (Immediate + Persisting)
Applies to: SOOrder, ARInvoice, INRegister, custom DACs

Goal:
- Validate date relationships as soon as user changes a field
- Block save if invalid values remain

Recommended Handlers:
- Events.FieldVerifying<YourDAC.yourDateField>
  - show PXSetPropertyException for immediate feedback
- Events.RowPersisting<YourDAC>
  - throw PXRowPersistingException to block invalid save

Common Operators:
- newValue < row.OtherDateField
- newValue > row.OtherDateField

Error Handling:
- Immediate: PXSetPropertyException("Invalid Date Entry")
- Persisting: PXRowPersistingException(nameof(row.YourDateField), row.YourDateField, "Invalid Date Entry")
""".strip(),
        )
    )

    patterns.append(
        (
            "dac_extension_usr_fields",
            """
Pattern: DAC Extension - Usr Fields

Goal:
- Add custom fields via PXCacheExtension<TDAC>

Required Elements:
- [PXDB*] attribute appropriate for type
- [PXUIField(DisplayName="...")]
- Property: public virtual <type> UsrMyField { get; set; }
- BQL field: public abstract class usrMyField : PX.Data.BQL.Bql<...>.Field<usrMyField> { }

Notes:
- Prefer DateTime? with [PXDBDate] for date-only fields
- Prefer PXDBString(length) for text fields
""".strip(),
        )
    )

    patterns.append(
        (
            "graph_extension_events",
            """
Pattern: Graph Extension - Event Handlers

Goal:
- Implement business rules using graph extensions

Common Handlers:
- Events.FieldUpdated<TDAC.field>
- Events.FieldVerifying<TDAC.field>
- Events.RowSelected<TDAC>
- Events.RowPersisting<TDAC>

Best Practice:
- Keep validations in shared methods invoked by both FieldVerifying and RowPersisting
- Use nameof(row.Field) in exceptions
""".strip(),
        )
    )

    patterns.append(
        (
            "workflow_update",
            """
Pattern: Workflow Update (Screen-based)

Goal:
- Modify workflow states/transitions safely

Approach:
- Use workflow extension for the screen/graph
- Add conditions and transitions explicitly
- Keep state names and actions deterministic

Warning:
- Workflow modifications are highly version-sensitive; validate against installed site version
""".strip(),
        )
    )

    return patterns


def main() -> int:
    ingestor = KnowledgeIngestor()
    patterns = build_patterns()
    try:
        result = ingestor.ingest_text_document(
            document_name="Acumatica_Pattern_Library",
            texts=patterns,
            section_type="pattern",
            source_label="curated_patterns",
            overwrite=False,
        )
        # Avoid Unicode output issues on Windows consoles (cp1252).
        print(f"Injected pattern library: {result}")
    except FileExistsError:
        print("Pattern library already exists: Acumatica_Pattern_Library (no changes made).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



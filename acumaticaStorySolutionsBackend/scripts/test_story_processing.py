#!/usr/bin/env python3
"""
Test Story Processing Script

Processes a sample JIRA story through the complete pipeline and analyzes the output.
"""

import sys
import json
import asyncio
from pathlib import Path

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root and src to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.core.pipeline_controller import PipelineController
from src.utils.logger_utils import get_logger
from src.utils.markdown_formatter import MarkdownFormatter
from src.config.config import config

logger = get_logger("STORY_TEST")


# Sample story data
SAMPLE_STORY = {
    "description": """A Rental Operations Coordinator needs the system to automatically assess and apply a damage fee when rental equipment is returned in a damaged or severely damaged condition. The system must pull the base damage fee from the equipment type, calculate the surcharge if necessary, insert the fee line on the Rental Return document, and enforce that the fee is present before the document can be saved or released.""",
    "acceptance_criteria": [
        "AC1 — Auto-Add Damage Fee: Given a Rental Return Order And Condition = Damaged When the user saves the document Then a Damage Fee line is automatically added And Unit Price uses EquipmentTypeExt.UsrDamageFee.",
        "AC2 — Severe Damage Surcharge: Given Condition = Severely Damaged When the fee is added Then DamageAmount = BaseFee × 1.5 × Qty.",
        "AC3 — Recalculation on Quantity Change: Given a Damage Fee line exists When the Qty on the damage fee line is updated Then the fee amount recalculates using: Qty × BaseFee × SeverityMultiplier.",
        "AC4 — Persist-Time Validation: Given Condition = Damaged or Severely Damaged And no Damage Fee line exists When the user clicks Save Then the system shows the error: “Missing Damage Fee Item – Rental Return cannot be saved.” And the record is not saved.",
        "AC5 — Audit Fields: Given a Damage Fee line is inserted automatically When it is created Then the system populates: UsrDamageFeeAdded = true UsrDamageFeeTimestamp = current datetime UsrDamageFeeTechnician = current user."
    ],
    "requirements": [
        "When the equipment condition is set to Damaged or Severely Damaged, the system must automatically insert a Damage Fee transaction line on the Rental Return document.",
        "The base damage fee amount must be read from EquipmentTypeExt.UsrDamageFee.",
        "If the condition is Severely Damaged, the fee must apply a 1.5× surcharge multiplier.",
        "DamageAmount = Qty * BaseFee * SeverityMultiplier",
        "When the quantity on the damage fee line changes, the system must recalculate the amount automatically.",
        "If the equipment condition is Damaged or Severely Damaged and the document has no damage fee line, the system must block saving and show an error: “Missing Damage Fee Item – Rental Return cannot be saved.”",
        "When a fee is added, the system must populate audit fields: UsrDamageFeeAdded=true, UsrDamageFeeTimestamp=current datetime, UsrDamageFeeTechnician=current user"
    ],
    "story_id": "TEST-RENTAL-DAMAGE-FEE",
    "title": "Auto Damage Fee on Rental Return"
}


async def test_story_processing():
    """Process the sample story through the pipeline"""
    print("=" * 80)
    print("🧪 TESTING STORY PROCESSING PIPELINE")
    print("=" * 80)
    print("\n📋 Sample Story:")
    print(f"Title: {SAMPLE_STORY['title']}")
    print(f"Story ID: {SAMPLE_STORY['story_id']}")
    print(f"Description Length: {len(SAMPLE_STORY['description'])} chars")
    print(f"Acceptance Criteria Count: {len(SAMPLE_STORY['acceptance_criteria'])}")
    print(f"Requirements Count: {len(SAMPLE_STORY['requirements'])}")
    
    try:
        # Initialize pipeline controller
        print("\n" + "=" * 80)
        print("🚀 Initializing Pipeline Controller...")
        print("=" * 80)
        pipeline_controller = PipelineController()
        
        # Process story
        print("\n" + "=" * 80)
        print("⚙️  Processing Story Through Pipeline...")
        print("=" * 80)
        
        result = await pipeline_controller.process_story(
            story_request=SAMPLE_STORY,
            request_id="test-request-001"
        )
        
        # Extract key information
        print("\n" + "=" * 80)
        print("📊 PROCESSING RESULTS")
        print("=" * 80)
        
        # Technical Entities
        technical_entities = result.get('technical_entities', {})
        print("\n🔍 Technical Entities Extracted:")
        for entity_type, entities in technical_entities.items():
            if entities:
                # technical_entities may contain lists (e.g., forms) and dicts (e.g., confidence_scores)
                if isinstance(entities, list):
                    print(f"  - {entity_type}: {len(entities)} found")
                    for entity in entities[:5]:  # Show first 5
                        print(f"    • {entity}")
                elif isinstance(entities, dict):
                    print(f"  - {entity_type}: {len(entities)} found")
                    for k, v in list(entities.items())[:5]:
                        print(f"    • {k}: {v}")
                else:
                    print(f"  - {entity_type}: {entities}")
        
        # Validated Entities
        validated_entities = result.get('validated_entities', {})
        print("\n✅ Validated Entities:")
        for entity_type, entities in validated_entities.items():
            if entities:
                if isinstance(entities, list) or isinstance(entities, dict):
                    print(f"  - {entity_type}: {len(entities)} validated")
                else:
                    print(f"  - {entity_type}: validated")
        
        # Fact Table
        fact_table = result.get('fact_table', {})
        print("\n📋 Fact Table:")
        for key, value in fact_table.items():
            if value:
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
                else:
                    print(f"  - {key}: {value}")
        
        # AC Mappings
        ac_mappings = result.get('ac_mappings', [])
        print(f"\n🎯 AC Mappings: {len(ac_mappings)} criteria mapped")
        for i, mapping in enumerate(ac_mappings[:3], 1):  # Show first 3
            print(f"  Mapping {i}:")
            print(f"    - Related Forms: {mapping.get('related_forms', [])}")
            print(f"    - Related DACs: {mapping.get('related_dacs', [])}")
            print(f"    - Related Fields: {mapping.get('related_fields', [])}")
            print(f"    - Required Handlers: {mapping.get('required_handlers', [])}")
        
        # Solution Narrative
        solution_narrative = result.get('solution_narrative', '')
        print(f"\n📝 Solution Narrative Length: {len(solution_narrative)} chars")
        print(f"Preview (first 500 chars):")
        print("-" * 80)
        print(solution_narrative[:500] + "..." if len(solution_narrative) > 500 else solution_narrative)
        print("-" * 80)
        
        # Sources
        sources = result.get('sources', [])
        print(f"\n📚 Sources Used: {len(sources)}")
        for source in sources[:5]:  # Show first 5
            print(f"  - {source.get('document', 'Unknown')} (Page {source.get('page', 'N/A')}, Confidence: {source.get('confidence', 0.0):.2f})")
        
        # Confidence Score
        confidence_score = result.get('confidence_score', 0.0)
        confidence_details = result.get('confidence_details', {})
        
        if isinstance(confidence_score, (int, float)):
            print(f"\n🎯 Confidence Score: {confidence_score:.2f}")
        else:
            print(f"\n🎯 Confidence Score: {confidence_score.get('confidence_score', 0.0):.2f}")
        
        if confidence_details and isinstance(confidence_details, dict):
            if 'component_scores' in confidence_details:
                print("  Component Scores:")
                for component, score in confidence_details['component_scores'].items():
                    print(f"    - {component}: {score:.2f}")
        
        # Save output
        output_dir = _project_root / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{SAMPLE_STORY['story_id']}_result.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Full result saved to: {output_file}")
        
        # Save markdown solution if available
        if solution_narrative:
            # Write EXACTLY ONE markdown output for the story.
            # Control which style is produced via OUTPUT_STYLE env var ("final" | "precision").
            try:
                formatter = MarkdownFormatter()

                output_style = getattr(config, "OUTPUT_STYLE", "final") or "final"
                output_style = str(output_style).lower().strip()

                if output_style == "precision":
                    md_content = formatter.format_precision_solution(
                        title=SAMPLE_STORY.get("title", "Story Solution"),
                        story_id=SAMPLE_STORY.get("story_id"),
                        narrative=solution_narrative,
                        fact_table=fact_table or {},
                        ac_mappings=ac_mappings or [],
                        technical_entities=validated_entities or technical_entities or {},
                        acceptance_criteria=SAMPLE_STORY.get("acceptance_criteria", []) or [],
                        sources=sources or [],
                        metadata=result.get("metadata", {}) or {},
                        confidence_score=result.get("confidence_score"),
                        confidence_details=result.get("confidence_details", {}) or {},
                        story_classification=result.get("story_classification", {}) or {},
                    )
                else:
                    md_content = formatter.format_final_solution(
                        title=SAMPLE_STORY.get("title", "Story Solution"),
                        story_id=SAMPLE_STORY.get("story_id"),
                        narrative=solution_narrative,
                        acceptance_criteria=SAMPLE_STORY.get("acceptance_criteria", []) or [],
                        sources=sources or [],
                    )

                # Always write a single markdown file: <STORY_ID>.md
                md_file = output_dir / f"{SAMPLE_STORY['story_id']}.md"
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(md_content)

                # Safety: if older runs created extra markdown variants, remove them to keep ONE .md per story.
                # (Only deletes generated artifacts for this story ID.)
                extra_files = [
                    output_dir / f"{SAMPLE_STORY['story_id']}_precision.md",
                    output_dir / f"{SAMPLE_STORY['story_id']}_solution.md",
                ]
                for extra in extra_files:
                    try:
                        if extra.exists():
                            extra.unlink()
                    except Exception:
                        # Non-fatal: keep going
                        pass

                print(f"📄 Markdown ({output_style}) saved to: {md_file}")
            except Exception as e:
                # Fallback: keep legacy simple markdown (do not delete; keep for compatibility)
                md_file = output_dir / f"{SAMPLE_STORY['story_id']}.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Solution for {SAMPLE_STORY['title']}\n\n")
                    f.write(f"**Story ID:** {SAMPLE_STORY['story_id']}\n\n")
                    f.write("## Solution\n\n")
                    f.write(solution_narrative)
                    f.write("\n\n## Technical Details\n\n")
                    f.write("### Extracted Entities\n")
                    for entity_type, entities in technical_entities.items():
                        if not entities:
                            continue
                        if isinstance(entities, list):
                            f.write(f"- **{entity_type}**: {', '.join(str(x) for x in entities[:10])}\n")
                        elif isinstance(entities, dict):
                            # Write only first few keys
                            keys = list(entities.keys())[:10]
                            f.write(f"- **{entity_type}**: {', '.join(str(k) for k in keys)}\n")
                        else:
                            f.write(f"- **{entity_type}**: {str(entities)}\n")
                    f.write("\n### Fact Table\n")
                    f.write(f"- Forms: {fact_table.get('forms', [])}\n")
                    f.write(f"- DACs: {fact_table.get('dacs', [])}\n")
                    f.write(f"- Graphs: {fact_table.get('graphs', [])}\n")
                    f.write(f"- Fields: {fact_table.get('fields', [])}\n")

                print(f"📄 Markdown solution saved to: {md_file}")
                print(f"⚠️ Markdown formatting failed, used fallback. Error: {e}")
        
        return result
        
    except Exception as e:
        logger.error("Story processing failed", extra={"error": str(e)})
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_story_processing())
    
    if result:
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        sys.exit(1)


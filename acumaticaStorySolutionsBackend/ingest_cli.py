#!/usr/bin/env python3
"""
CLI: Ingest Acumatica manuals (PDF) into the local knowledge base.

Replaces the outdated README references to a missing ingest_cli.py.

Usage:
  python ingest_cli.py path/to/manual.pdf
  python ingest_cli.py path/to/folder/of/pdfs --recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure `src/` is importable when running as a script
PROJECT_ROOT = Path(__file__).parent
# We need PROJECT_ROOT on sys.path so `import src...` resolves to {PROJECT_ROOT}/src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.knowledge_ingestor import KnowledgeIngestor


def _iter_pdfs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(path.glob(pattern))
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF manuals into knowledge_base/manuals/")
    parser.add_argument("path", type=str, help="Path to a PDF file or a directory containing PDFs")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--name", type=str, default=None, help="Override document folder name (single PDF only)")
    parser.add_argument(
        "--suffix",
        type=str,
        default="__text",
        help="Suffix to append to output document name (default: __text). Use empty string to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing knowledge-base document folder (DANGEROUS).",
    )
    args = parser.parse_args()

    target = Path(args.path)
    pdfs = _iter_pdfs(target, recursive=args.recursive)
    if not pdfs:
        print(f"❌ No PDFs found at: {target}")
        return 2

    ingestor = KnowledgeIngestor()

    for i, pdf in enumerate(pdfs, start=1):
        base_name = args.name if (args.name and len(pdfs) == 1) else pdf.stem
        suffix = args.suffix or ""

        # Safety: never overwrite existing docs by default.
        # If the target exists, we auto-increment to keep everything preserved.
        if args.overwrite:
            doc_name = base_name
        else:
            doc_name = base_name
            if suffix and not doc_name.endswith(suffix):
                doc_name = f"{doc_name}{suffix}"

            # If still exists, increment: <name>__text2, __text3, ...
            kb_root = PROJECT_ROOT / "knowledge_base" / "manuals"
            candidate = doc_name
            n = 2
            while (kb_root / candidate).exists():
                candidate = f"{doc_name}{n}"
                n += 1
            doc_name = candidate

        print(f"\n[{i}/{len(pdfs)}] 📚 Ingesting: {pdf.name}")
        result = ingestor.ingest_pdf(str(pdf), document_name=doc_name, overwrite=args.overwrite)
        print(f"✅ Done: {result}")

    print("\n✅ All ingestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



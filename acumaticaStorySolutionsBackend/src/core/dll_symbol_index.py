#!/usr/bin/env python3
"""
DLL Symbol Index

Purpose:
- Convert decompiled/flattened DLL text content (from KB) into a deterministic symbol index.
- Provide fast, explainable lookups for:
  - Graphs / GraphExtensions
  - DACs / DACExtensions
  - BQL fields/constants
  - Event handler signatures (when present)

This avoids "LLM guessing" and lets the pipeline anchor billing stories to real classes/namespaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable, Tuple
import json
import re

from src.config.config import config
from src.utils.logger_utils import get_logger


@dataclass
class SymbolRecord:
    doc_name: str
    section_id: str
    namespace: str
    class_name: str
    full_name: str
    inherits: str = ""
    implements: str = ""
    raw: str = ""

    def is_graph(self) -> bool:
        return "PX.Data.PXGraph" in self.inherits

    def is_graph_extension(self) -> bool:
        return "PX.Data.PXGraphExtension" in self.inherits

    def is_dac(self) -> bool:
        return "PX.Data.PXBqlTable" in self.inherits

    def is_bql_field(self) -> bool:
        return "BqlType" in self.inherits and "+Field" in self.inherits


@dataclass
class DllSymbolIndex:
    """In-memory symbol index for one or more DLL docs."""

    records: List[SymbolRecord] = field(default_factory=list)
    by_full_name: Dict[str, SymbolRecord] = field(default_factory=dict)

    def add(self, rec: SymbolRecord) -> None:
        self.records.append(rec)
        if rec.full_name:
            self.by_full_name.setdefault(rec.full_name, rec)

    def search(self, keywords: List[str], *, limit: int = 25) -> List[SymbolRecord]:
        """Heuristic keyword search over full_name/class_name/namespace/raw."""
        if not keywords:
            return []
        kws = [k.lower() for k in keywords if k]
        scored: List[Tuple[int, SymbolRecord]] = []
        for r in self.records:
            hay = f"{r.full_name} {r.class_name} {r.namespace} {r.inherits} {r.raw}".lower()
            score = 0
            for kw in kws:
                if kw in hay:
                    score += 1
            if score:
                scored.append((score, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def find_graphs_related_to(self, term: str, *, limit: int = 10) -> List[SymbolRecord]:
        term = (term or "").strip().lower()
        if not term:
            return []
        matches = []
        for r in self.records:
            if not (r.is_graph() or r.is_graph_extension()):
                continue
            hay = f"{r.full_name} {r.class_name} {r.namespace}".lower()
            if term in hay:
                matches.append(r)
        return matches[:limit]

    def find_fields_like(self, term: str, *, limit: int = 25) -> List[SymbolRecord]:
        term = (term or "").strip().lower()
        if not term:
            return []
        matches = []
        for r in self.records:
            if not r.is_bql_field():
                continue
            hay = f"{r.full_name} {r.class_name}".lower()
            if term in hay:
                matches.append(r)
        return matches[:limit]


class DllSymbolIndexBuilder:
    """Build symbol indexes from KB DLL metadata (text-first)."""

    CLASS_RE = re.compile(r"\*\*Class:\s*(?P<class>.+?)\*\*", re.IGNORECASE)
    FULL_NAME_RE = re.compile(r"Full Name:\s*`(?P<full>[^`]+)`", re.IGNORECASE)
    INHERITS_RE = re.compile(r"Inherits from:\s*`(?P<inh>[^`]+)`", re.IGNORECASE)
    IMPLEMENTS_RE = re.compile(r"Implements:\s*`(?P<impl>[^`]+)`", re.IGNORECASE)
    NAMESPACE_RE = re.compile(r"###\s*Namespace:\s*(?P<ns>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

    def __init__(self):
        self.logger = get_logger("DLL_SYMBOL_INDEX")

    def build_for_documents(self, doc_names: List[str]) -> DllSymbolIndex:
        idx = DllSymbolIndex()
        for doc in doc_names:
            try:
                self._ingest_doc(idx, doc)
            except Exception as e:
                self.logger.warning("Failed to index DLL doc", extra={"doc": doc, "error": str(e)})
        self.logger.info("DLL symbol index built", extra={"docs": len(doc_names), "records": len(idx.records)})
        return idx

    def _ingest_doc(self, idx: DllSymbolIndex, doc_name: str) -> None:
        meta_path = Path(config.LOCAL_BASE_PATH) / doc_name / "metadata" / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json for {doc_name}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            return

        current_ns = ""
        for entry in metadata:
            if not isinstance(entry, dict):
                continue
            if entry.get("section_type") != "dll_text":
                continue
            text = entry.get("text_content") or ""
            if not text:
                continue

            # Track namespace headings when present
            ns_match = self.NAMESPACE_RE.search(text)
            if ns_match:
                current_ns = ns_match.group("ns").strip()

            class_match = self.CLASS_RE.search(text)
            full_match = self.FULL_NAME_RE.search(text)
            if not (class_match and full_match):
                continue

            inh = ""
            impl = ""
            inh_m = self.INHERITS_RE.search(text)
            if inh_m:
                inh = inh_m.group("inh").strip()
            impl_m = self.IMPLEMENTS_RE.search(text)
            if impl_m:
                impl = impl_m.group("impl").strip()

            full_name = full_match.group("full").strip()
            class_name = class_match.group("class").strip()
            namespace = current_ns or ".".join(full_name.split(".")[:-1])

            idx.add(
                SymbolRecord(
                    doc_name=doc_name,
                    section_id=str(entry.get("section_id") or ""),
                    namespace=namespace,
                    class_name=class_name,
                    full_name=full_name,
                    inherits=inh,
                    implements=impl,
                    raw=text,
                )
            )



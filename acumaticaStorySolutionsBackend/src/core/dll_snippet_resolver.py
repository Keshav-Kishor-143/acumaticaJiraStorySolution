#!/usr/bin/env python3
"""
DLL Snippet Resolver

Purpose:
- Extract "code-like" snippets from the KB DLL exports (knowledge_base/manuals/*_DLL/data/dll_content.txt)
- Provide deterministic, explainable snippets for the markdown solution output.

Important limitation:
- The current DLL ingestion is reflection/text based (signatures + metadata). It does NOT include method bodies/IL.
  So "snippets" are primarily:
  - Full method signatures
  - Class full names + inheritance
  - BQL field type definitions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple
import re

from src.config.config import config
from src.utils.logger_utils import get_logger
from src.core.dll_symbol_index import DllSymbolIndexBuilder, SymbolRecord
from src.core.dll_reflection_index import DllReflectionIndexBuilder


@dataclass(frozen=True)
class DllSnippet:
    doc_name: str
    start_line: int
    end_line: int
    snippet: str
    kind: str  # "method_signature" | "class_record"
    score: int = 0


class DllSnippetResolver:
    """
    Deterministic resolver for DLL snippets.

    Strategy:
    1) Prefer symbol records (class blocks) via `DllSymbolIndexBuilder` where possible.
    2) Also scan `dll_content.txt` for method signatures (`public ...`) and extract nearby lines.
    """

    _METHOD_LINE_RE = re.compile(r"`\s*public\s+.+?`", re.IGNORECASE)

    def __init__(self):
        self.logger = get_logger("DLL_SNIPPET_RESOLVER")
        self._doc_lines_cache: Dict[str, List[str]] = {}
        self._symbol_index_builder = DllSymbolIndexBuilder()

    def list_dll_documents(self) -> List[str]:
        base = Path(config.LOCAL_BASE_PATH)
        if not base.exists():
            return []
        docs = []
        for p in base.iterdir():
            if p.is_dir() and p.name.endswith("_DLL"):
                docs.append(p.name)
        return sorted(docs)

    def _load_doc_lines(self, doc_name: str) -> List[str]:
        if doc_name in self._doc_lines_cache:
            return self._doc_lines_cache[doc_name]
        txt_path = Path(config.LOCAL_BASE_PATH) / doc_name / "data" / "dll_content.txt"
        if not txt_path.exists():
            self._doc_lines_cache[doc_name] = []
            return []
        lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        self._doc_lines_cache[doc_name] = lines
        return lines

    @staticmethod
    def _normalize_keywords(keywords: Iterable[str]) -> List[str]:
        out = []
        for k in keywords:
            kk = (k or "").strip()
            if not kk:
                continue
            out.append(kk.lower())
        # Deduplicate while preserving order
        return list(dict.fromkeys(out))

    def find_method_signature_snippets(
        self,
        *,
        keywords: List[str],
        doc_names: Optional[List[str]] = None,
        limit: int = 8,
        context_lines: int = 0,
    ) -> List[DllSnippet]:
        kws = self._normalize_keywords(keywords)
        if not kws:
            return []
        docs = doc_names or self.list_dll_documents()

        matches: List[Tuple[int, DllSnippet]] = []
        for doc in docs:
            lines = self._load_doc_lines(doc)
            if not lines:
                continue
            for i, line in enumerate(lines):
                hay = line.lower()
                if not any(kw in hay for kw in kws):
                    continue
                # Only treat as a "code snippet" if it looks like a method signature line
                if "public" not in hay:
                    continue
                if not self._METHOD_LINE_RE.search(line):
                    continue

                start = max(0, i - context_lines)
                end = min(len(lines) - 1, i + context_lines)
                snippet_text = "\n".join(lines[start : end + 1]).strip()
                score = sum(1 for kw in kws if kw in hay)

                matches.append((
                    score,
                    DllSnippet(
                        doc_name=doc,
                        start_line=start + 1,
                        end_line=end + 1,
                        snippet=snippet_text,
                        kind="method_signature",
                        score=score,
                    ),
                ))

        matches.sort(key=lambda t: t[0], reverse=True)
        # Deduplicate by (doc, snippet)
        seen = set()
        out: List[DllSnippet] = []
        for _, snip in matches:
            key = (snip.doc_name, snip.snippet)
            if key in seen:
                continue
            seen.add(key)
            out.append(snip)
            if len(out) >= limit:
                break
        return out

    def find_class_record_snippets(
        self,
        *,
        keywords: List[str],
        doc_names: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[DllSnippet]:
        kws = self._normalize_keywords(keywords)
        if not kws:
            return []
        docs = doc_names or self.list_dll_documents()

        idx = self._symbol_index_builder.build_for_documents(docs)
        recs: List[SymbolRecord] = idx.search(kws, limit=limit * 3)

        out: List[DllSnippet] = []
        for r in recs:
            # Score based on keyword overlap
            hay = f"{r.full_name} {r.class_name} {r.namespace} {r.inherits} {r.raw}".lower()
            score = sum(1 for kw in kws if kw in hay)
            raw = (r.raw or "").strip()
            if not raw:
                continue
            out.append(
                DllSnippet(
                    doc_name=r.doc_name,
                    start_line=0,
                    end_line=0,
                    snippet=raw,
                    kind="class_record",
                    score=score,
                )
            )
            if len(out) >= limit:
                break

        # Fallback: if symbol index didn't find anything useful, use reflection index (more complete)
        if not out:
            try:
                ridx = DllReflectionIndexBuilder().load_or_build(force_rebuild=False)
                # Map namespace prefix -> doc folder name (best-effort)
                def _doc_for_full_name(full: str) -> str:
                    if not full:
                        return ""
                    parts = full.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}_DLL"
                    return f"{parts[0]}_DLL" if parts else ""

                matches: List[Tuple[int, DllSnippet]] = []
                for r in ridx.records:
                    full = (r.full_name or "")
                    hay = f"{r.full_name} {r.name} {r.cache_name} {r.base_type}".lower()
                    score = sum(1 for kw in kws if kw in hay)
                    if score <= 0:
                        continue
                    snippet = "\n".join([
                        f"**Class: {r.name}**",
                        f"- Full Name: `{r.full_name}`",
                        f"- Inherits from: `{r.base_type}`" if r.base_type else "- Inherits from: (unknown)",
                        f"- Cache Name: `{r.cache_name}`" if r.cache_name else "- Cache Name: (none)",
                        f"- Primary Graphs: {', '.join(f'`{g}`' for g in (r.primary_graphs or [])[:3])}" if r.primary_graphs else "- Primary Graphs: (none)",
                    ]).strip()
                    matches.append((
                        score,
                        DllSnippet(
                            doc_name=_doc_for_full_name(full),
                            start_line=0,
                            end_line=0,
                            snippet=snippet,
                            kind="class_record",
                            score=score,
                        ),
                    ))
                matches.sort(key=lambda t: t[0], reverse=True)
                seen = set()
                for _, snip in matches:
                    key = (snip.doc_name, snip.snippet)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(snip)
                    if len(out) >= limit:
                        break
            except Exception:
                pass
        return out

    def resolve_best_snippets(
        self,
        *,
        keywords: List[str],
        doc_names: Optional[List[str]] = None,
        method_limit: int = 8,
        class_limit: int = 3,
    ) -> List[DllSnippet]:
        """
        Return a combined list of best snippets (classes + method signatures).
        """
        classes = self.find_class_record_snippets(
            keywords=keywords, doc_names=doc_names, limit=class_limit
        )
        methods = self.find_method_signature_snippets(
            keywords=keywords, doc_names=doc_names, limit=method_limit
        )
        return classes + methods



#!/usr/bin/env python3
"""
Dynamic Entity Resolver (Graph/DAC inference)

Goal:
- Infer the best-matching DAC + Graph for a story without hardcoding.
- Use deterministic DLL reflection index (primary graph, cache name, fields/properties)
- Optionally return multiple candidates with scores.

This is designed for "template filling" use-cases where we need a compile-ready
extension class target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

from src.utils.logger_utils import get_logger
from src.core.dll_reflection_index import DllReflectionIndexBuilder, TypeRecord


@dataclass(frozen=True)
class ResolvedTarget:
    dac_full_name: str
    graph_full_name: str
    dac_name: str
    graph_name: str
    dac_namespace: str
    graph_namespace: str
    confidence: float
    reasoning: str


class DynamicEntityResolver:
    def __init__(self):
        self.logger = get_logger("DYNAMIC_ENTITY_RESOLVER")
        self._idx = None
        self._builder = DllReflectionIndexBuilder()

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        """
        Tokenize with light camel-case splitting so "ReturnTicket" matches "return" and "ticket".
        """
        raw = (s or "").strip()
        if not raw:
            return []
        # Insert spaces on camel-case boundaries before lowercasing
        raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
        tokens = re.split(r"[^A-Za-z0-9]+", raw.lower())
        return [t for t in tokens if t and len(t) > 1]

    def _get_index(self):
        if self._idx is None:
            self._idx = self._builder.load_or_build(force_rebuild=False)
        return self._idx

    def resolve_graph_and_dac(
        self,
        *,
        story_text: str,
        field_names: List[str],
        prefer_assemblies: Optional[List[str]] = None,
        min_confidence: float = 0.55,
    ) -> Optional[ResolvedTarget]:
        """
        Returns best ResolvedTarget or None if confidence too low.
        """
        idx = self._get_index()
        text_tokens = self._tokenize(story_text)
        norm_story = self._norm(story_text)
        norm_fields = [self._norm(f) for f in (field_names or []) if f]

        # Intent hints to prevent cross-domain false positives
        intent_is_rental = any(k in text_tokens for k in ["rental", "return", "equipment", "fee", "damage", "damaged"])
        banned_tokens = {"changeorder", "change", "pmchangeorder", "project", "construction"} if intent_is_rental else set()

        # Candidate DACs include:
        # - dac (PXBqlTable)
        # - dac_extension (PXCacheExtension<>) where we can map to base DAC
        dacs: List[TypeRecord] = [r for r in idx.records if r.kind in ("dac", "dac_extension")]

        if prefer_assemblies:
            pref = set(prefer_assemblies)
            dacs = sorted(dacs, key=lambda r: (0 if r.assembly in pref else 1, r.full_name))

        scored: List[Tuple[float, TypeRecord, str]] = []

        for r in dacs:
            # Compare story to cache name (high signal when present)
            cache = r.cache_name or ""
            cache_tokens = self._tokenize(cache)
            cache_overlap = len(set(cache_tokens) & set(text_tokens))
            cache_score = 0.0
            if cache_tokens:
                cache_score = cache_overlap / max(len(set(cache_tokens)), 1)

            # Field match using properties or nested BQL field type names.
            prop_norms = {self._norm(p) for p in (r.properties or [])}
            nested_norms = {self._norm(n) for n in (r.nested_types or [])}

            field_hits = 0
            for nf in norm_fields:
                if not nf:
                    continue
                if nf in prop_norms or nf in nested_norms:
                    field_hits += 1

            field_score = 0.0
            if norm_fields:
                field_score = field_hits / max(len(set(norm_fields)), 1)

            # Lightweight name match: class name contains “event/order/quote”
            name_hay = self._norm(r.full_name)
            name_score = 0.0
            for kw in ("event", "order", "quote", "extension", "rental", "return", "equipment", "fee", "damage"):
                if kw in norm_story and kw in name_hay:
                    name_score += 0.10
            name_score = min(name_score, 0.30)

            # Hard guardrail: if rental intent, require at least some evidence (cache overlap or field hit)
            if intent_is_rental and cache_overlap == 0 and field_hits == 0:
                continue

            # Penalize clearly unrelated domains when rental intent is present
            if intent_is_rental:
                # Penalize namespaces/modules like ChangeOrders
                if any(bt in (r.full_name or "").lower() for bt in banned_tokens):
                    continue

            # Weighted sum
            score = (cache_score * 0.45) + (field_score * 0.45) + name_score

            if score > 0.05:
                reason = f"cache_score={cache_score:.2f}, field_score={field_score:.2f}, name_score={name_score:.2f}"
                scored.append((score, r, reason))

        scored.sort(key=lambda t: t[0], reverse=True)
        if not scored:
            return None

        best_score, best_rec, reason = scored[0]
        self.logger.info(
            "Best DAC candidate (pre-threshold)",
            extra={
                "candidate": best_rec.full_name,
                "assembly": best_rec.assembly,
                "kind": best_rec.kind,
                "cache_name": best_rec.cache_name,
                "primary_graphs": best_rec.primary_graphs[:3],
                "score": best_score,
                "reason": reason,
            },
        )

        # Resolve to base DAC if extension
        dac_full = best_rec.full_name
        if best_rec.kind == "dac_extension" and best_rec.dac_of_extension:
            dac_full = best_rec.dac_of_extension

        # Resolve graph:
        graph_full = ""
        # Primary graphs are best truth
        if best_rec.primary_graphs:
            graph_full = best_rec.primary_graphs[0]

        # If no primary graph, pick a graph by naming convention (Maint/Entry)
        if not graph_full:
            dac_short = dac_full.split(".")[-1]
            candidates = []
            for r in idx.records:
                if r.kind != "graph":
                    continue
                cn = r.name.lower()
                if dac_short.lower() in cn and (cn.endswith("maint") or cn.endswith("entry") or "maint" in cn or "entry" in cn):
                    candidates.append(r)
            if candidates:
                graph_full = candidates[0].full_name

        # If still empty, bail
        if not graph_full:
            if best_score < min_confidence:
                return None
            # Provide best-effort placeholder graph
            graph_full = "PX.Objects.PXGraph"

        dac_name = dac_full.split(".")[-1]
        dac_ns = ".".join(dac_full.split(".")[:-1])
        graph_name = graph_full.split(".")[-1]
        graph_ns = ".".join(graph_full.split(".")[:-1])

        confidence = min(0.99, max(0.0, best_score))
        if confidence < min_confidence:
            return None

        reasoning = f"Best DAC match: {best_rec.full_name} ({reason}). Graph: {graph_full or '[unresolved]'}"

        return ResolvedTarget(
            dac_full_name=dac_full,
            graph_full_name=graph_full,
            dac_name=dac_name,
            graph_name=graph_name,
            dac_namespace=dac_ns,
            graph_namespace=graph_ns,
            confidence=confidence,
            reasoning=reasoning,
        )



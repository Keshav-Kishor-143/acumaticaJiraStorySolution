import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger_utils import get_logger
from src.core.dll_reflection_index import DllReflectionIndexBuilder, TypeRecord


@dataclass(frozen=True)
class SemanticBusinessObject:
    """
    A best-effort semantic binding between business language and implementation objects.

    NOTE: This is derived dynamically from KB materials (DLL reflection + cache_name/primary_graphs).
    It is not a static/customer-specific mapping file authored by a human.
    """
    concept: str
    module_tag: str
    dac_full_name: str
    graph_full_name: str
    line_dac_full_name: str = ""
    confidence: float = 0.0
    phrases: List[str] = None
    evidence: List[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["evidence"] = self.evidence or []
        d["phrases"] = self.phrases or []
        return d


class SemanticObjectIndex:
    def __init__(self, objects: List[SemanticBusinessObject]):
        self.objects = objects or []

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        raw = (s or "").strip()
        if not raw:
            return []
        raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
        tokens = re.split(r"[^A-Za-z0-9]+", raw.lower())
        return [t for t in tokens if t and len(t) > 1]

    def resolve(self, query: str, field_names: Optional[List[str]] = None) -> Optional[Tuple[SemanticBusinessObject, float]]:
        """
        Resolve a story/query to a best matching business object using token overlap.
        Returns (object, score) or None.
        """
        q_tokens = set(self._tokenize(query))
        f_tokens = set(self._tokenize(" ".join(field_names or [])))
        if not q_tokens and not f_tokens:
            return None

        has_return = "return" in q_tokens
        has_rental = "rental" in q_tokens
        has_document = any(t in q_tokens for t in ["document", "order", "ticket", "entry", "release", "save"])

        best: Optional[Tuple[SemanticBusinessObject, float]] = None
        for obj in self.objects:
            phrase_blob = " ".join([obj.concept] + (obj.phrases or []))
            c_tokens = set(self._tokenize(phrase_blob))
            g_tokens = set(self._tokenize(obj.graph_full_name))
            d_tokens = set(self._tokenize(obj.dac_full_name))

            # Concept overlap is primary; graph/dac tokens help as secondary evidence.
            c_overlap = len(q_tokens & c_tokens)
            g_overlap = len(q_tokens & g_tokens)
            d_overlap = len(q_tokens & d_tokens)
            f_overlap = len(f_tokens & (d_tokens | g_tokens | c_tokens))

            score = 0.0
            if c_tokens:
                score += (c_overlap / max(len(c_tokens), 1)) * 0.55
            score += (g_overlap / max(len(g_tokens), 1)) * 0.20 if g_tokens else 0.0
            score += (d_overlap / max(len(d_tokens), 1)) * 0.15 if d_tokens else 0.0
            score += min(0.10, f_overlap * 0.03)
            
            # Boost score for business_objects.json entries (manually curated)
            if any("business_objects.json" in ev for ev in (obj.evidence or [])):
                score += 0.15  # Significant boost for curated entries

            # Module hint boosts (rental/return stories tend to live in NV.Rental360)
            q_is_rental = any(t in q_tokens for t in ["rental", "return", "equipment", "damage", "fee", "ticket"])
            if q_is_rental and "rental360" in obj.module_tag.lower():
                score += 0.12

            # Strong intent boosts/penalties for document-like concepts:
            # If story mentions "return" we should prefer concepts/graphs/dacs that also include "return".
            if has_return:
                if "return" in c_tokens:
                    score += 0.15
                elif "return" in (g_tokens | d_tokens):
                    score += 0.05
                else:
                    score -= 0.10

            # If story explicitly looks like a document workflow (save/release/entry),
            # de-prioritize master-data concepts that don't include return/rental.
            if has_document:
                concept_lc = (obj.concept or "").lower()
                if concept_lc in {"equipment", "equipment service", "equipment system codes"} and not ("return" in c_tokens or "rental" in c_tokens):
                    score -= 0.08

            # If the story says "rental return", ensure we prefer objects that cover both words.
            if has_rental and has_return:
                if "rental" in c_tokens and "return" in c_tokens:
                    score += 0.10

            if best is None or score > best[1]:
                best = (obj, score)

        return best


class SemanticObjectIndexBuilder:
    """
    Builds SemanticObjectIndex from:
    1. DLL reflection metadata (auto-generated)
    2. business_objects.json (manual + auto-generated entries)
    """
    """
    Build a semantic index dynamically from DLL reflection index.
    The output is cached under knowledge_base/manuals/_semantic_object_index.json.
    """

    def __init__(self):
        self.logger = get_logger("SEMANTIC_OBJECT_INDEX")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.kb_dir = self.project_root / "knowledge_base" / "manuals"
        self.cache_path = self.kb_dir / "_semantic_object_index.json"
        self.business_objects_path = self.kb_dir / "business_objects.json"

    @staticmethod
    def _parse_generic_type_names(base_type: str) -> List[str]:
        """
        Extract inner type full names from strings like:
        PX.Data.PXGraph`2[[Graph, NV.Rental360,...],[DAC, NV.Rental360,...]]
        """
        if not base_type:
            return []
        # Capture "Namespace.Type, Assembly" blocks inside [[...]]
        matches = re.findall(r"\[\[([A-Za-z0-9_.`+]+),\s*[A-Za-z0-9_.]+\s*,", base_type)
        # Deduplicate while preserving order
        out: List[str] = []
        seen = set()
        for m in matches:
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out

    @staticmethod
    def _module_tag(full_name: str) -> str:
        fn = full_name or ""
        # Use first 2 namespace segments as a module tag (e.g., NV.Rental360)
        parts = fn.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")

    def _guess_line_dac(self, dac: TypeRecord, all_dacs: List[TypeRecord]) -> str:
        """
        Heuristic: find a sibling DAC in same module/namespace with *Tran/*Line naming.
        """
        if not dac or not dac.full_name:
            return ""
        mod = self._module_tag(dac.full_name)
        base_ns = ".".join(dac.full_name.split(".")[:-1])
        candidates: List[str] = []
        for d in all_dacs:
            if d.kind != "dac":
                continue
            if not d.full_name:
                continue
            # Avoid choosing extension/helper DACs as "line" DACs
            if "extension" in d.full_name.lower():
                continue
            if self._module_tag(d.full_name) != mod:
                continue
            if not d.full_name.startswith(base_ns):
                continue
            nm = (d.name or "").lower()
            if nm.endswith("tran") or nm.endswith("line") or "detail" in nm:
                candidates.append(d.full_name)
        # Prefer same prefix + "Tran"
        dac_name = (dac.name or "")
        prefer = [c for c in candidates if (dac_name and dac_name.lower() in c.lower() and c.lower().endswith("tran"))]
        return (prefer[0] if prefer else (candidates[0] if candidates else ""))

    def build(self) -> SemanticObjectIndex:
        idx = DllReflectionIndexBuilder().load_or_build(force_rebuild=False)
        dacs = [r for r in idx.records if r.kind == "dac" and (r.cache_name or "").strip()]
        all_dacs = [r for r in idx.records if r.kind == "dac"]
        graphs = [r for r in idx.records if r.kind == "graph"]

        graph_by_full: Dict[str, TypeRecord] = {g.full_name: g for g in graphs if g.full_name}

        objects: List[SemanticBusinessObject] = []

        for dac in dacs:
            concept = (dac.cache_name or "").strip()
            if not concept:
                continue

            module_tag = self._module_tag(dac.full_name)
            evidence: List[str] = [f"dac.cache_name={concept}"]
            phrases: List[str] = []

            # Heuristic semantic aliases (dynamic) to bridge common business phrasing:
            # - "ReturnTicket" frequently represents a "Rental Return" document in rental modules.
            # - "RentalTicket" frequently represents "Rental Order/Ticket".
            cn = concept.lower().replace(" ", "")
            dn = (dac.name or "").lower()
            if ("return" in cn and "ticket" in cn) or ("return" in dn and "ticket" in dn):
                phrases.extend(["return ticket", "rental return", "rental return ticket", "return document"])
                evidence.append("alias:return_ticket->rental_return")
            if ("rental" in cn and "ticket" in cn) or ("rental" in dn and "ticket" in dn):
                phrases.extend(["rental ticket", "rental order", "rental document"])
                evidence.append("alias:rental_ticket->rental_order")

            # Prefer primary graphs if available (PXPrimaryGraphAttribute extraction)
            graph_full = ""
            if dac.primary_graphs:
                graph_full = dac.primary_graphs[0]
                evidence.append("dac.primary_graphs[0]")

            # Otherwise, try to infer a graph by scanning graphs that reference this DAC in PXGraph`2 base type.
            if not graph_full:
                for g in graphs:
                    base = g.base_type or ""
                    inners = self._parse_generic_type_names(base)
                    # For PXGraph`2 the 2nd generic is usually the primary DAC
                    if len(inners) >= 2 and inners[1] == dac.full_name:
                        graph_full = g.full_name
                        evidence.append("graph.base_type PXGraph`2 match")
                        break

            if not graph_full:
                # Can't bind to a graph, keep it out (otherwise it becomes noisy)
                continue

            # Try to infer a likely line DAC
            line_dac = self._guess_line_dac(dac, all_dacs)
            if line_dac:
                evidence.append("line_dac heuristic")

            # Confidence: graph binding + cache_name are decent; line DAC optional
            conf = 0.60
            if "primary_graphs" in " ".join(evidence):
                conf += 0.10
            if line_dac:
                conf += 0.05
            conf = min(conf, 0.85)

            objects.append(SemanticBusinessObject(
                concept=concept,
                phrases=phrases,
                module_tag=module_tag,
                dac_full_name=dac.full_name,
                graph_full_name=graph_full,
                line_dac_full_name=line_dac,
                confidence=conf,
                evidence=evidence,
            ))

        return SemanticObjectIndex(objects)
    
    def _load_from_business_objects(self) -> List[SemanticBusinessObject]:
        """Load semantic objects from business_objects.json"""
        objects = []
        
        if not self.business_objects_path.exists():
            return objects
        
        try:
            with open(self.business_objects_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for obj_data in data.get("objects", []):
                # Convert business_objects.json format to SemanticBusinessObject
                concept = obj_data.get("concept", "")
                if not concept:
                    continue
                
                # Build phrases from aliases
                phrases = obj_data.get("aliases", [])
                phrases.extend(obj_data.get("retrieval_tags", []))
                
                # Build evidence
                evidence = [
                    f"business_objects.json: {concept}",
                    f"module: {obj_data.get('module', '')}",
                    f"type: {obj_data.get('type', '')}",
                    f"priority: {obj_data.get('priority', 0)}"
                ]
                
                objects.append(SemanticBusinessObject(
                    concept=concept,
                    phrases=phrases,
                    module_tag=obj_data.get("module", ""),
                    dac_full_name=obj_data.get("primary_dac", ""),
                    graph_full_name=obj_data.get("primary_graph", ""),
                    line_dac_full_name=obj_data.get("line_dac", ""),
                    confidence=0.85,  # Higher confidence for manually curated entries
                    evidence=evidence,
                ))
            
            self.logger.info(f"Loaded {len(objects)} objects from business_objects.json")
            
        except Exception as e:
            self.logger.warning(f"Error loading business_objects.json: {e}")
        
        return objects
    
    def build_with_business_objects(self) -> SemanticObjectIndex:
        """Build index from both DLL metadata and business_objects.json"""
        # Build from DLL (existing logic)
        dll_index = self.build()
        dll_objects = dll_index.objects
        
        # Load from business_objects.json
        business_objects = self._load_from_business_objects()
        
        # Merge: business_objects.json entries take precedence (higher confidence)
        # Create lookup by concept
        merged_objects = []
        concept_map = {}
        
        # Add DLL objects first
        for obj in dll_objects:
            concept_key = obj.concept.lower()
            if concept_key not in concept_map:
                merged_objects.append(obj)
                concept_map[concept_key] = obj
        
        # Add business_objects.json entries (override if exists)
        for obj in business_objects:
            concept_key = obj.concept.lower()
            if concept_key in concept_map:
                # Replace existing entry
                idx = merged_objects.index(concept_map[concept_key])
                merged_objects[idx] = obj
            else:
                merged_objects.append(obj)
                concept_map[concept_key] = obj
        
        self.logger.info(f"Merged index: {len(dll_objects)} DLL + {len(business_objects)} business_objects = {len(merged_objects)} total")
        
        return SemanticObjectIndex(merged_objects)

    def load_or_build(self, force_rebuild: bool = False) -> SemanticObjectIndex:
        # Always merge with business_objects.json, even when loading from cache
        # This ensures manually curated entries are always included
        cache_objects = []
        if self.cache_path.exists() and not force_rebuild:
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for o in data.get("objects", []):
                    # Backward compatible with older cache versions
                    if "phrases" not in o:
                        o["phrases"] = []
                    cache_objects.append(SemanticBusinessObject(**o))
            except Exception:
                pass
        
        # Load business_objects.json entries
        business_objects = self._load_from_business_objects()
        
        if cache_objects and not force_rebuild:
            # Merge cache + business_objects.json (business_objects take precedence)
            merged_objects = []
            concept_map = {}
            
            # Add cache objects first
            for obj in cache_objects:
                concept_key = obj.concept.lower()
                if concept_key not in concept_map:
                    merged_objects.append(obj)
                    concept_map[concept_key] = obj
            
            # Add business_objects.json entries (override if exists)
            for obj in business_objects:
                concept_key = obj.concept.lower()
                if concept_key in concept_map:
                    # Replace existing entry
                    idx = merged_objects.index(concept_map[concept_key])
                    merged_objects[idx] = obj
                else:
                    merged_objects.append(obj)
                    concept_map[concept_key] = obj
            
            self.logger.info(f"Loaded from cache + business_objects: {len(cache_objects)} cached + {len(business_objects)} business_objects = {len(merged_objects)} total")
            return SemanticObjectIndex(merged_objects)

        # Use merged build that includes business_objects.json
        index = self.build_with_business_objects()
        try:
            payload = {
                "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "objects": [o.to_dict() for o in index.objects],
            }
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # Cache failure is non-fatal
            pass
        return index



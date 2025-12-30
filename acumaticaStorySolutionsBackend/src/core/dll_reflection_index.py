#!/usr/bin/env python3
"""
DLL Reflection Index (High-Truth)

Builds a deterministic index from the actual DLL binaries stored in:
- knowledge_base/manuals/*_DLL/data/*.dll
- knowledge_base/Dll's/*.dll (dependencies)

Why:
- The existing dll_content.txt exports only include attribute *names* (e.g. PXPrimaryGraphAttribute),
  not attribute *arguments* (e.g. typeof(MyGraph)).
- For dynamic inference of Graph/DAC (screen business objects) we need:
  - PXCacheNameAttribute.DisplayName (human name, often matches screen/business terms)
  - PXPrimaryGraphAttribute.Type (DAC -> Primary Graph mapping)
  - PXCacheExtension<TDac> (DAC extension -> base DAC mapping)
  - PXGraph / PXGraphExtension (graph discovery)

Output:
- A cached JSON file under knowledge_base/manuals/_dll_reflection_index.json
  to avoid re-loading large assemblies per request.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import hashlib

from src.config.config import config
from src.utils.logger_utils import get_logger


@dataclass
class TypeRecord:
    assembly: str
    full_name: str
    namespace: str
    name: str
    base_type: str = ""
    kind: str = ""  # "dac" | "dac_extension" | "graph" | "graph_extension" | ""
    dac_of_extension: str = ""  # full name of base DAC if kind == dac_extension
    cache_name: str = ""  # PXCacheNameAttribute.DisplayName (if accessible)
    primary_graphs: List[str] = None  # list of full type names
    properties: List[str] = None  # property names
    nested_types: List[str] = None  # nested type names

    def __post_init__(self):
        if self.primary_graphs is None:
            self.primary_graphs = []
        if self.properties is None:
            self.properties = []
        if self.nested_types is None:
            self.nested_types = []


class DllReflectionIndex:
    def __init__(self):
        self.records: List[TypeRecord] = []
        self.by_full_name: Dict[str, TypeRecord] = {}

    def add(self, rec: TypeRecord) -> None:
        self.records.append(rec)
        if rec.full_name and rec.full_name not in self.by_full_name:
            self.by_full_name[rec.full_name] = rec


class DllReflectionIndexBuilder:
    CACHE_FILENAME = "_dll_reflection_index.json"

    def __init__(self):
        self.logger = get_logger("DLL_REFLECTION_INDEX")

    def _cache_path(self) -> Path:
        return Path(config.LOCAL_BASE_PATH) / self.CACHE_FILENAME

    def _hash_file(self, path: Path, *, max_mb: int = 5) -> str:
        """
        Hash the first N MB for a stable cache key without reading huge files fully.
        """
        h = hashlib.sha256()
        if not path.exists():
            return ""
        max_bytes = max_mb * 1024 * 1024
        with path.open("rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()

    def _discover_dll_paths(self) -> List[Path]:
        base = Path(config.LOCAL_BASE_PATH)
        dlls: List[Path] = []
        if base.exists():
            for doc_dir in base.iterdir():
                if not doc_dir.is_dir():
                    continue
                if not doc_dir.name.endswith("_DLL"):
                    continue
                data_dir = doc_dir / "data"
                if not data_dir.exists():
                    continue
                for dll in data_dir.glob("*.dll"):
                    # Skip framework DLLs to reduce noise (still available as dependencies via resolver)
                    if dll.stem.lower() in {"system"}:
                        continue
                    dlls.append(dll)
        # Prefer stable order
        return sorted(dlls, key=lambda p: p.name.lower())

    def _build_cache_signature(self, dll_paths: List[Path]) -> Dict[str, Any]:
        # Minimal signature: file names + partial hashes
        items = []
        for p in dll_paths:
            items.append({"name": p.name, "size": p.stat().st_size if p.exists() else 0, "hash": self._hash_file(p)})
        return {"dlls": items}

    def load_or_build(self, *, force_rebuild: bool = False) -> DllReflectionIndex:
        dll_paths = self._discover_dll_paths()
        cache_path = self._cache_path()
        signature = self._build_cache_signature(dll_paths)

        if not force_rebuild and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("signature") == signature and isinstance(cached.get("records"), list):
                    idx = DllReflectionIndex()
                    for r in cached["records"]:
                        try:
                            rec = TypeRecord(**r)
                            idx.add(rec)
                        except Exception:
                            continue
                    self.logger.info("Loaded DLL reflection index from cache", extra={"records": len(idx.records)})
                    return idx
            except Exception as e:
                self.logger.warning("Failed to load DLL reflection cache", extra={"error": str(e)})

        idx = self.build(dll_paths)
        try:
            payload = {
                "generated_at": time.time(),
                "signature": signature,
                "records": [asdict(r) for r in idx.records],
            }
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info("Wrote DLL reflection index cache", extra={"path": str(cache_path), "records": len(idx.records)})
        except Exception as e:
            self.logger.warning("Failed to write DLL reflection cache", extra={"error": str(e)})
        return idx

    def build(self, dll_paths: List[Path]) -> DllReflectionIndex:
        """
        Reflection build using pythonnet.
        """
        idx = DllReflectionIndex()
        if not dll_paths:
            return idx

        # Late import pythonnet to keep module import cheap.
        import clr  # type: ignore
        clr.AddReference("System")  # type: ignore
        import System  # type: ignore
        from System.Reflection import Assembly  # type: ignore
        from System import AppDomain  # type: ignore

        # Build a resolver search path list (manuals + knowledge_base/Dll's)
        manual_base = Path(config.LOCAL_BASE_PATH)
        # Prefer dependency folder adjacent to manuals (workspace/knowledge_base/Dll's)
        dep_dir = manual_base.parent / "Dll's"
        search_dirs: List[Path] = []
        if dep_dir.exists():
            search_dirs.append(dep_dir)
        # Add each *_DLL/data directory
        for p in dll_paths:
            search_dirs.append(p.parent)

        # Assembly resolve handler to load dependencies from known dirs
        def _resolve(sender, args):  # noqa: ANN001
            try:
                name = str(args.Name).split(",")[0]  # AssemblyName
                for d in search_dirs:
                    cand = d / f"{name}.dll"
                    if cand.exists():
                        # Load from bytes to avoid locking and path issues
                        dll_bytes = cand.read_bytes()
                        net_bytes = System.Array[System.Byte](dll_bytes)
                        return Assembly.Load(net_bytes)
            except Exception:
                return None
            return None

        domain = AppDomain.CurrentDomain
        domain.AssemblyResolve += _resolve

        try:
            for dll_path in dll_paths:
                try:
                    asm_name = dll_path.stem
                    dll_bytes = dll_path.read_bytes()
                    net_bytes = System.Array[System.Byte](dll_bytes)
                    asm = Assembly.Load(net_bytes)

                    # Enumerate types
                    try:
                        types = list(asm.GetTypes())
                    except Exception as e:
                        # Handle ReflectionTypeLoadException-style partial loads
                        types = []
                        try:
                            partial = getattr(e, "Types", None)
                            if partial is not None:
                                types = [t for t in list(partial) if t is not None]
                        except Exception:
                            types = []
                        if not types:
                            continue

                    for t in types:
                        try:
                            rec = self._extract_type_record(asm_name, t)
                            if rec:
                                idx.add(rec)
                        except Exception:
                            continue

                except Exception as e:
                    self.logger.warning("Failed to load DLL for reflection index", extra={"dll": dll_path.name, "error": str(e)})
                    continue
        finally:
            # Detach handler to avoid leaking
            try:
                domain.AssemblyResolve -= _resolve
            except Exception:
                pass

        self.logger.info("Built DLL reflection index", extra={"records": len(idx.records), "dlls": len(dll_paths)})
        return idx

    def _extract_type_record(self, asm_name: str, t) -> Optional[TypeRecord]:  # noqa: ANN001
        full_name = str(getattr(t, "FullName", "") or "")
        name = str(getattr(t, "Name", "") or "")
        namespace = str(getattr(t, "Namespace", "") or "")
        base_type = ""
        try:
            bt = getattr(t, "BaseType", None)
            base_type = str(bt.FullName) if bt is not None else ""
        except Exception:
            base_type = ""

        if not full_name or full_name.startswith("<") or full_name.startswith("__StaticArrayInitTypeSize"):
            return None

        # Determine kind based on inheritance patterns
        kind = ""
        dac_of_extension = ""

        # Helper: walk base types for string contains
        def _base_chain_contains(token: str) -> bool:
            try:
                cur = t
                for _ in range(12):
                    bt = getattr(cur, "BaseType", None)
                    if bt is None:
                        return False
                    bfn = str(bt.FullName or "")
                    if token in bfn:
                        return True
                    cur = bt
            except Exception:
                return False
            return False

        if _base_chain_contains("PX.Data.PXBqlTable"):
            kind = "dac"
        elif "PX.Data.PXGraphExtension" in base_type:
            kind = "graph_extension"
        elif "PX.Data.PXGraph" in base_type:
            kind = "graph"
        elif "PX.Data.PXCacheExtension" in base_type:
            kind = "dac_extension"
            # Try to extract generic arg (TDac)
            try:
                bt = getattr(t, "BaseType", None)
                if bt is not None and bt.IsGenericType:
                    args = list(bt.GetGenericArguments())
                    if args:
                        dac_of_extension = str(args[0].FullName or "")
            except Exception:
                pass

        # Extract properties and nested types (for field matching)
        props: List[str] = []
        nested: List[str] = []
        try:
            for p in list(t.GetProperties()):
                pn = str(p.Name or "")
                if pn:
                    props.append(pn)
        except Exception:
            pass
        try:
            for nt in list(t.GetNestedTypes()):
                nn = str(nt.Name or "")
                if nn:
                    nested.append(nn)
        except Exception:
            pass

        cache_name = ""
        primary_graphs: List[str] = []

        # Prefer CustomAttributeData to read constructor arguments deterministically (includes PXPrimaryGraph typeof args)
        try:
            from System.Reflection import CustomAttributeData  # type: ignore
            cad_list = list(CustomAttributeData.GetCustomAttributes(t))
        except Exception:
            cad_list = []

        for cad in cad_list:
            try:
                at = cad.AttributeType
                an = str(at.Name or "")
                if an == "PXCacheNameAttribute" and not cache_name:
                    # Usually constructor arg0 is display name (string)
                    try:
                        if cad.ConstructorArguments and len(cad.ConstructorArguments) >= 1:
                            v = cad.ConstructorArguments[0].Value
                            if v:
                                cache_name = str(v)
                    except Exception:
                        pass
                if an == "PXPrimaryGraphAttribute":
                    try:
                        for arg in list(cad.ConstructorArguments or []):
                            v = getattr(arg, "Value", None)
                            if v is None:
                                continue
                            if hasattr(v, "FullName"):
                                primary_graphs.append(str(v.FullName or str(v)))
                            elif hasattr(v, "__iter__"):
                                for it in list(v):
                                    if it is None:
                                        continue
                                    if hasattr(it, "FullName"):
                                        primary_graphs.append(str(it.FullName or str(it)))
                    except Exception:
                        pass
            except Exception:
                continue

        primary_graphs = list(dict.fromkeys([g for g in primary_graphs if g]))
        # Filter out obvious noise / self-references
        primary_graphs = [g for g in primary_graphs if not g.endswith("PXPrimaryGraphAttribute")]

        # Fallback: object instances (may not expose args; keep best-effort)
        try:
            attrs = list(t.GetCustomAttributes(False))
        except Exception:
            attrs = []

        for a in attrs:
            try:
                an = str(a.GetType().Name or "")
                if an == "PXCacheNameAttribute":
                    # Try common property names
                    for prop_name in ("DisplayName", "Name", "Displayname"):
                        try:
                            v = getattr(a, prop_name, None)
                            if v:
                                cache_name = str(v)
                                break
                        except Exception:
                            continue

                    # If not, try string-valued properties (best-effort)
                    if not cache_name:
                        try:
                            for p in list(a.GetType().GetProperties()):
                                pn = str(p.Name or "").lower()
                                if pn in ("displayname", "name", "displaynamevalue"):
                                    val = p.GetValue(a, None)
                                    if val:
                                        cache_name = str(val)
                                        break
                        except Exception:
                            pass

                if an == "PXPrimaryGraphAttribute":
                    # Extract referenced graph type(s). Implementations vary; collect any Type / IEnumerable[Type].
                    try:
                        for p in list(a.GetType().GetProperties()):
                            pn = str(p.Name or "")
                            val = p.GetValue(a, None)
                            if val is None:
                                continue

                            # Single Type
                            try:
                                if hasattr(val, "FullName"):
                                    primary_graphs.append(str(val.FullName or str(val)))
                                    continue
                            except Exception:
                                pass

                            # Arrays / enumerables of Type
                            try:
                                # .NET arrays have Length and indexer; try iterate
                                if hasattr(val, "__iter__"):
                                    for it in list(val):
                                        if it is None:
                                            continue
                                        if hasattr(it, "FullName"):
                                            primary_graphs.append(str(it.FullName or str(it)))
                            except Exception:
                                pass

                            # Heuristic: property name contains 'type' and value has FullName
                            if "type" in pn.lower() and hasattr(val, "FullName"):
                                primary_graphs.append(str(val.FullName or str(val)))

                        # Also try fields (some versions store Type in a field)
                        try:
                            for f in list(a.GetType().GetFields()):
                                fn = str(f.Name or "")
                                fv = f.GetValue(a)
                                if fv is None:
                                    continue
                                if hasattr(fv, "FullName"):
                                    primary_graphs.append(str(fv.FullName or str(fv)))
                                elif hasattr(fv, "__iter__"):
                                    for it in list(fv):
                                        if it is None:
                                            continue
                                        if hasattr(it, "FullName"):
                                            primary_graphs.append(str(it.FullName or str(it)))
                        except Exception:
                            pass
                        # De-dup
                        primary_graphs = list(dict.fromkeys([g for g in primary_graphs if g]))
                    except Exception:
                        pass
            except Exception:
                continue

        return TypeRecord(
            assembly=asm_name,
            full_name=full_name,
            namespace=namespace,
            name=name,
            base_type=base_type,
            kind=kind,
            dac_of_extension=dac_of_extension,
            cache_name=cache_name,
            primary_graphs=primary_graphs,
            properties=list(dict.fromkeys(props)),
            nested_types=list(dict.fromkeys(nested)),
        )



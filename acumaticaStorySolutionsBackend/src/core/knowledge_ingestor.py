#!/usr/bin/env python3
"""
Knowledge Ingestor - Build a structured, high-signal knowledge base for Acumatica RAG.

This ingestor addresses the core accuracy issues:
- PDFs are ingested as REAL TEXT (not filename/page placeholders)
- Content is chunked into retrieval-friendly units
- Each chunk is tagged with lightweight metadata (forms/DACs/graphs/events)
- Vectors are generated from chunk text and stored in the existing local KB format:
  knowledge_base/manuals/{document}/(data|images|vectors|metadata)

Design constraints:
- Keep backward compatibility with existing retrievers that expect:
  - vectors.json: {"embeddings": [...]}
  - metadata.json: list entries, each entry contains "chunks": [{"vector_index": i, ...}]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re

from src.config.config import config
from src.utils.logger_utils import get_logger


def _get_huggingface_embedding(model_name: str, device: str = "cpu", logger=None):
    """
    Lightweight HuggingFace embedding loader with fallbacks.
    Mirrors the fallback logic in `src/core/ingest.py`, but scoped to text ingestion.
    """
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
    except Exception:
        try:
            from llama_index.embeddings import HuggingFaceEmbedding
            return HuggingFaceEmbedding(model_name=model_name, device=device, normalize=True)
        except Exception:
            from sentence_transformers import SentenceTransformer
            if logger:
                logger.info("Using sentence-transformers directly for embeddings")
            model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

            class EmbeddingWrapper:
                def __init__(self, m):
                    self.model = m

                def get_text_embedding(self, text: str):
                    return self.model.encode(text, normalize_embeddings=True).tolist()

            return EmbeddingWrapper(model)


@dataclass(frozen=True)
class TextChunk:
    document_name: str
    page_number: int
    chunk_id: str
    text: str
    tags: Dict[str, List[str]]


class KnowledgeIngestor:
    """
    Ingest PDFs and text artifacts into the local knowledge base with chunk-level metadata.
    """

    def __init__(self):
        self.logger = get_logger("KNOWLEDGE_INGESTOR")
        self.base_path = Path(config.LOCAL_BASE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Embeddings
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        self.embedding_model = _get_huggingface_embedding(
            model_name=config.EMBEDDING_MODEL,
            device=device,
            logger=self.logger,
        )

        # Regex taggers (fast, deterministic)
        self._re_form = re.compile(r"\b[A-Z]{2}\d{6}\b")
        self._re_graph = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:Entry|Maint|Setup|Inquiry)\b")
        self._re_event = re.compile(r"\b(FieldVerifying|FieldUpdated|RowSelected|RowPersisting)\b")
        # DAC names are hard to infer purely; keep it conservative.
        self._re_dac = re.compile(r"\b[A-Z][A-Za-z0-9]+\b")  # filtered by heuristics below

    def _doc_dirs(self, document_name: str) -> Dict[str, Path]:
        base = self.base_path / document_name
        return {
            "base": base,
            "data": base / "data",
            "images": base / "images",
            "vectors": base / "vectors",
            "metadata": base / "metadata",
        }

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize whitespace; remove repeated blank lines.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_page_text(self, document_name: str, page_number: int, page_text: str) -> List[TextChunk]:
        """
        Chunk a page into retrieval-friendly blocks.
        Simple, deterministic heuristic:
        - Split by blank lines into paragraphs
        - Accumulate paragraphs to ~1200-1800 chars per chunk
        """
        cleaned = self._clean_text(page_text)
        if not cleaned:
            return []

        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        target = 1500

        for p in paragraphs:
            if buf_len + len(p) + 2 > target and buf:
                chunks.append("\n\n".join(buf).strip())
                buf = [p]
                buf_len = len(p)
            else:
                buf.append(p)
                buf_len += len(p) + 2
        if buf:
            chunks.append("\n\n".join(buf).strip())

        out: List[TextChunk] = []
        for i, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"p{page_number}_c{i:03d}"
            tags = self._extract_tags(chunk_text)
            out.append(
                TextChunk(
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    tags=tags,
                )
            )
        return out

    def _extract_tags(self, text: str) -> Dict[str, List[str]]:
        forms = sorted(set(self._re_form.findall(text)))
        graphs = sorted(set(self._re_graph.findall(text)))
        events = sorted(set(self._re_event.findall(text)))

        # Conservative DAC heuristic:
        # - CamelCase tokens >= 3 chars
        # - Exclude common English/Acumatica UI words
        stop = {
            "Acumatica",
            "System",
            "Setup",
            "Screen",
            "Field",
            "Fields",
            "Graph",
            "DAC",
            "PXGraph",
            "PXCache",
            "PXSelect",
            "PXResult",
            "PXData",
            "PX",
            "Note",
            "Notes",
            "Example",
            "Examples",
        }
        candidates = set(self._re_dac.findall(text))
        dacs = []
        for c in candidates:
            if len(c) < 3:
                continue
            if c in stop:
                continue
            # Stronger signal for DACs: typical Acumatica object prefixes or common doc types
            if c.startswith(("SO", "AR", "AP", "PO", "IN", "CR", "EP", "PM", "SM")) and c[0].isupper():
                dacs.append(c)
        dacs = sorted(set(dacs))

        return {"forms": forms, "graphs": graphs, "events": events, "dacs": dacs}

    def ingest_pdf(
        self,
        pdf_path: str,
        document_name: Optional[str] = None,
        *,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF into knowledge_base/manuals.
        Creates:
        - images/pageN.jpg
        - metadata/metadata.json (chunk-level entries)
        - vectors/vectors.json (embeddings aligned with vector_index)
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = document_name or pdf_path_obj.stem
        dirs = self._doc_dirs(doc_name)
        # Safety: never overwrite an existing knowledge-base document unless explicitly allowed.
        if dirs["base"].exists() and not overwrite:
            existing_metadata = (dirs["metadata"] / "metadata.json").exists()
            existing_vectors = (dirs["vectors"] / "vectors.json").exists()
            if existing_metadata or existing_vectors:
                raise FileExistsError(
                    f"Refusing to overwrite existing knowledge base document '{doc_name}'. "
                    f"Pass overwrite=True or choose a different document_name."
                )

        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        # Lazy imports so non-PDF ingestion (e.g., pattern library injection) doesn't require these deps.
        import fitz  # PyMuPDF
        from PIL import Image

        # Copy source PDF for traceability
        try:
            target_pdf = dirs["data"] / pdf_path_obj.name
            if not target_pdf.exists():
                target_pdf.write_bytes(pdf_path_obj.read_bytes())
        except Exception:
            # Keep ingestion resilient; PDF copy is helpful but not critical.
            self.logger.warning("Failed to copy source PDF into data/", extra={"pdf": str(pdf_path_obj)})

        doc = fitz.open(str(pdf_path_obj))
        self.logger.info("Ingesting PDF (text-first)", extra={"document": doc_name, "pages": doc.page_count})

        embeddings: List[List[float]] = []
        metadata_entries: List[Dict[str, Any]] = []
        vector_index = 0

        # Create page images (for optional vision fallback / UI)
        for page_idx in range(doc.page_count):
            page_number = page_idx + 1
            page = doc[page_idx]

            # Render image
            try:
                pix = page.get_pixmap(dpi=config.DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if max(img.size) > config.MAX_IMAGE_SIZE:
                    ratio = config.MAX_IMAGE_SIZE / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                image_path = dirs["images"] / f"page{page_number}.jpg"
                img.save(str(image_path), "JPEG", quality=90)
                rel_image_path = f"images/page{page_number}.jpg"
            except Exception as e:
                self.logger.warning("Failed to render page image", extra={"page": page_number, "error": str(e)})
                rel_image_path = ""

            # Extract text
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""

            chunks = self._chunk_page_text(doc_name, page_number, page_text)
            if not chunks:
                continue

            for chunk in chunks:
                # Embed chunk text (real content)
                emb = self.embedding_model.get_text_embedding(chunk.text)
                if len(emb) > config.EMBEDDING_DIMENSION:
                    emb = emb[: config.EMBEDDING_DIMENSION]
                embeddings.append(emb)

                entry = {
                    "id": f"{doc_name}_{chunk.chunk_id}",
                    "pdf_name": doc_name,
                    "document_name": doc_name,
                    "page_number": chunk.page_number,
                    "section_type": "text_chunk",
                    "section_id": chunk.chunk_id,
                    "image_path": rel_image_path,
                    "text_content": chunk.text,
                    "tags": chunk.tags,
                    "chunks": [
                        {
                            "vector_index": vector_index,
                            # Keep coordinates for compatibility (unused for text chunks)
                            "coordinates": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                        }
                    ],
                }
                metadata_entries.append(entry)
                vector_index += 1

        # Persist
        vectors_payload = {
            "embeddings": embeddings,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimension": config.EMBEDDING_DIMENSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "document_name": doc_name,
        }
        (dirs["vectors"] / "vectors.json").write_text(json.dumps(vectors_payload, indent=2), encoding="utf-8")
        (dirs["metadata"] / "metadata.json").write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")

        self.logger.info(
            "PDF ingestion complete",
            extra={"document": doc_name, "chunks": len(metadata_entries), "vectors": len(embeddings)},
        )
        return {"document_name": doc_name, "chunks": len(metadata_entries), "vectors": len(embeddings)}

    def upgrade_existing_pdf_document(
        self,
        document_name: str,
        *,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        In-place upgrade:
        - Reads existing knowledge_base/manuals/{document_name}/data/*.pdf
        - Appends REAL text chunks to existing metadata.json
        - Appends corresponding embeddings to existing vectors.json

        IMPORTANT:
        - Does NOT delete or replace existing image-based metadata/vectors.
        - Does NOT create new document folders.
        - Is idempotent: if text chunks already exist for a page, it skips them.
        """
        dirs = self._doc_dirs(document_name)
        if not dirs["base"].exists():
            raise FileNotFoundError(f"Document folder not found: {dirs['base']}")

        # Locate source PDF
        pdf_files = list(dirs["data"].glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF found under: {dirs['data']}")
        pdf_path = pdf_files[0]

        metadata_path = dirs["metadata"] / "metadata.json"
        vectors_path = dirs["vectors"] / "vectors.json"

        if not metadata_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"Missing metadata/vectors for document: {document_name}")

        # Load existing artifacts
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(existing_metadata, list):
            raise ValueError("Expected metadata.json to be a list")

        vectors_payload = json.loads(vectors_path.read_text(encoding="utf-8"))
        embeddings = vectors_payload.get("embeddings", [])
        if not isinstance(embeddings, list):
            raise ValueError("Expected vectors.json to contain embeddings as a list")

        # Find already-ingested text chunks (by section_type/text_chunk)
        existing_section_ids = set()
        max_vector_index = -1
        for entry in existing_metadata:
            if isinstance(entry, dict):
                sid = entry.get("section_id")
                stype = entry.get("section_type")
                if sid and stype == "text_chunk":
                    existing_section_ids.add(sid)
                for ch in entry.get("chunks", []) or []:
                    vi = ch.get("vector_index")
                    if isinstance(vi, int):
                        max_vector_index = max(max_vector_index, vi)

        next_vector_index = max_vector_index + 1

        # Lazy imports for PDF processing
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        new_entries = 0
        new_vectors = 0

        self.logger.info(
            "Upgrading document in-place with text chunks",
            extra={"document": document_name, "pdf": str(pdf_path), "pages": total_pages, "vector_start": next_vector_index},
        )

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            page = doc[page_idx]
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""

            chunks = self._chunk_page_text(document_name, page_number, page_text)
            if not chunks:
                continue

            for chunk in chunks:
                # section_id must be stable and unique; we use chunk_id.
                if chunk.chunk_id in existing_section_ids:
                    continue

                emb = self.embedding_model.get_text_embedding(chunk.text)
                if len(emb) > config.EMBEDDING_DIMENSION:
                    emb = emb[: config.EMBEDDING_DIMENSION]
                embeddings.append(emb)
                new_vectors += 1

                existing_metadata.append(
                    {
                        "id": f"{document_name}_{chunk.chunk_id}",
                        "pdf_name": document_name,
                        "document_name": document_name,
                        "page_number": chunk.page_number,
                        "section_type": "text_chunk",
                        "section_id": chunk.chunk_id,
                        # Reuse existing page image if present
                        "image_path": f"images/page{chunk.page_number}.jpg" if (dirs["images"] / f"page{chunk.page_number}.jpg").exists() else "",
                        "text_content": chunk.text,
                        "tags": chunk.tags,
                        "chunks": [
                            {
                                "vector_index": next_vector_index,
                                "coordinates": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                            }
                        ],
                    }
                )
                existing_section_ids.add(chunk.chunk_id)
                next_vector_index += 1
                new_entries += 1

        # Persist updates (same files, no new files)
        vectors_payload["embeddings"] = embeddings
        vectors_payload.setdefault("embedding_model", config.EMBEDDING_MODEL)
        vectors_payload.setdefault("embedding_dimension", config.EMBEDDING_DIMENSION)
        vectors_payload["updated_at"] = datetime.utcnow().isoformat() + "Z"

        metadata_path.write_text(json.dumps(existing_metadata, indent=2), encoding="utf-8")
        vectors_path.write_text(json.dumps(vectors_payload, indent=2), encoding="utf-8")

        self.logger.info(
            "In-place upgrade complete",
            extra={"document": document_name, "new_metadata_entries": new_entries, "new_vectors": new_vectors, "total_vectors": len(embeddings)},
        )
        return {
            "document_name": document_name,
            "pages_scanned": total_pages,
            "new_text_chunks": new_entries,
            "new_embeddings": new_vectors,
            "total_embeddings": len(embeddings),
        }

    def ingest_text_document(
        self,
        document_name: str,
        texts: List[Tuple[str, str]],
        *,
        section_type: str = "pattern",
        source_label: str = "pattern_library",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a synthetic/text-only document (e.g., pattern library).

        Args:
            document_name: Target KB folder name
            texts: List of (section_id, text)
        """
        dirs = self._doc_dirs(document_name)
        # Safety: avoid overwriting unless explicitly allowed.
        if dirs["base"].exists() and not overwrite:
            existing_metadata = (dirs["metadata"] / "metadata.json").exists()
            existing_vectors = (dirs["vectors"] / "vectors.json").exists()
            if existing_metadata or existing_vectors:
                raise FileExistsError(
                    f"Refusing to overwrite existing knowledge base document '{document_name}'. "
                    f"Pass overwrite=True to replace it."
                )
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        embeddings: List[List[float]] = []
        metadata_entries: List[Dict[str, Any]] = []

        for vector_index, (section_id, text) in enumerate(texts):
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            tags = self._extract_tags(cleaned)
            emb = self.embedding_model.get_text_embedding(cleaned)
            if len(emb) > config.EMBEDDING_DIMENSION:
                emb = emb[: config.EMBEDDING_DIMENSION]
            embeddings.append(emb)

            metadata_entries.append(
                {
                    "id": f"{document_name}_{section_id}",
                    "pdf_name": document_name,
                    "document_name": document_name,
                    "page_number": 1,
                    "section_type": section_type,
                    "section_id": section_id,
                    "image_path": "",
                    "source": source_label,
                    "text_content": cleaned,
                    "tags": tags,
                    "chunks": [
                        {
                            "vector_index": vector_index,
                            "coordinates": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                        }
                    ],
                }
            )

        vectors_payload = {
            "embeddings": embeddings,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dimension": config.EMBEDDING_DIMENSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "document_name": document_name,
        }
        (dirs["vectors"] / "vectors.json").write_text(json.dumps(vectors_payload, indent=2), encoding="utf-8")
        (dirs["metadata"] / "metadata.json").write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")

        self.logger.info(
            "Text document ingestion complete",
            extra={"document": document_name, "sections": len(metadata_entries), "vectors": len(embeddings)},
        )
        return {"document_name": document_name, "sections": len(metadata_entries), "vectors": len(embeddings)}



"""Thematic document collections (folders) for scoped RAG retrieval."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config
from core.database import db
from core.document_processor import doc_processor
from core.embeddings.chain import EmbeddingChain
from core.ingestion import PDF_EXTENSIONS, SUPPORTED_EXTENSIONS, TEXT_EXTENSIONS
from core.pdf_processor import pdf_processor

log = logging.getLogger(__name__)

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def source_prefix(collection_id: str) -> str:
    return f"collections/{collection_id}/"


def make_source(collection_id: str, filename: str) -> str:
    return f"{source_prefix(collection_id)}{filename}"


def slugify(name: str, fallback: str = "collection") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:64] or fallback
    if not SLUG_RE.match(slug):
        slug = f"{fallback}-{uuid.uuid4().hex[:8]}"
    return slug


@dataclass(frozen=True)
class CollectionDocument:
    filename: str
    source: str
    uploaded_at: str
    chunk_count: int
    size_bytes: int


@dataclass(frozen=True)
class CollectionInfo:
    id: str
    name: str
    description: str
    created_at: str
    documents: list[CollectionDocument]


class CollectionError(ValueError):
    """Raised when collection operations fail validation."""


class CollectionService:
    """Manage thematic folders, documents, and active RAG scope."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or (config.project_root / "data" / "collections")
        self._registry_path = self._root / "registry.json"
        self._root.mkdir(parents=True, exist_ok=True)

    def list_collections(self) -> list[CollectionInfo]:
        registry = self._load_registry()
        return [self._to_info(item) for item in registry["collections"].values()]

    def get_collection(self, collection_id: str) -> CollectionInfo:
        registry = self._load_registry()
        item = registry["collections"].get(collection_id)
        if item is None:
            raise CollectionError(f"collection not found: {collection_id}")
        return self._to_info(item)

    def create_collection(
        self,
        name: str,
        *,
        collection_id: str | None = None,
        description: str = "",
    ) -> CollectionInfo:
        registry = self._load_registry()
        cid = collection_id or slugify(name)
        if not SLUG_RE.match(cid):
            raise CollectionError(
                "collection id must match [a-z0-9][a-z0-9_-]{0,63}"
            )
        if cid in registry["collections"]:
            raise CollectionError(f"collection already exists: {cid}")

        now = _utc_now()
        registry["collections"][cid] = {
            "id": cid,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": now,
            "documents": {},
        }
        (self._root / cid / "files").mkdir(parents=True, exist_ok=True)
        self._save_registry(registry)
        log.info("collection created id=%s name=%s", cid, name)
        return self._to_info(registry["collections"][cid])

    def delete_collection(self, collection_id: str) -> None:
        registry = self._load_registry()
        if collection_id not in registry["collections"]:
            raise CollectionError(f"collection not found: {collection_id}")

        prefix = source_prefix(collection_id)
        removed = db.vector_store.delete_by_source_prefix(prefix)
        shutil.rmtree(self._root / collection_id, ignore_errors=True)
        del registry["collections"][collection_id]
        registry["selected_collection_ids"] = [
            cid for cid in registry["selected_collection_ids"] if cid != collection_id
        ]
        self._save_registry(registry)
        log.info("collection deleted id=%s chunks_removed=%s", collection_id, removed)

    def delete_all_collections(self) -> list[str]:
        """Delete every collection with files and related vector chunks."""
        registry = self._load_registry()
        collection_ids = list(registry["collections"].keys())
        for collection_id in collection_ids:
            self.delete_collection(collection_id)
        return collection_ids

    def list_documents(self, collection_id: str) -> list[CollectionDocument]:
        return self.get_collection(collection_id).documents

    def ingest_document(
        self,
        collection_id: str,
        filename: str,
        content: bytes,
    ) -> CollectionDocument:
        registry = self._load_registry()
        collection = registry["collections"].get(collection_id)
        if collection is None:
            raise CollectionError(f"collection not found: {collection_id}")

        safe_name = Path(filename).name
        if not safe_name:
            raise CollectionError("filename must not be empty")
        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise CollectionError(
                f"unsupported file type: {extension or '(none)'}; "
                f"allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        files_dir = self._root / collection_id / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        target_path = files_dir / safe_name
        target_path.write_bytes(content)

        source = make_source(collection_id, safe_name)
        db.vector_store.delete_by_source(source)

        start_id = _next_chunk_id()
        if extension in PDF_EXTENSIONS:
            chunks = pdf_processor.process_document(str(target_path), source, start_id)
        else:
            chunks = doc_processor.process_document(str(target_path), source, start_id)

        if not chunks:
            target_path.unlink(missing_ok=True)
            raise CollectionError(f"no text extracted from file: {safe_name}")

        for chunk in chunks:
            chunk["collection_id"] = collection_id
            chunk["filename"] = safe_name

        texts = [str(chunk["chunk"]) for chunk in chunks]
        embeddings = EmbeddingChain(config).embed(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        db.insert_batch(chunks, dataset_kind="collection")

        now = _utc_now()
        doc_meta = {
            "filename": safe_name,
            "source": source,
            "uploaded_at": now,
            "chunk_count": len(chunks),
            "size_bytes": len(content),
        }
        collection["documents"][safe_name] = doc_meta
        self._save_registry(registry)
        log.info(
            "document ingested collection=%s file=%s chunks=%s",
            collection_id,
            safe_name,
            len(chunks),
        )
        return CollectionDocument(**doc_meta)

    def delete_document(self, collection_id: str, filename: str) -> None:
        registry = self._load_registry()
        collection = registry["collections"].get(collection_id)
        if collection is None:
            raise CollectionError(f"collection not found: {collection_id}")

        safe_name = Path(filename).name
        if safe_name not in collection["documents"]:
            raise CollectionError(
                f"document not found in collection {collection_id}: {safe_name}"
            )

        source = make_source(collection_id, safe_name)
        db.vector_store.delete_by_source(source)
        (self._root / collection_id / "files" / safe_name).unlink(missing_ok=True)
        del collection["documents"][safe_name]
        self._save_registry(registry)
        log.info("document deleted collection=%s file=%s", collection_id, safe_name)

    def get_selection(self) -> list[str]:
        registry = self._load_registry()
        return list(registry["selected_collection_ids"])

    def set_selection(self, collection_ids: list[str]) -> list[str]:
        registry = self._load_registry()
        known = set(registry["collections"])
        unknown = [cid for cid in collection_ids if cid not in known]
        if unknown:
            raise CollectionError(f"unknown collection ids: {', '.join(unknown)}")

        registry["selected_collection_ids"] = list(dict.fromkeys(collection_ids))
        self._save_registry(registry)
        return registry["selected_collection_ids"]

    def active_source_prefixes(
        self,
        collection_ids: list[str] | None = None,
    ) -> list[str] | None:
        """Return source prefixes for scoped search, or None for all collections."""
        ids = collection_ids if collection_ids is not None else self.get_selection()
        if not ids:
            return None
        return [source_prefix(cid) for cid in ids]

    def _load_registry(self) -> dict[str, Any]:
        if not self._registry_path.exists():
            return {"version": 1, "selected_collection_ids": [], "collections": {}}
        payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"version": 1, "selected_collection_ids": [], "collections": {}}
        payload.setdefault("version", 1)
        payload.setdefault("selected_collection_ids", [])
        payload.setdefault("collections", {})
        return payload

    def _save_registry(self, registry: dict[str, Any]) -> None:
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._registry_path)

    @staticmethod
    def _to_info(raw: dict[str, Any]) -> CollectionInfo:
        documents = [
            CollectionDocument(
                filename=str(doc["filename"]),
                source=str(doc["source"]),
                uploaded_at=str(doc["uploaded_at"]),
                chunk_count=int(doc["chunk_count"]),
                size_bytes=int(doc.get("size_bytes", 0)),
            )
            for doc in raw.get("documents", {}).values()
        ]
        documents.sort(key=lambda item: item.filename)
        return CollectionInfo(
            id=str(raw["id"]),
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            created_at=str(raw["created_at"]),
            documents=documents,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_chunk_id() -> int:
    max_id = 0
    for record in db.vector_store.load_all_records():
        try:
            max_id = max(max_id, int(record.id))
        except ValueError:
            continue
    return max_id + 1


collection_service = CollectionService()

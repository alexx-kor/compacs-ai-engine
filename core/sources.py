"""Source registry: list, download, delete indexed documents."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import config
from core.collections import CollectionError, collection_service, make_source
from core.database import db

_COLLECTION_SOURCE_RE = re.compile(r"^collections/(?P<collection_id>[^/]+)/(?P<filename>.+)$")


@dataclass(frozen=True)
class SourceInfo:
    id: str
    source: str
    collection_id: str | None
    filename: str | None
    uploaded_at: str | None
    chunk_count: int
    size_bytes: int | None
    kind: str


class SourceError(ValueError):
    """Raised when source operations fail."""


def encode_source_id(source: str) -> str:
    """URL-safe opaque id for ``/sources/{id}`` routes."""
    return base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii").rstrip("=")


def decode_source_id(source_id: str) -> str:
    """Decode opaque source id back to vector ``source`` path."""
    padding = "=" * (-len(source_id) % 4)
    try:
        return base64.urlsafe_b64decode(source_id + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise SourceError(f"invalid source id: {source_id}") from error


def _chunk_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in db.vector_store.load_all_records():
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


class SourceService:
    """Unified view over collection documents and legacy indexed sources."""

    def list_sources(self) -> list[SourceInfo]:
        counts = _chunk_counts()
        seen: set[str] = set()
        items: list[SourceInfo] = []

        for collection in collection_service.list_collections():
            for doc in collection.documents:
                seen.add(doc.source)
                items.append(
                    SourceInfo(
                        id=encode_source_id(doc.source),
                        source=doc.source,
                        collection_id=collection.id,
                        filename=doc.filename,
                        uploaded_at=doc.uploaded_at,
                        chunk_count=counts.get(doc.source, doc.chunk_count),
                        size_bytes=doc.size_bytes,
                        kind="collection",
                    )
                )

        for source, chunk_count in sorted(counts.items()):
            if source in seen:
                continue
            match = _COLLECTION_SOURCE_RE.match(source)
            items.append(
                SourceInfo(
                    id=encode_source_id(source),
                    source=source,
                    collection_id=match.group("collection_id") if match else None,
                    filename=match.group("filename") if match else Path(source).name,
                    uploaded_at=None,
                    chunk_count=chunk_count,
                    size_bytes=None,
                    kind="legacy",
                )
            )

        items.sort(key=lambda item: item.source)
        return items

    def get_source(self, source_id: str) -> SourceInfo:
        source = decode_source_id(source_id)
        for item in self.list_sources():
            if item.source == source:
                return item
        raise SourceError(f"source not found: {source_id}")

    def resolve_file_path(self, source: str) -> Path:
        match = _COLLECTION_SOURCE_RE.match(source)
        if not match:
            raise SourceError(f"original file unavailable for source: {source}")
        collection_id = match.group("collection_id")
        filename = match.group("filename")
        path = config.project_root / "data" / "collections" / collection_id / "files" / filename
        if not path.is_file():
            raise SourceError(f"original file missing on disk: {source}")
        return path

    def delete_source(self, source_id: str) -> dict[str, Any]:
        source = decode_source_id(source_id)
        match = _COLLECTION_SOURCE_RE.match(source)
        if match:
            collection_service.delete_document(
                match.group("collection_id"),
                match.group("filename"),
            )
            return {"deleted": source_id, "source": source, "reindexed": True}

        removed = db.vector_store.delete_by_source(source)
        if removed <= 0:
            raise SourceError(f"source not found: {source_id}")
        return {"deleted": source_id, "source": source, "reindexed": True, "chunks_removed": removed}

    def clear_knowledge_base(self) -> dict[str, Any]:
        """Remove all collections (files + registry) and every chunk in the vector index."""
        chunks_before = db.get_chunk_count()
        legacy_sources = [
            source
            for source in db.vector_store.list_sources()
            if not source.startswith("collections/")
        ]
        collection_ids = [info.id for info in collection_service.list_collections()]

        db.init_database(force_recreate=True)
        for collection_id in collection_ids:
            collection_service.delete_collection(collection_id)

        self._clear_runtime_caches()
        return {
            "collections_removed": collection_ids,
            "legacy_sources_removed": legacy_sources,
            "chunks_removed": chunks_before,
            "chunks_remaining": db.get_chunk_count(),
        }

    def reset_index(self) -> dict[str, Any]:
        """Wipe the vector index only. Uploaded collection files stay on disk."""
        chunks_removed = db.get_chunk_count()
        db.init_database(force_recreate=True)
        self._clear_runtime_caches()
        return {
            "chunks_removed": chunks_removed,
            "chunks_remaining": db.get_chunk_count(),
            "collection_files_preserved": True,
        }

    @staticmethod
    def _clear_runtime_caches() -> None:
        from core.embeddings.chain import EmbeddingChain

        db._cache.clear()
        db._cache_time.clear()
        EmbeddingChain.embed_cached.cache_clear()


source_service = SourceService()

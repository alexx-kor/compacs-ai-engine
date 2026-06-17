"""Backward-compatible database facade over VectorStore."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from config import config
from core.storage.factory import create_vector_store
from core.storage.protocol import ChunkRecord, VectorStore

log = logging.getLogger(__name__)


class DatabaseManager:
    """Facade preserving legacy ``db.search`` and cache APIs."""

    def __init__(self) -> None:
        self._store: VectorStore = create_vector_store(config)
        self._cache: dict[str, str] = {}
        self._cache_time: dict[str, float] = {}

    @property
    def vector_store(self) -> VectorStore:
        """Return underlying vector store instance."""
        return self._store

    @property
    def backend_name(self) -> str:
        return self._store.backend_name

    def init_database(self, force_recreate: bool = False) -> None:
        """Initialize backing vector store."""
        self._store.init_store(force_recreate=force_recreate)

    def reload_store(self) -> None:
        """Recreate vector store handle after env/path changes."""
        self._store = create_vector_store(config)

    def save_bm25_index(self, chunks: list[dict[str, Any]]) -> None:
        """Legacy BM25 sidecar hook (dense-only path ignores this)."""
        _ = chunks
        log.debug("save_bm25_index skipped (BM25 sidecar not active)")

    def insert_batch(self, chunks: list[dict[str, Any]], dataset_kind: str = "raw") -> None:
        """Insert legacy chunk dictionaries."""
        if not chunks:
            return
        records = [ChunkRecord.from_legacy_dict(chunk, dataset_kind=dataset_kind) for chunk in chunks]
        self._store.insert_batch(records)

    def search(
        self,
        embedding: list[float],
        source_prefixes: list[str] | None = None,
    ) -> list[tuple[Any, ...]]:
        """Search and return legacy tuples ``(chunk, source, page, distance)``."""
        query = np.asarray(embedding, dtype=np.float64)
        records = self._store.search(
            query_embedding=query,
            limit=config.top_k,
            similarity_threshold=config.similarity_threshold,
            source_prefixes=source_prefixes,
        )
        tuples: list[tuple[Any, ...]] = []
        for record in records:
            distance = float(
                1.0
                - float(
                    np.dot(query, record.embedding)
                    / (np.linalg.norm(query) * np.linalg.norm(record.embedding) + 1e-12)
                )
            )
            tuples.append((record.chunk, record.source, record.page, distance))
        return tuples

    def get_chunk_count(self) -> int:
        return self._store.chunk_count()

    def get_cache(self, key: str) -> str | None:
        return self.resolve_cache(key)

    def resolve_cache(self, key: str) -> str | None:
        if not config.cache_enabled:
            return None
        cached = self._cache.get(key)
        if cached is None:
            return None
        if time.time() - self._cache_time.get(key, 0.0) >= config.cache_ttl:
            return None
        return cached

    def set_cache(self, key: str, value: str) -> None:
        if not config.cache_enabled:
            return
        self._cache[key] = value
        self._cache_time[key] = time.time()

    def store_stats(self) -> dict[str, Any]:
        return {
            "backend": self._store.backend_name,
            "chunk_count": self._store.chunk_count(),
            "sources": self._store.list_sources(),
        }


db = DatabaseManager()

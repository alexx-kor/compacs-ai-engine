"""Backward-compatible database facade over VectorStore."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from config import config
from core.bm25_index import Bm25Index
from core.hybrid_retriever import merge_dense_bm25
from core.storage.factory import create_vector_store
from core.storage.protocol import ChunkRecord, VectorStore

log = logging.getLogger(__name__)


class DatabaseManager:
    """Facade preserving legacy ``db.search`` and cache APIs."""

    def __init__(self) -> None:
        self._store: VectorStore = create_vector_store(config)
        self._cache: dict[str, str] = {}
        self._cache_time: dict[str, float] = {}
        self._bm25: Bm25Index | None = Bm25Index.load(
            config.local_vector_store_dir / "bm25_index.json"
        )

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
        """Reload vector store and BM25 sidecar from disk."""
        self._store = create_vector_store(config)
        self._bm25 = Bm25Index.load(config.local_vector_store_dir / "bm25_index.json")

    def insert_batch(self, chunks: list[dict[str, Any]], dataset_kind: str = "raw") -> None:
        """Insert legacy chunk dictionaries."""
        if not chunks:
            return
        records = [ChunkRecord.from_legacy_dict(chunk, dataset_kind=dataset_kind) for chunk in chunks]
        self._store.insert_batch(records)

    def search(self, embedding: list[float], query_text: str = "") -> list[tuple[Any, ...]]:
        """Search and return legacy tuples ``(chunk, source, page, distance)``."""
        query = np.asarray(embedding, dtype=np.float64)
        pool_limit = max(config.top_k * 3, config.top_k)
        records = self._store.search(
            query_embedding=query,
            limit=pool_limit,
            similarity_threshold=config.similarity_threshold,
        )
        if config.hybrid_search_enabled and self._bm25 is not None and query_text.strip():
            catalog = {row.id: row for row in self._store.load_all_records()}
            records = merge_dense_bm25(
                records,
                query_text,
                query,
                self._bm25,
                config,
                catalog,
            )
        else:
            records = records[: config.top_k]

        query_dim = int(query.shape[0])
        tuples: list[tuple[Any, ...]] = []
        for record in records:
            if int(record.embedding.shape[0]) != query_dim:
                continue
            distance = float(
                1.0
                - float(
                    np.dot(query, record.embedding)
                    / (np.linalg.norm(query) * np.linalg.norm(record.embedding) + 1e-12)
                )
            )
            tuples.append((record.chunk, record.source, record.page, distance))
        return tuples

    def reload_bm25_index(self) -> None:
        """Reload BM25 sidecar after re-indexing."""
        self._bm25 = Bm25Index.load(config.local_vector_store_dir / "bm25_index.json")

    def save_bm25_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build and persist BM25 index from legacy chunk dicts."""
        index = Bm25Index.from_chunks(chunks, lemmatize=config.bm25_lemmatize)
        path = config.local_vector_store_dir / "bm25_index.json"
        index.save(path)
        self._bm25 = index

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

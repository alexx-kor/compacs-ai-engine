"""JSON file-based vector store for development and offline use."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config import Config
from core.storage._similarity import cosine_distance
from core.storage.protocol import ChunkRecord, VectorStore

log = logging.getLogger(__name__)


class JsonVectorStore(VectorStore):
    """Persist chunks in a single JSON file under the configured directory."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._store_dir = Path(config.local_vector_store_dir)
        self._store_path = self._store_dir / "chunks.json"
        self._rows: list[ChunkRecord] = []
        self._embedding_dim: int | None = None
        self._embedding_provider: str | None = None
        self._load_from_disk()

    @property
    def backend_name(self) -> str:
        return "json"

    def init_store(self, force_recreate: bool = False) -> None:
        if force_recreate:
            self._rows = []
            self._atomic_write()
            log.warning("JSON vector store recreated path=%s", self._store_path)
            return
        self._store_dir.mkdir(parents=True, exist_ok=True)
        log.info("JSON vector store ready path=%s rows=%s", self._store_path, len(self._rows))

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int,
        similarity_threshold: float,
    ) -> list[ChunkRecord]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        query_dim = int(query_embedding.shape[0])
        scored: list[tuple[float, ChunkRecord]] = []
        for record in self._rows:
            if int(record.embedding.shape[0]) != query_dim:
                continue
            distance = cosine_distance(query_embedding.tolist(), record.embedding.tolist())
            if distance >= similarity_threshold:
                continue
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        return [record for _, record in scored[:limit]]

    def insert_batch(self, chunks: list[ChunkRecord]) -> int:
        if not chunks:
            raise ValueError("chunks list must not be empty")
        if self._rows:
            expected_dim = int(self._rows[0].embedding.shape[0])
            mismatched = [
                chunk.id
                for chunk in chunks
                if int(chunk.embedding.shape[0]) != expected_dim
            ]
            if mismatched:
                raise ValueError(
                    f"embedding dimension mismatch on insert: expected={expected_dim} "
                    f"ids={mismatched[:5]}. Re-index with --force-recreate."
                )
        self._rows.extend(chunks)
        self._atomic_write()
        log.info("Inserted chunks count=%s backend=json", len(chunks))
        return len(chunks)

    def delete_by_source(self, source: str) -> int:
        before = len(self._rows)
        self._rows = [row for row in self._rows if row.source != source]
        removed = before - len(self._rows)
        if removed:
            self._atomic_write()
        return removed

    def list_sources(self) -> list[str]:
        return sorted({row.source for row in self._rows})

    def chunk_count(self) -> int:
        return len(self._rows)

    def load_all_records(self) -> list[ChunkRecord]:
        return list(self._rows)

    def _load_from_disk(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if not self._store_path.exists():
            return
        try:
            with self._store_path.open("r", encoding="utf-8") as handle:
                payload: Any = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            log.warning("Could not read JSON store path=%s error=%s", self._store_path, error)
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            return
        dim = payload.get("embedding_dim")
        if isinstance(dim, int):
            self._embedding_dim = dim
        provider = payload.get("embedding_provider")
        if isinstance(provider, str):
            self._embedding_provider = provider
        rows: list[ChunkRecord] = []
        for raw in payload["rows"]:
            if not isinstance(raw, dict):
                continue
            rows.append(self._row_to_record(raw))
        self._rows = rows
        self._sanitize_embedding_dimensions(persist=True)

    def _sanitize_embedding_dimensions(self, *, persist: bool) -> None:
        """Drop orphan rows whose embedding size differs from the index majority."""
        if not self._rows:
            return
        dim_counts = Counter(int(record.embedding.shape[0]) for record in self._rows)
        if len(dim_counts) == 1:
            self._embedding_dim = next(iter(dim_counts))
            return

        target_dim, keep_count = dim_counts.most_common(1)[0]
        before = len(self._rows)
        self._rows = [
            record for record in self._rows if int(record.embedding.shape[0]) == target_dim
        ]
        removed = before - len(self._rows)
        self._embedding_dim = target_dim
        log.warning(
            "removed %s/%s chunks with non-dominant embedding dims %s (kept dim=%s)",
            removed,
            before,
            dict(dim_counts),
            target_dim,
        )
        if persist and removed:
            self._atomic_write()

    def _atomic_write(self) -> None:
        tmp_path = self._store_path.with_suffix(".json.tmp")
        if self._rows:
            self._embedding_dim = int(self._rows[0].embedding.shape[0])
            self._embedding_provider = os.getenv("EMBEDDING_PROVIDER", self._embedding_provider)
        payload: dict[str, Any] = {
            "version": 1,
            "rows": [self._record_to_row(record) for record in self._rows],
        }
        if self._embedding_dim is not None:
            payload["embedding_dim"] = self._embedding_dim
        if self._embedding_provider:
            payload["embedding_provider"] = self._embedding_provider
        body = json.dumps(payload, ensure_ascii=False)
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(self._store_path)

    @staticmethod
    def _record_to_row(record: ChunkRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "source": record.source,
            "page": record.page,
            "chunk": record.chunk,
            "embedding": [float(value) for value in record.embedding.tolist()],
            "metadata": record.metadata,
            "dataset_kind": record.dataset_kind,
            "created_at": record.created_at.isoformat(),
        }

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> ChunkRecord:
        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc)
        created_raw = row.get("created_at")
        if isinstance(created_raw, str):
            try:
                created_at = datetime.fromisoformat(created_raw)
            except ValueError:
                pass
        return ChunkRecord(
            id=str(row.get("id", "")),
            source=str(row.get("source", "")),
            page=int(row.get("page", 0)),
            chunk=str(row.get("chunk", "")),
            embedding=np.asarray(row.get("embedding", []), dtype=np.float64),
            metadata=dict(row.get("metadata", {})),
            dataset_kind=str(row.get("dataset_kind", "raw")),
            created_at=created_at,
        )

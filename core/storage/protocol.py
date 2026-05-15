"""Storage protocol definitions."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChunkRecord:
    """Immutable chunk record with embedding and metadata."""

    id: str
    source: str
    page: int
    chunk: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    dataset_kind: str = "raw"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chunk id must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")
        if self.page < 0:
            raise ValueError(f"page must be non-negative, got {self.page}")
        if self.created_at.tzinfo is None:
            object.__setattr__(
                self,
                "created_at",
                self.created_at.replace(tzinfo=timezone.utc),
            )

    @classmethod
    def create_now(
        cls,
        id: str,
        source: str,
        page: int,
        chunk: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        dataset_kind: str = "raw",
    ) -> ChunkRecord:
        """Create a chunk record with the current UTC timestamp."""
        return cls(
            id=id,
            source=source,
            page=page,
            chunk=chunk,
            embedding=embedding,
            metadata=metadata or {},
            dataset_kind=dataset_kind,
            created_at=datetime.now(timezone.utc),
        )

    @classmethod
    def from_legacy_dict(cls, row: dict[str, Any], dataset_kind: str = "raw") -> ChunkRecord:
        """Build a record from legacy ingestion chunk dictionaries."""
        embedding_raw = row.get("embedding", [])
        embedding = np.asarray(embedding_raw, dtype=np.float64)
        metadata = {
            key: value
            for key, value in row.items()
            if key
            not in {"id", "source", "page", "chunk", "embedding", "chunk_hash", "char_count"}
        }
        if "chunk_hash" in row:
            metadata["chunk_hash"] = row["chunk_hash"]
        if "char_count" in row:
            metadata["char_count"] = row["char_count"]
        return cls(
            id=str(row.get("id", "")),
            source=str(row.get("source", "")),
            page=int(row.get("page", 0)),
            chunk=str(row.get("chunk", "")),
            embedding=embedding,
            metadata=metadata,
            dataset_kind=dataset_kind,
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to legacy chunk dict used by older ingestion paths."""
        payload: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "chunk": self.chunk,
            "embedding": [float(value) for value in self.embedding.tolist()],
            "chunk_hash": str(self.metadata.get("chunk_hash", "")),
            "char_count": int(self.metadata.get("char_count", len(self.chunk))),
        }
        return payload


class VectorStore(ABC):
    """Vector storage protocol for RAG chunks."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int,
        similarity_threshold: float,
    ) -> list[ChunkRecord]:
        """Search for similar chunks ordered by relevance."""

    @abstractmethod
    def insert_batch(self, chunks: list[ChunkRecord]) -> int:
        """Insert multiple chunks."""

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a source."""

    @abstractmethod
    def list_sources(self) -> list[str]:
        """List distinct source identifiers."""

    @abstractmethod
    def chunk_count(self) -> int:
        """Return total stored chunk count."""

    @abstractmethod
    def init_store(self, force_recreate: bool = False) -> None:
        """Initialize or reset backing storage."""

    @abstractmethod
    def load_all_records(self) -> list[ChunkRecord]:
        """Return all stored chunk records."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier for logging and health checks."""


def metadata_to_json(metadata: dict[str, Any]) -> str:
    """Serialize metadata for ClickHouse storage."""
    return json.dumps(metadata, ensure_ascii=False, default=str)


def metadata_from_json(raw: str) -> dict[str, Any]:
    """Deserialize metadata from ClickHouse storage."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

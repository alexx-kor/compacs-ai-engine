from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.storage.protocol import ChunkRecord
from rag_service import RagService


class _FakeEmbeddingChain:
    active_provider = "test"

    def embed_cached(self, _text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


class _FakeLLMChain:
    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> tuple[str, str]:
        return "mock answer from docs", "ollama"


class _FakeVectorStore:
    backend_name = "json"

    def search(self, query_embedding: np.ndarray, limit: int, similarity_threshold: float) -> list[ChunkRecord]:
        return [
            ChunkRecord.create_now(
                id="1",
                source="instructions/raw/sample.txt",
                page=1,
                chunk="ClickHouse configuration details",
                embedding=query_embedding,
            )
        ]

    def insert_batch(self, chunks: list[ChunkRecord]) -> int:
        return len(chunks)

    def delete_by_source(self, source: str) -> int:
        return 0

    def list_sources(self) -> list[str]:
        return ["instructions/raw/sample.txt"]

    def chunk_count(self) -> int:
        return 1

    def init_store(self, force_recreate: bool = False) -> None:
        return None

    def load_all_records(self) -> list[ChunkRecord]:
        return []


def test_rag_pipeline_returns_answer_with_sources(monkeypatch: Any) -> None:
    fake_store = _FakeVectorStore()

    class _FakeDatabase:
        backend_name = "json"

        def search(self, embedding: list[float]) -> list[tuple[Any, ...]]:
            records = fake_store.search(np.asarray(embedding), limit=5, similarity_threshold=0.5)
            return [(record.chunk, record.source, record.page, 0.1) for record in records]

        def resolve_cache(self, _key: str) -> str | None:
            return None

        def set_cache(self, _key: str, _value: str) -> None:
            return None

    monkeypatch.setattr("rag_service.db", _FakeDatabase())
    service = RagService(embedding_chain=_FakeEmbeddingChain(), llm_chain=_FakeLLMChain())
    result = service.ask("How do I configure ClickHouse?")
    assert result["answer"] == "mock answer from docs"
    assert result["sources"]
    assert result["provider_used"] == "ollama"

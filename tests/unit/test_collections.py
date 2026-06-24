from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from config import Config
from core.collections import CollectionService, make_source
from core.database import DatabaseManager
from core.storage.json_store import JsonVectorStore
from core.storage.protocol import ChunkRecord


def _store_config(tmp_path: Path) -> Config:
    base = Config.from_env()
    return Config(
        project_root=base.project_root,
        instructions_dir=base.instructions_dir,
        local_vector_store_dir=tmp_path / "vectors",
        few_shot_folder=base.few_shot_folder,
        results_folder=tmp_path / "results",
        storage_backend="json",
        clickhouse_host=base.clickhouse_host,
        clickhouse_port=base.clickhouse_port,
        clickhouse_user=base.clickhouse_user,
        clickhouse_password=base.clickhouse_password,
        clickhouse_secure=base.clickhouse_secure,
        llm_provider=base.llm_provider,
        llm_fallback_enabled=base.llm_fallback_enabled,
        embedding_provider=base.embedding_provider,
        embedding_fallback_enabled=base.embedding_fallback_enabled,
        openai_api_key=base.openai_api_key,
        openai_model=base.openai_model,
        openai_embedding_model=base.openai_embedding_model,
        openai_max_tokens=base.openai_max_tokens,
        openai_max_requests_per_min=base.openai_max_requests_per_min,
        openai_daily_budget_usd=base.openai_daily_budget_usd,
        ollama_host=base.ollama_host,
        ollama_model=base.ollama_model,
        ollama_keep_alive=base.ollama_keep_alive,
        ollama_client_timeout=base.ollama_client_timeout,
        ollama_context_chunks=base.ollama_context_chunks,
        ollama_chunk_chars=base.ollama_chunk_chars,
        embed_model=base.embed_model,
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        rerank_top_k=3,
        similarity_threshold=0.99,
        batch_size=base.batch_size,
        max_text_length=base.max_text_length,
        min_chunk_size=10,
        max_chunks_per_doc=base.max_chunks_per_doc,
        num_ctx=base.num_ctx,
        num_predict=base.num_predict,
        temperature=base.temperature,
        top_p=base.top_p,
        repeat_penalty=base.repeat_penalty,
        cache_enabled=base.cache_enabled,
        cache_ttl=base.cache_ttl,
    )


@pytest.fixture()
def collection_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _store_config(tmp_path)
    store = JsonVectorStore(cfg)
    db = DatabaseManager.__new__(DatabaseManager)
    db._store = store
    db._cache = {}
    db._cache_time = {}
    monkeypatch.setattr("core.collections.db", db)
    service = CollectionService(root=tmp_path / "collections")
    return service, store, db


def test_create_list_and_delete_collection(collection_env) -> None:
    service, _store, _db = collection_env
    created = service.create_collection("Operators", collection_id="ops")
    assert created.id == "ops"
    assert len(service.list_collections()) == 1

    service.delete_collection("ops")
    assert service.list_collections() == []


def test_ingest_document_and_search_scope(collection_env) -> None:
    service, store, _db = collection_env
    service.create_collection("Manual", collection_id="manual")

    other = ChunkRecord.create_now(
        id="99",
        source="legacy/other.txt",
        page=1,
        chunk="legacy content about unrelated topic",
        embedding=np.array([0.0, 1.0], dtype=np.float64),
    )
    store.insert_batch([other])

    text = (
        "Кнопка «Новый документ» предназначена для формирования нового документа, "
        "в котором отсутствуют CDPL-процедуры и связи."
    )
    fake_vectors = [[1.0, 0.0]]

    with patch("core.collections.EmbeddingChain") as mock_chain:
        mock_chain.return_value.embed.return_value = fake_vectors
        doc = service.ingest_document("manual", "ui.txt", text.encode("utf-8"))

    assert doc.chunk_count >= 1
    assert doc.source == make_source("manual", "ui.txt")

    service.set_selection(["manual"])
    scoped = store.search(
        np.array([1.0, 0.0], dtype=np.float64),
        limit=5,
        similarity_threshold=0.5,
        source_prefixes=service.active_source_prefixes(),
    )
    assert len(scoped) == 1
    assert scoped[0].source.startswith("collections/manual/")

    service.delete_document("manual", "ui.txt")
    assert service.list_documents("manual") == []


def test_selection_empty_searches_all(collection_env) -> None:
    service, store, _db = collection_env
    record = ChunkRecord.create_now(
        id="1",
        source="global/doc.txt",
        page=1,
        chunk="global answer",
        embedding=np.array([1.0, 0.0], dtype=np.float64),
    )
    store.insert_batch([record])
    assert service.active_source_prefixes() is None
    assert service.active_source_prefixes([]) is None

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from config import Config
from core.collections import CollectionService
from core.database import DatabaseManager
from core.sources import SourceService, decode_source_id, encode_source_id
from core.storage.json_store import JsonVectorStore
from core.storage.protocol import ChunkRecord


def _store_config(tmp_path: Path) -> Config:
    base = Config.from_env()
    return Config(
        project_root=tmp_path,
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
def source_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _store_config(tmp_path)
    store = JsonVectorStore(cfg)
    db = DatabaseManager.__new__(DatabaseManager)
    db._store = store
    db._cache = {}
    db._cache_time = {}
    monkeypatch.setattr("core.collections.db", db)
    monkeypatch.setattr("core.sources.db", db)
    collections_root = tmp_path / "data" / "collections"
    collections = CollectionService(root=collections_root)
    monkeypatch.setattr("core.sources.collection_service", collections)
    monkeypatch.setattr("core.sources.config", cfg)
    sources = SourceService()
    return collections, sources, store, db


def test_encode_decode_source_id() -> None:
    source = "collections/manual/ui.txt"
    source_id = encode_source_id(source)
    assert decode_source_id(source_id) == source


def test_list_and_delete_collection_source(source_env) -> None:
    collections, sources, _store, _db = source_env
    collections.create_collection("Manual", collection_id="manual")
    text = (
        "Кнопка «Новый документ» предназначена для формирования нового документа, "
        "в котором отсутствуют CDPL-процедуры и связи."
    )
    with patch("core.collections.EmbeddingChain") as mock_chain:
        mock_chain.return_value.embed.return_value = [[1.0, 0.0]]
        collections.ingest_document("manual", "ui.txt", text.encode("utf-8"))

    listed = sources.list_sources()
    assert len(listed) == 1
    assert listed[0].collection_id == "manual"
    assert listed[0].filename == "ui.txt"

    path = sources.resolve_file_path(listed[0].source)
    assert path.is_file()

    result = sources.delete_source(listed[0].id)
    assert result["reindexed"] is True
    assert sources.list_sources() == []

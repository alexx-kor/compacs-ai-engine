from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import Config
from core.storage.json_store import JsonVectorStore
from core.storage.protocol import ChunkRecord


def _store_config(tmp_path: Path) -> Config:
    base = Config.from_env()
    return Config(
        project_root=base.project_root,
        instructions_dir=base.instructions_dir,
        local_vector_store_dir=tmp_path,
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
        ollama_embed_max_chars=base.ollama_embed_max_chars,
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        top_k=base.top_k,
        rerank_top_k=base.rerank_top_k,
        similarity_threshold=0.99,
        batch_size=base.batch_size,
        max_text_length=base.max_text_length,
        min_chunk_size=base.min_chunk_size,
        max_chunks_per_doc=base.max_chunks_per_doc,
        num_ctx=base.num_ctx,
        num_predict=base.num_predict,
        temperature=base.temperature,
        top_p=base.top_p,
        repeat_penalty=base.repeat_penalty,
        cache_enabled=base.cache_enabled,
        cache_ttl=base.cache_ttl,
        chunk_strategies=base.chunk_strategies,
        hybrid_search_enabled=base.hybrid_search_enabled,
        hybrid_dense_weight=base.hybrid_dense_weight,
        hybrid_bm25_weight=base.hybrid_bm25_weight,
        rerank_lemmatize=base.rerank_lemmatize,
        bm25_lemmatize=base.bm25_lemmatize,
    )


def test_json_store_insert_and_search(tmp_path: Path) -> None:
    store = JsonVectorStore(_store_config(tmp_path))
    record = ChunkRecord.create_now(
        id="1",
        source="doc.txt",
        page=1,
        chunk="payment api parameter",
        embedding=np.array([1.0, 0.0], dtype=np.float64),
    )
    store.insert_batch([record])
    results = store.search(np.array([1.0, 0.0], dtype=np.float64), limit=5, similarity_threshold=0.5)
    assert len(results) == 1
    assert results[0].source == "doc.txt"


def test_json_store_rejects_empty_insert(tmp_path: Path) -> None:
    store = JsonVectorStore(_store_config(tmp_path))
    with pytest.raises(ValueError):
        store.insert_batch([])

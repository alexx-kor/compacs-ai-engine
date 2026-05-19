from __future__ import annotations

from pathlib import Path

import pytest

from config import Config
from core.cost_guard import BudgetExceededError, CostGuard, RateLimitExceededError


def _test_config(tmp_path: Path) -> Config:
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
        llm_provider="ollama",
        llm_fallback_enabled=True,
        embedding_provider="ollama",
        embedding_fallback_enabled=True,
        openai_api_key=None,
        openai_model=base.openai_model,
        openai_embedding_model=base.openai_embedding_model,
        openai_max_tokens=base.openai_max_tokens,
        openai_max_requests_per_min=2,
        openai_daily_budget_usd=0.01,
        ollama_host=base.ollama_host,
        ollama_model=base.ollama_model,
        embed_model=base.embed_model,
        ollama_embed_max_chars=base.ollama_embed_max_chars,
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        top_k=base.top_k,
        rerank_top_k=base.rerank_top_k,
        similarity_threshold=base.similarity_threshold,
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


def test_cost_guard_raises_on_rate_limit(tmp_path: Path) -> None:
    guard = CostGuard(_test_config(tmp_path), usage_file=tmp_path / "usage.json")
    guard.record_request()
    guard.record_request()
    with pytest.raises(RateLimitExceededError):
        guard.check_limits()


def test_cost_guard_raises_on_budget(tmp_path: Path) -> None:
    guard = CostGuard(_test_config(tmp_path), usage_file=tmp_path / "usage.json")
    guard._append_usage(total_tokens=1000, cost_usd=1.0)
    with pytest.raises(BudgetExceededError):
        guard.check_limits()

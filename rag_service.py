"""Unified RAG service used by CLI and HTTP APIs."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np

from config import config
from core.cost_guard import CostGuard
from core.database import db
from core.embeddings.chain import EmbeddingChain
from core.llm.chain import LLMChain
from core.reranker import reranker
from router.smart_router import select_prompt

log = logging.getLogger(__name__)


class RagService:
    """Orchestrate retrieval, reranking, prompt routing, and completion."""

    def __init__(
        self,
        embedding_chain: EmbeddingChain | None = None,
        llm_chain: LLMChain | None = None,
        cost_guard: CostGuard | None = None,
    ) -> None:
        self._embeddings = embedding_chain or EmbeddingChain(config)
        self._cost_guard = cost_guard or CostGuard(config)
        self._llm = llm_chain or LLMChain(config, self._cost_guard)

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a question using the configured storage and LLM chain."""
        started_at = time.time()
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = db.resolve_cache(cache_key)
        if cached:
            payload = json.loads(cached)
            payload["cached"] = True
            return payload

        query_embedding = np.asarray(self._embeddings.embed_cached(question), dtype=np.float64)
        search_results = db.search(query_embedding.tolist())
        if not search_results:
            return {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "time_total": round(time.time() - started_at, 2),
                "provider_used": "none",
                "storage_backend": db.backend_name,
            }

        reranked = reranker.rerank(question, search_results)
        context_parts: list[str] = []
        sources: list[tuple[Any, ...]] = []
        for chunk, source, page, _distance in reranked[: config.rerank_top_k]:
            context_parts.append("[%s, p.%s]\n%s" % (source, page, str(chunk)[:800]))
            sources.append((source, page))
        context = "\n\n".join(context_parts)
        system_prompt, _num_predict, temperature = select_prompt(question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "CONTEXT:\n%s\n\nQUESTION: %s" % (context, question)},
        ]
        answer, provider_used = self._llm.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=config.openai_max_tokens,
        )
        payload: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "time_total": round(time.time() - started_at, 2),
            "cached": False,
            "provider_used": provider_used,
            "embedding_provider": self._embeddings.active_provider,
            "storage_backend": db.backend_name,
            "model": config.openai_model if provider_used == "openai" else config.ollama_model,
        }
        db.set_cache(cache_key, json.dumps(payload))
        return payload


rag_service = RagService()

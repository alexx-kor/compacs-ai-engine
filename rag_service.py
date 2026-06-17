"""Unified RAG service used by CLI and HTTP APIs."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
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

    def _build_context(
        self,
        question: str,
        *,
        collection_ids: list[str] | None,
    ) -> dict[str, Any]:
        from core.collections import collection_service

        source_prefixes = collection_service.active_source_prefixes(collection_ids)
        query_embedding = np.asarray(self._embeddings.embed_cached(question), dtype=np.float64)
        search_results = db.search(query_embedding.tolist(), source_prefixes=source_prefixes)
        if not search_results:
            return {
                "question": question,
                "search_results": [],
                "context": "",
                "sources": [],
                "messages": [],
                "temperature": 0.0,
                "collection_ids": collection_ids or collection_service.get_selection(),
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
        return {
            "question": question,
            "search_results": search_results,
            "context": context,
            "sources": sources,
            "messages": messages,
            "temperature": temperature,
            "collection_ids": collection_ids or collection_service.get_selection(),
        }

    def ask(
        self,
        question: str,
        *,
        collection_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Answer a question using the configured storage and LLM chain."""
        started_at = time.time()
        scope_key = ",".join(sorted(collection_ids or []))
        cache_key = hashlib.md5(f"{scope_key}:{question}".encode()).hexdigest()
        cached = db.resolve_cache(cache_key)
        if cached:
            payload = json.loads(cached)
            payload["cached"] = True
            return payload

        built = self._build_context(question, collection_ids=collection_ids)
        if not built["search_results"]:
            return {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "time_total": round(time.time() - started_at, 2),
                "provider_used": "none",
                "storage_backend": db.backend_name,
                "collection_ids": built["collection_ids"],
            }

        answer, provider_used = self._llm.complete(
            messages=built["messages"],
            temperature=built["temperature"],
            max_tokens=config.openai_max_tokens,
        )
        payload: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": built["sources"],
            "time_total": round(time.time() - started_at, 2),
            "cached": False,
            "provider_used": provider_used,
            "embedding_provider": self._embeddings.active_provider,
            "storage_backend": db.backend_name,
            "model": config.openai_model if provider_used == "openai" else config.ollama_model,
            "collection_ids": built["collection_ids"],
        }
        db.set_cache(cache_key, json.dumps(payload))
        return payload

    def ask_stream(
        self,
        question: str,
        *,
        collection_ids: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-style events: status → token* → done."""
        started_at = time.time()
        scope_key = ",".join(sorted(collection_ids or []))
        cache_key = hashlib.md5(f"{scope_key}:{question}".encode()).hexdigest()
        cached = db.resolve_cache(cache_key)
        if cached:
            payload = json.loads(cached)
            payload["cached"] = True
            yield {"event": "status", "data": {"phase": "cache"}}
            yield {"event": "token", "data": {"text": str(payload.get("answer", ""))}}
            yield {"event": "done", "data": payload}
            return

        yield {"event": "status", "data": {"phase": "retrieval"}}
        built = self._build_context(question, collection_ids=collection_ids)
        if not built["search_results"]:
            payload = {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "time_total": round(time.time() - started_at, 2),
                "provider_used": "none",
                "storage_backend": db.backend_name,
                "collection_ids": built["collection_ids"],
                "cached": False,
            }
            yield {"event": "done", "data": payload}
            return

        provider_used = "unknown"
        parts: list[str] = []
        yield {"event": "status", "data": {"phase": "generation"}}
        for token, provider_used in self._llm.stream_complete(
            messages=built["messages"],
            temperature=built["temperature"],
            max_tokens=config.openai_max_tokens,
        ):
            parts.append(token)
            yield {"event": "token", "data": {"text": token}}

        answer = "".join(parts)
        payload = {
            "question": question,
            "answer": answer,
            "sources": built["sources"],
            "time_total": round(time.time() - started_at, 2),
            "cached": False,
            "provider_used": provider_used,
            "embedding_provider": self._embeddings.active_provider,
            "storage_backend": db.backend_name,
            "model": config.openai_model if provider_used == "openai" else config.ollama_model,
            "collection_ids": built["collection_ids"],
        }
        db.set_cache(cache_key, json.dumps(payload))
        yield {"event": "done", "data": payload}


rag_service = RagService()

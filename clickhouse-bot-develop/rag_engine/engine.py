"""RAG engine: retrieve, rerank, and answer with OpenAI chat completions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, cast

from config import config
from core.database import db
from core.embeddings import embedder
from core.openai_client import get_openai_client
from core.reranker import reranker
from router.smart_router import select_prompt


log = logging.getLogger(__name__)

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))


class RAGEngine:
    """Run retrieval-augmented generation with optional in-process response cache."""

    @staticmethod
    def ask(question: str) -> dict[str, Any]:
        """Answer a question using vector search, reranking, and GPT.

        Args:
            question: Natural-language query.

        Returns:
            Result payload including ``question``, ``answer``, ``sources``,
            ``time_total``, and optional ``cached`` / ``status`` / ``model``.
        """
        t_start = time.time()

        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = db.resolve_cache(cache_key)
        if cached:
            result = cast(dict[str, Any], json.loads(cached))
            result["cached"] = True
            return result

        q_emb = list(embedder.generate_cached(question))
        results = db.search(q_emb)

        if not results:
            return {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "time_total": round(time.time() - t_start, 2),
            }

        reranked = reranker.rerank(question, results)

        context_parts: list[str] = []
        sources: list[tuple[Any, ...]] = []
        for row in reranked[: config.rerank_top_k]:
            chunk, source, page = row[0], row[1], row[2]
            context_parts.append("[%s, p.%s]\n%s" % (source, page, str(chunk)[:800]))
            sources.append((source, page))

        context = "\n\n".join(context_parts)

        system_prompt, _num_predict, temperature = select_prompt(question)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "CONTEXT:\n%s\n\nQUESTION: %s" % (context, question)},
        ]

        answer = ""
        status = "error"

        try:
            response = get_openai_client().chat.completions.create(
                model=GPT_MODEL,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                top_p=config.top_p,
            )
            raw_answer = response.choices[0].message.content
            answer = raw_answer if raw_answer is not None else ""
            status = "success"
            usage = response.usage
            if usage is not None:
                log.info(
                    "   [TOKENS] prompt=%s completion=%s total=%s",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )
        except Exception as exc:
            answer = "ERROR: %s" % exc
            status = "error"

        payload: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "time_total": round(time.time() - t_start, 2),
            "cached": False,
            "status": status,
            "model": GPT_MODEL,
        }

        db.set_cache(cache_key, json.dumps(payload))
        return payload


rag = RAGEngine()

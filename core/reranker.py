"""Lexical reranker blending dense score, token overlap, and optional lemmas."""

from __future__ import annotations

import logging
import re

from config import config
from core.text_processing import expand_query_tokens, tokenize_for_search

log = logging.getLogger(__name__)

_CHUNK_TYPE_BOOST = {
    "graph": 0.12,
    "qa": 0.10,
    "section": 0.06,
    "definition": 0.05,
    "lemma_hint": 0.04,
}


class Reranker:
    @staticmethod
    def rerank(question: str, results: list[tuple]) -> list[tuple]:
        if not results:
            return results

        q_words = set(re.findall(r"\b\w{3,}\b", question.lower(), flags=re.UNICODE))
        q_lemma = set(expand_query_tokens(question)) if config.rerank_lemmatize else q_words

        scored: list[tuple[float, tuple]] = []
        for result in results:
            chunk, source, page, distance = result[:4]
            chunk_text = str(chunk).lower()
            c_words = set(re.findall(r"\b\w{3,}\b", chunk_text, flags=re.UNICODE))
            c_lemma = set(tokenize_for_search(chunk_text, lemmatize=config.rerank_lemmatize))

            overlap = len(q_words & c_words) / max(len(q_words), 1)
            lemma_overlap = len(q_lemma & c_lemma) / max(len(q_lemma), 1)
            similarity = 1.0 - float(distance)
            type_boost = _chunk_type_boost(chunk_text, str(source))

            final_score = (
                similarity * 0.50
                + overlap * 0.25
                + lemma_overlap * 0.15
                + type_boost * 0.10
            )
            scored.append((final_score, result))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: config.rerank_top_k]]


def _chunk_type_boost(chunk_text: str, source: str) -> float:
    for marker, boost in _CHUNK_TYPE_BOOST.items():
        if f"[{marker}" in chunk_text.lower() or source.startswith(f"{marker}/"):
            return boost
    if source.startswith("graph/"):
        return _CHUNK_TYPE_BOOST["graph"]
    if source.startswith("qa/"):
        return _CHUNK_TYPE_BOOST["qa"]
    return 0.0


reranker = Reranker()

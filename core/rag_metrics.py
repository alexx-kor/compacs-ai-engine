"""RAG evaluation metrics: retrieval hit@k, faithfulness, token estimates."""

from __future__ import annotations

import re
from typing import Any

from core.evaluation_utils import tokenize_overlap

_KEY_FACT_BACKTICK = re.compile(r"`([^`]{2,})`")
_KEY_FACT_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_KEY_FACT_ENV = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_KEY_FACT_NUMBER = re.compile(r"\b\d{3,5}\b")


def extract_key_facts(text: str) -> list[str]:
    """Pull high-signal literals from golden/answer text (IPs, env vars, codes)."""
    if not text or not text.strip():
        return []
    facts: list[str] = []
    seen: set[str] = set()
    for pattern in (_KEY_FACT_BACKTICK, _KEY_FACT_IP, _KEY_FACT_ENV, _KEY_FACT_NUMBER):
        for match in pattern.findall(text):
            normalized = str(match).strip().lower()
            if len(normalized) >= 2 and normalized not in seen:
                seen.add(normalized)
                facts.append(normalized)
    return facts


def _fact_in_text(fact: str, haystack: str) -> bool:
    return fact.lower() in haystack.lower()


def hit_at_k(
    reference_text: str,
    context_chunks: list[str],
    *,
    k: int = 3,
) -> float:
    """
    Fraction of key facts from ``reference_text`` found in any of the top-k chunks.

    Used for retrieval: ``reference_text`` is the golden expected answer.
    Returns value in [0, 1].
    """
    facts = extract_key_facts(reference_text)
    if not facts:
        overlap = tokenize_overlap(reference_text)
        if not overlap:
            return 1.0 if context_chunks[:k] else 0.0
        pool = " ".join(context_chunks[:k]).lower()
        hits = sum(1 for token in overlap if token in pool)
        return hits / len(overlap)

    top = context_chunks[:k]
    if not top:
        return 0.0
    pool = "\n".join(top)
    hits = sum(1 for fact in facts if _fact_in_text(fact, pool))
    return hits / len(facts)


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    Fraction of key facts in the generated answer that appear in retrieved context.

    Returns value in [0, 1]. Empty answer → 0.
    """
    if not answer or not answer.strip():
        return 0.0
    facts = extract_key_facts(answer)
    if not facts:
        answer_tokens = tokenize_overlap(answer)
        if not answer_tokens:
            return 0.0
        pool = "\n".join(context_chunks).lower()
        hits = sum(1 for token in answer_tokens if token in pool)
        return hits / len(answer_tokens)

    pool = "\n".join(context_chunks)
    if not pool.strip():
        return 0.0
    hits = sum(1 for fact in facts if _fact_in_text(fact, pool))
    return hits / len(facts)


def estimate_tokens(text: str, *, provider: str = "openai") -> int:
    """Estimate token count when the API does not return usage."""
    if not text:
        return 0
    if provider == "openai":
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def merge_usage_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    """Sum token and cost fields from multiple usage dicts."""
    prompt = 0
    completion = 0
    cost = 0.0
    for part in parts:
        if not part:
            continue
        prompt += int(part.get("prompt_tokens", 0) or 0)
        completion += int(part.get("completion_tokens", 0) or 0)
        cost += float(part.get("cost_usd", 0.0) or 0.0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": round(cost, 6),
    }

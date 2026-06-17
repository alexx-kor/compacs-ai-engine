"""Input query filtering to drop irrelevant noise before retrieval."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_SPLIT_RE = re.compile(
    r"(?:\n+|[!?;]+|\b(?:а ещё|а еще|и ещё|и еще|кстати|еще|также|also|and\s+also)\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterResult:
    original: str
    filtered: str
    applied: bool
    reason: str
    bm25_score: float = 0.0


def split_candidates(text: str) -> list[str]:
    """Split a noisy user message into candidate question segments."""
    parts = [segment.strip() for segment in _SPLIT_RE.split(text) if segment and segment.strip()]
    return [part for part in parts if len(part) >= 6] or [text.strip()]


def _is_enabled() -> bool:
    return os.environ.get("QUERY_FILTER_ENABLED", "false").lower() in ("1", "true", "yes")


def filter_query(text: str, **_: object) -> FilterResult:
    """Return the best domain-relevant segment or the original query."""
    original = (text or "").strip()
    if not original:
        return FilterResult(
            original=text,
            filtered=text,
            applied=False,
            reason="empty",
        )
    if not _is_enabled():
        return FilterResult(
            original=original,
            filtered=original,
            applied=False,
            reason="disabled",
        )

    candidates = split_candidates(original)
    if len(candidates) <= 1:
        return FilterResult(
            original=original,
            filtered=original,
            applied=False,
            reason="single_segment",
        )

    # Without BM25 index, prefer the longest question-like segment.
    def score(segment: str) -> tuple[int, int]:
        question_bonus = 1 if "?" in segment else 0
        return (question_bonus, len(segment))

    best = max(candidates, key=score)
    if best == original:
        return FilterResult(
            original=original,
            filtered=original,
            applied=False,
            reason="unchanged",
        )
    return FilterResult(
        original=original,
        filtered=best,
        applied=True,
        reason="longest_question_segment",
    )

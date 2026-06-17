"""Shared helpers for golden-set evaluation scoring."""

from __future__ import annotations

import re

NOT_FOUND_PHRASES = (
    "information not found in the current documentation index",
    "not found in documentation",
    "not found in the documentation",
)


def is_not_found_answer(text: str) -> bool:
    """
    Return True only when the answer is *predominantly* a refusal.

    Structured answers that fill sections 2–7 with "not found" while section 1
    has real content must NOT be treated as a global refusal.
    """
    normalized = text.strip().lower()
    if not any(phrase in normalized for phrase in NOT_FOUND_PHRASES):
        return False
    if normalized in {phrase.lower() for phrase in NOT_FOUND_PHRASES}:
        return True

    cleaned = normalized
    for phrase in NOT_FOUND_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"[\s\-–—•.]+", " ", cleaned).strip()
    # Substantive content remains after stripping boilerplate refusals
    return len(cleaned) < 80


def is_unanswerable_expected(expected_answer: str) -> bool:
    """Return True when the golden reference says the docs have no answer."""
    return is_not_found_answer(expected_answer)


def tokenize_overlap(text: str) -> set[str]:
    """Tokenize RU/EN words (length >= 3) for lexical overlap metrics."""
    return set(re.findall(r"[a-zA-Zа-яёА-ЯЁ0-9_]{3,}", text.lower()))

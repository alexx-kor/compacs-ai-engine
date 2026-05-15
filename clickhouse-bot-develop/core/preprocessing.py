"""Text preprocessing profiles for embedding cleanliness experiments."""

from __future__ import annotations

import re


def preprocess_text(text: str, profile: str) -> str:
    """Normalize text by profile name.

    Supported profiles:
    - ``baseline``: light cleanup
    - ``experiment``: stricter normalization
    """
    if profile == "baseline":
        return _baseline_clean(text)
    if profile == "experiment":
        return _experiment_clean(text)
    raise ValueError(f"Unknown preprocessing profile: {profile}")


def _baseline_clean(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _experiment_clean(text: str) -> str:
    cleaned = _baseline_clean(text)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\.,:;!?/\-()]", " ", cleaned)
    cleaned = re.sub(r"\b\d{16,}\b", " ", cleaned)
    cleaned = re.sub(r"([!?.,:;])\1{1,}", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


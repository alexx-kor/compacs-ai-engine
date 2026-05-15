from __future__ import annotations

from core.preprocessing import preprocess_text


def test_baseline_preprocess_compacts_whitespace() -> None:
    raw = "line one\r\n\r\nline   two"
    cleaned = preprocess_text(raw, "baseline")
    assert "\r" not in cleaned
    assert "line one" in cleaned
    assert "line two" in cleaned


def test_experiment_preprocess_strips_urls_and_email() -> None:
    raw = "Visit https://example.com or email user@example.com"
    cleaned = preprocess_text(raw, "experiment")
    assert "https://example.com" not in cleaned
    assert "user@example.com" not in cleaned

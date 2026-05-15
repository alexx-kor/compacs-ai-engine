from __future__ import annotations

from core.preprocessing import preprocess_text


def test_baseline_preprocess_compacts_whitespace() -> None:
    raw = "Line 1\r\n\r\n\r\nLine\t\t2    with   spaces"
    cleaned = preprocess_text(raw, "baseline")
    assert "\r" not in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Line 1" in cleaned
    assert "Line 2 with spaces" in cleaned


def test_experiment_preprocess_strips_urls_and_email() -> None:
    raw = "Reach me at user@example.com and visit https://example.com NOW!!!"
    cleaned = preprocess_text(raw, "experiment")
    assert "example.com" not in cleaned
    assert "@" not in cleaned
    assert "!!!" not in cleaned


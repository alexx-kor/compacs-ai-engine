from __future__ import annotations

from typing import Any

import app


def test_run_judge_single_returns_summary(monkeypatch: Any) -> None:
    rows = [
        {"id": 1, "question": "Q1", "answer": "A1"},
        {"id": 2, "question": "Q2", "answer": "A2"},
    ]
    monkeypatch.setattr(app, "load_json", lambda _path: rows)
    monkeypatch.setattr(app, "llm_evaluate_percent", lambda _q, _a: {"total": 80.0})

    payload = app.run_judge_single("fake.json")

    assert payload["mode"] == "single"
    assert payload["summary"]["count"] == 2
    assert payload["summary"]["avg_total"] == 80.0
    assert len(payload["results"]) == 2


def test_run_judge_pairwise_counts_wins(monkeypatch: Any) -> None:
    main: list[dict[str, Any]] = [{"id": 1, "question": "Q1", "answer": "A-main"}]
    hyp: list[dict[str, Any]] = [{"id": 1, "question": "Q1", "answer": "A-hyp"}]

    def fake_load(path: str) -> list[dict[str, Any]]:
        return main if "main" in path else hyp

    monkeypatch.setattr(app, "load_json", fake_load)
    monkeypatch.setattr(app, "llm_evaluate_percent", lambda _q, _a: {"total": 75.0})
    monkeypatch.setattr(
        app,
        "compare_two_answers_percent",
        lambda _q, _a, _b: {"winner": "A", "winner_percent": 15.0},
    )

    payload = app.run_judge_pairwise("main.json", "hyp.json")

    assert payload["mode"] == "pairwise"
    assert payload["summary"]["count"] == 1
    assert payload["summary"]["wins_main"] == 1
    assert payload["summary"]["wins_hypothesis"] == 0


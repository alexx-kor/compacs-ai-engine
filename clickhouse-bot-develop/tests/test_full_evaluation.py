from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

import full_evaluation as fe


def test_metric_scores_is_frozen() -> None:
    scores = fe.MetricScores(relevance=5.0)
    with pytest.raises(FrozenInstanceError):
        scores.relevance = 9.0  # type: ignore[misc]


def test_evaluation_result_is_frozen_and_replaceable() -> None:
    result = fe.AnswerEvaluator.evaluate(
        question="What is a merchant control key?",
        answer="The merchant control key is used for signing requests.",
        time_seconds=1.5,
        tokens=42,
    )
    updated = replace(result, id=13, sources=[("api.md", 1)])
    assert updated.id == 13
    assert updated.sources == [("api.md", 1)]
    assert result.id == 0


def test_calculate_relevance_zero_for_error_answer() -> None:
    assert fe.metrics.calculate_relevance("question text", "ERROR: timeout") == 0.0


def test_calculate_relevance_positive_for_matching_words() -> None:
    score = fe.metrics.calculate_relevance(
        "merchant control key signing",
        "The merchant control key is used when signing API requests.",
    )
    assert score > 0.0


def test_answer_evaluator_grade_excellent() -> None:
    answer = (
        "Follow these steps to integrate the sale form. "
        "First configure the merchant control key. "
        "For example, use the signing parameter in the request. "
        "Therefore the integration completes successfully."
    )
    result = fe.AnswerEvaluator.evaluate(
        question="Create a step by step guide how to integrate sale form",
        answer=answer,
        time_seconds=2.0,
        tokens=100,
    )
    assert result.grade in (
        fe.GradeLevel.EXCELLENT,
        fe.GradeLevel.GOOD,
        fe.GradeLevel.SATISFACTORY,
    )
    assert result.final_score >= 5.0
    assert result.scores.toxicity == 0.0


def test_rag_ask_empty_search_returns_not_found(monkeypatch: Any) -> None:
    class _FakeEmbedder:
        def generate_cached(self, _q: str) -> object:
            return iter([0.1, 0.2])

    class _FakeDB:
        def search(self, _emb: object) -> list[object]:
            return []

    monkeypatch.setattr(fe, "embedder", _FakeEmbedder())
    monkeypatch.setattr(fe, "db", _FakeDB())

    rag = fe.RAGWithGPT()
    response = rag.ask("What is PCI?")

    assert response["answer"] == "NOT FOUND in documentation"
    assert response["sources"] == []
    assert response["tokens"] == 0

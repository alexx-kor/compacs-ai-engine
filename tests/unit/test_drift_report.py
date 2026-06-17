from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.drift_report import collect_quality_metrics, compare_datasets


def test_collect_quality_metrics_with_golden_and_empty_index(tmp_path: Path) -> None:
    golden = tmp_path / "baseline" / "golden_set.json"
    golden.parent.mkdir(parents=True)
    golden.write_text(
        json.dumps(
            [
                {"id": 1, "question": "Q?", "expected_answer": "A" * 500},
                {"id": 2, "question": "Q2?", "expected_answer": "B" * 600},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = collect_quality_metrics(tmp_path, chunk_lengths=[], chunk_count=0, source_count=0)

    assert report["overall_verdict"] in ("significant_drift", "moderate_drift", "unavailable")
    index_check = next(c for c in report["checks"] if c["id"] == "index_vs_golden")
    assert index_check.get("skipped") is True
    assert report["index"]["chunk_count"] == 0


def test_collect_quality_metrics_detects_index_drift(tmp_path: Path) -> None:
    golden = tmp_path / "baseline" / "golden_set.json"
    golden.parent.mkdir(parents=True)
    golden.write_text(
        json.dumps(
            [{"id": i, "question": "Q", "expected_answer": "X" * 2000} for i in range(10)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    short_chunks = [50.0, 60.0, 55.0, 45.0, 52.0]

    report = collect_quality_metrics(
        tmp_path,
        chunk_lengths=short_chunks,
        chunk_count=len(short_chunks),
        source_count=1,
    )

    index_check = next(c for c in report["checks"] if c["id"] == "index_vs_golden")
    assert index_check.get("skipped") is not True
    assert index_check["overall_verdict"] in ("moderate_drift", "significant_drift")


def test_compare_datasets_eval(tmp_path: Path) -> None:
    def _write(path: Path, scores: list[float]) -> None:
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "id": i + 1,
                            "question": "Q?",
                            "expected_answer": "A" * 100,
                            "final_score": score,
                            "llm_judge": {"total": score},
                            "rag_metrics": {"faithfulness": score / 100},
                        }
                        for i, score in enumerate(scores)
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    ref = tmp_path / "base.json"
    cur = tmp_path / "ft.json"
    _write(ref, [70.0, 72.0, 74.0, 76.0])
    _write(cur, [90.0, 92.0, 94.0, 96.0])

    report = compare_datasets(ref, cur, metrics=["judge_percent"], bins=4, alpha=0.05)
    judge = report["metrics"]["judge_percent"]
    assert judge["skipped"] is False
    assert judge["psi"]["psi"] > 0.05

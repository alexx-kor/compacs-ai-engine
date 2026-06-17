from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import load_script_module

monitor = load_script_module("monitor_metric_dispersion")


def test_dispersion_stats_single_value() -> None:
    stats = monitor.dispersion_stats([85.0])
    assert stats["n"] == 1
    assert stats["mean"] == 85.0
    assert stats["std"] == 0.0
    assert stats["variance"] == 0.0


def test_bootstrap_curve_variance_decreases_with_n() -> None:
    values = [60.0, 70.0, 80.0, 90.0, 85.0, 75.0, 88.0, 92.0]
    curve = monitor.bootstrap_curve(values, [4, 8], repeats=300, seed=42)
    assert curve[0]["n"] == 4
    assert curve[1]["n"] == 8
    assert curve[1]["std_of_mean"] <= curve[0]["std_of_mean"]


def test_analyze_eval_from_fixture(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "config": {"model": "llama3.2:3b", "golden_path": "val.json"},
                "results": [
                    {"id": 1, "final_score": 8.0, "llm_judge": {"total": 80.0}, "rag_metrics": {"faithfulness": 0.5, "hit_at_3": 1.0}},
                    {"id": 2, "final_score": 7.0, "llm_judge": {"total": 70.0}, "rag_metrics": {"faithfulness": 0.4, "hit_at_3": 0.0}},
                    {"id": 3, "final_score": 9.0, "llm_judge": {"total": 90.0}, "rag_metrics": {"faithfulness": 0.6, "hit_at_3": 1.0}},
                    {"id": 4, "final_score": 6.0, "llm_judge": {"total": 60.0}, "rag_metrics": {"faithfulness": 0.3, "hit_at_3": 1.0}},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = monitor.analyze_eval(
        eval_path,
        metrics=["judge_percent", "final_score"],
        min_n=2,
        step=2,
        bootstrap_repeats=50,
        seed=42,
    )
    assert report["total_questions"] == 4
    assert report["metrics"]["judge_percent"]["full_sample"]["mean"] == 75.0
    assert len(report["metrics"]["judge_percent"]["by_sample_size"]) == 2

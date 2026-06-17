from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import load_script_module

drift = load_script_module("monitor_data_drift")


def _write_eval(path: Path, scores: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {"model": "test"},
                "results": [
                    {
                        "id": i + 1,
                        "question": f"Q{i}?",
                        "expected_answer": "A" * (50 + i),
                        "final_score": score,
                        "llm_judge": {"total": score},
                        "rag_metrics": {"faithfulness": score / 100, "hit_at_3": 1.0},
                    }
                    for i, score in enumerate(scores)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_compare_eval_json_with_drift(tmp_path: Path) -> None:
    ref = tmp_path / "base.json"
    cur = tmp_path / "ft.json"
    _write_eval(ref, [70.0, 72.0, 74.0, 76.0, 78.0, 80.0])
    _write_eval(cur, [85.0, 87.0, 89.0, 91.0, 93.0, 95.0])

    report = drift.compare_datasets(
        ref,
        cur,
        metrics=["judge_percent", "final_score"],
        bins=4,
        alpha=0.05,
    )

    judge = report["metrics"]["judge_percent"]
    assert judge["skipped"] is False
    assert judge["psi"]["psi"] > 0.05
    assert judge["ks"]["drift_detected"] is True
    assert judge["overall_verdict"] in ("moderate_drift", "significant_drift")

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import load_script_module

summarize = load_script_module("summarize_ui_split_eval")


def test_metric_delta() -> None:
    base = {"llm_judge_avg_percent": 70.0, "avg_score": 0.5}
    ft = {"llm_judge_avg_percent": 78.5, "avg_score": 0.62}
    assert summarize._metric_delta(base, ft, "llm_judge_avg_percent") == 8.5
    assert summarize._metric_delta(base, ft, "avg_score") == 0.12
    assert summarize._metric_delta(base, {}, "avg_score") is None


def test_stats_reads_statistics_block(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps({"statistics": {"llm_judge_avg_percent": 82.0, "avg_score": 0.7}}),
        encoding="utf-8",
    )
    assert summarize._stats(path)["llm_judge_avg_percent"] == 82.0

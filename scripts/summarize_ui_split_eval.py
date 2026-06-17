#!/usr/bin/env python3
"""Summarize VAL/TEST baseline vs finetuned evaluation deltas."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("statistics", {})


def _metric_delta(base: dict, ft: dict, key: str) -> float | None:
    if base.get(key) is None or ft.get(key) is None:
        return None
    return round(float(ft[key]) - float(base[key]), 3)


def main() -> int:
    files = {
        "val_base": ROOT / "evaluation_ui_extension_val_baseline.json",
        "val_ft": ROOT / "evaluation_ui_extension_val_finetuned.json",
        "test_base": ROOT / "evaluation_ui_extension_test_baseline.json",
        "test_ft": ROOT / "evaluation_ui_extension_test_finetuned.json",
    }
    for path in files.values():
        if not path.exists():
            print(f"Missing: {path}")
            return 1

    val_base, val_ft = _stats(files["val_base"]), _stats(files["val_ft"])
    test_base, test_ft = _stats(files["test_base"]), _stats(files["test_ft"])
    summary = {
        "val": {
            "baseline_judge": val_base.get("llm_judge_avg_percent"),
            "finetuned_judge": val_ft.get("llm_judge_avg_percent"),
            "delta_judge": _metric_delta(val_base, val_ft, "llm_judge_avg_percent"),
            "baseline_score": val_base.get("avg_score"),
            "finetuned_score": val_ft.get("avg_score"),
            "delta_score": _metric_delta(val_base, val_ft, "avg_score"),
        },
        "test": {
            "baseline_judge": test_base.get("llm_judge_avg_percent"),
            "finetuned_judge": test_ft.get("llm_judge_avg_percent"),
            "delta_judge": _metric_delta(test_base, test_ft, "llm_judge_avg_percent"),
            "baseline_score": test_base.get("avg_score"),
            "finetuned_score": test_ft.get("avg_score"),
            "delta_score": _metric_delta(test_base, test_ft, "avg_score"),
        },
    }
    out = ROOT / "data" / "finetune" / "ui_split_eval_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

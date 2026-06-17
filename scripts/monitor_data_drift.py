#!/usr/bin/env python3
"""Monitor data drift between reference and current datasets (PSI + KS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from core.drift_report import compare_datasets, dataset_kind, metrics_for_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSI + Kolmogorov–Smirnov data drift monitor")
    parser.add_argument("--reference", help="Reference JSON (QA, eval, or qa_audit)")
    parser.add_argument("--current", help="Current JSON to compare against reference")
    parser.add_argument(
        "--preset",
        choices=["splits", "eval-val", "eval-test"],
        help="Built-in comparisons (splits / baseline vs finetuned val|test)",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metrics (auto if empty)",
    )
    parser.add_argument("--bins", type=int, default=10, help="PSI bin count")
    parser.add_argument("--alpha", type=float, default=0.05, help="KS significance level")
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "finetune" / "data_drift_report.json"),
    )
    return parser.parse_args()


def _default_metrics_for_preset(preset: str) -> list[str]:
    if preset == "splits":
        return ["question_len", "answer_len"]
    return ["judge_percent", "final_score", "faithfulness", "hit_at_3"]


def main() -> int:
    args = parse_args()
    output: dict[str, Any] = {}

    if args.reference and args.current:
        ref_path = Path(args.reference)
        cur_path = Path(args.current)
        for path in (ref_path, cur_path):
            if not path.exists():
                print(f"Missing: {path}")
                return 1
        ref_kind = dataset_kind(ref_path)
        metrics = (
            [m.strip() for m in args.metrics.split(",") if m.strip()]
            or metrics_for_dataset(ref_kind)
        )
        output["comparison"] = compare_datasets(
            ref_path,
            cur_path,
            metrics=metrics,
            bins=args.bins,
            alpha=args.alpha,
        )
    elif args.preset == "splits":
        splits = ROOT / "instructions" / "golden" / "splits"
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or ["question_len", "answer_len"]
        pairs = {
            "train_vs_val": (splits / "ui_extension_qa_train.json", splits / "ui_extension_qa_val.json"),
            "train_vs_test": (splits / "ui_extension_qa_train.json", splits / "ui_extension_qa_test.json"),
            "val_vs_test": (splits / "ui_extension_qa_val.json", splits / "ui_extension_qa_test.json"),
        }
        output["preset"] = "splits"
        output["comparisons"] = {}
        for name, (ref, cur) in pairs.items():
            if not ref.exists() or not cur.exists():
                print(f"Missing split files for {name}")
                return 1
            output["comparisons"][name] = compare_datasets(
                ref, cur, metrics=metrics, bins=args.bins, alpha=args.alpha
            )
    elif args.preset in ("eval-val", "eval-test"):
        split = "val" if args.preset == "eval-val" else "test"
        ref_path = ROOT / f"evaluation_ui_extension_{split}_baseline.json"
        cur_path = ROOT / f"evaluation_ui_extension_{split}_finetuned.json"
        if not ref_path.exists() or not cur_path.exists():
            print(f"Missing eval files for {args.preset}")
            return 1
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or _default_metrics_for_preset(
            args.preset
        )
        output["preset"] = args.preset
        output["comparison"] = compare_datasets(
            ref_path, cur_path, metrics=metrics, bins=args.bins, alpha=args.alpha
        )
    else:
        print("Provide --reference + --current, or --preset splits|eval-val|eval-test")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

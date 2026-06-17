#!/usr/bin/env python3
"""Monitor metric mean/variance as evaluation sample size grows (bootstrap curve)."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

MetricFn = Callable[[dict[str, Any]], float | None]

METRIC_EXTRACTORS: dict[str, MetricFn] = {}


def _register_metrics() -> None:
    def judge(row: dict[str, Any]) -> float | None:
        judge_payload = row.get("llm_judge")
        if not isinstance(judge_payload, dict) or judge_payload.get("error"):
            return None
        total = judge_payload.get("total")
        return float(total) if total is not None else None

    def final_score(row: dict[str, Any]) -> float | None:
        value = row.get("final_score")
        return float(value) if value is not None else None

    def rag_metric(key: str) -> MetricFn:
        def _extract(row: dict[str, Any]) -> float | None:
            metrics = row.get("rag_metrics")
            if not isinstance(metrics, dict) or key not in metrics:
                return None
            return float(metrics[key])

        return _extract

    METRIC_EXTRACTORS.update(
        {
            "judge_percent": judge,
            "final_score": final_score,
            "faithfulness": rag_metric("faithfulness"),
            "hit_at_3": rag_metric("hit_at_3"),
        }
    )


_register_metrics()


def load_eval_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"unsupported evaluation JSON: {path}")


def extract_metric_values(results: list[dict[str, Any]], metric: str) -> list[float]:
    if metric not in METRIC_EXTRACTORS:
        raise KeyError(f"unknown metric: {metric}")
    values: list[float] = []
    for row in results:
        value = METRIC_EXTRACTORS[metric](row)
        if value is not None:
            values.append(value)
    return values


def dispersion_stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "variance": 0.0, "sem": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.fmean(values)
    if n == 1:
        std = 0.0
    else:
        std = statistics.stdev(values)
    variance = std * std
    sem = std / math.sqrt(n) if n else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "variance": round(variance, 4),
        "sem": round(sem, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def sample_sizes(total: int, *, min_n: int, step: int) -> list[int]:
    if total <= 0:
        return []
    start = max(2, min(min_n, total))
    sizes: list[int] = []
    current = start
    while current < total:
        sizes.append(current)
        current += step
    sizes.append(total)
    return sorted(set(sizes))


def bootstrap_curve(
    values: list[float],
    sizes: list[int],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not values:
        return []
    rng = random.Random(seed)
    total = len(values)
    curve: list[dict[str, Any]] = []

    for n in sizes:
        if n > total:
            continue
        means: list[float] = []
        variances: list[float] = []
        for _ in range(repeats):
            sample = rng.choices(values, k=n)
            means.append(statistics.fmean(sample))
            if n > 1:
                variances.append(statistics.variance(sample))
            else:
                variances.append(0.0)
        mean_of_means = statistics.fmean(means)
        std_of_means = statistics.stdev(means) if len(means) > 1 else 0.0
        mean_variance = statistics.fmean(variances)
        curve.append(
            {
                "n": n,
                "mean": round(mean_of_means, 4),
                "std_of_mean": round(std_of_means, 4),
                "variance_of_sample": round(mean_variance, 4),
                "sem": round(std_of_means, 4),
                "bootstrap_repeats": repeats,
            }
        )
    return curve


def analyze_eval(
    path: Path,
    *,
    metrics: list[str],
    min_n: int,
    step: int,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    results = load_eval_results(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {}) if isinstance(payload, dict) else {}

    metric_reports: dict[str, Any] = {}
    for metric in metrics:
        values = extract_metric_values(results, metric)
        sizes = sample_sizes(len(values), min_n=min_n, step=step)
        metric_reports[metric] = {
            "full_sample": dispersion_stats(values),
            "by_sample_size": bootstrap_curve(
                values,
                sizes,
                repeats=bootstrap_repeats,
                seed=seed,
            ),
        }

    return {
        "source": str(path.resolve()),
        "model": config.get("model"),
        "golden_path": config.get("golden_path"),
        "total_questions": len(results),
        "metrics": metric_reports,
    }


def compare_dispersion(
    baseline_report: dict[str, Any],
    finetuned_report: dict[str, Any],
    *,
    drift_bins: int = 10,
    drift_alpha: float = 0.05,
) -> dict[str, Any]:
    from core.drift_metrics import drift_summary

    comparison: dict[str, Any] = {}
    for metric in baseline_report.get("metrics", {}):
        if metric not in finetuned_report.get("metrics", {}):
            continue
        base_full = baseline_report["metrics"][metric]["full_sample"]
        ft_full = finetuned_report["metrics"][metric]["full_sample"]
        entry: dict[str, Any] = {
            "baseline_full": base_full,
            "finetuned_full": ft_full,
            "delta_mean": round(float(ft_full["mean"]) - float(base_full["mean"]), 4),
            "delta_std": round(float(ft_full["std"]) - float(base_full["std"]), 4),
            "delta_variance": round(float(ft_full["variance"]) - float(base_full["variance"]), 4),
        }
        comparison[metric] = entry

    if baseline_report.get("source") and finetuned_report.get("source"):
        base_path = Path(baseline_report["source"])
        ft_path = Path(finetuned_report["source"])
        if base_path.exists() and ft_path.exists():
            base_results = load_eval_results(base_path)
            ft_results = load_eval_results(ft_path)
            drift: dict[str, Any] = {}
            for metric in comparison:
                ref_vals = extract_metric_values(base_results, metric)
                cur_vals = extract_metric_values(ft_results, metric)
                drift[metric] = drift_summary(
                    ref_vals,
                    cur_vals,
                    metric_name=metric,
                    bins=drift_bins,
                    alpha=drift_alpha,
                )
            comparison["_drift"] = drift
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap learning curve: mean and variance vs sample size."
    )
    parser.add_argument("--eval-json", action="append", default=[], help="Evaluation JSON path")
    parser.add_argument(
        "--baseline-json",
        default="",
        help="Baseline eval JSON (pair with --finetuned-json for comparison)",
    )
    parser.add_argument(
        "--finetuned-json",
        default="",
        help="Fine-tuned eval JSON",
    )
    parser.add_argument(
        "--metrics",
        default="judge_percent,final_score,faithfulness,hit_at_3",
        help="Comma-separated metrics",
    )
    parser.add_argument("--min-n", type=int, default=4, help="Smallest subsample size")
    parser.add_argument("--step", type=int, default=4, help="Step between subsample sizes")
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-bins", type=int, default=10, help="PSI bins for baseline vs FT drift")
    parser.add_argument("--drift-alpha", type=float, default=0.05, help="KS alpha for drift test")
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "finetune" / "metric_dispersion_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    reports: list[dict[str, Any]] = []
    paths: list[Path] = [Path(p) for p in args.eval_json]
    if args.baseline_json:
        paths.append(Path(args.baseline_json))
    if args.finetuned_json:
        paths.append(Path(args.finetuned_json))

    for path in paths:
        if not path.exists():
            print(f"Missing: {path}")
            return 1
        reports.append(
            analyze_eval(
                path,
                metrics=metrics,
                min_n=args.min_n,
                step=args.step,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
            )
        )

    output: dict[str, Any] = {"reports": reports}
    if args.baseline_json and args.finetuned_json:
        base_report = next(r for r in reports if r["source"] == str(Path(args.baseline_json).resolve()))
        ft_report = next(r for r in reports if r["source"] == str(Path(args.finetuned_json).resolve()))
        output["comparison"] = compare_dispersion(
            base_report,
            ft_report,
            drift_bins=args.drift_bins,
            drift_alpha=args.drift_alpha,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

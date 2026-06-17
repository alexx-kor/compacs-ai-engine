"""Quality / drift metrics for monitoring (PSI + KS)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.drift_metrics import drift_summary

RowFn = Callable[[dict[str, Any]], Optional[float]]

VERDICT_RANK = {
    "unavailable": 0,
    "no_drift": 1,
    "moderate_drift": 2,
    "significant_drift": 3,
    "drift": 3,
}


def _load_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return "qa", payload
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return "eval", payload["results"]
        if isinstance(payload.get("rows"), list):
            return "qa_audit", payload["rows"]
    raise ValueError(f"unsupported JSON format: {path}")


def _metric_registry(kind: str) -> dict[str, RowFn]:
    def eval_judge(row: dict[str, Any]) -> float | None:
        judge = row.get("llm_judge")
        if not isinstance(judge, dict) or judge.get("error"):
            return None
        total = judge.get("total")
        return float(total) if total is not None else None

    def eval_rag(key: str) -> RowFn:
        def _fn(row: dict[str, Any]) -> float | None:
            metrics = row.get("rag_metrics")
            if not isinstance(metrics, dict) or key not in metrics:
                return None
            return float(metrics[key])

        return _fn

    common: dict[str, RowFn] = {
        "question_len": lambda r: float(len(str(r.get("question", "")))),
        "answer_len": lambda r: float(
            len(str(r.get("expected_answer", r.get("answer_preview", ""))))
        ),
    }

    if kind == "eval":
        return {
            **common,
            "judge_percent": eval_judge,
            "final_score": lambda r: float(r["final_score"])
            if r.get("final_score") is not None
            else None,
            "faithfulness": eval_rag("faithfulness"),
            "hit_at_3": eval_rag("hit_at_3"),
            "latency_seconds": eval_rag("latency_seconds"),
        }
    if kind == "qa_audit":
        return {
            **common,
            "heuristic_quality_pct": lambda r: float(r["heuristic_quality_pct"])
            if r.get("heuristic_quality_pct") is not None
            else None,
            "faithfulness_overlap_pct": lambda r: float(r["faithfulness_overlap_pct"])
            if r.get("faithfulness_overlap_pct") is not None
            else None,
        }
    return common


def extract_values(rows: list[dict[str, Any]], metric: str, registry: dict[str, RowFn]) -> list[float]:
    if metric not in registry:
        raise KeyError(f"unknown metric: {metric}")
    values: list[float] = []
    for row in rows:
        value = registry[metric](row)
        if value is not None:
            values.append(value)
    return values


def compare_datasets(
    reference_path: Path,
    current_path: Path,
    *,
    metrics: list[str],
    bins: int = 10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    ref_kind, ref_rows = _load_rows(reference_path)
    cur_kind, cur_rows = _load_rows(current_path)
    ref_registry = _metric_registry(ref_kind)
    cur_registry = _metric_registry(cur_kind)

    metric_reports: dict[str, Any] = {}
    for metric in metrics:
        if metric not in ref_registry or metric not in cur_registry:
            metric_reports[metric] = {
                "skipped": True,
                "reason": f"metric not available for {ref_kind}/{cur_kind}",
            }
            continue
        ref_values = extract_values(ref_rows, metric, ref_registry)
        cur_values = extract_values(cur_rows, metric, cur_registry)
        metric_reports[metric] = drift_summary(
            ref_values,
            cur_values,
            metric_name=metric,
            bins=bins,
            alpha=alpha,
        )

    return {
        "reference": {"path": str(reference_path.resolve()), "kind": ref_kind, "rows": len(ref_rows)},
        "current": {"path": str(current_path.resolve()), "kind": cur_kind, "rows": len(cur_rows)},
        "metrics": metric_reports,
    }


def dataset_kind(path: Path) -> str:
    kind, _ = _load_rows(path)
    return kind


def metrics_for_dataset(kind: str) -> list[str]:
    return list(_metric_registry(kind).keys())


def default_pair_report(project_root: Path) -> dict[str, Any]:
    splits = project_root / "instructions" / "golden" / "splits"
    return {
        "train_vs_val": compare_datasets(
            splits / "ui_extension_qa_train.json",
            splits / "ui_extension_qa_val.json",
            metrics=["question_len", "answer_len"],
            bins=8,
            alpha=0.05,
        ),
        "train_vs_test": compare_datasets(
            splits / "ui_extension_qa_train.json",
            splits / "ui_extension_qa_test.json",
            metrics=["question_len", "answer_len"],
            bins=8,
            alpha=0.05,
        ),
    }


def _worst_verdict(verdicts: list[str]) -> str:
    if not verdicts:
        return "unavailable"
    return max(verdicts, key=lambda v: VERDICT_RANK.get(v, 0))


def _metric_overall(report: dict[str, Any]) -> Optional[str]:
    if report.get("skipped"):
        return None
    return str(report.get("overall_verdict", report.get("verdict")))


def _compare_length_distributions(
    reference: list[float],
    current: list[float],
    *,
    name: str,
    bins: int = 10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    return drift_summary(reference, current, metric_name=name, bins=bins, alpha=alpha)


def _golden_paths(project_root: Path, instructions_dir: Path) -> list[Path]:
    candidates = [
        instructions_dir / "golden" / "golden_set.json",
        instructions_dir / "golden" / "ui_extension_qa_150.json",
        project_root / "baseline" / "golden_set.json",
    ]
    return [path for path in candidates if path.is_file()]


def _index_chunk_lengths(records: list[Any]) -> list[float]:
    return [float(len(str(getattr(record, "chunk", "")))) for record in records]


def collect_quality_metrics(
    project_root: Path,
    *,
    instructions_dir: Optional[Path] = None,
    chunk_lengths: Optional[list[float]] = None,
    chunk_count: Optional[int] = None,
    source_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build PSI/KS quality block for ``GET /v1/metrics``."""
    root = project_root.resolve()
    instr = instructions_dir or (root / "instructions")
    checks: list[dict[str, Any]] = []
    verdicts: list[str] = []

    saved_path = root / "data" / "finetune" / "data_drift_report.json"
    saved_report: Optional[dict[str, Any]] = None
    if saved_path.is_file():
        try:
            saved_report = json.loads(saved_path.read_text(encoding="utf-8"))
            checks.append(
                {
                    "id": "saved_report",
                    "source": str(saved_path),
                    "available": True,
                    "note": "offline report from monitor_data_drift.py",
                }
            )
        except (OSError, json.JSONDecodeError):
            checks.append({"id": "saved_report", "available": False, "source": str(saved_path)})

    splits_dir = instr / "golden" / "splits"
    split_pairs = {
        "train_vs_val": (
            splits_dir / "ui_extension_qa_train.json",
            splits_dir / "ui_extension_qa_val.json",
        ),
        "train_vs_test": (
            splits_dir / "ui_extension_qa_train.json",
            splits_dir / "ui_extension_qa_test.json",
        ),
    }
    for pair_id, (ref_path, cur_path) in split_pairs.items():
        if not ref_path.is_file() or not cur_path.is_file():
            continue
        comparison = compare_datasets(
            ref_path,
            cur_path,
            metrics=["question_len", "answer_len"],
            bins=8,
            alpha=0.05,
        )
        pair_verdicts = [
            v
            for metric in comparison.get("metrics", {}).values()
            if (v := _metric_overall(metric))
        ]
        overall = _worst_verdict(pair_verdicts)
        verdicts.append(overall)
        checks.append(
            {
                "id": f"qa_splits_{pair_id}",
                "overall_verdict": overall,
                "comparison": comparison,
            }
        )

    for eval_id, ref_name, cur_name in (
        ("eval_val", "evaluation_ui_extension_val_baseline.json", "evaluation_ui_extension_val_finetuned.json"),
        ("eval_test", "evaluation_ui_extension_test_baseline.json", "evaluation_ui_extension_test_finetuned.json"),
    ):
        ref_path = root / ref_name
        cur_path = root / cur_name
        if not ref_path.is_file() or not cur_path.is_file():
            continue
        comparison = compare_datasets(
            ref_path,
            cur_path,
            metrics=["judge_percent", "final_score", "faithfulness", "hit_at_3"],
            bins=8,
            alpha=0.05,
        )
        pair_verdicts = [
            v
            for metric in comparison.get("metrics", {}).values()
            if (v := _metric_overall(metric))
        ]
        overall = _worst_verdict(pair_verdicts)
        verdicts.append(overall)
        checks.append({"id": eval_id, "overall_verdict": overall, "comparison": comparison})

    index_block: dict[str, Any] = {
        "chunk_count": chunk_count if chunk_count is not None else len(chunk_lengths or []),
        "source_count": source_count,
    }
    if chunk_lengths:
        index_block["avg_chunk_len"] = round(sum(chunk_lengths) / len(chunk_lengths), 2)

    for golden_path in _golden_paths(root, instr):
        try:
            _kind, rows = _load_rows(golden_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        golden_answer_lens = [
            float(len(str(row.get("expected_answer", row.get("answer_preview", "")))))
            for row in rows
            if row.get("expected_answer") or row.get("answer_preview")
        ]
        if len(golden_answer_lens) < 2:
            continue
        if chunk_lengths and len(chunk_lengths) >= 2:
            chunk_drift = _compare_length_distributions(
                golden_answer_lens,
                chunk_lengths,
                name="golden_answer_len_vs_index_chunk_len",
            )
            verdicts.append(str(chunk_drift.get("overall_verdict", "unavailable")))
            index_block["golden_reference"] = str(golden_path)
            index_block["golden_rows"] = len(golden_answer_lens)
            checks.append(
                {
                    "id": "index_vs_golden",
                    "overall_verdict": chunk_drift.get("overall_verdict"),
                    "metric": chunk_drift,
                }
            )
        else:
            checks.append(
                {
                    "id": "index_vs_golden",
                    "skipped": True,
                    "reason": "vector index empty or too small for PSI",
                    "golden_reference": str(golden_path),
                    "golden_rows": len(golden_answer_lens),
                }
            )
            verdicts.append("significant_drift" if (chunk_count or 0) == 0 else "moderate_drift")
        break

    overall = _worst_verdict(verdicts) if verdicts else "unavailable"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_verdict": overall,
        "psi_thresholds": {"no_drift": 0.1, "moderate": 0.25},
        "index": index_block,
        "checks": checks,
        "saved_report": saved_report,
        "saved_report_path": str(saved_path) if saved_path.is_file() else None,
    }

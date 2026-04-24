#!/usr/bin/env python3
"""Compare old and new evaluation results."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure command line logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_existing_path(path_pattern: str) -> Path | None:
    """Resolve direct path or first glob match."""
    candidate = Path(path_pattern)
    if candidate.exists():
        return candidate
    matches = glob.glob(path_pattern)
    return Path(matches[0]) if matches else None


def load_old_results(path_pattern: str) -> pd.DataFrame | None:
    """Load old results from CSV or JSON."""
    resolved_path = resolve_existing_path(path_pattern)
    if resolved_path is None:
        log.error("old results not found: %s", path_pattern)
        return None

    log.info("loading old results from=%s", resolved_path)
    if resolved_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(resolved_path)
        log.info("old records loaded=%s format=csv", len(dataframe))
        return dataframe

    if resolved_path.suffix.lower() == ".json":
        with resolved_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        dataframe = pd.DataFrame(data)
        log.info("old records loaded=%s format=json", len(dataframe))
        return dataframe

    log.error("unsupported old file format: %s", resolved_path.suffix)
    return None


def load_new_results(path_pattern: str) -> list[dict[str, Any]] | None:
    """Load new results from JSON."""
    resolved_path = resolve_existing_path(path_pattern)
    if resolved_path is None:
        log.error("new results not found: %s", path_pattern)
        return None

    log.info("loading new results from=%s", resolved_path)
    with resolved_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    log.info("new records loaded=%s", len(data))
    return data


def compare_scores(old_df: pd.DataFrame, new_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare old and new scores by question id."""
    id_column = "id" if "id" in old_df.columns else "Id" if "Id" in old_df.columns else old_df.columns[0]
    score_column = (
        "similarity_score" if "similarity_score" in old_df.columns
        else "score" if "score" in old_df.columns
        else "total_score"
    )

    new_scores: dict[Any, dict[str, Any]] = {}
    for item in new_data:
        question_id = item.get("id")
        if question_id:
            new_scores[question_id] = {
                "score": item.get("score", item.get("total_score", 0)),
                "time": item.get("time", item.get("time_seconds", 0)),
                "question": item.get("question", ""),
            }

    results: list[dict[str, Any]] = []
    for _, row in old_df.iterrows():
        question_id = row.get(id_column)
        if question_id and question_id in new_scores:
            old_score = row.get(score_column, 0)
            new_score = new_scores[question_id]["score"]
            difference = new_score - old_score
            results.append(
                {
                    "id": question_id,
                    "question": str(row.get("question", ""))[:80],
                    "old_score": float(old_score) if old_score else 0.0,
                    "new_score": float(new_score) if new_score else 0.0,
                    "difference": difference,
                    "improved": difference > 0,
                    "old_time": row.get("time_seconds", row.get("time", 0)),
                    "new_time": new_scores[question_id]["time"],
                }
            )

    return results


def save_comparison_results(results: list[dict[str, Any]]) -> None:
    """Save comparison report to CSV and JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("comparison")
    output_dir.mkdir(exist_ok=True)

    dataframe = pd.DataFrame(results)
    csv_path = output_dir / f"comparison_{timestamp}.csv"
    json_path = output_dir / f"comparison_{timestamp}.json"

    dataframe.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, ensure_ascii=False, indent=2)

    log.info("saved csv report=%s", csv_path)
    log.info("saved json report=%s", json_path)


def print_comparison(results: list[dict[str, Any]]) -> None:
    """Print human-readable comparison summary."""
    if not results:
        log.error("no results to compare")
        return

    improved = [item for item in results if item["improved"]]
    degraded = [item for item in results if not item["improved"]]
    avg_old = sum(item["old_score"] for item in results) / len(results)
    avg_new = sum(item["new_score"] for item in results) / len(results)

    log.info("summary total=%s improved=%s degraded=%s", len(results), len(improved), len(degraded))
    log.info("avg old score=%.3f", avg_old)
    log.info("avg new score=%.3f", avg_new)
    log.info("avg delta=%+.3f", avg_new - avg_old)

    for item in sorted(improved, key=lambda row: row["difference"], reverse=True)[:5]:
        log.info("top improved qid=%s old=%.3f new=%.3f diff=%.3f", item["id"], item["old_score"], item["new_score"], item["difference"])
    for item in sorted(degraded, key=lambda row: row["difference"])[:5]:
        log.info("top degraded qid=%s old=%.3f new=%.3f diff=%.3f", item["id"], item["old_score"], item["new_score"], item["difference"])

    save_comparison_results(results)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare old and new evaluation results.")
    parser.add_argument("--old", "-o", required=True, help="Old results file (CSV or JSON)")
    parser.add_argument("--new", "-n", required=True, help="New results JSON file")
    return parser.parse_args()


def main() -> int:
    """Run comparison CLI entry point."""
    configure_logging()
    args = parse_args()

    old_df = load_old_results(args.old)
    new_data = load_new_results(args.new)
    if old_df is None or new_data is None:
        return 1

    results = compare_scores(old_df, new_data)
    if not results:
        log.error("no matching questions found")
        return 1

    print_comparison(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
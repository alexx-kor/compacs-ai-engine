"""Compare legacy evaluation exports (CSV/JSON) against a newer JSON evaluation run."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "compare_scores",
    "emit_comparison",
    "load_new_results",
    "load_old_results",
    "main",
    "print_comparison",
]


def load_old_results(file_path: str) -> pd.DataFrame | None:
    """Load historical results from a CSV or JSON file path.

    If ``file_path`` is missing, a single glob match is attempted. Unknown
    extensions return ``None``.

    Args:
        file_path: Path or glob pattern for the legacy export.

    Returns:
        A DataFrame of old rows, or ``None`` when the file is missing or the
        format is not CSV/JSON.
    """
    resolved = file_path
    if not Path(resolved).exists():
        matches = glob.glob(file_path)
        if matches:
            resolved = matches[0]
        else:
            log.error("[ERROR] Old results not found: %s", file_path)
            return None

    log.info("[LOAD] Loading from: %s", resolved)

    if resolved.endswith(".csv"):
        frame = pd.read_csv(resolved)
        log.info("[LOAD] Old results: %s records from CSV", len(frame))
        return frame
    if resolved.endswith(".json"):
        path = Path(resolved)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        frame = pd.DataFrame(data)
        log.info("[LOAD] Old results: %s records from JSON", len(frame))
        return frame

    log.error("[ERROR] Unknown format: %s", resolved)
    return None


def load_new_results(file_path: str) -> Any | None:
    """Load the new evaluation JSON payload from disk.

    Args:
        file_path: Path or glob pattern for the new JSON file.

    Returns:
        Parsed JSON (typically a list of records), or ``None`` when missing.
    """
    resolved = file_path
    if not Path(resolved).exists():
        matches = glob.glob(file_path)
        if matches:
            resolved = matches[0]
        else:
            log.error("[ERROR] New results not found: %s", file_path)
            return None

    log.info("[LOAD] Loading from: %s", resolved)

    path = Path(resolved)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    log.info("[LOAD] New results: %s records", len(data))
    return data


def compare_scores(old_df: pd.DataFrame, new_data: Any) -> list[dict[str, Any]]:
    """Align rows by question id and compute score deltas.

    Args:
        old_df: Legacy table with id and score-like columns.
        new_data: Iterable of new result dicts, or a mapping (same iteration
            semantics as the previous implementation).

    Returns:
        One dict per matched id with old/new scores and timing fields.
    """
    id_col = "id" if "id" in old_df.columns else "Id" if "Id" in old_df.columns else old_df.columns[0]
    score_col = (
        "similarity_score"
        if "similarity_score" in old_df.columns
        else "score"
        if "score" in old_df.columns
        else "total_score"
    )

    new_scores: dict[Any, dict[str, Any]] = {}
    for item in new_data:
        qid = item.get("id")
        if qid:
            new_scores[qid] = {
                "score": item.get("score", item.get("total_score", 0)),
                "time": item.get("time", item.get("time_seconds", 0)),
                "question": item.get("question", ""),
            }

    results: list[dict[str, Any]] = []
    for _, row in old_df.iterrows():
        qid = row.get(id_col)
        if qid and qid in new_scores:
            old_score = row.get(score_col, 0)
            new_score = new_scores[qid]["score"]
            diff = new_score - old_score

            results.append(
                {
                    "id": qid,
                    "question": str(row.get("question", ""))[:80],
                    "old_score": float(old_score) if old_score else 0.0,
                    "new_score": float(new_score) if new_score else 0.0,
                    "difference": diff,
                    "improved": diff > 0,
                    "old_time": row.get("time_seconds", row.get("time", 0)),
                    "new_time": new_scores[qid]["time"],
                }
            )

    return results


def emit_comparison(results: Sequence[Mapping[str, Any]]) -> None:
    """Log a summary and write comparison CSV/JSON under ``comparison/``.

    Args:
        results: Rows produced by :func:`compare_scores`.
    """
    log.info("")
    log.info("=" * 80)
    log.info("COMPARISON RESULTS")
    log.info("=" * 80)

    if not results:
        log.error("[ERROR] No results to compare!")
        return

    rows = list(results)
    improved = [row for row in rows if row["improved"]]
    degraded = [row for row in rows if not row["improved"]]

    avg_old = sum(float(row["old_score"]) for row in rows) / len(rows)
    avg_new = sum(float(row["new_score"]) for row in rows) / len(rows)

    log.info("")
    log.info("SUMMARY:")
    log.info("  Total questions: %s", len(rows))
    log.info("  Improved: %s", len(improved))
    log.info("  Degraded: %s", len(degraded))
    log.info("  Average old score: %.3f", avg_old)
    log.info("  Average new score: %.3f", avg_new)
    log.info("  Difference: %+.3f", avg_new - avg_old)

    if improved:
        log.info("")
        log.info("TOP IMPROVED:")
        for row in sorted(improved, key=lambda item: item["difference"], reverse=True)[:5]:
            log.info(
                "  Q%s: %.3f -> %.3f (+%.3f)",
                row["id"],
                row["old_score"],
                row["new_score"],
                row["difference"],
            )
            log.info("    %s...", str(row["question"])[:60])

    if degraded:
        log.info("")
        log.info("TOP DEGRADED:")
        for row in sorted(degraded, key=lambda item: item["difference"])[:5]:
            log.info(
                "  Q%s: %.3f -> %.3f (%.3f)",
                row["id"],
                row["old_score"],
                row["new_score"],
                row["difference"],
            )
            log.info("    %s...", str(row["question"])[:60])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("comparison")
    output_dir.mkdir(exist_ok=True)

    frame_out = pd.DataFrame(rows)
    csv_path = output_dir / ("comparison_%s.csv" % timestamp)
    frame_out.to_csv(csv_path, index=False)
    log.info("")
    log.info("[SAVE] %s", csv_path)

    json_path = output_dir / ("comparison_%s.json" % timestamp)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    log.info("[SAVE] %s", json_path)


def print_comparison(results: Sequence[Mapping[str, Any]]) -> None:
    """Backward-compatible alias for :func:`emit_comparison`."""
    emit_comparison(results)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the results comparator.

    Returns:
        Parsed arguments with paths to old and new exports.
    """
    parser = argparse.ArgumentParser(description="Compare old and new results")
    parser.add_argument(
        "--old",
        "-o",
        type=str,
        required=True,
        help="Old results file (CSV or JSON)",
    )
    parser.add_argument(
        "--new",
        "-n",
        type=str,
        required=True,
        help="New results JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Load exports, compare aligned ids, write artifacts, and log the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_args()

    old_df = load_old_results(arguments.old)
    new_data = load_new_results(arguments.new)

    if old_df is None or new_data is None:
        return

    results = compare_scores(old_df, new_data)

    if not results:
        log.error("[ERROR] No matching questions found!")
        return

    print_comparison(results)


if __name__ == "__main__":
    main()

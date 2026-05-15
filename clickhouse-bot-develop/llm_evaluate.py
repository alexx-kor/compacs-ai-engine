"""Percentage-based LLM evaluation of RAG answers via Ollama (no emoji output)."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import config  # noqa: E402

_ollama: Any = importlib.import_module("ollama")


log = logging.getLogger(__name__)

__all__ = [
    "calculate_statistics",
    "compare_two_answers_percent",
    "emit_summary",
    "evaluate_all",
    "llm_evaluate_percent",
    "load_json",
    "main",
    "print_summary",
    "save_results",
]


def load_json(file_path: str) -> list[dict[str, Any]]:
    """Load a JSON array of answer records from disk.

    Args:
        file_path: UTF-8 JSON file path (typically a list of objects).

    Returns:
        Parsed list (same object as ``json.load`` when root is a JSON array).

    Raises:
        FileNotFoundError: When the path does not exist.
        json.JSONDecodeError: When the file is not valid JSON.
        OSError: When the file cannot be read.
    """
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    return cast(list[dict[str, Any]], data)


def llm_evaluate_percent(question: str, answer: str) -> dict[str, Any]:
    """Score a single answer with dimension percentages and a total (0–100).

    Args:
        question: User question text.
        answer: Model answer text (truncated internally for the prompt).

    Returns:
        Parsed scores including ``relevance``, ``accuracy``, ``completeness``,
        ``clarity``, and ``total``, or a dict with ``error`` and ``total`` set
        to ``0`` when parsing or the Ollama call fails.
    """
    answer = answer[:2000] if answer else ""

    prompt = f"""You are an expert evaluator of RAG system answers. Rate the answer on a scale of 0-100%.

QUESTION: {question}

ANSWER: {answer}

Rate on these criteria (0-100%, where 0=terrible, 100=perfect):
1. RELEVANCE: Does the answer directly address the question?
2. ACCURACY: Is the information factually correct?
3. COMPLETENESS: Does it provide sufficient detail?
4. CLARITY: Is it well-structured and easy to understand?

CRITICAL: Reply ONLY in JSON format with numbers between 0 and 100.
Example: {{"relevance": 85, "accuracy": 90, "completeness": 75, "clarity": 80, "total": 82.5}}

Do NOT output any text outside the JSON.
Do NOT use numbers below 0 or above 100.
"""

    try:
        response = _ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 300,
            },
        )
        content = response["message"]["content"]
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            result = cast(dict[str, Any], json.loads(json_match.group()))
            for key in ("relevance", "accuracy", "completeness", "clarity", "total"):
                if key in result:
                    result[key] = max(0, min(100, float(result[key])))
            return result
        return {"error": "Failed to parse", "total": 0}
    except Exception as exc:
        return {"error": str(exc), "total": 0}


def compare_two_answers_percent(question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
    """Compare two answers and return winner metadata and scores.

    Args:
        question: User question text.
        answer_a: First answer (MAIN / system A).
        answer_b: Second answer (HYPOTHESIS / system B).

    Returns:
        Parsed comparison JSON, or a minimal error-shaped dict when parsing
        fails or the Ollama call raises.
    """
    answer_a = answer_a[:2000] if answer_a else ""
    answer_b = answer_b[:2000] if answer_b else ""

    prompt = f"""You are an expert evaluator comparing two RAG answers.

QUESTION: {question}

ANSWER A: {answer_a}

ANSWER B: {answer_b}

Compare these answers and decide:

1. Which answer is BETTER overall? (A or B or TIE)
2. By what PERCENTAGE is the better answer better? (0-100%)
3. Why?

Reply ONLY in JSON format:
{{"winner": "A" or "B" or "TIE", "winner_percent": 0-100, "reason": "brief explanation", "a_score": 0-100, "b_score": 0-100}}
"""

    try:
        response = _ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 300,
            },
        )
        content = response["message"]["content"]
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            result = cast(dict[str, Any], json.loads(json_match.group()))
            if "winner_percent" in result:
                result["winner_percent"] = max(0, min(100, float(result["winner_percent"])))
            return result
        return {"winner": "UNKNOWN", "winner_percent": 0}
    except Exception as exc:
        return {"winner": "ERROR", "winner_percent": 0, "error": str(exc)}


def evaluate_all(
    answers_main: Sequence[Mapping[str, Any]],
    answers_hypothesis: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair MAIN and HYPOTHESIS answers by ``id`` and run all scoring steps.

    Args:
        answers_main: Records with ``id``, ``question``, ``answer``, optional ``sources``/``time``.
        answers_hypothesis: Same shape for the hypothesis run.

    Returns:
        One result dict per matched question id (same pairing rules as before).
    """
    results: list[dict[str, Any]] = []

    main_dict = {item["id"]: item for item in answers_main}
    hyp_dict = {item["id"]: item for item in answers_hypothesis}

    for qid in main_dict:
        if qid not in hyp_dict:
            continue

        question = main_dict[qid]["question"]
        answer_main = main_dict[qid]["answer"]
        answer_hyp = hyp_dict[qid]["answer"]

        log.info("")
        log.info("=" * 70)
        log.info("Q%s: %s...", qid, question[:70])
        log.info("=" * 70)

        log.info("  [1/3] Evaluating MAIN answer...")
        eval_main = llm_evaluate_percent(question, answer_main)
        log.info("  done (%.0f%%)", float(eval_main.get("total", 0)))

        log.info("  [2/3] Evaluating HYPOTHESIS answer...")
        eval_hyp = llm_evaluate_percent(question, answer_hyp)
        log.info("  done (%.0f%%)", float(eval_hyp.get("total", 0)))

        log.info("  [3/3] Comparing answers...")
        comparison = compare_two_answers_percent(question, answer_main, answer_hyp)
        log.info("  done (Winner: %s)", comparison.get("winner", "?"))

        results.append(
            {
                "id": qid,
                "question": question,
                "main": {
                    "answer": answer_main[:500],
                    "sources": main_dict[qid].get("sources", []),
                    "time": main_dict[qid].get("time", 0),
                    "evaluation": eval_main,
                },
                "hypothesis": {
                    "answer": answer_hyp[:500],
                    "sources": hyp_dict[qid].get("sources", []),
                    "time": hyp_dict[qid].get("time", 0),
                    "evaluation": eval_hyp,
                },
                "comparison": comparison,
            }
        )

        winner = comparison.get("winner", "?")
        winner_pct = float(comparison.get("winner_percent", 0))
        if winner == "A":
            winner_icon = "[MAIN]"
        elif winner == "B":
            winner_icon = "[HYP]"
        else:
            winner_icon = "[TIE]"

        log.info("")
        log.info("  RESULT:")
        log.info("     MAIN: %.1f%%", float(eval_main.get("total", 0)))
        log.info("     HYP:  %.1f%%", float(eval_hyp.get("total", 0)))
        log.info("     WINNER: %s (+%.0f%%)", winner_icon, winner_pct)
        log.info("     Reason: %s...", str(comparison.get("reason", ""))[:80])

    return results


def calculate_statistics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate win counts and average percentage scores.

    Args:
        results: Rows produced by :func:`evaluate_all`.

    Returns:
        Summary statistics dict (empty aggregates when ``results`` is empty).
    """
    main_scores = [float(r["main"]["evaluation"].get("total", 0)) for r in results]
    hyp_scores = [float(r["hypothesis"]["evaluation"].get("total", 0)) for r in results]

    winners = [str(r["comparison"].get("winner", "UNKNOWN")) for r in results]

    wins_main = winners.count("A")
    wins_hyp = winners.count("B")
    ties = winners.count("TIE")

    avg_main = sum(main_scores) / len(main_scores) if main_scores else 0.0
    avg_hyp = sum(hyp_scores) / len(hyp_scores) if hyp_scores else 0.0

    if avg_main > avg_hyp:
        advantage = avg_main - avg_hyp
        advantage_pct = (advantage / avg_main) * 100 if avg_main > 0 else 0.0
        better = "MAIN"
    else:
        advantage = avg_hyp - avg_main
        advantage_pct = (advantage / avg_hyp) * 100 if avg_hyp > 0 else 0.0
        better = "HYPOTHESIS"

    count = len(results)
    return {
        "total_questions": count,
        "main_avg_score": avg_main,
        "hyp_avg_score": avg_hyp,
        "wins_main": wins_main,
        "wins_hyp": wins_hyp,
        "ties": ties,
        "win_rate_main": (wins_main / count) * 100 if results else 0.0,
        "win_rate_hyp": (wins_hyp / count) * 100 if results else 0.0,
        "tie_rate": (ties / count) * 100 if results else 0.0,
        "advantage_percent": advantage_pct,
        "better_system": better,
        "improvement": advantage_pct if better == "HYPOTHESIS" else -advantage_pct,
    }


def emit_summary(results: Sequence[Mapping[str, Any]]) -> None:
    """Emit the final human-readable summary to the configured logger.

    Args:
        results: Rows produced by :func:`evaluate_all`.
    """
    stats = calculate_statistics(results)

    log.info("")
    log.info("=" * 80)
    log.info("FINAL SUMMARY - PERCENTAGE BASED")
    log.info("=" * 80)

    log.info("")
    log.info("OVERALL SCORES:")
    log.info("   MAIN system:       %.1f%%", stats["main_avg_score"])
    log.info("   HYPOTHESIS system: %.1f%%", stats["hyp_avg_score"])

    bar_main = "#" * int(stats["main_avg_score"] / 2)
    bar_hyp = "#" * int(stats["hyp_avg_score"] / 2)
    log.info("   MAIN:       [%s] %.1f%%", f"{bar_main:<50}", stats["main_avg_score"])
    log.info("   HYPOTHESIS: [%s] %.1f%%", f"{bar_hyp:<50}", stats["hyp_avg_score"])

    log.info("")
    log.info("WIN/LOSS STATISTICS:")
    log.info("   MAIN wins:       %s (%.1f%%)", stats["wins_main"], stats["win_rate_main"])
    log.info("   HYPOTHESIS wins: %s (%.1f%%)", stats["wins_hyp"], stats["win_rate_hyp"])
    log.info("   Ties:            %s (%.1f%%)", stats["ties"], stats["tie_rate"])

    log.info("")
    log.info("IMPROVEMENT:")
    if stats["better_system"] == "HYPOTHESIS":
        log.info("   HYPOTHESIS is better by %.1f%%", stats["advantage_percent"])
    else:
        log.info("   MAIN is better by %.1f%%", stats["advantage_percent"])

    log.info("")
    log.info("DETAILED RESULTS:")
    log.info("-" * 80)
    log.info("%-4s %-10s %-10s %-12s %s", "ID", "MAIN %", "HYP %", "WINNER", "ADVANTAGE")
    log.info("-" * 80)

    for row in results:
        main_score = float(row["main"]["evaluation"].get("total", 0))
        hyp_score = float(row["hypothesis"]["evaluation"].get("total", 0))
        winner = str(row["comparison"].get("winner", "?"))
        winner_pct = float(row["comparison"].get("winner_percent", 0))

        if winner == "A":
            winner_text = "[MAIN]"
        elif winner == "B":
            winner_text = "[HYP]"
        else:
            winner_text = "[TIE]"

        adv = "+%.0f%%" % winner_pct if winner_pct > 0 else "0%"
        log.info(
            "%-4s %-10.1f %-10.1f %-12s %s",
            row["id"],
            main_score,
            hyp_score,
            winner_text,
            adv,
        )

    log.info("-" * 80)


def print_summary(results: Sequence[Mapping[str, Any]]) -> None:
    """Backward-compatible alias for :func:`emit_summary`."""
    emit_summary(results)


def save_results(results: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> None:
    """Persist detailed JSON, statistics JSON, and a plain-text report.

    Args:
        results: Rows produced by :func:`evaluate_all`.
        stats: Aggregates from :func:`calculate_statistics`.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("llm_evaluation")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / ("llm_evaluation_%s.json" % timestamp)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, ensure_ascii=False, indent=2)

    stats_file = output_dir / ("statistics_%s.json" % timestamp)
    with stats_file.open("w", encoding="utf-8") as handle:
        json.dump(dict(stats), handle, ensure_ascii=False, indent=2)

    report_file = output_dir / ("report_%s.txt" % timestamp)
    with report_file.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("LLM EVALUATION REPORT (PERCENTAGE BASED)\n")
        handle.write("Generated: %s\n" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        handle.write("=" * 80 + "\n\n")

        handle.write("OVERALL SCORES:\n")
        handle.write("  MAIN system:       %.1f%%\n" % stats["main_avg_score"])
        handle.write("  HYPOTHESIS system: %.1f%%\n\n" % stats["hyp_avg_score"])

        handle.write("WIN/LOSS:\n")
        handle.write("  MAIN wins:       %s (%.1f%%)\n" % (stats["wins_main"], stats["win_rate_main"]))
        handle.write("  HYPOTHESIS wins: %s (%.1f%%)\n" % (stats["wins_hyp"], stats["win_rate_hyp"]))
        handle.write("  Ties:            %s (%.1f%%)\n\n" % (stats["ties"], stats["tie_rate"]))

        handle.write(
            "CONCLUSION: %s is better by %.1f%%\n\n"
            % (stats["better_system"], stats["advantage_percent"])
        )

        handle.write("DETAILS BY QUESTION:\n")
        for row in results:
            handle.write("\nQ%s: %s\n" % (row["id"], str(row["question"])[:80]))
            handle.write("  MAIN score: %.1f%%\n" % float(row["main"]["evaluation"].get("total", 0)))
            handle.write("  HYP score:  %.1f%%\n" % float(row["hypothesis"]["evaluation"].get("total", 0)))
            handle.write("  Winner: %s\n" % row["comparison"].get("winner", "?"))
            handle.write("  Advantage: +%.0f%%\n" % float(row["comparison"].get("winner_percent", 0)))

    log.info("")
    log.info("[SAVE] Results saved to: %s", output_dir)
    log.info("   - %s", output_file.name)
    log.info("   - %s", stats_file.name)
    log.info("   - %s", report_file.name)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Ollama percentage evaluator.

    Returns:
        Parsed arguments including paths to MAIN and HYPOTHESIS JSON exports.
    """
    parser = argparse.ArgumentParser(description="LLM Evaluation - Percentage Based")
    parser.add_argument(
        "--main",
        "-m",
        type=str,
        required=True,
        help="Main answers JSON file",
    )
    parser.add_argument(
        "--hypothesis",
        "-hyp",
        type=str,
        required=True,
        help="Hypothesis answers JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Load answer files, verify Ollama, run evaluations, save artifacts, and emit summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arguments = parse_args()

    log.info("")
    log.info("=" * 80)
    log.info("LLM EVALUATION - PERCENTAGE BASED (0-100%%)")
    log.info("=" * 80)

    log.info("")
    log.info("[LOAD] Loading MAIN answers: %s", arguments.main)
    answers_main = load_json(arguments.main)

    log.info("[LOAD] Loading HYPOTHESIS answers: %s", arguments.hypothesis)
    answers_hypothesis = load_json(arguments.hypothesis)

    log.info("")
    log.info("[INFO] MAIN: %s answers", len(answers_main))
    log.info("[INFO] HYPOTHESIS: %s answers", len(answers_hypothesis))

    try:
        _ollama.list()
        log.info("[OK] Ollama is running")
        log.info("")
    except Exception as exc:
        log.error("[ERROR] Ollama is not running: %s", exc)
        return

    results = evaluate_all(answers_main, answers_hypothesis)
    stats = calculate_statistics(results)
    save_results(results, stats)
    emit_summary(results)

    log.info("")
    log.info("=" * 80)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

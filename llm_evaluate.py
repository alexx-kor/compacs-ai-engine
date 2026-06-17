#!/usr/bin/env python3
"""Evaluate two answer sets with an LLM judge and save percentage reports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama

from config import config

log = logging.getLogger(__name__)
JSON_BLOCK_PATTERN = r"\{[^{}]*\}"
METRIC_KEYS = ("relevance", "accuracy", "completeness", "clarity", "total")


def configure_logging() -> None:
    """Configure command line logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json_file(file_path: str) -> list[dict[str, Any]]:
    """Load list payload from JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return payload if isinstance(payload, list) else []


def clamp_0_100(value: Any) -> float:
    """Clamp metric value to [0, 100]."""
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_first_json_block(response_text: str) -> dict[str, Any]:
    """Extract first JSON object from model response."""
    match = re.search(JSON_BLOCK_PATTERN, response_text, re.DOTALL)
    if match is None:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _judge_prompt(question: str, answer: str) -> str:
    short_answer = (answer or "")[:2000]
    return f"""You are an expert evaluator of RAG system answers. Rate the answer on a scale of 0-100%.

QUESTION: {question}

ANSWER: {short_answer}

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


def _normalize_judge_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {"error": "Failed to parse", "total": 0.0}
    for key in METRIC_KEYS:
        if key in metrics:
            metrics[key] = clamp_0_100(metrics[key])
    return metrics


def _empty_judge_result(reason: str = "empty answer") -> dict[str, Any]:
    return {
        "relevance": 0.0,
        "accuracy": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "total": 0.0,
        "skipped": reason,
    }


def evaluate_answer_percent_openai(
    question: str,
    answer: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate one answer with OpenAI judge (default: gpt-4o-mini)."""
    if not (answer or "").strip():
        return _empty_judge_result()

    from dotenv import load_dotenv

    load_dotenv(".env.rag", override=True)
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed", "total": 0.0}

    from config import Config

    runtime = Config.from_env()
    key = api_key or runtime.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not key or key.strip() == "user_provided":
        return {"error": "OPENAI_API_KEY not configured", "total": 0.0}

    judge_model = model or os.getenv("JUDGE_MODEL") or runtime.openai_model or "gpt-4o-mini"
    client = OpenAI(api_key=key)
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": _judge_prompt(question, answer)}],
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as error:
        return {"error": str(error), "total": 0.0}

    content = response.choices[0].message.content or ""
    metrics = _normalize_judge_metrics(parse_first_json_block(content))
    metrics["judge_model"] = judge_model
    metrics["judge_backend"] = "openai"
    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    from core.cost_guard import CostGuard

    metrics["judge_tokens"] = prompt_tokens + completion_tokens
    metrics["judge_cost_usd"] = round(
        CostGuard._estimate_cost(prompt_tokens, completion_tokens, judge_model),
        6,
    )
    return metrics


def evaluate_answer_percent(question: str, answer: str) -> dict[str, Any]:
    """Evaluate one answer with local Ollama judge."""
    if not (answer or "").strip():
        return _empty_judge_result()

    try:
        response = ollama.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": _judge_prompt(question, answer)}],
            options={"temperature": 0.1, "num_predict": 300},
        )
    except (RuntimeError, OSError, ValueError) as error:
        return {"error": str(error), "total": 0.0}

    content = (response.get("message") or {}).get("content", "")
    metrics = _normalize_judge_metrics(parse_first_json_block(content))
    metrics["judge_model"] = config.ollama_model
    metrics["judge_backend"] = "ollama"
    return metrics


def compare_answers_percent(question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
    """Compare two answers and return winner information."""
    short_a = (answer_a or "")[:2000]
    short_b = (answer_b or "")[:2000]
    prompt = f"""You are an expert evaluator comparing two RAG answers.

QUESTION: {question}

ANSWER A: {short_a}

ANSWER B: {short_b}

Compare these answers and decide:

1. Which answer is BETTER overall? (A or B or TIE)
2. By what PERCENTAGE is the better answer better? (0-100%)
3. Why?

Reply ONLY in JSON format:
{{"winner": "A" or "B" or "TIE", "winner_percent": 0-100, "reason": "brief explanation", "a_score": 0-100, "b_score": 0-100}}
"""
    try:
        response = ollama.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300},
        )
    except (RuntimeError, OSError, ValueError) as error:
        return {"winner": "ERROR", "winner_percent": 0.0, "error": str(error)}

    content = (response.get("message") or {}).get("content", "")
    comparison = parse_first_json_block(content)
    if not comparison:
        return {"winner": "UNKNOWN", "winner_percent": 0.0}
    comparison["winner_percent"] = clamp_0_100(comparison.get("winner_percent", 0))
    return comparison


def evaluate_all(main_answers: list[dict[str, Any]], hypothesis_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate and compare all matching question ids."""
    results: list[dict[str, Any]] = []
    main_by_id = {item["id"]: item for item in main_answers if "id" in item}
    hypothesis_by_id = {item["id"]: item for item in hypothesis_answers if "id" in item}

    for question_id, main_item in main_by_id.items():
        if question_id not in hypothesis_by_id:
            continue
        hypothesis_item = hypothesis_by_id[question_id]
        question = main_item.get("question", "")
        main_answer = main_item.get("answer", "")
        hypothesis_answer = hypothesis_item.get("answer", "")

        log.info("qid=%s step=1/3 evaluate MAIN", question_id)
        main_eval = evaluate_answer_percent(question, main_answer)
        log.info("qid=%s step=2/3 evaluate HYPOTHESIS", question_id)
        hypothesis_eval = evaluate_answer_percent(question, hypothesis_answer)
        log.info("qid=%s step=3/3 compare", question_id)
        comparison = compare_answers_percent(question, main_answer, hypothesis_answer)

        results.append(
            {
                "id": question_id,
                "question": question,
                "main": {
                    "answer": main_answer[:500],
                    "sources": main_item.get("sources", []),
                    "time": main_item.get("time", 0),
                    "evaluation": main_eval,
                },
                "hypothesis": {
                    "answer": hypothesis_answer[:500],
                    "sources": hypothesis_item.get("sources", []),
                    "time": hypothesis_item.get("time", 0),
                    "evaluation": hypothesis_eval,
                },
                "comparison": comparison,
            }
        )
    return results


def calculate_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate statistics from evaluation results."""
    main_scores = [item["main"]["evaluation"].get("total", 0) for item in results]
    hypothesis_scores = [item["hypothesis"]["evaluation"].get("total", 0) for item in results]
    winners = [item["comparison"].get("winner", "UNKNOWN") for item in results]
    wins_main = winners.count("A")
    wins_hypothesis = winners.count("B")
    ties = winners.count("TIE")

    avg_main = sum(main_scores) / len(main_scores) if main_scores else 0
    avg_hypothesis = sum(hypothesis_scores) / len(hypothesis_scores) if hypothesis_scores else 0
    if avg_main > avg_hypothesis:
        advantage = avg_main - avg_hypothesis
        advantage_pct = (advantage / avg_main) * 100 if avg_main > 0 else 0
        better_system = "MAIN"
    else:
        advantage = avg_hypothesis - avg_main
        advantage_pct = (advantage / avg_hypothesis) * 100 if avg_hypothesis > 0 else 0
        better_system = "HYPOTHESIS"

    total = len(results)
    return {
        "total_questions": total,
        "main_avg_score": avg_main,
        "hyp_avg_score": avg_hypothesis,
        "wins_main": wins_main,
        "wins_hyp": wins_hypothesis,
        "ties": ties,
        "win_rate_main": (wins_main / total) * 100 if total else 0,
        "win_rate_hyp": (wins_hypothesis / total) * 100 if total else 0,
        "tie_rate": (ties / total) * 100 if total else 0,
        "advantage_percent": advantage_pct,
        "better_system": better_system,
        "improvement": advantage_pct if better_system == "HYPOTHESIS" else -advantage_pct,
    }


def save_results(results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    """Save detailed evaluation artifacts to llm_evaluation/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("llm_evaluation")
    output_dir.mkdir(exist_ok=True)

    details_path = output_dir / f"llm_evaluation_{timestamp}.json"
    stats_path = output_dir / f"statistics_{timestamp}.json"
    report_path = output_dir / f"report_{timestamp}.txt"

    with details_path.open("w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, ensure_ascii=False, indent=2)
    with stats_path.open("w", encoding="utf-8") as file_handle:
        json.dump(stats, file_handle, ensure_ascii=False, indent=2)

    with report_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("LLM EVALUATION REPORT (PERCENTAGE BASED)\n")
        file_handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write("=" * 80 + "\n\n")
        file_handle.write("OVERALL SCORES:\n")
        file_handle.write(f"  MAIN system:       {stats['main_avg_score']:.1f}%\n")
        file_handle.write(f"  HYPOTHESIS system: {stats['hyp_avg_score']:.1f}%\n\n")
        file_handle.write("WIN/LOSS:\n")
        file_handle.write(f"  MAIN wins:       {stats['wins_main']} ({stats['win_rate_main']:.1f}%)\n")
        file_handle.write(f"  HYPOTHESIS wins: {stats['wins_hyp']} ({stats['win_rate_hyp']:.1f}%)\n")
        file_handle.write(f"  Ties:            {stats['ties']} ({stats['tie_rate']:.1f}%)\n\n")
        file_handle.write(f"CONCLUSION: {stats['better_system']} is better by {stats['advantage_percent']:.1f}%\n\n")
        file_handle.write("DETAILS BY QUESTION:\n")
        for item in results:
            file_handle.write(f"\nQ{item['id']}: {item['question'][:80]}\n")
            file_handle.write(f"  MAIN score: {item['main']['evaluation'].get('total', 0):.1f}%\n")
            file_handle.write(f"  HYP score:  {item['hypothesis']['evaluation'].get('total', 0):.1f}%\n")
            file_handle.write(f"  Winner: {item['comparison'].get('winner', '?')}\n")
            file_handle.write(f"  Advantage: +{item['comparison'].get('winner_percent', 0):.0f}%\n")

    log.info("saved details=%s", details_path)
    log.info("saved stats=%s", stats_path)
    log.info("saved report=%s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="LLM evaluation (percentage-based).")
    parser.add_argument("--main", "-m", required=True, help="Main answers JSON file")
    parser.add_argument("--hypothesis", "-hyp", required=True, help="Hypothesis answers JSON file")
    return parser.parse_args()


def main() -> int:
    """Run evaluation command."""
    configure_logging()
    args = parse_args()

    main_answers = load_json_file(args.main)
    hypothesis_answers = load_json_file(args.hypothesis)
    log.info("loaded MAIN=%s HYPOTHESIS=%s", len(main_answers), len(hypothesis_answers))

    try:
        ollama.list()
    except (RuntimeError, OSError, ValueError):
        log.error("ollama is not running")
        return 1

    results = evaluate_all(main_answers, hypothesis_answers)
    stats = calculate_statistics(results)
    save_results(results, stats)
    log.info("evaluation complete total=%s better=%s", stats["total_questions"], stats["better_system"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
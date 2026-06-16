#!/usr/bin/env python3
"""Run GPT-4o-mini judge on baseline answer sets vs golden reference."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import config  # noqa: E402

log = logging.getLogger(__name__)

BASELINE_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = BASELINE_DIR / "golden_set.json"
OLLAMA_PLAIN_FILE = BASELINE_DIR / "rag_answers_gpu.json"
OLLAMA_HYBRID_FILE = BASELINE_DIR / "evaluation_llama3b_gpt_judge_v3.json"
GPT_HYBRID_FILE = BASELINE_DIR / "evaluation_results_20260518_160608.json"
DEFAULT_OUTPUT = BASELINE_DIR / "llm_judge_scores.json"

JSON_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)
METRIC_KEYS = ("relevance", "accuracy", "completeness", "clarity", "total")
JUDGE_MODEL = "gpt-4o-mini"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def answers_by_id(payload: Any) -> dict[int, dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("results", [])
    return {int(row["id"]): row for row in rows}


def clamp_0_100(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_first_json_block(response_text: str) -> dict[str, Any]:
    match = JSON_BLOCK_PATTERN.search(response_text)
    if match is None:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def build_judge_prompt(question: str, expected: str, answer: str) -> str:
    ref = (expected or "")[:2500]
    ans = (answer or "")[:2500]
    return f"""You are an expert evaluator of RAG answers. Compare the ANSWER to the REFERENCE (golden) for the QUESTION.

QUESTION:
{question}

REFERENCE (golden expected answer):
{ref}

ANSWER (to evaluate):
{ans}

Rate on 0-100% (0=terrible, 100=perfect):
1. RELEVANCE: Does the answer address the question?
2. ACCURACY: Factual agreement with the reference (penalize hallucinations and wrong facts).
3. COMPLETENESS: Covers key points from the reference.
4. CLARITY: Clear and well structured.

If the reference says information is not in documentation, a correct "not found" answer should score high on accuracy.

Reply ONLY with JSON:
{{"relevance": 0-100, "accuracy": 0-100, "completeness": 0-100, "clarity": 0-100, "total": 0-100}}
"""


def judge_with_ollama(question: str, expected: str, answer: str) -> dict[str, Any]:
    import ollama

    prompt = build_judge_prompt(question, expected, answer)
    try:
        response = ollama.chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300},
        )
    except (RuntimeError, OSError, ValueError) as error:
        return {"error": str(error), "total": 0.0, "judge_backend": "ollama"}

    content = (response.get("message") or {}).get("content", "")
    metrics = parse_first_json_block(content)
    if not metrics:
        return {"error": "Failed to parse", "raw": content[:200], "total": 0.0, "judge_backend": "ollama"}

    for key in METRIC_KEYS:
        if key in metrics:
            metrics[key] = clamp_0_100(metrics[key])
    if "total" not in metrics:
        parts = [metrics[k] for k in ("relevance", "accuracy", "completeness", "clarity") if k in metrics]
        if parts:
            metrics["total"] = round(sum(parts) / len(parts), 1)
    metrics["judge_model"] = config.ollama_model
    metrics["judge_backend"] = "ollama"
    return metrics


def judge_with_openai(client: Any, question: str, expected: str, answer: str) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": build_judge_prompt(question, expected, answer)}],
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as error:
        log.warning("openai judge failed, fallback to ollama: %s", error)
        return judge_with_ollama(question, expected, answer)

    content = response.choices[0].message.content or ""
    metrics = parse_first_json_block(content)
    if not metrics:
        return judge_with_ollama(question, expected, answer)

    for key in METRIC_KEYS:
        if key in metrics:
            metrics[key] = clamp_0_100(metrics[key])
    if "total" not in metrics:
        parts = [metrics[k] for k in ("relevance", "accuracy", "completeness", "clarity") if k in metrics]
        if parts:
            metrics["total"] = round(sum(parts) / len(parts), 1)
    metrics["judge_model"] = JUDGE_MODEL
    metrics["judge_backend"] = "openai"
    return metrics


def extract_existing_hybrid_scores() -> dict[int, dict[str, Any]]:
    """Reuse llm_judge from evaluation_llama3b_gpt_judge_v3.json."""
    by_id: dict[int, dict[str, Any]] = {}
    for qid, row in answers_by_id(load_json(OLLAMA_HYBRID_FILE)).items():
        judge = row.get("llm_judge")
        if isinstance(judge, dict) and judge.get("total") is not None:
            by_id[qid] = dict(judge)
    return by_id


def save_checkpoint(output_path: Path, result: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_set(
    client: Any | None,
    label: str,
    golden: dict[int, dict[str, Any]],
    answers: dict[int, dict[str, Any]],
    existing: dict[int, dict[str, Any]] | None = None,
    delay_s: float = 0.2,
    on_progress: Callable[[dict[int, dict[str, Any]]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    existing = existing or {}
    scores: dict[int, dict[str, Any]] = dict(existing)
    for qid in sorted(golden):
        cached_total = (
            scores[qid].get("total")
            if qid in scores
            else None
        )
        has_metrics = qid in scores and any(
            scores[qid].get(k) is not None for k in ("relevance", "accuracy", "completeness", "clarity", "total")
        )
        if qid in scores and has_metrics and "error" not in scores[qid] and (
            cached_total is not None and cached_total > 0
            or any(scores[qid].get(k) for k in ("relevance", "accuracy", "completeness", "clarity"))
        ):
            log.info("%s qid=%s cached total=%s", label, qid, scores[qid].get("total"))
            continue
        gold = golden[qid]
        ans_row = answers.get(qid, {})
        answer = str(ans_row.get("answer", ""))
        if not answer:
            scores[qid] = {"error": "empty answer", "total": 0.0}
        else:
            log.info("%s qid=%s judging...", label, qid)
            if client is not None:
                scores[qid] = judge_with_openai(
                    client,
                    str(gold.get("question", "")),
                    str(gold.get("expected_answer", "")),
                    answer,
                )
            else:
                scores[qid] = judge_with_ollama(
                    str(gold.get("question", "")),
                    str(gold.get("expected_answer", "")),
                    answer,
                )
            time.sleep(delay_s)
        if on_progress is not None:
            on_progress(scores)
    return scores


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="LLM judge for baseline comparison.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-openai", action="store_true", help="Only export cached hybrid scores")
    args = parser.parse_args()

    golden_list = load_json(GOLDEN_FILE)
    golden = {int(item["id"]): item for item in golden_list}

    output_path = Path(args.output)
    cached: dict[str, Any] = {}
    if output_path.exists():
        cached = load_json(output_path)

    hybrid_existing = extract_existing_hybrid_scores()
    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "golden_file": GOLDEN_FILE.name,
        "runs": {},
    }

    if args.skip_openai:
        result["runs"]["ollama_hybrid"] = {str(k): v for k, v in hybrid_existing.items()}
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("wrote cached hybrid only -> %s", output_path)
        return 0

    client: Any | None = None
    api_key = config.openai_api_key
    if api_key and not api_key.startswith("user_") and "provided" not in api_key.lower():
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            log.info("judge backend: openai (%s)", JUDGE_MODEL)
        except ImportError:
            log.warning("openai package missing, using ollama judge")
    else:
        log.warning("OPENAI_API_KEY invalid or placeholder — using ollama judge (%s)", config.ollama_model)

    def cached_run(key: str) -> dict[int, dict[str, Any]]:
        raw = cached.get("runs", {}).get(key) or {}
        return {int(k): v for k, v in raw.items() if isinstance(v, dict)}

    plain_cached = cached_run("ollama_plain")
    gpt_cached = cached_run("gpt_hybrid")
    result["runs"]["ollama_hybrid"] = {str(k): v for k, v in hybrid_existing.items()}
    save_checkpoint(output_path, result)

    def persist_plain(scores: dict[int, dict[str, Any]]) -> None:
        result["runs"]["ollama_plain"] = {str(k): v for k, v in scores.items()}
        save_checkpoint(output_path, result)

    result["runs"]["ollama_plain"] = {
        str(k): v
        for k, v in run_set(
            client,
            "ollama_plain",
            golden,
            answers_by_id(load_json(OLLAMA_PLAIN_FILE)),
            plain_cached,
            on_progress=persist_plain,
        ).items()
    }
    save_checkpoint(output_path, result)

    def persist_gpt(scores: dict[int, dict[str, Any]]) -> None:
        result["runs"]["gpt_hybrid"] = {str(k): v for k, v in scores.items()}
        save_checkpoint(output_path, result)

    result["runs"]["gpt_hybrid"] = {
        str(k): v
        for k, v in run_set(
            client,
            "gpt_hybrid",
            golden,
            answers_by_id(load_json(GPT_HYBRID_FILE)),
            gpt_cached,
            on_progress=persist_gpt,
        ).items()
    }

    save_checkpoint(output_path, result)
    for label, scores in result["runs"].items():
        totals = [v.get("total", 0) for v in scores.values() if isinstance(v, dict)]
        avg = sum(totals) / len(totals) if totals else 0
        log.info("%s avg_judge=%.1f%% n=%s", label, avg, len(totals))
    log.info("saved %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

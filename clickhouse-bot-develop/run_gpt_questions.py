#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run benchmark questions with OpenAI chat models and local RAG context."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config  # noqa: E402
from core.database import db  # noqa: E402
from core.embeddings import embedder  # noqa: E402
from core.openai_client import get_openai_client  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from core.reranker import reranker  # noqa: E402
from evaluator.folder_scanner import FolderScanner  # noqa: E402
from evaluator.qa_loader import QALoader  # noqa: E402
from router.smart_router import select_prompt  # noqa: E402

log = logging.getLogger(__name__)

AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "gpt-4o": {"name": "GPT-4o", "speed": "fast", "quality": "excellent", "cost_per_1m": 2.50},
    "gpt-4o-mini": {"name": "GPT-4o Mini", "speed": "very fast", "quality": "good", "cost_per_1m": 0.15},
    "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "speed": "fastest", "quality": "medium", "cost_per_1m": 0.50},
    "gpt-4-turbo": {"name": "GPT-4 Turbo", "speed": "medium", "quality": "excellent", "cost_per_1m": 10.00},
}

DEFAULT_QUESTIONS = {
    3: "Create a step by step guide how to integrate sale form",
    9: "What is a Connecting Party?",
    13: "What is a merchant control key? Is it included in request?",
    16: "Do I need private key for v4/transfer?",
    19: "What is the difference between v2/sale and v2/sale-form?",
    22: "Should I implement both status and callback handling?",
    25: "How to calculate control parameter for v2/sale?",
    30: "How to make a reversal?",
    49: "What is the difference between RPI and card number?",
    52: "Do I need PCI for v2/sale?",
}

SYSTEM_PROMPT = """You are a technical documentation expert. Answer based ONLY on the provided context.

If information not found: "NOT FOUND in documentation"

Be concise and accurate."""


def ask_gpt(question: str, model: str) -> dict[str, Any]:
    """Ask GPT with retrieved local RAG context."""
    q_emb = list(embedder.generate_cached(question))
    results = db.search(q_emb)
    if not results:
        return {
            "answer": "NOT FOUND in documentation",
            "time": 0.0,
            "tokens": 0,
            "status": "no_context",
        }

    reranked = reranker.rerank(question, results)
    context_parts: list[str] = []
    for row in reranked[: config.rerank_top_k]:
        chunk, source, page = row[0], row[1], row[2]
        context_parts.append("[%s, p.%s]\n%s" % (source, page, str(chunk)[:800]))
    context = "\n\n".join(context_parts)
    system_prompt, _num_predict, _temperature = select_prompt(question)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": "CONTEXT:\n%s\n\nQUESTION: %s" % (context, question)},
    ]

    start = time.time()
    try:
        response = get_openai_client().chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            temperature=0.1,
            max_tokens=800,
        )
        elapsed = time.time() - start
        answer = response.choices[0].message.content
        usage = response.usage
        tokens = usage.total_tokens if usage is not None else 0
        return {
            "answer": answer,
            "time": elapsed,
            "tokens": tokens,
            "status": "success",
        }
    except Exception as exc:
        log.error("GPT request failed: %s", exc)
        return {
            "answer": "ERROR: %s" % exc,
            "time": time.time() - start,
            "tokens": 0,
            "status": "error",
        }


def load_local_questions(docs_dir: str, limit: int) -> list[tuple[int, str]]:
    """Load questions from local docs folder question/answer files."""
    scanner = FolderScanner(docs_dir)
    folders = scanner.scan()
    questions: list[tuple[int, str]] = []
    next_id = 1

    for folder in folders:
        qa_pairs = QALoader.load_qa_pairs(folder["questions_file"], folder["answers_file"])
        for question, _expected in qa_pairs:
            text = question.strip()
            if not text:
                continue
            questions.append((next_id, text))
            next_id += 1
            if len(questions) >= limit:
                return questions
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run questions with GPT")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o-mini",
        choices=list(AVAILABLE_MODELS.keys()),
        help="GPT model to use",
    )
    parser.add_argument(
        "--source",
        choices=["local", "default"],
        default="local",
        help="Question source: local docs Q/A files or bundled defaults",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=config.docs_folder,
        help="Docs directory for --source local",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of questions to run",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    model = args.model
    model_info = AVAILABLE_MODELS[model]

    log.info("%s", "\n" + "=" * 70)
    log.info("RUNNING QUESTIONS WITH %s", str(model_info["name"]).upper())
    log.info("  Speed: %s", model_info["speed"])
    log.info("  Quality: %s", model_info["quality"])
    log.info("  Cost: $%s/1M tokens", model_info["cost_per_1m"])
    log.info("%s", "=" * 70)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found in .env")
        return

    log.info("[CHECK] OpenAI API key prefix=%s...", api_key[:10])
    chunk_count = db.load_chunk_count()
    log.info("[INFO] Local chunks in active store=%s", chunk_count)
    if chunk_count == 0:
        log.error("No indexed chunks found. Run ingestion first, e.g.:")
        log.error(
            '  uv run python app.py pipeline --profile mixed '
            '--input-dir "./doc-2.0-sources" --force-reload'
        )
        return

    if args.source == "local":
        question_rows = load_local_questions(args.docs_dir, args.limit)
        if not question_rows:
            log.warning("Local questions not found, fallback to default set")
            question_rows = list(DEFAULT_QUESTIONS.items())[: args.limit]
    else:
        question_rows = list(DEFAULT_QUESTIONS.items())[: args.limit]
    log.info("[INFO] Question source=%s count=%s", args.source, len(question_rows))

    results: list[dict[str, Any]] = []
    for qid, question in question_rows:
        log.info("%s", "\n" + "=" * 70)
        log.info("Q%s: %s...", qid, question[:70])
        log.info("%s", "=" * 70)

        result = ask_gpt(question, model)

        answer_preview = (result["answer"] or "")[:500]
        log.info("ANSWER (preview): %s", answer_preview)
        log.info("TIME: %.2fs", float(result["time"]))
        log.info("TOKENS: %s", result["tokens"])

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": result["answer"],
                "time": result["time"],
                "tokens": result["tokens"],
                "status": result["status"],
                "model": model,
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_gpt")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / ("answers_gpt_%s_%s.json" % (model, timestamp))
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    log.info("[SAVE] Results saved path=%s", output_file)

    avg_time = sum(float(r["time"]) for r in results) / len(results)
    total_tokens = sum(int(r["tokens"]) for r in results)
    estimated_cost = total_tokens * float(model_info["cost_per_1m"]) / 1_000_000

    log.info("%s", "\nSTATISTICS:")
    log.info("  Model: %s", model_info["name"])
    log.info("  Average time: %.2fs", avg_time)
    log.info("  Total tokens: %s", total_tokens)
    log.info("  Estimated cost: $%.4f", estimated_cost)

    log.info("[DONE]")


if __name__ == "__main__":
    main()

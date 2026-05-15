#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a fixed set of questions against the main vector store."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.logger import init_logger, setup_logging
from rag_engine.engine import rag

log = logging.getLogger(__name__)

QUESTIONS = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10 questions on main table")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    init_logger(config.llm_model)
    db.set_active_table("main")
    log.info("Using table=%s chunks=%s", db.load_active_table(), db.load_chunk_count())

    log.info("%s", "\n" + "=" * 70)
    log.info("RUNNING 10 QUESTIONS ON MAIN TABLE (OLD DATA)")
    log.info("%s", "=" * 70)

    results: list[dict[str, object]] = []
    for qid, question in QUESTIONS.items():
        log.info("%s", "\n" + "=" * 70)
        log.info("Q%s: %s", qid, question)
        log.info("%s", "=" * 70)

        start = time.time()
        result = rag.ask(question)
        elapsed = time.time() - start

        answer_text = str(result["answer"])
        log.info("ANSWER (preview): %s", answer_text[:500])
        log.info("SOURCES: %s", result["sources"])
        log.info("TIME: %.2fs", elapsed)

        results.append({
            "id": qid,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "time": elapsed,
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_main")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"answers_main_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    log.info("[SAVE] Results saved path=%s", output_file)
    log.info("[DONE]")


if __name__ == "__main__":
    main()

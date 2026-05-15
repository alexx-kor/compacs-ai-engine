#!/usr/bin/env python3
"""Unified CLI entrypoint with mode selection via subcommands."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config
from core.logger import setup_logging
from llm_evaluate_percent import compare_two_answers_percent, llm_evaluate_percent, load_json
from load_graph_chunks import load_graph_chunks
from core.preprocess_server import run_preprocess_server

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build root parser with subcommands for all common actions."""
    parser = argparse.ArgumentParser(description="RAG System unified CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest folder into local vector store")
    ingest_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input folder for ingestion (supports pdf/txt/md/rst)",
    )
    ingest_parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Reset active store before ingestion",
    )
    ingest_parser.add_argument(
        "--preprocess-url",
        type=str,
        default=None,
        help="Optional preprocess service URL, e.g. http://127.0.0.1:8080",
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess-server",
        help="Run preprocessing HTTP server (use 8080 baseline, 8090 experiment)",
    )
    preprocess_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    preprocess_parser.add_argument("--port", type=int, default=8080, help="Bind port")
    preprocess_parser.add_argument(
        "--profile",
        choices=["baseline", "experiment"],
        default=None,
        help="Preprocessing profile (auto-selected by port when omitted)",
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run predefined ingestion pipeline profile")
    pipeline_parser.add_argument(
        "--profile",
        choices=["pdf", "rst", "graph", "mixed"],
        default="mixed",
        help="Pipeline profile to execute",
    )
    pipeline_parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input folder for file-based profiles (pdf/rst/mixed)",
    )
    pipeline_parser.add_argument(
        "--graph-file-path",
        type=str,
        default=None,
        help="Optional graph JSONL path for graph profile",
    )
    pipeline_parser.add_argument(
        "--preprocess-url",
        type=str,
        default=None,
        help="Optional preprocess service URL for file-based profiles",
    )
    pipeline_parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Reset active store before file-based ingestion",
    )

    query_parser = subparsers.add_parser("query", help="Ask one question")
    query_parser.add_argument("--question", "-q", type=str, required=True, help="Question text")

    eval_parser = subparsers.add_parser("evaluate", help="Run QA evaluation")
    eval_parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions")

    judge_parser = subparsers.add_parser("judge", help="Evaluate answers with LLM judge")
    judge_parser.add_argument(
        "--mode",
        choices=["single", "pairwise"],
        default="pairwise",
        help="Single-file scoring or pairwise comparison",
    )
    judge_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON (required for mode=single)",
    )
    judge_parser.add_argument(
        "--main-file",
        type=str,
        default=None,
        help="Main answers JSON (required for mode=pairwise)",
    )
    judge_parser.add_argument(
        "--hyp-file",
        type=str,
        default=None,
        help="Hypothesis answers JSON (required for mode=pairwise)",
    )
    judge_parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap of compared/evaluated items",
    )
    judge_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path (JSON).",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive question loop")
    chat_parser.add_argument("--prompt", type=str, default="Ask> ", help="CLI prompt text")

    return parser


def run_chat(prompt: str = "Ask> ") -> None:
    """Run interactive chat mode over the current indexed data."""
    from rag_engine.engine import rag

    log.info("%s", "\n" + "=" * 60 + "\nINTERACTIVE MODE (type 'exit' to stop)\n" + "=" * 60)
    while True:
        try:
            question = input(f"\n{prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            log.info("[STOP] Interactive session ended")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            log.info("[STOP] Interactive session ended")
            break

        result = rag.ask(question)
        log.info("A: %s", result.get("answer", ""))
        log.info("Sources: %s", result.get("sources", []))
        log.info("Time seconds=%s", result.get("time_total", 0))


def run_judge_single(input_file: str, max_items: int | None = None) -> dict[str, Any]:
    """Evaluate a single answers file and return report payload."""
    rows = load_json(input_file)
    if max_items is not None:
        rows = rows[:max_items]

    results: list[dict[str, Any]] = []
    for index, item in enumerate(rows, 1):
        question = str(item.get("question", ""))
        answer = str(item.get("answer", item.get("generated_answer", "")))
        log.debug("Judge single progress index=%s total=%s", index, len(rows))
        evaluation = llm_evaluate_percent(question, answer)
        results.append(
            {
                "id": item.get("id", index),
                "question": question,
                "answer": answer[:500],
                "evaluation": evaluation,
            }
        )

    totals = [float(r["evaluation"].get("total", 0.0)) for r in results]
    summary = {
        "count": len(results),
        "avg_total": (sum(totals) / len(totals)) if totals else 0.0,
        "max_total": max(totals) if totals else 0.0,
        "min_total": min(totals) if totals else 0.0,
    }
    return {"mode": "single", "input": input_file, "summary": summary, "results": results}


def run_judge_pairwise(
    main_file: str,
    hyp_file: str,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Compare two answer files with the LLM judge and return report payload."""
    main_rows = load_json(main_file)
    hyp_rows = load_json(hyp_file)
    main_dict = {str(item.get("id", idx)): item for idx, item in enumerate(main_rows)}
    hyp_dict = {str(item.get("id", idx)): item for idx, item in enumerate(hyp_rows)}

    common_ids = [item_id for item_id in main_dict if item_id in hyp_dict]
    if max_items is not None:
        common_ids = common_ids[:max_items]

    results: list[dict[str, Any]] = []
    for index, item_id in enumerate(common_ids, 1):
        main_item = main_dict[item_id]
        hyp_item = hyp_dict[item_id]
        question = str(main_item.get("question", hyp_item.get("question", "")))
        main_answer = str(main_item.get("answer", main_item.get("generated_answer", "")))
        hyp_answer = str(hyp_item.get("answer", hyp_item.get("generated_answer", "")))

        log.debug("Judge pairwise progress index=%s total=%s", index, len(common_ids))
        main_eval = llm_evaluate_percent(question, main_answer)
        hyp_eval = llm_evaluate_percent(question, hyp_answer)
        comparison = compare_two_answers_percent(question, main_answer, hyp_answer)
        results.append(
            {
                "id": item_id,
                "question": question,
                "main": {"answer": main_answer[:500], "evaluation": main_eval},
                "hypothesis": {"answer": hyp_answer[:500], "evaluation": hyp_eval},
                "comparison": comparison,
            }
        )

    winners = [str(row["comparison"].get("winner", "UNKNOWN")) for row in results]
    summary = {
        "count": len(results),
        "wins_main": winners.count("A"),
        "wins_hypothesis": winners.count("B"),
        "ties": winners.count("TIE"),
    }
    return {
        "mode": "pairwise",
        "main_file": main_file,
        "hyp_file": hyp_file,
        "summary": summary,
        "results": results,
    }


def save_judge_report(payload: dict[str, Any], output_path: str | None) -> str:
    """Save judge report to JSON and return file path."""
    if output_path:
        target = Path(output_path)
    else:
        target_dir = Path(config.results_folder) / "judge_reports"
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target = target_dir / f"judge_{payload['mode']}_{stamp}.json"

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(target)


def main() -> None:
    """Dispatch command selected by the user."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    start_time = time.time()
    log.info("[START] %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

    if args.command == "ingest":
        from run import load_documents

        count = load_documents(
            force_reload=args.force_reload,
            input_dir=args.input_dir,
            preprocess_url=args.preprocess_url,
        )
        log.info("[DONE] Loaded chunks=%s", count)
    elif args.command == "preprocess-server":
        profile = args.profile
        if profile is None:
            profile = "baseline" if args.port == 8080 else "experiment" if args.port == 8090 else "baseline"
        run_preprocess_server(host=args.host, port=args.port, profile=profile)
    elif args.command == "pipeline":
        extension_map: dict[str, set[str] | None] = {
            "pdf": {".pdf"},
            "rst": {".rst"},
            "mixed": None,
        }
        if args.profile == "graph":
            loaded = load_graph_chunks(args.graph_file_path) if args.graph_file_path else load_graph_chunks()
            log.info("[DONE] Loaded graph chunks=%s", len(loaded))
        else:
            from run import load_documents

            allowed = extension_map[args.profile]
            count = load_documents(
                force_reload=args.force_reload,
                input_dir=args.input_dir,
                preprocess_url=args.preprocess_url,
                allowed_extensions=allowed,
            )
            log.info("[DONE] Loaded chunks=%s", count)
    elif args.command == "query":
        from rag_engine.engine import rag

        result = rag.ask(args.question)
        log.info("Q: %s", args.question)
        log.info("A: %s", result.get("answer", ""))
        log.info("Sources: %s", result.get("sources", []))
        log.info("Time seconds=%s", result.get("time_total", 0))
    elif args.command == "evaluate":
        from run import run_evaluation

        run_evaluation(max_questions=args.max_questions)
    elif args.command == "judge":
        if args.mode == "single":
            if not args.input:
                raise SystemExit("--input is required for judge --mode single")
            payload = run_judge_single(args.input, max_items=args.max_items)
        else:
            if not args.main_file or not args.hyp_file:
                raise SystemExit("--main-file and --hyp-file are required for judge --mode pairwise")
            payload = run_judge_pairwise(
                args.main_file,
                args.hyp_file,
                max_items=args.max_items,
            )
        saved_path = save_judge_report(payload, args.output)
        log.info("[DONE] Judge report saved path=%s", saved_path)
        log.info("Summary: %s", payload["summary"])
    elif args.command == "chat":
        run_chat(prompt=args.prompt)

    log.info("[FINISH] Completed in %.2fs", time.time() - start_time)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI entrypoint for loading, querying, and evaluating the RAG system."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
from core.ingestion import ingestion_service
from core.logger import setup_logging
from core.preprocess_client import preprocess_via_service
from evaluator.folder_scanner import FolderScanner
from evaluator.qa_loader import QALoader
from evaluator.results import ResultsAnalyzer
from rag_engine.engine import rag
from router.smart_router import SmartPromptRouter

log = logging.getLogger(__name__)

_BANNER = """
    ============================================================
    RAG SYSTEM - KNOWLEDGE BASE
    ============================================================

    Commands:
        python run.py                    - Load docs + interactive
        python run.py --evaluate        - Evaluate on Q&A pairs
        python run.py --query "..."     - Ask a single question
        python run.py --load-only       - Only load documents
        python run.py --force-reload     - Force reload all documents
    """.strip()


def load_documents(
    force_reload: bool = False,
    input_dir: str | None = None,
    preprocess_url: str | None = None,
    allowed_extensions: set[str] | None = None,
) -> int:
    """Load source documents into the vector database.

    Args:
        force_reload: Recreate DB tables before loading when True.

    Returns:
        Number of chunks inserted during this run.
    """
    log.info("%s", "\n" + "=" * 60 + "\nLOADING KNOWLEDGE BASE\n" + "=" * 60)

    existing_chunks = db.get_chunk_count()

    if existing_chunks > 0 and not force_reload:
        log.info("Database already has chunks count=%s; skipping document loading", existing_chunks)
        return existing_chunks

    if force_reload:
        log.info("Force reload enabled")

    db.init_database(force_recreate=force_reload)

    selected_dir = input_dir or config.docs_folder
    chunks, report = ingestion_service.collect_chunks(
        selected_dir,
        start_id=0,
        allowed_extensions=allowed_extensions,
    )
    if not chunks:
        log.warning("No supported files or chunks in dir=%s", selected_dir)
        return 0

    log.info(
        "Ingestion summary input_dir=%s discovered=%s processed=%s skipped=%s",
        report.input_dir,
        report.files_discovered,
        report.files_processed,
        report.files_skipped,
    )

    texts = [chunk["chunk"] for chunk in chunks]
    if preprocess_url:
        log.info("Using preprocess service url=%s", preprocess_url)
        texts = [preprocess_via_service(text, preprocess_url) for text in texts]
        for chunk, cleaned_text in zip(chunks, texts):
            chunk["chunk"] = cleaned_text
    log.info("Generating embeddings count=%s", len(texts))
    embeddings = embedder.generate(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    db.insert_batch(chunks)

    log.info("Total chunks inserted=%s", report.chunks_created)
    return report.chunks_created


def find_all_qa_pairs() -> list[tuple[str, str]]:
    """Find and load all question/answer pairs from docs folders.

    Returns:
        List of (question, expected_answer) pairs.
    """
    log.info("%s", "\n" + "=" * 60 + "\nSEARCHING FOR QUESTIONS & ANSWERS\n" + "=" * 60)

    scanner = FolderScanner(config.docs_folder)
    folders = scanner.scan()

    if not folders:
        log.warning("No questions or answers files found under docs_folder=%s", config.docs_folder)
        return []

    all_pairs = []
    for folder in folders:
        log.info(
            "QA folder name=%s questions_file=%s answers_file=%s",
            folder["folder_name"],
            folder["questions_file"],
            folder["answers_file"],
        )

        qa_pairs = QALoader.load_qa_pairs(
            folder["questions_file"],
            folder["answers_file"],
        )
        log.info("Pairs loaded in folder count=%s", len(qa_pairs))
        all_pairs.extend(qa_pairs)

    log.info("Total QA pairs found=%s", len(all_pairs))
    return all_pairs


def run_evaluation(
    qa_pairs: list[tuple[str, str]] | None = None,
    max_questions: int | None = None,
) -> list[dict[str, Any]] | None:
    """Run evaluation over QA pairs and save results.

    Args:
        qa_pairs: Optional preloaded QA pairs.
        max_questions: Optional max number of questions.

    Returns:
        Evaluation rows, or None when no data found.
    """
    log.info("%s", "\n" + "=" * 60 + "\nQA EVALUATION\n" + "=" * 60)

    if qa_pairs is None:
        qa_pairs = find_all_qa_pairs()

    if not qa_pairs:
        log.error("No QA pairs found for evaluation")
        return None

    if max_questions:
        qa_pairs = qa_pairs[:max_questions]
        log.info("Limited evaluation to max_questions=%s", max_questions)

    examples_count = (
        SmartPromptRouter.get_examples_count()
        if hasattr(SmartPromptRouter, "get_examples_count")
        else 0
    )
    if examples_count > 0:
        log.info("Few-shot examples count=%s", examples_count)

    all_results = []

    for i, (question, expected_answer) in enumerate(qa_pairs):
        log.debug("evaluation_progress index=%s total=%s", i + 1, len(qa_pairs))

        result = rag.ask(question)

        words_q = set(question.lower().split())
        words_a = set(result["answer"].lower().split())
        similarity = len(words_q & words_a) / max(len(words_q), 1) if words_q else 0

        all_results.append({
            "question": question[:200],
            "expected_answer": expected_answer[:200],
            "generated_answer": result["answer"][:300],
            "similarity_score": round(similarity, 3),
            "time_seconds": result["time_total"],
            "sources": str(result["sources"]),
            "status": result.get("status", "success"),
        })

    if all_results:
        ResultsAnalyzer.save(all_results)
        success_results = [r for r in all_results if r["status"] == "success"]
        if success_results:
            avg_score = sum(r["similarity_score"] for r in success_results) / len(success_results)
            log.info("Average similarity score=%.3f", avg_score)
        log.info(
            "Evaluation finished success=%s total=%s",
            len(success_results),
            len(all_results),
        )

    return all_results


def main() -> None:
    """Parse CLI args and execute selected workflow."""
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument("--query", "-q", type=str, help="Ask a question")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run QA evaluation")
    parser.add_argument("--load-only", action="store_true", help="Only load documents and exit")
    parser.add_argument("--force-reload", action="store_true", help="Recreate tables and reload docs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input folder for ingestion (supports pdf/txt/md/rst)",
    )
    parser.add_argument(
        "--preprocess-url",
        type=str,
        default=None,
        help="Optional preprocess service URL, e.g. http://127.0.0.1:8080",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions for evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    start_time = time.time()
    start_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log.info("[START] %s", start_datetime)

    log.info("%s", _BANNER)

    loaded_chunks = load_documents(
        force_reload=args.force_reload,
        input_dir=args.input_dir,
        preprocess_url=args.preprocess_url,
    )
    if args.load_only:
        log.info("[DONE] Loaded chunks=%s", loaded_chunks)
        return

    if args.query:
        log.info("%s", "\n" + "=" * 60 + "\nSINGLE QUESTION MODE\n" + "=" * 60)
        result = rag.ask(args.query)
        log.info("Q: %s", args.query)
        log.info("A: %s", result.get("answer", ""))
        log.info("Sources: %s", result.get("sources", []))
        log.info("Time seconds=%s", result.get("time_total", 0))
    elif args.evaluate:
        run_evaluation(max_questions=args.max_questions)
    else:
        log.info("%s", "\n" + "=" * 60 + "\nINTERACTIVE MODE (type 'exit' to stop)\n" + "=" * 60)
        while True:
            try:
                question = input("\nAsk> ").strip()
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

    elapsed_seconds = time.time() - start_time
    log.info("[FINISH] Completed in %.2fs", elapsed_seconds)


if __name__ == "__main__":
    main()

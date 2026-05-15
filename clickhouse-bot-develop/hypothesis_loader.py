#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load document chunks into the hypothesis vector store with structured logging."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.document_processor import doc_processor
from core.embeddings import embedder
from core.logger import init_logger, setup_logging

log = logging.getLogger(__name__)


def load_hypothesis(
    hypothesis_name: str,
    hypothesis_params: dict[str, Any] | None = None,
    force_reload: bool = False,
) -> int:
    """Load documents into the hypothesis table with structured logging."""
    log.info("%s", "\n" + "=" * 70)
    log.info(" HYPOTHESIS LOADER: %s", hypothesis_name)
    log.info("%s", "=" * 70)

    structured = init_logger(config.llm_model)
    start_total = time.time()

    params_str = json.dumps(hypothesis_params) if hypothesis_params else ""

    if not config.doc_files:
        log.error("No document files configured under docs_folder=%s", config.docs_folder)
        return 0

    log.info("Hypothesis name=%s files=%s", hypothesis_name, len(config.doc_files))

    if force_reload:
        db.init_hypothesis_database(force_recreate=True)
    else:
        db.init_hypothesis_database(force_recreate=False)

    chunk_id = 0
    all_chunks: list[dict[str, Any]] = []

    log.info("%s", "\n[STEP 1] CHUNKING")
    log.info("%s", "-" * 40)

    chunk_start = time.time()

    for i, (file_path, source_name) in enumerate(config.doc_files):
        file_start = time.time()
        chunks = doc_processor.process_document(file_path, source_name, chunk_id)

        if chunks:
            log.debug(
                "Chunked file index=%s total=%s source=%s chunk_count=%s",
                i + 1,
                len(config.doc_files),
                source_name[:50],
                len(chunks),
            )

            file_time_ms = (time.time() - file_start) * 1000
            structured.log_ingest_file(
                filename=source_name,
                source_type="TXT",
                chunk_count=len(chunks),
                embedding_time_ms=0,
                total_time_ms=file_time_ms,
            )

            all_chunks.extend(chunks)
            chunk_id += len(chunks)
        else:
            structured.log_ingest_soft_skip(
                filename=source_name,
                reason="No valid chunks",
                file_size_bytes=os.path.getsize(file_path),
            )

    chunk_time = time.time() - chunk_start
    log.info("[CHUNKING] Done in %.2fs", chunk_time)
    log.info("[CHUNKING] Total chunks=%s", len(all_chunks))

    log.info("%s", "\n[STEP 2] EMBEDDING GENERATION")
    log.info("%s", "-" * 40)

    embed_start = time.time()

    texts = [c["chunk"] for c in all_chunks]
    log.info("Embedding texts count=%s", len(texts))

    embeddings = embedder.generate(texts)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    embed_time = time.time() - embed_start
    structured.log_ingest_backend_summary()

    log.info("%s", "\n[STEP 3] DATABASE INSERT")
    log.info("%s", "-" * 40)

    insert_start = time.time()

    db.insert_hypothesis_batch(all_chunks, hypothesis_name, params_str)

    insert_time = time.time() - insert_start

    total_time_ms = (time.time() - start_total) * 1000
    structured.log_ingest_batch_summary(total_time_ms)

    log.info("[EMBEDDING] Done in %.2fs", embed_time)
    log.info("[INSERT] Done in %.2fs", insert_time)

    total_time = time.time() - start_total

    log.info("%s", "\n" + "=" * 70)
    log.info(" HYPOTHESIS LOADING COMPLETE!")
    log.info("%s", "=" * 70)
    log.info("   Hypothesis: %s", hypothesis_name)
    log.info("   Files: %s", len(config.doc_files))
    log.info("   Chunks: %s", len(all_chunks))
    log.info("   Total time: %.2fs (%.2f min)", total_time, total_time / 60)
    log.info("   Log file: %s", structured.log_file)
    log.info("%s", "=" * 70)

    return len(all_chunks)


def list_hypotheses() -> list[str]:
    """List hypothesis names and chunk counts in the database."""
    log.info("%s", "\n" + "=" * 70)
    log.info(" HYPOTHESES IN DATABASE")
    log.info("%s", "=" * 70)

    hypotheses = db.load_all_hypotheses()

    if not hypotheses:
        log.warning("No hypotheses found")
    else:
        log.info("Found hypotheses count=%s", len(hypotheses))
        for name in hypotheses:
            count = db.load_hypothesis_chunk_count(name)
            log.info("   - %s: %s chunks", name, count)

    log.info("%s", "=" * 70)
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypothesis Loader for RAG System")
    parser.add_argument("--name", "-n", type=str, help="Hypothesis name")
    parser.add_argument("--params", "-p", type=str, help="Hypothesis parameters (JSON string)")
    parser.add_argument("--force", action="store_true", help="Force reload")
    parser.add_argument("--list", "-l", action="store_true", help="List all hypotheses")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    if args.list:
        list_hypotheses()
        return

    if not args.name:
        log.error("Please specify --name for hypothesis")
        return

    params: dict[str, Any] | None = None
    if args.params:
        try:
            raw = json.loads(args.params)
            params = raw if isinstance(raw, dict) else None
        except json.JSONDecodeError:
            log.error("Invalid JSON params: %s", args.params)
            return

    load_hypothesis(
        hypothesis_name=args.name,
        hypothesis_params=params,
        force_reload=args.force,
    )


if __name__ == "__main__":
    main()

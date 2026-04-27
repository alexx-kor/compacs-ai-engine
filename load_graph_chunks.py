#!/usr/bin/env python3
"""Preprocess `instructions` documents and load vector chunks into ClickHouse."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INSTRUCTIONS_DIR = ROOT_DIR / "instructions"

# Ensure config discovers the right source folder before imports.
os.environ.setdefault("DOCS_FOLDER", str(DEFAULT_INSTRUCTIONS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from config import config  # noqa: E402
from core.database import db  # noqa: E402
from core.document_processor import doc_processor  # noqa: E402
from core.embeddings import embedder  # noqa: E402

log = logging.getLogger(__name__)


def configure_logging(is_debug: bool) -> None:
    """Configure application logging."""
    level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_instruction_files(instructions_dir: Path) -> list[Path]:
    """Collect source files from instructions directory."""
    if not instructions_dir.exists():
        raise FileNotFoundError(f"instructions directory not found: {instructions_dir}")
    files = sorted(instructions_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files found in: {instructions_dir}")
    return files


def process_files_to_chunks(files: Sequence[Path]) -> list[dict]:
    """Transform source files into chunk records."""
    chunks: list[dict] = []
    next_id = 0
    for source_file in files:
        source_name = source_file.name
        file_chunks = doc_processor.process_document(str(source_file), source_name, next_id)
        if not file_chunks:
            log.warning("skip file with no chunks: source=%s", source_name)
            continue
        chunks.extend(file_chunks)
        next_id += len(file_chunks)
        log.info("prepared chunks: source=%s count=%s", source_name, len(file_chunks))
    return chunks


def load_chunks_to_clickhouse(chunks: list[dict], force_recreate: bool) -> int:
    """Create embeddings and load chunks into ClickHouse table."""
    if not chunks:
        return 0

    db.init_database(force_recreate=force_recreate)
    batch_size = max(1, int(config.batch_size))
    inserted_total = 0

    for start_idx in range(0, len(chunks), batch_size):
        batch = chunks[start_idx : start_idx + batch_size]
        texts = [item["chunk"] for item in batch]
        started_at = time.perf_counter()
        vectors = embedder.generate(texts)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: vectors={len(vectors)} chunks={len(batch)} start_idx={start_idx}"
            )

        for item, vector in zip(batch, vectors):
            item["embedding"] = vector

        db.insert_batch(batch)
        inserted_total += len(batch)
        log.info(
            "batch loaded: inserted=%s/%s latency_ms=%.2f",
            inserted_total,
            len(chunks),
            elapsed_ms,
        )

    return inserted_total


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess instructions and load chunks into ClickHouse."
    )
    parser.add_argument(
        "--instructions-dir",
        default=str(DEFAULT_INSTRUCTIONS_DIR),
        help="Path to instructions directory with .txt files.",
    )
    parser.add_argument(
        "--force-recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop and recreate ClickHouse table before loading.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run preprocessing pipeline."""
    args = parse_args()
    configure_logging(is_debug=args.debug)

    instructions_dir = Path(args.instructions_dir).resolve()
    log.info("pipeline start: instructions_dir=%s", instructions_dir)
    log.info("clickhouse host=%s secure=%s", config.ch_host, config.ch_secure)
    log.info("embedding model=%s batch_size=%s", config.embed_model, config.batch_size)

    files = collect_instruction_files(instructions_dir)
    log.info("source files discovered=%s", len(files))

    chunks = process_files_to_chunks(files)
    log.info("total chunks prepared=%s", len(chunks))
    if not chunks:
        log.warning("nothing to load; exiting")
        return 0

    inserted = load_chunks_to_clickhouse(chunks, force_recreate=args.force_recreate)
    db_count = db.get_chunk_count()

    log.info("pipeline done: inserted=%s db_count=%s", inserted, db_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

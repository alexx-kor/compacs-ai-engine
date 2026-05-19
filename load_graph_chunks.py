#!/usr/bin/env python3
"""Preprocess instructions with multi-strategy chunks and load into local JSON + BM25."""

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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "vectors"
DEFAULT_GRAPH_DIR = ROOT_DIR / "data" / "graph"


def _apply_bootstrap_env() -> None:
    """Apply storage paths from early CLI flags before config import."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--instructions-dir",
        default=str(DEFAULT_INSTRUCTIONS_DIR),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    bootstrap, _ = parser.parse_known_args()
    instructions_path = str(Path(bootstrap.instructions_dir).resolve())
    output_path = str(Path(bootstrap.output_dir).resolve())
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["EMBEDDING_PROVIDER"] = "ollama"
    os.environ["DOCS_FOLDER"] = instructions_path
    os.environ["INSTRUCTIONS_DIR"] = instructions_path
    os.environ["LOCAL_VECTOR_STORE_DIR"] = output_path


_apply_bootstrap_env()
sys.path.insert(0, str(ROOT_DIR))

from config import config  # noqa: E402
from core.chunk_pipeline import build_all_chunks  # noqa: E402
from core.database import db  # noqa: E402
from core.embedding_alignment import reset_embedder  # noqa: E402
from core.embeddings import embedder  # noqa: E402

# Lock Ollama after .env load (load_graph_chunks indexes with nomic-embed-text / 768-dim).
os.environ["EMBEDDING_PROVIDER"] = "ollama"
os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"
reset_embedder()

log = logging.getLogger(__name__)


def configure_logging(is_debug: bool) -> None:
    level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_instruction_files(instructions_dir: Path) -> list[Path]:
    if not instructions_dir.exists():
        raise FileNotFoundError(f"instructions directory not found: {instructions_dir}")
    files = sorted(instructions_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files found in: {instructions_dir}")
    return files


def load_chunks_to_store(chunks: list[dict], force_recreate: bool) -> int:
    if not chunks:
        return 0

    db.init_database(force_recreate=force_recreate)
    batch_size = max(1, int(config.batch_size))
    inserted_total = 0

    for start_idx in range(0, len(chunks), batch_size):
        batch = chunks[start_idx : start_idx + batch_size]
        texts = [item["chunk"] for item in batch]
        started_at = time.perf_counter()
        vectors = embedder.embed(texts)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: vectors={len(vectors)} chunks={len(batch)} "
                f"start_idx={start_idx}"
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

    db.save_bm25_index(chunks)
    db.reload_store()
    return inserted_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-strategy ingest: sliding, sections, defs, graph, QA + BM25."
    )
    parser.add_argument("--instructions-dir", default=str(DEFAULT_INSTRUCTIONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument(
        "--strategies",
        default=None,
        help="Override CHUNK_STRATEGIES (comma-separated).",
    )
    parser.add_argument(
        "--force-recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(is_debug=args.debug)

    if args.strategies:
        os.environ["CHUNK_STRATEGIES"] = args.strategies

    instructions_dir = Path(args.instructions_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    graph_dir = Path(args.graph_dir).resolve()
    store_path = output_dir / "chunks.json"

    log.info("pipeline start: instructions_dir=%s", instructions_dir)
    log.info("storage backend=%s path=%s", db.backend_name, store_path)
    log.info("strategies=%s", config.chunk_strategies)
    log.info("hybrid=%s bm25_lemma=%s rerank_lemma=%s", config.hybrid_search_enabled, config.bm25_lemmatize, config.rerank_lemmatize)
    log.info("embedding model=%s batch_size=%s", config.embed_model, config.batch_size)

    files = collect_instruction_files(instructions_dir)
    log.info("source files discovered=%s", len(files))

    chunks = build_all_chunks(files, instructions_dir, graph_dir)
    if not chunks:
        log.warning("nothing to load; exiting")
        return 0

    by_type: dict[str, int] = {}
    for chunk in chunks:
        ctype = str(chunk.get("chunk_type", "unknown"))
        by_type[ctype] = by_type.get(ctype, 0) + 1
    log.info("chunks by type: %s", by_type)

    inserted = load_chunks_to_store(chunks, force_recreate=args.force_recreate)
    db_count = db.get_chunk_count()
    log.info("pipeline done: inserted=%s db_count=%s path=%s", inserted, db_count, store_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

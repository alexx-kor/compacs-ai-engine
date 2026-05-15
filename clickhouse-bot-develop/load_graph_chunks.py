"""Load graph chunks from JSONL and insert them into the hypothesis table."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.database import db
from core.embeddings import embedder


log = logging.getLogger(__name__)

DEFAULT_GRAPH_CHUNKS_PATH = str(Path(__file__).resolve().parent / "data" / "graph_chunks.jsonl")

__all__ = [
    "DEFAULT_GRAPH_CHUNKS_PATH",
    "assign_embeddings_to_chunks",
    "load_graph_chunk_rows",
    "load_graph_chunks",
    "main",
    "parse_args",
]


def load_graph_chunks(file_path: str = DEFAULT_GRAPH_CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load graph chunks from JSONL, generate embeddings, and persist them.

    Reads each JSONL row, normalizes fields, requests embeddings in batch order,
    attaches vectors to rows, and inserts the batch into the hypothesis table.

    Args:
        file_path: Path to the source JSONL file.

    Returns:
        List of chunk dictionaries including the ``embedding`` field after generation.

    Raises:
        FileNotFoundError: When ``file_path`` does not exist.
        json.JSONDecodeError: When a line is not valid JSON.
        OSError: When the file cannot be read.
        ValueError: When ``page`` or other normalized fields cannot be coerced.
    """
    log.info("Loading graph chunks from %s", file_path)
    chunks = load_graph_chunk_rows(file_path)
    log.info("Loaded %s chunks from source file", len(chunks))

    chunk_texts = [str(chunk["chunk"]) for chunk in chunks]
    embeddings = embedder.generate(chunk_texts)
    assign_embeddings_to_chunks(chunks, embeddings)
    db.insert_hypothesis_batch(chunks)

    log.info("Loaded %s chunks into database", len(chunks))
    return chunks


def load_graph_chunk_rows(file_path: str) -> list[dict[str, Any]]:
    """Read and normalize graph chunk rows from a JSONL file.

    Each line must be a JSON object. Row index (0-based) becomes ``id``.
    Missing keys use defaults: ``source`` → ``"unknown"``, ``page`` → ``1``,
    ``chunk`` / ``answer`` → empty string.

    Args:
        file_path: Path to the source JSONL file.

    Returns:
        Normalized chunk rows without embeddings.

    Raises:
        FileNotFoundError: When ``file_path`` does not exist.
        json.JSONDecodeError: When a line is not valid JSON.
        OSError: When the file cannot be read.
        ValueError: When ``page`` cannot be converted to ``int``.
    """
    chunks: list[dict[str, Any]] = []
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file_handle:
        for row_index, line in enumerate(file_handle):
            row_data: dict[str, Any] = json.loads(line)
            chunk_text = str(row_data.get("chunk", ""))
            chunks.append(
                {
                    "id": row_index,
                    "source": str(row_data.get("source", "unknown")),
                    "page": int(row_data.get("page", 1)),
                    "chunk": chunk_text,
                    "answer": str(row_data.get("answer", "")),
                    "chunk_hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                    "char_count": len(chunk_text),
                }
            )
    return chunks


def assign_embeddings_to_chunks(
    chunks: list[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Attach generated embeddings to chunk dictionaries in place.

    Args:
        chunks: Prepared chunk dictionaries (mutated).
        embeddings: Embedding vectors in the same order as ``chunks``.

    Note:
        If lengths differ, only the overlapping prefix is updated (same as
        ``zip`` semantics in previous versions).
    """
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the graph chunk loader.

    Returns:
        Parsed namespace with ``file_path`` for the JSONL source.
    """
    parser = argparse.ArgumentParser(description="Load graph chunks into hypothesis table")
    parser.add_argument(
        "--file-path",
        type=str,
        default=DEFAULT_GRAPH_CHUNKS_PATH,
        help="Path to graph chunks JSONL file",
    )
    return parser.parse_args()


def main() -> None:
    """Run the graph chunks loader as a CLI command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    arguments = parse_args()
    load_graph_chunks(arguments.file_path)


if __name__ == "__main__":
    main()

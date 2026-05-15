"""Split plain-text documents into chunk records for indexing."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from config import config


log = logging.getLogger(__name__)


class DocumentProcessor:
    """Load UTF-8 text files and emit chunk dicts aligned with the DB schema."""

    @staticmethod
    def load_document(file_path: str, source_name: str) -> list[tuple[int, str]]:
        """Read a text file and return (page, text) segments.

        Args:
            file_path: Path to a UTF-8 text file.
            source_name: Logical source label (used by callers for logging).

        Returns:
            Non-overlapping page segments; empty when the file cannot be read.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            chunks: list[tuple[int, str]] = []
            if len(content) > config.chunk_size:
                parts: list[str] = []
                current: list[str] = []
                current_len = 0

                for line in content.split("\n"):
                    if current_len + len(line) > config.chunk_size:
                        parts.append("\n".join(current))
                        current = [line]
                        current_len = len(line)
                    else:
                        current.append(line)
                        current_len += len(line)

                if current:
                    parts.append("\n".join(current))

                for index, part in enumerate(parts):
                    if len(part.strip()) > config.min_chunk_size:
                        chunks.append((index + 1, part.strip()))
            else:
                if len(content.strip()) > config.min_chunk_size:
                    chunks.append((1, content.strip()))

            return chunks
        except Exception as exc:
            log.warning("Error loading path=%s error=%s", file_path, exc)
            return []

    @staticmethod
    def split_chunks(text: str) -> list[str]:
        """Split text into overlapping word windows up to ``config.chunk_size``."""
        size = config.chunk_size
        words = text.split()
        out: list[str] = []
        step = size - config.chunk_overlap

        for offset in range(0, len(words), step):
            chunk = " ".join(words[offset : offset + size])
            if len(chunk) > config.min_chunk_size:
                out.append(chunk)
                if len(out) >= config.max_chunks_per_doc:
                    break
        return out

    @staticmethod
    def process_document(file_path: str, source_name: str, start_id: int) -> list[dict[str, Any]]:
        """Produce chunk dicts with ids and hashes for one document path."""
        log.info("Processing source=%s", source_name)
        pages = DocumentProcessor.load_document(file_path, source_name)

        if not pages:
            return []

        chunks: list[dict[str, Any]] = []
        for page_num, text in pages:
            text_chunks = DocumentProcessor.split_chunks(text)

            for chunk in text_chunks:
                if len(chunk) > config.max_text_length:
                    chunk = chunk[: config.max_text_length]
                chunks.append(
                    {
                        "id": start_id + len(chunks),
                        "source": source_name,
                        "page": page_num,
                        "chunk": chunk,
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest(),
                        "char_count": len(chunk),
                    }
                )
                if len(chunks) >= config.max_chunks_per_doc:
                    break
            if len(chunks) >= config.max_chunks_per_doc:
                break

        log.info("Created %s chunks from source=%s", len(chunks), source_name)
        return chunks


doc_processor = DocumentProcessor()

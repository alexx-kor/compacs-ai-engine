"""Multi-strategy chunking for legacy ingest and UI extension pipelines."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Sequence

from config import config
from core.document_processor import doc_processor

log = logging.getLogger(__name__)

DEFAULT_STRATEGIES = ("sliding", "section", "definition")
_SECTION_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+|[A-ZА-ЯЁ][A-ZА-ЯЁ0-9\s\-]{4,}$)")
_DEFINITION_RE = re.compile(r"(?:^|\n)(?:«[^»]+»|[A-Za-zА-Яа-яЁё0-9_.-]+)\s*[—\-–:]\s*.+", re.MULTILINE)


def _relative_source(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _chunk_dict(
    chunk_id: int,
    source: str,
    page: int,
    text: str,
    chunk_type: str,
) -> dict[str, Any]:
    cleaned = text.strip()
    if len(cleaned) <= config.min_chunk_size:
        return {}
    if len(cleaned) > config.max_text_length:
        cleaned = cleaned[: config.max_text_length]
    return {
        "id": chunk_id,
        "source": source,
        "page": page,
        "chunk": cleaned,
        "chunk_hash": hashlib.md5(cleaned.encode()).hexdigest(),
        "char_count": len(cleaned),
        "chunk_type": chunk_type,
    }


def _sliding_chunks(file_path: Path, source: str, start_id: int) -> list[dict[str, Any]]:
    raw = doc_processor.process_document(str(file_path), source, start_id)
    for item in raw:
        item["chunk_type"] = "sliding"
    return raw


def _section_chunks(text: str, source: str, start_id: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        block = "\n".join(current).strip()
        if len(block) > config.min_chunk_size:
            sections.append(block)
        current.clear()

    for line in lines:
        if _SECTION_RE.match(line.strip()) and current:
            flush()
        current.append(line)
    flush()

    if not sections and len(text.strip()) > config.min_chunk_size:
        sections = [text.strip()]

    chunks: list[dict[str, Any]] = []
    for index, block in enumerate(sections):
        if len(block) > config.chunk_size:
            for part in doc_processor.split_chunks(block):
                item = _chunk_dict(start_id + len(chunks), source, index + 1, part, "section")
                if item:
                    chunks.append(item)
        else:
            item = _chunk_dict(start_id + len(chunks), source, index + 1, block, "section")
            if item:
                chunks.append(item)
        if len(chunks) >= config.max_chunks_per_doc:
            break
    return chunks


def _definition_chunks(text: str, source: str, start_id: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for match in _DEFINITION_RE.finditer(text):
        snippet = match.group(0).strip()
        if len(snippet) > config.max_text_length:
            snippet = snippet[: config.max_text_length]
        if len(snippet) <= config.min_chunk_size:
            continue
        item = _chunk_dict(start_id + len(chunks), source, 1, snippet, "definition")
        if item:
            chunks.append(item)
        if len(chunks) >= config.max_chunks_per_doc:
            break
    return chunks


def build_all_chunks(
    files: Sequence[Path],
    instructions_dir: Path,
    graph_dir: Path,
    *,
    strategies: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build chunk dictionaries using one or more chunking strategies."""
    _ = graph_dir  # reserved for graph/RST strategies
    active = tuple(strategies or DEFAULT_STRATEGIES)
    all_chunks: list[dict[str, Any]] = []
    seen_hashes: set[tuple[str, str]] = set()
    next_id = 0

    for file_path in files:
        if not file_path.is_file():
            continue
        source = _relative_source(file_path.resolve(), instructions_dir.resolve())
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as error:
            log.warning("skip unreadable file=%s error=%s", file_path, error)
            continue

        file_chunks: list[dict[str, Any]] = []
        if "sliding" in active:
            file_chunks.extend(_sliding_chunks(file_path, source, next_id))
        if "section" in active:
            file_chunks.extend(_section_chunks(text, source, next_id + len(file_chunks)))
        if "definition" in active:
            file_chunks.extend(_definition_chunks(text, source, next_id + len(file_chunks)))

        for chunk in file_chunks:
            key = (str(chunk.get("source", source)), str(chunk.get("chunk_hash", "")))
            if not chunk.get("chunk_hash") or key in seen_hashes:
                continue
            seen_hashes.add(key)
            chunk["id"] = next_id
            next_id += 1
            all_chunks.append(chunk)

        log.info("chunked file=%s chunks=%s strategies=%s", source, len(file_chunks), active)

    return all_chunks

"""Multi-strategy chunk builder: sliding, sections, definitions, graph, Q&A."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

from config import config
from core.datasets import DatasetScanner
from core.document_processor import doc_processor
from core.knowledge_graph import (
    extract_graph_from_instruction_files,
    save_graph_artifacts,
    triples_to_graph_chunks,
)
from core.text_processing import (
    PreprocessProfile,
    extract_definition_lines,
    format_chunk_markup,
    preprocess_text,
    split_sections,
)

log = logging.getLogger(__name__)


def _make_chunk(
    *,
    chunk_id: int,
    source: str,
    page: int,
    body: str,
    chunk_type: str,
    section: str = "",
    profile: str = "baseline",
    extra: dict | None = None,
) -> dict:
    marked = format_chunk_markup(
        body,
        chunk_type=chunk_type,
        source=source,
        section=section,
        profile=profile,
    )
    payload: dict = {
        "id": chunk_id,
        "source": source,
        "page": page,
        "chunk": marked,
        "chunk_hash": hashlib.md5(marked.encode()).hexdigest(),
        "char_count": len(marked),
        "chunk_type": chunk_type,
        "preprocess_profile": profile,
    }
    if section:
        payload["section"] = section
    if extra:
        payload.update(extra)
    return payload


def build_sliding_chunks(files: Sequence[Path], start_id: int) -> list[dict]:
    """Classic overlapping word windows."""
    chunks: list[dict] = []
    next_id = start_id
    for source_file in files:
        raw = source_file.read_text(encoding="utf-8")
        cleaned = preprocess_text(raw, PreprocessProfile.BASELINE)
        temp_path = source_file
        file_chunks = doc_processor.process_document(str(temp_path), source_file.name, next_id)
        for item in file_chunks:
            item["chunk_type"] = "sliding"
            item["preprocess_profile"] = PreprocessProfile.BASELINE.value
            item["chunk"] = format_chunk_markup(
                preprocess_text(item["chunk"], PreprocessProfile.BASELINE),
                chunk_type="sliding",
                source=source_file.name,
            )
            item["chunk_hash"] = hashlib.md5(item["chunk"].encode()).hexdigest()
        chunks.extend(file_chunks)
        next_id += len(file_chunks)
        log.info("sliding chunks: source=%s count=%s", source_file.name, len(file_chunks))
    return chunks


def build_section_chunks(files: Sequence[Path], start_id: int) -> list[dict]:
    """One chunk per document section (numbered headings, tables)."""
    chunks: list[dict] = []
    chunk_id = start_id
    for source_file in files:
        raw = preprocess_text(source_file.read_text(encoding="utf-8"), PreprocessProfile.NORMALIZED)
        section_count = 0
        for page_num, (title, body) in enumerate(split_sections(raw), start=1):
            if len(body.strip()) < config.min_chunk_size:
                continue
            text = body if len(body) <= config.max_text_length else body[: config.max_text_length]
            chunks.append(
                _make_chunk(
                    chunk_id=chunk_id,
                    source=source_file.name,
                    page=page_num,
                    body=text,
                    chunk_type="section",
                    section=title,
                    profile=PreprocessProfile.MARKDOWN.value,
                )
            )
            chunk_id += 1
            section_count += 1
        log.info("section chunks: source=%s count=%s", source_file.name, section_count)
    return chunks


def build_definition_chunks(files: Sequence[Path], start_id: int) -> list[dict]:
    """Glossary / term-definition mini chunks."""
    chunks: list[dict] = []
    chunk_id = start_id
    for source_file in files:
        raw = preprocess_text(source_file.read_text(encoding="utf-8"), PreprocessProfile.BASELINE)
        for item in extract_definition_lines(raw, source_file.name):
            body = f"Термин: {item['term']}\nОпределение: {item['definition']}"
            chunks.append(
                _make_chunk(
                    chunk_id=chunk_id,
                    source=f"defs/{source_file.name}",
                    page=1,
                    body=body,
                    chunk_type="definition",
                    section=item["term"],
                )
            )
            chunk_id += 1
    log.info("definition chunks total=%s", len(chunks))
    return chunks


def build_lemma_hint_chunks(files: Sequence[Path], start_id: int) -> list[dict]:
    """Chunks with SEARCH_HINTS suffix for better lexical overlap."""
    chunks: list[dict] = []
    chunk_id = start_id
    for source_file in files:
        hinted = preprocess_text(source_file.read_text(encoding="utf-8"), PreprocessProfile.LEMMA_HINT)
        if len(hinted) < config.min_chunk_size:
            continue
        if len(hinted) > config.max_text_length:
            hinted = hinted[: config.max_text_length]
        chunks.append(
            _make_chunk(
                chunk_id=chunk_id,
                source=f"hints/{source_file.name}",
                page=1,
                body=hinted,
                chunk_type="lemma_hint",
                profile=PreprocessProfile.LEMMA_HINT.value,
            )
        )
        chunk_id += 1
    log.info("lemma_hint chunks total=%s", len(chunks))
    return chunks


def build_graph_chunks(files: Sequence[Path], start_id: int, graph_dir: Path) -> tuple[list[dict], int]:
    triples = extract_graph_from_instruction_files(files)
    if triples:
        save_graph_artifacts(triples, graph_dir)
    graph_chunks = triples_to_graph_chunks(triples, start_id=start_id)
    for item in graph_chunks:
        item["preprocess_profile"] = "graph"
    log.info("graph chunks=%s triples=%s", len(graph_chunks), len(triples))
    return graph_chunks, len(triples)


def build_qa_chunks(instructions_dir: Path, start_id: int) -> list[dict]:
    """Build Q&A chunks from instructions/graph/*/questions.txt + answer.txt."""
    pairs = DatasetScanner(instructions_dir).scan_graph_qa_pairs()
    chunks: list[dict] = []
    chunk_id = start_id
    for pair in pairs:
        topic = pair.topic_dir.name
        questions = [
            line.strip()
            for line in pair.questions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        answer = pair.answer_path.read_text(encoding="utf-8").strip()
        for index, question in enumerate(questions, start=1):
            body = f"Вопрос: {question}\nОтвет: {answer}"
            chunks.append(
                _make_chunk(
                    chunk_id=chunk_id,
                    source=f"qa/{topic}",
                    page=index,
                    body=body,
                    chunk_type="qa",
                    section=topic,
                )
            )
            chunk_id += 1
    log.info("qa chunks=%s pairs=%s", len(chunks), len(pairs))
    return chunks


def build_all_chunks(
    files: Sequence[Path],
    instructions_dir: Path,
    graph_dir: Path,
    *,
    strategies: tuple[str, ...] | None = None,
) -> list[dict]:
    """Build chunks for all enabled strategies."""
    enabled = strategies or config.chunk_strategies
    all_chunks: list[dict] = []
    next_id = 0

    if "sliding" in enabled:
        sliding = build_sliding_chunks(files, next_id)
        all_chunks.extend(sliding)
        next_id += len(sliding)

    if "section" in enabled:
        sections = build_section_chunks(files, next_id)
        all_chunks.extend(sections)
        next_id += len(sections)

    if "definition" in enabled:
        defs = build_definition_chunks(files, next_id)
        all_chunks.extend(defs)
        next_id += len(defs)

    if "lemma_hint" in enabled:
        hints = build_lemma_hint_chunks(files, next_id)
        all_chunks.extend(hints)
        next_id += len(hints)

    if "graph" in enabled:
        graph_chunks, _ = build_graph_chunks(files, next_id, graph_dir)
        all_chunks.extend(graph_chunks)
        next_id += len(graph_chunks)

    if "qa" in enabled:
        qa = build_qa_chunks(instructions_dir, next_id)
        all_chunks.extend(qa)
        next_id += len(qa)

    log.info(
        "chunk pipeline done total=%s strategies=%s",
        len(all_chunks),
        ",".join(enabled),
    )
    return all_chunks

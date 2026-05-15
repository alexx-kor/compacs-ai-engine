#!/usr/bin/env python3
"""Build a graph-style Q&A dataset from doc-2.0-sources for local vector indexing."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph dataset from doc-2.0-sources Q&A pairs"
    )
    parser.add_argument("--docs-dir", default="./doc-2.0-sources", help="Path to doc-2.0-sources")
    parser.add_argument("--out-dir", default="./data", help="Output directory for artifacts")
    parser.add_argument("--min-answer-len", default=20, type=int, help="Minimum answer length in chars")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log each processed folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no files written")
    return parser.parse_args()


def read_file_safe(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            return path.read_text(encoding=enc).strip()
        except (UnicodeDecodeError, OSError):
            continue
    log.warning("Could not read file path=%s", path)
    return ""


def topic_label_from_path(rel_path: str) -> str:
    """Convert a folder path to a human-readable topic label (last two segments)."""
    parts = Path(rel_path).parts
    clean = []
    for part in parts:
        cleaned = re.sub(r"^\d+_\d+_?", "", part)
        clean.append(cleaned or part)
    return " / ".join(clean[-2:]) if len(clean) >= 2 else "/".join(clean)


def build_graph_dataset(
    docs_dir: Path,
    out_dir: Path,
    min_answer_len: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    if not docs_dir.exists():
        log.error("docs-dir not found: %s", docs_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    graph_index = {}       # directory → {questions, answer, topic, rel_path}
    chunks = []            # для graph_chunks.jsonl
    dir_tree_lines = []    # для directory_tree.txt

    stats = {
        "dirs_visited": 0,
        "dirs_with_qa": 0,
        "dirs_skipped_no_questions": 0,
        "dirs_skipped_no_answer": 0,
        "dirs_skipped_short_answer": 0,
        "total_questions": 0,
        "total_chunks": 0,
    }

    chunk_id = 0

    # Обходим все директории
    for dirpath in sorted(docs_dir.rglob("*")):
        if not dirpath.is_dir():
            continue

        stats["dirs_visited"] += 1
        rel = dirpath.relative_to(docs_dir)
        rel_str = str(rel)

        q_file = dirpath / "questions.txt"
        a_file = dirpath / "answer.txt"

        # Пропускаем папки без Q&A
        if not q_file.exists() and not a_file.exists():
            continue

        if not q_file.exists():
            stats["dirs_skipped_no_questions"] += 1
            continue

        if not a_file.exists():
            stats["dirs_skipped_no_answer"] += 1
            continue

        questions_raw = read_file_safe(q_file)
        answer = read_file_safe(a_file)

        if len(answer) < min_answer_len:
            stats["dirs_skipped_short_answer"] += 1
            if verbose:
                log.debug("SKIP (short answer) dir=%s", rel_str)
            continue

        # Вопросы — каждая непустая строка
        questions = [
            q.strip() for q in questions_raw.splitlines()
            if q.strip()
        ]

        if not questions:
            stats["dirs_skipped_no_questions"] += 1
            continue

        topic = topic_label_from_path(rel_str)

        # Запись в graph_index
        graph_index[rel_str] = {
            "topic": topic,
            "rel_path": rel_str,
            "questions": questions,
            "answer": answer,
            "answer_chars": len(answer),
            "question_count": len(questions),
        }

        # Строка в directory_tree.txt
        dir_tree_lines.append(rel_str)

        # Чанки для векторного индекса
        # Один чанк на Q&A пару — вопрос как текст, ответ как метаданные
        # Такой формат позволяет искать по вопросу и сразу отдавать ответ
        for q in questions:
            chunk = {
                "id": chunk_id,
                "source": rel_str,          # путь директории как source
                "topic": topic,
                "page": 1,
                "chunk": q,                 # текст для эмбеддинга = вопрос
                "answer": answer,           # ответ хранится рядом
                "chunk_hash": "",           # заполнится при индексировании
                "char_count": len(q),
            }
            chunks.append(chunk)
            chunk_id += 1

        stats["dirs_with_qa"] += 1
        stats["total_questions"] += len(questions)
        stats["total_chunks"] += len(questions)

        if verbose:
            log.debug("OK [%3dq] %s", len(questions), rel_str)

    # --- Запись результатов ---

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    graph_index_path = out_dir / "graph_index.json"
    dir_tree_path = out_dir / "directory_tree.txt"
    chunks_path = out_dir / "graph_chunks.jsonl"
    report_path = out_dir / "build_report.txt"

    if dry_run:
        log.info("[dry-run] No files written. Statistics:")
    else:
        # graph_index.json
        with open(graph_index_path, "w", encoding="utf-8") as f:
            json.dump(graph_index, f, ensure_ascii=False, indent=2)

        # directory_tree.txt — плоский список для промпта
        with open(dir_tree_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(dir_tree_lines)))

        # graph_chunks.jsonl — один JSON на строку
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # build_report.txt
        report_lines = [
            "build_graph_dataset report",
            f"timestamp: {ts}",
            f"docs_dir:  {docs_dir}",
            f"out_dir:   {out_dir}",
            "",
            f"dirs visited:              {stats['dirs_visited']}",
            f"dirs with Q&A:             {stats['dirs_with_qa']}",
            f"skipped (no questions):    {stats['dirs_skipped_no_questions']}",
            f"skipped (no answer):       {stats['dirs_skipped_no_answer']}",
            f"skipped (short answer):    {stats['dirs_skipped_short_answer']}",
            f"total questions indexed:   {stats['total_questions']}",
            f"total chunks written:      {stats['total_chunks']}",
            "",
            "output files:",
            f"  {graph_index_path}",
            f"  {dir_tree_path}",
            f"  {chunks_path}",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    log.info("%s", "=" * 55)
    log.info("  docs_dir   : %s", docs_dir)
    log.info("  out_dir    : %s", out_dir)
    log.info("%s", "=" * 55)
    log.info("  dirs with Q&A          : %s", stats["dirs_with_qa"])
    log.info("  skipped (no questions) : %s", stats["dirs_skipped_no_questions"])
    log.info("  skipped (no answer)    : %s", stats["dirs_skipped_no_answer"])
    log.info("  skipped (short answer) : %s", stats["dirs_skipped_short_answer"])
    log.info("  total questions        : %s", stats["total_questions"])
    log.info("  total chunks           : %s", stats["total_chunks"])
    log.info("%s", "=" * 55)

    if not dry_run:
        log.info("  graph_index.json    -> %s", graph_index_path)
        log.info("  directory_tree.txt  -> %s", dir_tree_path)
        log.info("  graph_chunks.jsonl  -> %s", chunks_path)
        log.info("  build_report.txt    -> %s", report_path)
        log.info("Next step: run indexing with graph_chunks.jsonl instead of doc-2.0-sources")


def main() -> None:
    args = parse_args()

    import logging as _logging
    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    docs_dir = Path(args.docs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    log.info("build_graph_dataset.py")
    log.info("docs_dir : %s", docs_dir)
    log.info("out_dir  : %s", out_dir)
    log.info("dry_run  : %s", args.dry_run)

    build_graph_dataset(
        docs_dir=docs_dir,
        out_dir=out_dir,
        min_answer_len=args.min_answer_len,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

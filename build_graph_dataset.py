#!/usr/bin/env python3
"""
build_graph_dataset.py

Превращает doc-2.0-sources (questions.txt + answer.txt по папкам)
в размеченный датасет для ClickHouse с графовой схемой.

Что делает:
1. Обходит все листовые папки doc-2.0-sources
2. Читает questions.txt + answer.txt из каждой
3. Строит graph_index.json  — маппинг директория → вопросы → ответ
4. Строит directory_tree.txt — плоский список для промпта define-api-directory
5. Строит graph_chunks.jsonl — готовый датасет для заливки в ClickHouse
   (каждый чанк = один Q&A блок с метаданными топика)

Запуск ПЕРЕД основным индексированием:
  python3 build_graph_dataset.py
  python3 build_graph_dataset.py --docs-dir ./doc-2.0-sources --out-dir ./data

После запуска:
  data/graph_index.json     — индекс для graph_router
  data/directory_tree.txt   — дерево директорий для промпта
  data/graph_chunks.jsonl   — чанки для ClickHouse (заменяет стандартный индекс)
  data/build_report.txt     — отчёт о сборке
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Build graph dataset from doc-2.0-sources Q&A pairs"
    )
    p.add_argument(
        "--docs-dir", default="./doc-2.0-sources",
        help="Путь к doc-2.0-sources (default: ./doc-2.0-sources)"
    )
    p.add_argument(
        "--out-dir", default="./data",
        help="Куда сохранять результаты (default: ./data)"
    )
    p.add_argument(
        "--min-answer-len", default=20, type=int,
        help="Минимальная длина ответа в символах (default: 20)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Печатать каждую обработанную папку"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Только показать что будет сделано, без записи файлов"
    )
    return p.parse_args()


def read_file_safe(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            return path.read_text(encoding=enc).strip()
        except (UnicodeDecodeError, Exception):
            continue
    return ""


def topic_label_from_path(rel_path: str) -> str:
    """
    Превращает путь папки в читаемый топик-лейбл.
    integration/1_api_use_cases/1_10_payout_by_reference/1_10_2_payout_by_ref_flow
    → payout_by_reference / payout_by_ref_flow
    """
    parts = Path(rel_path).parts
    # убираем числовые префиксы вида 1_10_ из имён папок
    clean = []
    for p in parts:
        # удаляем ведущие числовые сегменты типа "1_10_"
        import re
        cleaned = re.sub(r'^\d+_\d+_?', '', p)
        if not cleaned:
            cleaned = p
        clean.append(cleaned)
    return " / ".join(clean[-2:]) if len(clean) >= 2 else "/".join(clean)


def build_graph_dataset(docs_dir: Path, out_dir: Path, min_answer_len: int,
                        verbose: bool, dry_run: bool):
    if not docs_dir.exists():
        print(f"ERROR: docs-dir не найден: {docs_dir}")
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
                print(f"  SKIP (short answer) {rel_str}")
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

        # Чанки для ClickHouse
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
            print(f"  OK [{len(questions):3d}q] {rel_str}")

    # --- Запись результатов ---

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    graph_index_path = out_dir / "graph_index.json"
    dir_tree_path = out_dir / "directory_tree.txt"
    chunks_path = out_dir / "graph_chunks.jsonl"
    report_path = out_dir / "build_report.txt"

    if dry_run:
        print("[dry-run] Файлы не записаны. Статистика:")
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
            f"build_graph_dataset report",
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
            f"output files:",
            f"  {graph_index_path}",
            f"  {dir_tree_path}",
            f"  {chunks_path}",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    # Печатаем итог
    print()
    print("=" * 55)
    print(f"  docs_dir   : {docs_dir}")
    print(f"  out_dir    : {out_dir}")
    print("=" * 55)
    print(f"  dirs with Q&A          : {stats['dirs_with_qa']}")
    print(f"  skipped (no questions) : {stats['dirs_skipped_no_questions']}")
    print(f"  skipped (no answer)    : {stats['dirs_skipped_no_answer']}")
    print(f"  skipped (short answer) : {stats['dirs_skipped_short_answer']}")
    print(f"  total questions        : {stats['total_questions']}")
    print(f"  total chunks           : {stats['total_chunks']}")
    print("=" * 55)

    if not dry_run:
        print(f"  graph_index.json    -> {graph_index_path}")
        print(f"  directory_tree.txt  -> {dir_tree_path}")
        print(f"  graph_chunks.jsonl  -> {chunks_path}")
        print(f"  build_report.txt    -> {report_path}")
        print()
        print("Следующий шаг:")
        print("  Запусти индексирование через graph_chunks.jsonl")
        print("  вместо стандартного обхода doc-2.0-sources.")


def main():
    args = parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"build_graph_dataset.py")
    print(f"docs_dir : {docs_dir}")
    print(f"out_dir  : {out_dir}")
    print(f"dry_run  : {args.dry_run}")
    print()

    build_graph_dataset(
        docs_dir=docs_dir,
        out_dir=out_dir,
        min_answer_len=args.min_answer_len,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

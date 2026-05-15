#!/usr/bin/env python3
"""CLI to add Q&A pairs into few-shot training CSV under config.few_shot_folder."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from core.logger import setup_logging

log = logging.getLogger(__name__)


def add_to_few_shot(question: str, answer: str, source: str = "training") -> bool:
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"
    examples_file.parent.mkdir(parents=True, exist_ok=True)

    if examples_file.exists():
        df = pd.read_csv(examples_file)
    else:
        df = pd.DataFrame(columns=['question', 'answer', 'source'])

    if not df[df['question'] == question].empty:
        log.info("Example already exists question_prefix=%s", question[:50])
        return False

    new_row = pd.DataFrame([{
        'question': question,
        'answer': answer,
        'source': source
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(examples_file, index=False, encoding='utf-8')
    log.info("Added to few-shot question_prefix=%s", question[:50])
    return True


def train_from_evaluation_results(results_file: str, min_score: float = 0.7) -> int:
    if not os.path.exists(results_file):
        log.error("File not found path=%s", results_file)
        return 0

    df = pd.read_csv(results_file)
    good_answers = df[df['similarity_score'] >= min_score]

    log.info(
        "Found good answers count=%s min_score=%s",
        len(good_answers),
        min_score,
    )

    added = 0
    for position, (_, row) in enumerate(good_answers.iterrows(), start=1):
        log.debug(
            "Processing evaluation row index=%s total=%s",
            position,
            len(good_answers),
        )
        if add_to_few_shot(row['question'], row['generated_answer'], source="evaluation"):
            added += 1

    log.info("Added new few-shot examples count=%s", added)
    return added


def train_from_txt(questions_txt: str, answers_txt: str, source: str = "txt") -> int:
    if not os.path.exists(questions_txt) or not os.path.exists(answers_txt):
        log.error(
            "Questions or answers file missing questions=%s answers=%s",
            questions_txt,
            answers_txt,
        )
        return 0

    with open(questions_txt, 'r', encoding='utf-8') as qhandle:
        questions = [line.strip() for line in qhandle if line.strip()]

    with open(answers_txt, 'r', encoding='utf-8') as ahandle:
        answers = [line.strip() for line in ahandle if line.strip()]

    min_len = min(len(questions), len(answers))

    added = 0
    for i in range(min_len):
        log.debug("TXT pair index=%s total=%s", i + 1, min_len)
        if add_to_few_shot(questions[i], answers[i], source=source):
            added += 1

    log.info("Added examples from TXT count=%s", added)
    return added


def list_examples() -> None:
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"

    if not examples_file.exists():
        log.warning("No examples file at path=%s", examples_file)
        return

    df = pd.read_csv(examples_file)
    log.info("Few-shot examples total=%s", len(df))
    log.info("%s", "=" * 60)
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        log.info("%s. Q: %s...", position, row['question'][:80])
        log.info("   A: %s...", row['answer'][:80])


def clear_examples() -> None:
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"
    if examples_file.exists():
        examples_file.unlink()
        log.info("All examples cleared path=%s", examples_file)
    else:
        log.warning("No examples file to clear path=%s", examples_file)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train RAG on answers')
    parser.add_argument('--file', '-f', type=str, help='CSV file with evaluation results')
    parser.add_argument('--questions-txt', type=str, help='TXT file with questions')
    parser.add_argument('--answers-txt', type=str, help='TXT file with answers')
    parser.add_argument('--question', '-q', type=str, help='Single question to add')
    parser.add_argument('--answer', '-a', type=str, help='Answer for the question')
    parser.add_argument('--source', '-s', type=str, default='manual', help='Source of the example')
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum score to include')
    parser.add_argument('--list', '-l', action='store_true', help='List all examples')
    parser.add_argument('--clear', action='store_true', help='Clear all examples')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable DEBUG logging',
    )

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    if args.list:
        list_examples()
    elif args.clear:
        clear_examples()
    elif args.file:
        train_from_evaluation_results(args.file, args.min_score)
    elif args.questions_txt and args.answers_txt:
        train_from_txt(args.questions_txt, args.answers_txt, args.source)
    elif args.question and args.answer:
        add_to_few_shot(args.question, args.answer, args.source)
    else:
        log.info(
            "%s",
            """

                            TRAIN ON ANSWERS - FEW-SHOT LEARNING


        Usage:
            # Train from evaluation results
            python train_on_answers.py --file data/results/evaluation_results.csv

            # Train from TXT files
            python train_on_answers.py --questions-txt questions.txt --answers-txt answers.txt

            # Add single example
            python train_on_answers.py --question "What is X?" --answer "X is Y"

            # List all examples
            python train_on_answers.py --list

            # Clear all examples
            python train_on_answers.py --clear
        """,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from evaluator.qa_loader import QALoader


def test_load_questions_txt(tmp_path: Path) -> None:
    path = tmp_path / "questions.txt"
    path.write_text("First?\n\nSecond?\n", encoding="utf-8")

    questions = QALoader.load_questions(str(path))

    assert questions == ["First?", "Second?"]


def test_load_questions_json(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps([{"question": "Q1"}, {"question": "Q2"}]),
        encoding="utf-8",
    )

    questions = QALoader.load_questions(str(path))

    assert questions == ["Q1", "Q2"]


def test_load_qa_pairs_from_txt(tmp_path: Path) -> None:
    questions = tmp_path / "q.txt"
    answers = tmp_path / "a.txt"
    questions.write_text("Question one\nQuestion two\n", encoding="utf-8")
    answers.write_text("Answer one\nAnswer two\n", encoding="utf-8")

    pairs = QALoader.load_qa_pairs(str(questions), str(answers))

    assert pairs == [("Question one", "Answer one"), ("Question two", "Answer two")]


def test_load_qa_pairs_empty_returns_empty(tmp_path: Path) -> None:
    questions = tmp_path / "q.txt"
    answers = tmp_path / "a.txt"
    questions.write_text("", encoding="utf-8")
    answers.write_text("only answer\n", encoding="utf-8")

    pairs = QALoader.load_qa_pairs(str(questions), str(answers))

    assert pairs == []

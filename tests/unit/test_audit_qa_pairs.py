from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import load_script_module

audit = load_script_module("audit_qa_pairs")


def test_overlap_faithfulness_full_overlap() -> None:
    answer = "усредненный счет импульсов автоматического экспонирования"
    chunk = "Усредненный счет импульсов получаемых от автоматического экспонирования"
    assert audit.overlap_faithfulness(answer, chunk) == 100.0


def test_overlap_faithfulness_empty_answer() -> None:
    assert audit.overlap_faithfulness("", "some chunk text here") == 0.0


def test_audit_pairs_detects_duplicate_and_noise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Как открыть экран тренда?",
                    "expected_answer": "Открыть экран тренда кнопкой тренд панели выбора экранов",
                    "chunk_id": "10",
                },
                {
                    "id": 2,
                    "question": "Как открыть экран тренда?",
                    "expected_answer": "Дублирующий ответ про экран тренда кнопкой панели",
                    "chunk_id": "10",
                },
                {
                    "id": 3,
                    "question": "арбуз что такое модуль?",
                    "expected_answer": "Короткий ответ",
                    "chunk_id": "11",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        audit,
        "load_chunk_catalog",
        lambda _vector_dir: {
            "10": "экран тренда кнопкой тренд панели выбора экранов",
            "11": "модуль работает в режиме slave",
        },
    )

    report = audit.audit_pairs(qa_path, tmp_path / "vectors", judge_sample=0)

    assert report["summary"]["pairs"] == 3
    assert report["summary"]["duplicates"] == 1
    assert report["summary"]["noise_questions"] == 1
    assert report["rows"][0]["duplicate_question"] is False
    assert report["rows"][1]["duplicate_question"] is True
    assert report["rows"][2]["noise_in_question"] is True
    assert report["summary"]["recommended_manual_review_pct"] in (12, 25, 50)

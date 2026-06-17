from __future__ import annotations

import json
from pathlib import Path

from export_hybrid_comparison_xlsx import build_metrics_rows, records_to_map
from tests.helpers import load_script_module

export_report = load_script_module("export_ui_extension_report")


def _sample_golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "question": "Q1?", "expected_answer": "A1"},
                {"id": 2, "question": "Q2?", "expected_answer": "A2"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_build_ui_comparison_rows_delta(tmp_path: Path) -> None:
    golden = _sample_golden(tmp_path)
    baseline = [
        {"id": 1, "answer": "base1", "llm_judge": {"total": 70}, "sources": [["doc.txt", 1]]},
        {"id": 2, "answer": "base2", "llm_judge": {"total": 60}},
    ]
    finetuned = [
        {"id": 1, "answer": "ft1", "llm_judge": {"total": 85}},
        {"id": 2, "answer": "ft2", "llm_judge": {"total": 55}},
    ]

    rows = export_report.build_ui_comparison_rows(golden, baseline, finetuned)
    assert len(rows) == 2
    assert rows[0]["Δ Judge (FT - base)"] == 15.0
    assert rows[1]["Δ Judge (FT - base)"] == -5.0
    assert "doc.txt" in rows[0]["Источники baseline"]


def test_build_qa_audit_rows_flags_manual_review() -> None:
    qa_audit = {
        "rows": [
            {"id": 1, "question": "ok?", "heuristic_quality_pct": 90, "duplicate_question": False},
            {
                "id": 2,
                "question": "bad?",
                "heuristic_quality_pct": 40,
                "duplicate_question": False,
                "gpt_audit": {"total": 30, "verdict": "reject"},
            },
        ]
    }
    rows = export_report.build_qa_audit_rows(qa_audit)
    assert rows[0]["Нужна ручная проверка"] == "нет"
    assert rows[1]["Нужна ручная проверка"] == "да"


def test_build_metrics_rows_from_eval_records(tmp_path: Path) -> None:
    golden = _sample_golden(tmp_path)
    baseline = [
        {
            "id": 1,
            "llm_judge": {"total": 80},
            "rag_metrics": {"faithfulness": 0.9, "hit_at_3": 1, "latency_seconds": 1.2},
        }
    ]
    finetuned = [
        {
            "id": 1,
            "llm_judge": {"total": 90},
            "rag_metrics": {"faithfulness": 0.95, "hit_at_3": 1, "latency_seconds": 1.0},
        }
    ]
    rows = build_metrics_rows(golden, baseline, finetuned)
    assert rows[0]["Judge % Ollama"] == 80
    assert rows[0]["Judge % GPT"] == 90
    assert rows[0]["Faithfulness Ollama"] == 0.9


def test_records_to_map_skips_missing_id() -> None:
    mapped = records_to_map([{"id": 5, "answer": "x"}, {"answer": "no id"}])
    assert mapped == {5: {"id": 5, "answer": "x"}}

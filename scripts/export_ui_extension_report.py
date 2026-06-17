#!/usr/bin/env python3
"""Export UI extension fine-tune report (baseline Excel template + QA audit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_hybrid_comparison_xlsx import (  # noqa: E402
    METADATA_COLUMNS,
    METRICS_COLUMNS,
    _autosize_columns,
    _display_score,
    _format_sources,
    _judge_total,
    _rag_metric,
    build_metrics_rows,
    load_eval_json,
    load_eval_payload,
    records_to_map,
)

COMPARISON_COLUMNS_UI = [
    "№",
    "Вопрос",
    "Эталон (синт. QA)",
    "Ollama llama3.2:3b (RAG)",
    "Оценка baseline",
    "Ollama compacs-ui-ft (RAG+FT)",
    "Оценка после FT",
    "Δ Judge (FT - base)",
    "Источники baseline",
    "Источники FT",
]

QA_AUDIT_COLUMNS = [
    "№",
    "Вопрос",
    "Heuristic quality %",
    "Faithfulness overlap %",
    "Дубликат",
    "Шум в вопросе",
    "GPT audit total",
    "GPT verdict",
    "Нужна ручная проверка",
]


def build_ui_comparison_rows(
    golden_path: Path,
    baseline_records: list[dict[str, Any]],
    finetuned_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    golden_list = json.loads(golden_path.read_text(encoding="utf-8"))
    base_map = records_to_map(baseline_records)
    ft_map = records_to_map(finetuned_records)
    rows: list[dict[str, Any]] = []
    for item in golden_list:
        qid = int(item["id"])
        base = base_map.get(qid, {})
        ft = ft_map.get(qid, {})
        j_base = _judge_total(base)
        j_ft = _judge_total(ft)
        delta = ""
        if j_base not in ("", None) and j_ft not in ("", None):
            try:
                delta = round(float(j_ft) - float(j_base), 1)
            except (TypeError, ValueError):
                delta = ""
        rows.append(
            {
                "№": qid,
                "Вопрос": item.get("question", ""),
                "Эталон (синт. QA)": item.get("expected_answer", ""),
                "Ollama llama3.2:3b (RAG)": base.get("answer", ""),
                "Оценка baseline": _display_score(base),
                "Ollama compacs-ui-ft (RAG+FT)": ft.get("answer", ""),
                "Оценка после FT": _display_score(ft),
                "Δ Judge (FT - base)": delta,
                "Источники baseline": _format_sources(base.get("sources")),
                "Источники FT": _format_sources(ft.get("sources")),
            }
        )
    return rows


def build_ui_metadata(
    golden_path: Path,
    baseline_payload: dict[str, Any],
    finetuned_payload: dict[str, Any],
    qa_audit: dict[str, Any],
    *,
    ui_chunks: int,
) -> list[dict[str, Any]]:
    def row(title: str, payload: dict[str, Any], note: str) -> dict[str, Any]:
        cfg = payload.get("config", {})
        stats = payload.get("statistics", {})
        return {
            "Прогон": title,
            "Файл": cfg.get("golden_path", ""),
            "Модель": cfg.get("model", "—"),
            "LLM provider": cfg.get("llm_provider", "—"),
            "Чанков": cfg.get("chunks", ui_chunks),
            "Вопросов": cfg.get("questions", "—"),
            "Средний балл (локальный)": stats.get("avg_score", "—"),
            "Средний балл LLM-судья (%)": stats.get("llm_judge_avg_percent", "—"),
            "Гибридный поиск": "да",
            "Примечание": note,
        }

    qa_sum = qa_audit.get("summary", {})
    rows = [
        {
            "Прогон": "UI extension QA (синт.)",
            "Файл": str(golden_path.name),
            "Модель": "gpt-4o-mini",
            "LLM provider": "openai",
            "Чанков": ui_chunks,
            "Вопросов": qa_sum.get("pairs", 150),
            "Средний балл (локальный)": qa_sum.get("avg_heuristic_quality_pct", "—"),
            "Средний балл LLM-судья (%)": qa_sum.get("avg_gpt_audit_pct", "—"),
            "Гибридный поиск": "—",
            "Примечание": qa_sum.get("manual_labeling_verdict", ""),
        },
        row(
            "Ollama baseline (llama3.2:3b)",
            baseline_payload,
            "RAG hybrid + query filter, до FT",
        ),
        row(
            "Ollama fine-tuned (compacs-ui-ft)",
            finetuned_payload,
            "Modelfile + ollama create compacs-ui-ft",
        ),
    ]
    return rows


def build_qa_audit_rows(qa_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in qa_audit.get("rows", []):
        gpt = item.get("gpt_audit") or {}
        verdict = gpt.get("verdict", "") if isinstance(gpt, dict) else ""
        gpt_total = gpt.get("total", "") if isinstance(gpt, dict) else ""
        need_manual = (
            item.get("heuristic_quality_pct", 100) < 55
            or item.get("duplicate_question")
            or verdict in ("review", "reject")
        )
        rows.append(
            {
                "№": item.get("id"),
                "Вопрос": item.get("question", ""),
                "Heuristic quality %": item.get("heuristic_quality_pct"),
                "Faithfulness overlap %": item.get("faithfulness_overlap_pct"),
                "Дубликат": "да" if item.get("duplicate_question") else "нет",
                "Шум в вопросе": "да" if item.get("noise_in_question") else "нет",
                "GPT audit total": gpt_total,
                "GPT verdict": verdict,
                "Нужна ручная проверка": "да" if need_manual else "нет",
            }
        )
    return rows


def build_conclusions_sheet(
    baseline_payload: dict[str, Any],
    finetuned_payload: dict[str, Any],
    qa_audit: dict[str, Any],
) -> pd.DataFrame:
    b_stats = baseline_payload.get("statistics", {})
    f_stats = finetuned_payload.get("statistics", {})
    qa_sum = qa_audit.get("summary", {})
    lines = [
        ("Метрика", "Значение"),
        ("Вопросов в оценке", baseline_payload.get("config", {}).get("questions", "")),
        ("Judge baseline (%)", b_stats.get("llm_judge_avg_percent", "—")),
        ("Judge после FT (%)", f_stats.get("llm_judge_avg_percent", "—")),
        (
            "Δ Judge (п.п.)",
            round(
                float(f_stats.get("llm_judge_avg_percent", 0) or 0)
                - float(b_stats.get("llm_judge_avg_percent", 0) or 0),
                1,
            )
            if f_stats.get("llm_judge_avg_percent") and b_stats.get("llm_judge_avg_percent")
            else "—",
        ),
        ("Локальный балл baseline", b_stats.get("avg_score", "—")),
        ("Локальный балл FT", f_stats.get("avg_score", "—")),
        ("Средн. faithfulness baseline", b_stats.get("avg_faithfulness", "—")),
        ("Средн. faithfulness FT", f_stats.get("avg_faithfulness", "—")),
        ("", ""),
        ("Качество QA (heuristic avg %)", qa_sum.get("avg_heuristic_quality_pct", "—")),
        ("Качество QA (GPT sample avg %)", qa_sum.get("avg_gpt_audit_pct", "—")),
        ("Пар heuristic ≥75", qa_sum.get("high_quality_heuristic_ge75", "—")),
        ("Пар heuristic <55", qa_sum.get("low_quality_heuristic_lt55", "—")),
        ("Дубликаты вопросов", qa_sum.get("duplicates", "—")),
        ("Рекомендуемая ручная проверка (%)", qa_sum.get("recommended_manual_review_pct", "—")),
        ("", ""),
        ("Вывод по ручной разметке", qa_sum.get("manual_labeling_verdict", "")),
    ]
    return pd.DataFrame(lines[1:], columns=lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(ROOT / "instructions" / "golden" / "ui_extension_qa_150.json"))
    parser.add_argument("--baseline-json", default=str(ROOT / "evaluation_ui_extension_baseline.json"))
    parser.add_argument("--finetuned-json", default=str(ROOT / "evaluation_ui_extension_finetuned.json"))
    parser.add_argument("--qa-audit", default=str(ROOT / "data" / "finetune" / "qa_audit_report.json"))
    parser.add_argument("-o", "--output", default=str(ROOT / "comparison_ui_extension_finetune.xlsx"))
    parser.add_argument("--ui-chunks", type=int, default=0)
    args = parser.parse_args()

    golden_path = Path(args.golden)
    baseline_path = Path(args.baseline_json)
    finetuned_path = Path(args.finetuned_json)
    qa_audit_path = Path(args.qa_audit)

    for path in (golden_path, baseline_path, finetuned_path, qa_audit_path):
        if not path.exists():
            print(f"Missing required file: {path}", file=sys.stderr)
            return 1

    baseline_payload = load_eval_payload(baseline_path)
    finetuned_payload = load_eval_payload(finetuned_path)
    qa_audit = json.loads(qa_audit_path.read_text(encoding="utf-8"))

    baseline_records = load_eval_json(baseline_path)
    finetuned_records = load_eval_json(finetuned_path)

    comparison_df = pd.DataFrame(
        build_ui_comparison_rows(golden_path, baseline_records, finetuned_records),
        columns=COMPARISON_COLUMNS_UI,
    )
    metrics_rows = build_metrics_rows(golden_path, baseline_records, finetuned_records)
    for row in metrics_rows:
        row["Judge % baseline"] = row.pop("Judge % Ollama", "")
        row["Judge % FT"] = row.pop("Judge % GPT", "")
    metrics_cols = [
        "№", "Вопрос", "Judge % baseline", "Judge % FT",
        "Faithfulness Ollama", "Faithfulness GPT", "Hit@3 Ollama", "Hit@3 GPT",
        "Latency Ollama (с)", "Latency GPT (с)", "Retrieval Ollama (с)", "Retrieval GPT (с)",
        "LLM Ollama (с)", "LLM GPT (с)", "Prompt tokens Ollama", "Prompt tokens GPT",
        "Completion tokens Ollama", "Completion tokens GPT", "Total tokens Ollama",
        "Total tokens GPT", "Cost USD Ollama", "Cost USD GPT",
    ]
    metrics_df = pd.DataFrame(metrics_rows, columns=metrics_cols)
    metadata_df = pd.DataFrame(
        build_ui_metadata(
            golden_path,
            baseline_payload,
            finetuned_payload,
            qa_audit,
            ui_chunks=args.ui_chunks,
        ),
        columns=METADATA_COLUMNS,
    )
    qa_df = pd.DataFrame(build_qa_audit_rows(qa_audit), columns=QA_AUDIT_COLUMNS)
    conclusions_df = build_conclusions_sheet(baseline_payload, finetuned_payload, qa_audit)

    output_path = Path(args.output).resolve()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, index=False, sheet_name="Сравнение")
        metrics_df.to_excel(writer, index=False, sheet_name="Метрики")
        metadata_df.to_excel(writer, index=False, sheet_name="Метаданные")
        qa_df.to_excel(writer, index=False, sheet_name="Качество QA")
        conclusions_df.to_excel(writer, index=False, sheet_name="Выводы")
        _autosize_columns(
            writer,
            "Сравнение",
            {
                "№": 6,
                "Вопрос": 42,
                "Эталон (синт. QA)": 48,
                "Ollama llama3.2:3b (RAG)": 42,
                "Оценка baseline": 12,
                "Ollama compacs-ui-ft (RAG+FT)": 42,
                "Оценка после FT": 12,
                "Δ Judge (FT - base)": 10,
                "Источники baseline": 28,
                "Источники FT": 28,
            },
        )

    print(f"Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

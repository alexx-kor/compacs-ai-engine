#!/usr/bin/env python3
"""Export comparison workbook in baseline_comparison.xlsx format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT_DIR / "instructions" / "golden" / "golden_set.json"
DEFAULT_VECTOR_STORE = ROOT_DIR / "data" / "vectors"
DEFAULT_BASELINE_JSON = ROOT_DIR / "baseline" / "rag_answers_gpu.json"
DEFAULT_BASELINE_XLSX = ROOT_DIR / "baseline_comparison.xlsx"

COMPARISON_COLUMNS = [
    "№",
    "Вопрос",
    "Золотой сет",
    "Ollama без гибридного поиска",
    "Оценка Ollama (без гибрида)",
    "Ollama с гибридным поиском",
    "Оценка Ollama (гибрид)",
    "GPT с гибридным поиском",
    "Оценка GPT (гибрид)",
    "Источники Ollama (без гибрида)",
    "Источники Ollama (гибрид)",
    "Источники GPT (гибрид)",
]

METADATA_COLUMNS = [
    "Прогон",
    "Файл",
    "Модель",
    "LLM provider",
    "Чанков",
    "Вопросов",
    "Средний балл (локальный)",
    "Средний балл LLM-судья (%)",
    "Гибридный поиск",
    "Примечание",
]

METRICS_COLUMNS = [
    "№",
    "Вопрос",
    "Judge % Ollama",
    "Judge % GPT",
    "Faithfulness Ollama",
    "Faithfulness GPT",
    "Hit@3 Ollama",
    "Hit@3 GPT",
    "Latency Ollama (с)",
    "Latency GPT (с)",
    "Retrieval Ollama (с)",
    "Retrieval GPT (с)",
    "LLM Ollama (с)",
    "LLM GPT (с)",
    "Prompt tokens Ollama",
    "Prompt tokens GPT",
    "Completion tokens Ollama",
    "Completion tokens GPT",
    "Total tokens Ollama",
    "Total tokens GPT",
    "Cost USD Ollama",
    "Cost USD GPT",
]


def _apply_bootstrap(vector_store_dir: Path) -> None:
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(vector_store_dir.resolve())
    os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")


def load_eval_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"results": payload, "statistics": {}, "config": {}}
    raise ValueError(f"unsupported JSON format: {path}")


def load_eval_json(path: Path) -> list[dict[str, Any]]:
    payload = load_eval_payload(path)
    records = payload.get("results")
    if isinstance(records, list):
        return records
    raise ValueError(f"unsupported JSON format: {path}")


def _format_sources(sources: Any) -> str:
    if not sources:
        return ""
    parts: list[str] = []
    for item in sources:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            parts.append(f"{item[0]} (стр. {item[1]})")
        else:
            parts.append(str(item))
    return "; ".join(parts)


def _judge_total(row: dict[str, Any]) -> Any:
    judge = row.get("llm_judge")
    if not isinstance(judge, dict) or "error" in judge:
        return ""
    return judge.get("total", "")


def _display_score(row: dict[str, Any]) -> Any:
    judge_total = _judge_total(row)
    if judge_total not in ("", None):
        return judge_total
    return row.get("final_score", "")


def _rag_metric(row: dict[str, Any], key: str, fallback: Any = "") -> Any:
    metrics = row.get("rag_metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    return fallback


def records_to_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in records if item.get("id") is not None}


def load_baseline_answers(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return {}
    return {int(item["id"]): item for item in payload if item.get("id") is not None}


def load_baseline_sheet(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    dataframe = pd.read_excel(path, sheet_name="Сравнение")
    rows: dict[int, dict[str, Any]] = {}
    for _, row in dataframe.iterrows():
        try:
            qid = int(row["№"])
        except Exception:
            continue
        rows[qid] = row.to_dict()
    return rows


def build_rows(
    golden_path: Path,
    baseline_xlsx_rows: dict[int, dict[str, Any]],
    baseline_json_rows: dict[int, dict[str, Any]],
    ollama_records: list[dict[str, Any]],
    gpt_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    with golden_path.open("r", encoding="utf-8") as handle:
        golden_list = json.load(handle)

    ollama_by_id = records_to_map(ollama_records)
    gpt_by_id = records_to_map(gpt_records)

    rows: list[dict[str, Any]] = []
    for item in golden_list:
        qid = int(item["id"])
        question = item.get("question", "")
        expected = item.get("expected_answer", "")
        baseline_sheet = baseline_xlsx_rows.get(qid, {})
        baseline_json = baseline_json_rows.get(qid, {})
        ollama = ollama_by_id.get(qid, {})
        gpt = gpt_by_id.get(qid, {})

        row = {
            "№": qid,
            "Вопрос": question,
            "Золотой сет": expected,
            "Ollama без гибридного поиска": baseline_sheet.get(
                "Ollama без гибридного поиска",
                baseline_json.get("answer", ""),
            ),
            "Оценка Ollama (без гибрида)": baseline_sheet.get("Оценка Ollama (без гибрида)", ""),
            "Ollama с гибридным поиском": ollama.get("answer", ""),
            "Оценка Ollama (гибрид)": _display_score(ollama),
            "GPT с гибридным поиском": gpt.get("answer", ""),
            "Оценка GPT (гибрид)": _display_score(gpt),
            "Источники Ollama (без гибрида)": baseline_sheet.get(
                "Источники Ollama (без гибрида)",
                _format_sources(baseline_json.get("sources")),
            ),
            "Источники Ollama (гибрид)": _format_sources(ollama.get("sources")),
            "Источники GPT (гибрид)": _format_sources(gpt.get("sources")),
        }
        rows.append(row)

    return rows


def build_metrics_rows(
    golden_path: Path,
    ollama_records: list[dict[str, Any]],
    gpt_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    with golden_path.open("r", encoding="utf-8") as handle:
        golden_list = json.load(handle)

    ollama_by_id = records_to_map(ollama_records)
    gpt_by_id = records_to_map(gpt_records)
    rows: list[dict[str, Any]] = []

    for item in golden_list:
        qid = int(item["id"])
        question = item.get("question", "")
        ollama = ollama_by_id.get(qid, {})
        gpt = gpt_by_id.get(qid, {})
        rows.append(
            {
                "№": qid,
                "Вопрос": question,
                "Judge % Ollama": _judge_total(ollama),
                "Judge % GPT": _judge_total(gpt),
                "Faithfulness Ollama": _rag_metric(ollama, "faithfulness"),
                "Faithfulness GPT": _rag_metric(gpt, "faithfulness"),
                "Hit@3 Ollama": _rag_metric(ollama, "hit_at_3"),
                "Hit@3 GPT": _rag_metric(gpt, "hit_at_3"),
                "Latency Ollama (с)": _rag_metric(
                    ollama, "latency_seconds", ollama.get("time", "")
                ),
                "Latency GPT (с)": _rag_metric(gpt, "latency_seconds", gpt.get("time", "")),
                "Retrieval Ollama (с)": _rag_metric(ollama, "retrieval_time_seconds"),
                "Retrieval GPT (с)": _rag_metric(gpt, "retrieval_time_seconds"),
                "LLM Ollama (с)": _rag_metric(ollama, "llm_time_seconds"),
                "LLM GPT (с)": _rag_metric(gpt, "llm_time_seconds"),
                "Prompt tokens Ollama": _rag_metric(ollama, "prompt_tokens"),
                "Prompt tokens GPT": _rag_metric(gpt, "prompt_tokens"),
                "Completion tokens Ollama": _rag_metric(ollama, "completion_tokens"),
                "Completion tokens GPT": _rag_metric(gpt, "completion_tokens"),
                "Total tokens Ollama": _rag_metric(
                    ollama, "total_tokens", ollama.get("tokens", "")
                ),
                "Total tokens GPT": _rag_metric(gpt, "total_tokens", gpt.get("tokens", "")),
                "Cost USD Ollama": _rag_metric(ollama, "cost_usd"),
                "Cost USD GPT": _rag_metric(gpt, "cost_usd"),
            }
        )
    return rows


def _metadata_row_from_payload(
    title: str,
    file_name: str,
    payload: dict[str, Any],
    *,
    hybrid: str,
    note: str,
) -> dict[str, Any]:
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    stats = payload.get("statistics", {}) if isinstance(payload, dict) else {}
    model = config.get("model", "—")
    provider = config.get("llm_provider", "—")
    questions = config.get("questions", "—")
    chunks = config.get("chunks", "—")
    local_score = stats.get("avg_score", "—")
    judge_score = stats.get("llm_judge_avg_percent", "—")
    return {
        "Прогон": title,
        "Файл": file_name,
        "Модель": model,
        "LLM provider": provider,
        "Чанков": chunks,
        "Вопросов": questions,
        "Средний балл (локальный)": local_score,
        "Средний балл LLM-судья (%)": judge_score,
        "Гибридный поиск": hybrid,
        "Примечание": note,
    }


def build_metadata_rows(
    baseline_xlsx_path: Path,
    ollama_path: Path,
    gpt_path: Path,
    ollama_payload: dict[str, Any],
    gpt_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if baseline_xlsx_path.exists():
        baseline_meta = pd.read_excel(baseline_xlsx_path, sheet_name="Метаданные")
        for _, row in baseline_meta.iterrows():
            run_name = str(row.get("Прогон", ""))
            if run_name in ("Золотой сет", "Ollama без гибридного поиска"):
                rows.append({column: row.get(column, "") for column in METADATA_COLUMNS})

    if not rows:
        rows.append(
            {
                "Прогон": "Золотой сет",
                "Файл": "golden_set.json",
                "Модель": "—",
                "LLM provider": "—",
                "Чанков": "—",
                "Вопросов": 28,
                "Средний балл (локальный)": "—",
                "Средний балл LLM-судья (%)": "—",
                "Гибридный поиск": "—",
                "Примечание": "Эталонные ответы (reference)",
            }
        )

    rows.append(
        _metadata_row_from_payload(
            "Ollama с гибридным поиском",
            ollama_path.name,
            ollama_payload,
            hybrid="да",
            note="full_evaluation + llama3.2:3b",
        )
    )
    rows.append(
        _metadata_row_from_payload(
            "GPT с гибридным поиском",
            gpt_path.name,
            gpt_payload,
            hybrid="да",
            note="full_evaluation + gpt-4o-mini",
        )
    )
    return rows


def run_provider_eval(provider: str, output: Path, limit: int) -> None:
    _apply_bootstrap(DEFAULT_VECTOR_STORE)
    sys.path.insert(0, str(ROOT_DIR))
    from dotenv import load_dotenv

    load_dotenv(".env.rag", override=True)
    os.environ["CACHE_ENABLED"] = "false"

    from core.database import db
    from core.embedding_alignment import configure_embeddings_for_index
    from full_evaluation import RagRunner, load_golden_cases

    configure_embeddings_for_index(DEFAULT_VECTOR_STORE)
    db.reload_store()
    golden_cases = load_golden_cases(DEFAULT_GOLDEN)
    if limit > 0:
        golden_cases = golden_cases[:limit]

    runner = RagRunner(llm_provider=provider)
    results: list[dict[str, Any]] = []
    for item in golden_cases:
        response = runner.ask(item.question)
        results.append(
            {
                "id": item.id,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "final_score": "",
                "grade": "",
                "time": response.get("time_total", 0),
                "llm_provider": response.get("llm_provider", provider),
            }
        )

    payload = {
        "config": {"llm_provider": provider, "questions": len(results)},
        "results": results,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _autosize_columns(writer: pd.ExcelWriter, sheet_name: str, widths: dict[str, int]) -> None:
    worksheet = writer.sheets[sheet_name]
    for index, column in enumerate(widths, start=1):
        worksheet.column_dimensions[chr(64 + index)].width = widths[column]
    worksheet.freeze_panes = "A2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export comparison workbook in baseline format.")
    parser.add_argument(
        "--baseline-json",
        default=str(DEFAULT_BASELINE_JSON),
        help="Baseline JSON with Ollama answers without hybrid search.",
    )
    parser.add_argument(
        "--baseline-xlsx",
        default=str(DEFAULT_BASELINE_XLSX),
        help="Existing baseline workbook used as layout/reference.",
    )
    parser.add_argument(
        "--ollama-json",
        default="evaluation_llama3b_gpt_judge_v3.json",
        help="Evaluation JSON with Ollama hybrid answers.",
    )
    parser.add_argument(
        "--gpt-json",
        default="evaluation_gpt_hybrid.json",
        help="Evaluation JSON with GPT hybrid answers.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_hybrid_gpt_ollama.xlsx",
        help="Output .xlsx path.",
    )
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN),
        help="golden_set.json path.",
    )
    parser.add_argument(
        "--run-gpt",
        action="store_true",
        help="Run GPT hybrid evaluation if --gpt-json is missing.",
    )
    parser.add_argument(
        "--run-ollama",
        action="store_true",
        help="Run Ollama hybrid evaluation if --ollama-json is missing.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit questions when --run-* is used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    golden_path = Path(args.golden).resolve()
    baseline_json_path = Path(args.baseline_json).resolve()
    baseline_xlsx_path = Path(args.baseline_xlsx).resolve()
    ollama_path = Path(args.ollama_json).resolve()
    gpt_path = Path(args.gpt_json).resolve()

    if args.run_ollama and not ollama_path.exists():
        print(f"Running Ollama evaluation -> {ollama_path}")
        run_provider_eval("ollama", ollama_path, args.limit)

    if args.run_gpt and not gpt_path.exists():
        print(f"Running GPT evaluation -> {gpt_path}")
        run_provider_eval("openai", gpt_path, args.limit)

    if not ollama_path.exists():
        print(f"Ollama results not found: {ollama_path}", file=sys.stderr)
        return 1
    if not gpt_path.exists():
        print(f"GPT results not found: {gpt_path}. Use --run-gpt or pass --gpt-json.", file=sys.stderr)
        return 1

    ollama_payload = load_eval_payload(ollama_path)
    gpt_payload = load_eval_payload(gpt_path)
    comparison_rows = build_rows(
        golden_path,
        load_baseline_sheet(baseline_xlsx_path),
        load_baseline_answers(baseline_json_path),
        load_eval_json(ollama_path),
        load_eval_json(gpt_path),
    )
    metrics_rows = build_metrics_rows(
        golden_path,
        load_eval_json(ollama_path),
        load_eval_json(gpt_path),
    )
    metadata_rows = build_metadata_rows(
        baseline_xlsx_path,
        ollama_path,
        gpt_path,
        ollama_payload,
        gpt_payload,
    )

    comparison_df = pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS)
    metrics_df = pd.DataFrame(metrics_rows, columns=METRICS_COLUMNS)
    metadata_df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, index=False, sheet_name="Сравнение")
        metrics_df.to_excel(writer, index=False, sheet_name="Метрики")
        metadata_df.to_excel(writer, index=False, sheet_name="Метаданные")
        _autosize_columns(
            writer,
            "Сравнение",
            {
                "№": 6,
                "Вопрос": 48,
                "Золотой сет": 52,
                "Ollama без гибридного поиска": 52,
                "Оценка Ollama (без гибрида)": 18,
                "Ollama с гибридным поиском": 52,
                "Оценка Ollama (гибрид)": 18,
                "GPT с гибридным поиском": 52,
                "Оценка GPT (гибрид)": 18,
                "Источники Ollama (без гибрида)": 42,
                "Источники Ollama (гибрид)": 42,
                "Источники GPT (гибрид)": 42,
            },
        )
        _autosize_columns(
            writer,
            "Метрики",
            {
                "№": 6,
                "Вопрос": 44,
                "Judge % Ollama": 14,
                "Judge % GPT": 12,
                "Faithfulness Ollama": 18,
                "Faithfulness GPT": 16,
                "Hit@3 Ollama": 13,
                "Hit@3 GPT": 11,
                "Latency Ollama (с)": 16,
                "Latency GPT (с)": 14,
                "Retrieval Ollama (с)": 18,
                "Retrieval GPT (с)": 16,
                "LLM Ollama (с)": 14,
                "LLM GPT (с)": 12,
                "Prompt tokens Ollama": 20,
                "Prompt tokens GPT": 18,
                "Completion tokens Ollama": 24,
                "Completion tokens GPT": 22,
                "Total tokens Ollama": 18,
                "Total tokens GPT": 16,
                "Cost USD Ollama": 16,
                "Cost USD GPT": 14,
            },
        )
        _autosize_columns(
            writer,
            "Метаданные",
            {
                "Прогон": 30,
                "Файл": 35,
                "Модель": 28,
                "LLM provider": 14,
                "Чанков": 10,
                "Вопросов": 10,
                "Средний балл (локальный)": 22,
                "Средний балл LLM-судья (%)": 24,
                "Гибридный поиск": 16,
                "Примечание": 46,
            },
        )

    print(f"Saved {len(comparison_rows)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

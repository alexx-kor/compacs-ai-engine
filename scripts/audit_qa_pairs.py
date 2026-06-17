#!/usr/bin/env python3
"""Audit synthetic QA pair quality (groundedness, duplicates, manual-review need)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

NOISE_MARKERS = ("арбуз", "кстати", "btw", "помидор", "биткоин", "???", "!!!", "блабла")


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[\w]{4,}", text.lower())}


def overlap_faithfulness(answer: str, chunk: str) -> float:
    a, c = _tokenize(answer), _tokenize(chunk)
    if not a:
        return 0.0
    return round(100.0 * len(a & c) / len(a), 1)


def load_chunk_catalog(vector_dir: Path) -> dict[str, str]:
    from config import Config
    from core.storage.json_store import JsonVectorStore

    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(vector_dir)
    store = JsonVectorStore(Config.from_env())
    return {str(record.id): str(record.chunk) for record in store.load_all_records()}


def audit_pairs(
    qa_path: Path,
    vector_dir: Path,
    *,
    judge_sample: int,
) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.rag", override=True)
    items = json.loads(qa_path.read_text(encoding="utf-8"))
    chunk_catalog = load_chunk_catalog(vector_dir)
    seen_questions: dict[str, int] = {}
    rows: list[dict] = []

    for item in items:
        q = str(item.get("question", "")).strip()
        a = str(item.get("expected_answer", "")).strip()
        chunk_id = str(item.get("chunk_id", ""))
        chunk_text = chunk_catalog.get(chunk_id, "")
        faith = overlap_faithfulness(a, chunk_text) if chunk_text else 0.0
        q_key = re.sub(r"\s+", " ", q.lower())[:80]
        duplicate = q_key in seen_questions
        seen_questions[q_key] = item.get("id", 0)
        noise = any(m in q.lower() for m in NOISE_MARKERS)
        heuristic = 100.0
        if len(q) < 15:
            heuristic -= 25
        if len(a) < 40:
            heuristic -= 25
        if faith < 25:
            heuristic -= 30
        elif faith < 45:
            heuristic -= 15
        if duplicate:
            heuristic -= 20
        if noise:
            heuristic -= 15
        if "?" not in q and "как" not in q.lower():
            heuristic -= 5
        heuristic = max(0.0, min(100.0, heuristic))
        rows.append(
            {
                "id": item.get("id"),
                "question": q,
                "answer_preview": a[:200],
                "source_chunk": item.get("source_chunk", ""),
                "chunk_id": chunk_id,
                "faithfulness_overlap_pct": faith,
                "duplicate_question": duplicate,
                "noise_in_question": noise,
                "heuristic_quality_pct": round(heuristic, 1),
                "gpt_audit": None,
            }
        )

    judge_fn = None
    if judge_sample > 0:
        from config import Config
        from openai import OpenAI

        cfg = Config.from_env()
        if cfg.openai_api_key and cfg.openai_api_key.strip() != "user_provided":
            client = OpenAI(api_key=cfg.openai_api_key)

            def _judge_one(question: str, answer: str, chunk: str) -> dict:
                prompt = (
                    "Оцени качество обучающей пары Q/A для RAG (0-100). Критерии: "
                    "ответ строго по фрагменту, вопрос чёткий без шума, пригодно для fine-tune. "
                    'JSON: {"total": number, "grounded": number, "clarity": number, "verdict": "ok|review|reject"}'
                )
                user = f"Фрагмент:\n{chunk[:2000]}\n\nВопрос: {question}\n\nОтвет: {answer[:1500]}"
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user},
                    ],
                )
                return json.loads(resp.choices[0].message.content or "{}")

            judge_fn = _judge_one

    import random

    rng = random.Random(42)
    sample_ids = set(rng.sample([r["id"] for r in rows], k=min(judge_sample, len(rows))))
    for row in rows:
        if row["id"] not in sample_ids or not judge_fn:
            continue
        chunk_text = chunk_catalog.get(str(row["chunk_id"]), "")
        try:
            row["gpt_audit"] = judge_fn(row["question"], row["answer_preview"], chunk_text)
        except Exception as error:
            row["gpt_audit"] = {"error": str(error)}

    gpt_scores = [
        float(r["gpt_audit"]["total"])
        for r in rows
        if isinstance(r.get("gpt_audit"), dict) and r["gpt_audit"].get("total") is not None
    ]
    heuristics = [r["heuristic_quality_pct"] for r in rows]
    high = sum(1 for h in heuristics if h >= 75)
    low = sum(1 for h in heuristics if h < 55)
    avg_heur = round(mean(heuristics), 1) if heuristics else 0
    avg_gpt = round(mean(gpt_scores), 1) if gpt_scores else None

    if avg_heur >= 80 and (avg_gpt is None or avg_gpt >= 78):
        manual_verdict = (
            "Ручная разметка всех 150 пар не обязательна. Достаточно выборочной проверки "
            "10–15% (15–23 пары) с фокусом на пары с heuristic < 55 или verdict=review/reject."
        )
        manual_need_pct = 12
    elif avg_heur >= 65:
        manual_verdict = (
            "Рекомендуется ручная правка 20–30% пар с низким faithfulness/overlap "
            "или дублирующимися вопросами перед дообучением."
        )
        manual_need_pct = 25
    else:
        manual_verdict = (
            "Качество синтетических пар недостаточное: необходима ручная разметка "
            "или перегенерация с усиленным контролем по фрагменту."
        )
        manual_need_pct = 50

    return {
        "summary": {
            "pairs": len(rows),
            "avg_heuristic_quality_pct": avg_heur,
            "avg_gpt_audit_pct": avg_gpt,
            "high_quality_heuristic_ge75": high,
            "low_quality_heuristic_lt55": low,
            "duplicates": sum(1 for r in rows if r["duplicate_question"]),
            "noise_questions": sum(1 for r in rows if r["noise_in_question"]),
            "gpt_judged_sample": len(gpt_scores),
            "recommended_manual_review_pct": manual_need_pct,
            "manual_labeling_verdict": manual_verdict,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-path", default=str(ROOT / "instructions" / "golden" / "ui_extension_qa_150.json"))
    parser.add_argument("--vector-dir", default=str(ROOT / "data" / "vectors"))
    parser.add_argument("--output", default=str(ROOT / "data" / "finetune" / "qa_audit_report.json"))
    parser.add_argument("--judge-sample", type=int, default=50)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        log.error("QA file not found: %s", qa_path)
        return 1

    report = audit_pairs(qa_path, Path(args.vector_dir), judge_sample=args.judge_sample)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("QA audit saved: %s", out)
    log.info("summary: %s", json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

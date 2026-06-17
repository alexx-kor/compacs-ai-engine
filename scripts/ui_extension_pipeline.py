#!/usr/bin/env python3
"""UI extension dataset: extract → index → 150 QA → fine-tune → evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

DEFAULT_UI_ROOT = ROOT / "UI extension" / "UI extension"
DEFAULT_TXT_OUT = ROOT / "instructions" / "ui_extension"
DEFAULT_VECTOR_DIR = ROOT / "data" / "vectors"
DEFAULT_QA_PATH = ROOT / "instructions" / "golden" / "ui_extension_qa_150.json"
DEFAULT_TRAIN_JSONL = ROOT / "data" / "finetune" / "ui_extension_train.jsonl"
DEFAULT_MODELFILE = ROOT / "data" / "finetune" / "Modelfile.compacs-ui-ft"
DEFAULT_FT_MODEL = "compacs-ui-ft"


@dataclass(frozen=True)
class PipelinePaths:
    ui_root: Path
    txt_out: Path
    vector_dir: Path
    qa_path: Path
    train_jsonl: Path
    modelfile: Path


def _paths(args: argparse.Namespace) -> PipelinePaths:
    return PipelinePaths(
        ui_root=Path(args.ui_root).resolve(),
        txt_out=Path(args.txt_out).resolve(),
        vector_dir=Path(args.vector_dir).resolve(),
        qa_path=Path(args.qa_path).resolve(),
        train_jsonl=Path(args.train_jsonl).resolve(),
        modelfile=Path(args.modelfile).resolve(),
    )


def cmd_extract(paths: PipelinePaths) -> int:
    from core.html_text import extract_html_directory

    if not paths.ui_root.exists():
        log.error("UI extension folder not found: %s", paths.ui_root)
        return 1
    written = extract_html_directory(paths.ui_root, paths.txt_out, languages=("ru",))
    log.info("extracted %s txt files -> %s", len(written), paths.txt_out)
    for path in written:
        log.info("  %s (%s chars)", path.name, path.stat().st_size)
    return 0 if written else 1


def _collect_ui_txt_files(txt_dir: Path) -> list[Path]:
    return sorted(txt_dir.rglob("*.txt"))


def cmd_index(paths: PipelinePaths, *, merge: bool) -> int:
    from config import config
    from core.chunk_pipeline import build_all_chunks
    from core.database import db
    from core.embeddings import embedder
    from core.storage.protocol import ChunkRecord

    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(paths.vector_dir)
    os.environ["EMBEDDING_PROVIDER"] = "ollama"
    os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"

    from core.embedding_alignment import reset_embedder

    reset_embedder()

    files = _collect_ui_txt_files(paths.txt_out)
    if not files:
        log.error("no txt in %s — run extract first", paths.txt_out)
        return 1

    graph_dir = ROOT / "data" / "graph"
    new_chunks = build_all_chunks(
        files,
        paths.txt_out,
        graph_dir,
        strategies=("sliding", "section", "definition"),
    )
    log.info("built %s chunks from UI extension", len(new_chunks))
    if not new_chunks:
        return 1

    db.reload_store()
    if merge:
        db.init_database(force_recreate=False)
        existing = db._store.load_all_records()  # noqa: SLF001
        max_id = max((int(r.id) for r in existing), default=-1)
        for offset, chunk in enumerate(new_chunks):
            chunk["id"] = max_id + 1 + offset
        log.info("merge mode: existing=%s new=%s", len(existing), len(new_chunks))
    else:
        db.init_database(force_recreate=True)

    batch_size = max(1, int(config.batch_size))
    inserted = 0
    for start in range(0, len(new_chunks), batch_size):
        batch = new_chunks[start : start + batch_size]
        texts = [item["chunk"] for item in batch]
        vectors = embedder.embed(texts)
        for item, vector in zip(batch, vectors):
            item["embedding"] = vector
        db.insert_batch(batch, dataset_kind="ui_extension")
        inserted += len(batch)
        log.info("embedded batch %s/%s", inserted, len(new_chunks))

    all_records = db._store.load_all_records()  # noqa: SLF001
    legacy = [ChunkRecord.from_legacy_dict(r.to_legacy_dict()).to_legacy_dict() for r in all_records]
    # use record's legacy dict directly
    legacy = [r.to_legacy_dict() for r in all_records]
    db.save_bm25_index(legacy)
    db.reload_store()
    log.info("index done: total_chunks=%s store=%s", db.get_chunk_count(), paths.vector_dir)
    return 0


def _sample_chunks_for_qa(vector_dir: Path, count: int, seed: int) -> list[dict[str, str]]:
    from core.storage.json_store import JsonVectorStore
    from config import Config

    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(vector_dir)
    store = JsonVectorStore(Config.from_env())
    records = store.load_all_records()
    ui_records = [r for r in records if "ui_extension" in str(r.source)]
    if not ui_records:
        ui_records = records[-500:]
    rng = random.Random(seed)
    picked = rng.sample(ui_records, k=min(count, len(ui_records)))
    return [
        {
            "chunk_id": r.id,
            "source": r.source,
            "text": str(r.chunk)[:2500],
        }
        for r in picked
    ]


def cmd_generate_qa(paths: PipelinePaths, count: int, seed: int) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.rag", override=True)
    from config import Config
    from openai import OpenAI

    cfg = Config.from_env()
    if not cfg.openai_api_key or cfg.openai_api_key.strip() == "user_provided":
        log.error("OPENAI_API_KEY required in .env.rag for QA generation")
        return 1

    samples = _sample_chunks_for_qa(paths.vector_dir, count * 2, seed)
    client = OpenAI(api_key=cfg.openai_api_key)
    pairs: list[dict[str, Any]] = []
    system = (
        "Ты создаёшь обучающие пары вопрос-ответ по документации ПО КОМПАКС (UI/руководство оператора). "
        "Вопрос — чёткий, без шума, оффтопа и лишних фраз. Ответ — фактологичный, по тексту фрагмента, "
        "на русском, 3–8 предложений. Не выдумывай факты вне фрагмента."
    )

    for index, sample in enumerate(samples):
        if len(pairs) >= count:
            break
        user = (
            f"Фрагмент документации (источник: {sample['source']}):\n\n"
            f"{sample['text']}\n\n"
            "Сформируй JSON: {\"question\": \"...\", \"answer\": \"...\"}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            if len(question) < 15 or len(answer) < 40:
                continue
            pairs.append(
                {
                    "id": len(pairs) + 1,
                    "question": question,
                    "expected_answer": answer,
                    "source_chunk": sample["source"],
                    "chunk_id": sample["chunk_id"],
                }
            )
            log.info("[%s/%s] qa id=%s", len(pairs), count, pairs[-1]["id"])
        except Exception as error:
            log.warning("qa generation failed: %s", error)

    if len(pairs) < count:
        log.warning("generated only %s/%s pairs", len(pairs), count)

    paths.qa_path.parent.mkdir(parents=True, exist_ok=True)
    paths.qa_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved QA set: %s (%s pairs)", paths.qa_path, len(pairs))
    return 0 if pairs else 1


def _qa_to_train_jsonl(qa_path: Path, out_path: Path) -> int:
    items = json.loads(qa_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты эксперт по ПО КОМПАКС. Отвечай точно по документации, "
                            "структурированно, на русском."
                        ),
                    },
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["expected_answer"]},
                ]
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(items)


def cmd_finetune(
    paths: PipelinePaths,
    base_model: str,
    ollama_only: bool,
    train_qa_path: Path | None = None,
) -> int:
    train_source = train_qa_path or paths.qa_path
    if not train_source.exists():
        log.error("QA file missing: %s", train_source)
        return 1

    n = _qa_to_train_jsonl(train_source, paths.train_jsonl)
    log.info("train jsonl: %s (%s examples)", paths.train_jsonl, n)

    items = json.loads(train_source.read_text(encoding="utf-8"))
    exemplars = items[: min(8, len(items))]
    lines = [
        f"FROM {base_model}",
        "",
        "PARAMETER temperature 0.2",
        "PARAMETER num_ctx 8192",
        "",
        "SYSTEM \"\"\"Ты эксперт по ПО КОМПАКС (руководство оператора, UI). "
        "Отвечай точно, по документации, на русском, без выдумок.\"\"\"",
        "",
    ]
    for ex in exemplars:
        q = ex["question"].replace('"""', "'")
        a = ex["expected_answer"].replace('"""', "'")[:1200]
        lines.append(f"MESSAGE user \"\"\"{q}\"\"\"")
        lines.append(f"MESSAGE assistant \"\"\"{a}\"\"\"")
        lines.append("")

    paths.modelfile.parent.mkdir(parents=True, exist_ok=True)
    paths.modelfile.write_text("\n".join(lines), encoding="utf-8")
    log.info("Modelfile written: %s", paths.modelfile)

    try:
        subprocess.run(
            ["ollama", "create", DEFAULT_FT_MODEL, "-f", str(paths.modelfile)],
            check=True,
            capture_output=False,
        )
        log.info("ollama model created: %s", DEFAULT_FT_MODEL)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        log.error("ollama create failed: %s", error)
        return 1

    if ollama_only:
        return 0

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.rag", override=True)
    from config import Config
    from openai import OpenAI

    cfg = Config.from_env()
    if not cfg.openai_api_key or cfg.openai_api_key.strip() == "user_provided":
        log.warning("skip OpenAI fine-tune: no API key")
        return 0

    client = OpenAI(api_key=cfg.openai_api_key)
    with paths.train_jsonl.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4o-mini-2024-07-18",
        suffix="compacs-ui",
    )
    meta_path = paths.train_jsonl.parent / "openai_finetune_job.json"
    meta_path.write_text(
        json.dumps({"job_id": job.id, "status": job.status, "model": job.model}, indent=2),
        encoding="utf-8",
    )
    log.info("OpenAI fine-tune job started: %s (poll with `openai api fine_tuning.jobs.retrieve -i %s`)", job.id, job.id)
    return 0


def cmd_evaluate(
    paths: PipelinePaths,
    *,
    baseline_model: str,
    finetuned_model: str,
    llm_judge: bool,
) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.rag", override=True)
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(paths.vector_dir)
    os.environ["CACHE_ENABLED"] = "false"
    os.environ["QUERY_FILTER_ENABLED"] = "true"

    from core.database import db
    from core.embedding_alignment import configure_embeddings_for_index
    from full_evaluation import MetricsCalculator, RagRunner, evaluate_response, load_golden_cases

    db.reload_store()
    if db.get_chunk_count() == 0:
        log.error("vector store empty — run index first")
        return 1
    configure_embeddings_for_index(paths.vector_dir)

    golden_cases = load_golden_cases(paths.qa_path)
    metrics = MetricsCalculator()
    llm_judge_fn = None
    if llm_judge:
        from config import Config
        from llm_evaluate import evaluate_answer_percent_openai

        cfg = Config.from_env()
        if cfg.openai_api_key and cfg.openai_api_key.strip() != "user_provided":
            llm_judge_fn = lambda q, a: evaluate_answer_percent_openai(q, a, model="gpt-4o-mini")

    def _run_eval(model_name: str, tag: str) -> dict[str, Any]:
        os.environ["OLLAMA_MODEL"] = model_name
        rag = RagRunner(llm_provider="ollama")
        results: list[dict[str, Any]] = []
        for item in golden_cases:
            response = rag.ask(item.question)
            ev = evaluate_response(item, response, metrics)
            judge = llm_judge_fn(item.question, str(response.get("answer", ""))) if llm_judge_fn else None
            results.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": response.get("answer"),
                    "final_score": ev.final_score,
                    "grade": ev.grade.value,
                    "llm_judge": judge,
                }
            )
        scores = [r["final_score"] for r in results]
        judges = [
            float(r["llm_judge"]["total"])
            for r in results
            if r.get("llm_judge") and r["llm_judge"].get("total") is not None
        ]
        summary = {
            "tag": tag,
            "model": model_name,
            "count": len(results),
            "avg_score": round(mean(scores), 2) if scores else 0,
            "avg_judge_percent": round(mean(judges), 1) if judges else None,
        }
        return {"summary": summary, "results": results}

    baseline = _run_eval(baseline_model, "baseline")
    finetuned = _run_eval(finetuned_model, "finetuned")

    out = paths.vector_dir.parent / "finetune" / "ui_extension_eval_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline["summary"],
        "finetuned": finetuned["summary"],
        "delta_score": round(
            finetuned["summary"]["avg_score"] - baseline["summary"]["avg_score"], 2
        ),
        "delta_judge": (
            round(
                (finetuned["summary"]["avg_judge_percent"] or 0)
                - (baseline["summary"]["avg_judge_percent"] or 0),
                1,
            )
            if finetuned["summary"]["avg_judge_percent"] and baseline["summary"]["avg_judge_percent"]
            else None
        ),
    }
    out.write_text(
        json.dumps(
            {"comparison": payload, "baseline_results": baseline, "finetuned_results": finetuned},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("eval comparison: %s", json.dumps(payload, ensure_ascii=False))
    log.info("saved %s", out)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI extension RAG + fine-tune pipeline")
    parser.add_argument("--ui-root", default=str(DEFAULT_UI_ROOT))
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT_OUT))
    parser.add_argument("--vector-dir", default=str(DEFAULT_VECTOR_DIR))
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL))
    parser.add_argument("--modelfile", default=str(DEFAULT_MODELFILE))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("extract", help="HTML → txt in instructions/ui_extension")

    p_index = sub.add_parser("index", help="Chunk + embed (merge into existing store by default)")
    p_index.add_argument("--no-merge", action="store_true", help="Replace store instead of append")

    p_qa = sub.add_parser("generate-qa", help="Generate 150 clean QA pairs via GPT")
    p_qa.add_argument("--count", type=int, default=150)
    p_qa.add_argument("--seed", type=int, default=42)

    p_ft = sub.add_parser("finetune", help="Build Modelfile + ollama create; optional OpenAI FT job")
    p_ft.add_argument("--base-model", default="llama3.2:3b")
    p_ft.add_argument("--ollama-only", action="store_true")
    p_ft.add_argument("--train-qa-path", default=None, help="Optional train split JSON path")

    p_ev = sub.add_parser("evaluate", help="Compare baseline vs fine-tuned on QA set")
    p_ev.add_argument("--baseline-model", default="llama3.2:3b")
    p_ev.add_argument("--finetuned-model", default=DEFAULT_FT_MODEL)
    p_ev.add_argument("--llm-judge", action="store_true")

    p_all = sub.add_parser("all", help="extract → index → generate-qa → finetune")
    p_all.add_argument("--qa-count", type=int, default=150)
    p_all.add_argument("--skip-openai-ft", action="store_true")

    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    paths = _paths(args)

    if args.command == "extract":
        return cmd_extract(paths)
    if args.command == "index":
        return cmd_index(paths, merge=not args.no_merge)
    if args.command == "generate-qa":
        return cmd_generate_qa(paths, args.count, args.seed)
    if args.command == "finetune":
        train_qa = Path(args.train_qa_path).resolve() if args.train_qa_path else None
        return cmd_finetune(paths, args.base_model, args.ollama_only, train_qa_path=train_qa)
    if args.command == "evaluate":
        return cmd_evaluate(
            paths,
            baseline_model=args.baseline_model,
            finetuned_model=args.finetuned_model,
            llm_judge=args.llm_judge,
        )
    if args.command == "all":
        steps = [
            ("extract", lambda: cmd_extract(paths)),
            ("index", lambda: cmd_index(paths, merge=True)),
            ("generate-qa", lambda: cmd_generate_qa(paths, args.qa_count, 42)),
            ("finetune", lambda: cmd_finetune(paths, "llama3.2:3b", args.skip_openai_ft, None)),
        ]
        for name, fn in steps:
            log.info("=== step: %s ===", name)
            code = fn()
            if code != 0:
                log.error("step %s failed", name)
                return code
        log.info("pipeline complete; run evaluate separately when ready")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

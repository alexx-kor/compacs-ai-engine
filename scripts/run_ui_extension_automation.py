#!/usr/bin/env python3
"""Full automated UI extension pipeline: index → QA → audit → FT → eval → Excel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

PYTHON = sys.executable
QA_PATH = ROOT / "instructions" / "golden" / "ui_extension_qa_150.json"
SPLIT_DIR = ROOT / "instructions" / "golden" / "splits"
TRAIN_QA = SPLIT_DIR / "ui_extension_qa_train.json"
VAL_QA = SPLIT_DIR / "ui_extension_qa_val.json"
TEST_QA = SPLIT_DIR / "ui_extension_qa_test.json"
BASELINE_JSON = ROOT / "evaluation_ui_extension_val_baseline.json"
FINETUNED_JSON = ROOT / "evaluation_ui_extension_val_finetuned.json"
BASELINE_TEST_JSON = ROOT / "evaluation_ui_extension_test_baseline.json"
FINETUNED_TEST_JSON = ROOT / "evaluation_ui_extension_test_finetuned.json"
QA_AUDIT_JSON = ROOT / "data" / "finetune" / "qa_audit_report.json"
REPORT_XLSX = ROOT / "comparison_ui_extension_finetune.xlsx"
PIPELINE = ROOT / "scripts" / "ui_extension_pipeline.py"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    log.info("RUN: %s", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=str(ROOT), env=merged, check=False).returncode


def _ui_chunk_count() -> int:
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(ROOT / "data" / "vectors")
    from core.database import db

    db.reload_store()
    return sum(
        1
        for r in db._store.load_all_records()  # noqa: SLF001
        if getattr(r, "dataset_kind", "") == "ui_extension"
        or "OG_1" in str(r.source)
    )


def _wait_ui_chunks(min_chunks: int, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        count = _ui_chunk_count()
        log.info("ui chunks indexed: %s (target min %s)", count, min_chunks)
        if count >= min_chunks:
            return True
        time.sleep(30)
    return _ui_chunk_count() >= min_chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-count", type=int, default=150)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--min-ui-chunks", type=int, default=500)
    parser.add_argument("--index-timeout-min", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    steps_failed = 0

    if not args.skip_index:
        count = _ui_chunk_count()
        if count < args.min_ui_chunks:
            log.info("starting/resuming index merge...")
            code = _run([PYTHON, str(PIPELINE), "index"])
            if code != 0:
                log.warning("index returned %s, waiting for partial chunks", code)
            if not _wait_ui_chunks(args.min_ui_chunks, args.index_timeout_min * 60):
                log.warning("index incomplete but continuing with %s ui chunks", _ui_chunk_count())

    if not QA_PATH.exists():
        code = _run(
            [PYTHON, str(PIPELINE), "generate-qa", "--count", str(args.qa_count), "--seed", "42"]
        )
        if code != 0:
            return code
    else:
        n = len(json.loads(QA_PATH.read_text(encoding="utf-8")))
        log.info("QA file exists: %s pairs", n)
        if n < args.qa_count:
            code = _run(
                [PYTHON, str(PIPELINE), "generate-qa", "--count", str(args.qa_count), "--seed", "42"]
            )
            if code != 0:
                return code

    if not TRAIN_QA.exists() or not VAL_QA.exists() or not TEST_QA.exists():
        code = _run(
            [
                PYTHON,
                str(ROOT / "scripts" / "split_ui_qa_dataset.py"),
                "--input",
                str(QA_PATH),
                "--output-dir",
                str(SPLIT_DIR),
                "--seed",
                str(args.seed),
            ]
        )
        if code != 0:
            return code

    code = _run(
        [
            PYTHON,
            str(ROOT / "scripts" / "audit_qa_pairs.py"),
            "--qa-path",
            str(QA_PATH),
            "--judge-sample",
            "50",
        ]
    )
    if code != 0:
        steps_failed += 1

    code = _run(
        [
            PYTHON,
            str(PIPELINE),
            "finetune",
            "--ollama-only",
            "--train-qa-path",
            str(TRAIN_QA),
        ]
    )
    if code != 0:
        log.error("finetune failed")
        return code

    if not BASELINE_JSON.exists():
        code = _run(
            [
                PYTHON,
                str(ROOT / "full_evaluation.py"),
                "--golden",
                str(VAL_QA),
                "--output",
                str(BASELINE_JSON),
                "--llm-judge",
                "--judge-backend",
                "openai",
                "--llm-provider",
                "ollama",
            ],
            {"OLLAMA_MODEL": "llama3.2:3b", "CACHE_ENABLED": "false", "QUERY_FILTER_ENABLED": "true"},
        )
        if code != 0:
            return code

    if not FINETUNED_JSON.exists():
        code = _run(
            [
                PYTHON,
                str(ROOT / "full_evaluation.py"),
                "--golden",
                str(VAL_QA),
                "--output",
                str(FINETUNED_JSON),
                "--llm-judge",
                "--judge-backend",
                "openai",
                "--llm-provider",
                "ollama",
            ],
            {
                "OLLAMA_MODEL": "compacs-ui-ft",
                "CACHE_ENABLED": "false",
                "QUERY_FILTER_ENABLED": "true",
            },
        )
        if code != 0:
            return code

    if not BASELINE_TEST_JSON.exists():
        code = _run(
            [
                PYTHON,
                str(ROOT / "full_evaluation.py"),
                "--golden",
                str(TEST_QA),
                "--output",
                str(BASELINE_TEST_JSON),
                "--llm-judge",
                "--judge-backend",
                "openai",
                "--llm-provider",
                "ollama",
            ],
            {"OLLAMA_MODEL": "llama3.2:3b", "CACHE_ENABLED": "false", "QUERY_FILTER_ENABLED": "true"},
        )
        if code != 0:
            return code

    if not FINETUNED_TEST_JSON.exists():
        code = _run(
            [
                PYTHON,
                str(ROOT / "full_evaluation.py"),
                "--golden",
                str(TEST_QA),
                "--output",
                str(FINETUNED_TEST_JSON),
                "--llm-judge",
                "--judge-backend",
                "openai",
                "--llm-provider",
                "ollama",
            ],
            {
                "OLLAMA_MODEL": "compacs-ui-ft",
                "CACHE_ENABLED": "false",
                "QUERY_FILTER_ENABLED": "true",
            },
        )
        if code != 0:
            return code

    code = _run(
        [
            PYTHON,
            str(ROOT / "scripts" / "export_ui_extension_report.py"),
            "--golden",
            str(VAL_QA),
            "--baseline-json",
            str(BASELINE_JSON),
            "--finetuned-json",
            str(FINETUNED_JSON),
            "-o",
            str(REPORT_XLSX),
            "--ui-chunks",
            str(_ui_chunk_count()),
        ]
    )
    if code != 0:
        return code

    log.info("=== AUTOMATION COMPLETE ===")
    log.info("Excel report: %s", REPORT_XLSX)
    log.info("VAL baseline eval: %s", BASELINE_JSON)
    log.info("VAL finetuned eval: %s", FINETUNED_JSON)
    log.info("TEST baseline eval: %s", BASELINE_TEST_JSON)
    log.info("TEST finetuned eval: %s", FINETUNED_TEST_JSON)
    log.info("QA audit: %s", QA_AUDIT_JSON)
    return 0 if steps_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

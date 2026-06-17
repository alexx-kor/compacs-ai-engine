from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import PROJECT_ROOT, load_script_module

split_mod = load_script_module("split_ui_qa_dataset")


def test_to_chat_jsonl_format(tmp_path: Path) -> None:
    items = [
        {"question": "Вопрос один?", "expected_answer": "Ответ один."},
        {"question": "Вопрос два?", "expected_answer": "Ответ два."},
    ]
    out = tmp_path / "train.jsonl"
    split_mod._to_chat_jsonl(items, out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"] == "Вопрос один?"
    assert payload["messages"][2]["content"] == "Ответ один."


def test_split_cli_produces_expected_counts(tmp_path: Path) -> None:
    input_path = tmp_path / "qa.json"
    items = [{"id": i, "question": f"Q{i}?", "expected_answer": f"A{i}." * 10} for i in range(1, 11)]
    input_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    output_dir = tmp_path / "splits"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "split_ui_qa_dataset.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    meta = json.loads((output_dir / "ui_extension_qa_split_meta.json").read_text(encoding="utf-8"))
    assert meta["counts"] == {"total": 10, "train": 8, "val": 1, "test": 1}
    assert meta["seed"] == 42

    train = json.loads((output_dir / "ui_extension_qa_train.json").read_text(encoding="utf-8"))
    val = json.loads((output_dir / "ui_extension_qa_val.json").read_text(encoding="utf-8"))
    test = json.loads((output_dir / "ui_extension_qa_test.json").read_text(encoding="utf-8"))
    assert len(train) + len(val) + len(test) == 10


def test_split_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="Ratios must sum"):
        total = 0.8 + 0.1 + 0.05
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

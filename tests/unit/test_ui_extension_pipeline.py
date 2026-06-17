from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import load_script_module

pipeline = load_script_module("ui_extension_pipeline")


def test_qa_to_train_jsonl(tmp_path: Path) -> None:
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        json.dumps(
            [{"id": 1, "question": "Что такое NvF?", "expected_answer": "Усредненный счет импульсов."}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "train.jsonl"

    count = pipeline._qa_to_train_jsonl(qa_path, out_path)

    assert count == 1
    line = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert line["messages"][1]["content"] == "Что такое NvF?"
    assert "КОМПАКС" in line["messages"][0]["content"]


def test_default_paths_and_ft_model() -> None:
    assert pipeline.DEFAULT_FT_MODEL == "compacs-ui-ft"
    assert pipeline.DEFAULT_QA_PATH.name == "ui_extension_qa_150.json"

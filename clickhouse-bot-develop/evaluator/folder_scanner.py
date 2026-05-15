"""Scan documentation folders for paired questions/answers files."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class FolderScanner:
    """Walk a tree and locate ``questions*`` / ``answers*`` file pairs."""

    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path)

    def scan(self) -> list[dict[str, Any]]:
        """Return metadata dicts for each folder that has both Q and A files."""
        results: list[dict[str, Any]] = []
        if not self.root_path.exists():
            return results

        for root, _dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            questions_file: Path | None = None
            answers_file: Path | None = None

            for name in files:
                file_lower = name.lower()
                if "question" in file_lower or file_lower == "q.txt":
                    questions_file = root_path / name
                if "answer" in file_lower or file_lower == "a.txt":
                    answers_file = root_path / name

            if questions_file and answers_file:
                folder_name = str(root_path.relative_to(self.root_path))
                if folder_name == ".":
                    folder_name = "root"

                results.append(
                    {
                        "folder_path": str(root_path),
                        "folder_name": folder_name,
                        "questions_file": str(questions_file),
                        "answers_file": str(answers_file),
                        "questions_count": count_records(questions_file),
                        "answers_count": count_records(answers_file),
                    }
                )

        return results


def count_records(file_path: Path) -> int:
    """Best-effort line/record count for txt, csv, or json sidecars."""
    try:
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as handle:
                return len([line for line in handle if line.strip()])
        if file_path.suffix == ".csv":
            return len(pd.read_csv(file_path))
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as handle:
                data: object = json.load(handle)
                return len(data) if isinstance(data, list) else 1
    except (OSError, json.JSONDecodeError, ValueError, pd.errors.ParserError) as exc:
        log.warning("Could not count records path=%s: %s", file_path, exc)
    return 0

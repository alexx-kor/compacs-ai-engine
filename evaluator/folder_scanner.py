"""Scan evaluation folders for matching question/answer file pairs."""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class FolderScanner:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)

    def scan(self) -> list[dict]:
        results: list[dict] = []
        if not self.root_path.exists():
            return results

        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            questions_file = None
            answers_file = None

            for file in files:
                file_lower = file.lower()
                if 'question' in file_lower or file_lower == 'q.txt':
                    questions_file = root_path / file
                if 'answer' in file_lower or file_lower == 'a.txt':
                    answers_file = root_path / file

            if questions_file and answers_file:
                folder_name = str(root_path.relative_to(self.root_path))
                if folder_name == '.':
                    folder_name = 'root'

                results.append({
                    'folder_path': str(root_path),
                    'folder_name': folder_name,
                    'questions_file': str(questions_file),
                    'answers_file': str(answers_file),
                    'questions_count': count_records(questions_file),
                    'answers_count': count_records(answers_file)
                })

        return results


def count_records(file_path: Path) -> int:
    try:
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return len([line for line in f if line.strip()])
        elif file_path.suffix == '.csv':
            import pandas as pd
            return len(pd.read_csv(file_path))
        elif file_path.suffix == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        log.exception("failed counting records path=%s: %s", file_path, exc)
        raise
    return 0

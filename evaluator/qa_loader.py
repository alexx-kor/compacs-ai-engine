"""Question/answer loaders for evaluation datasets."""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class QALoader:
    @staticmethod
    def load_questions(file_path: str) -> list[str]:
        path = Path(file_path)

        if path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        elif path.suffix == '.csv':
            df = pd.read_csv(file_path)
            for col in df.columns:
                if 'question' in col.lower() or 'query' in col.lower():
                    return df[col].dropna().astype(str).tolist()
            return df.iloc[:, 0].dropna().astype(str).tolist()

        elif path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [item.get('question', str(item)) for item in data]

        return []

    @staticmethod
    def load_answers(file_path: str) -> list[str]:
        path = Path(file_path)

        if path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        elif path.suffix == '.csv':
            df = pd.read_csv(file_path)
            for col in df.columns:
                if 'answer' in col.lower() or 'response' in col.lower():
                    return df[col].dropna().astype(str).tolist()
            return df.iloc[:, 0].dropna().astype(str).tolist()

        elif path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [item.get('answer', str(item)) for item in data]

        return []

    @staticmethod
    def load_qa_pairs(questions_file: str, answers_file: str) -> list[tuple[str, str]]:
        questions = QALoader.load_questions(questions_file)
        answers = QALoader.load_answers(answers_file)

        min_len = min(len(questions), len(answers))
        return list(zip(questions[:min_len], answers[:min_len]))

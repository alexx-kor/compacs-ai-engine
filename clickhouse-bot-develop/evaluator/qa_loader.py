"""Load questions and answers from TXT, CSV, or JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def _series_to_str_list(series: pd.Series) -> list[str]:
    return [str(value) for value in series.dropna().tolist()]


def _first_matching_column(frame: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in frame.columns:
        lowered = str(column).lower()
        if any(keyword in lowered for keyword in keywords):
            return str(column)
    return None


class QALoader:
    """Load question and answer lists and pair them by row index."""

    @staticmethod
    def load_questions(file_path: str) -> list[str]:
        path = Path(file_path)

        if path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]

        if path.suffix == ".csv":
            frame = pd.read_csv(file_path)
            column = _first_matching_column(frame, ("question", "query"))
            if column is not None:
                return _series_to_str_list(frame[column])
            return _series_to_str_list(frame.iloc[:, 0])

        if path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as handle:
                data: object = json.load(handle)
            if isinstance(data, list):
                return [
                    str(item.get("question", item)) if isinstance(item, dict) else str(item)
                    for item in data
                ]

        log.warning("No questions loaded from path=%s", file_path)
        return []

    @staticmethod
    def load_answers(file_path: str) -> list[str]:
        path = Path(file_path)

        if path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]

        if path.suffix == ".csv":
            frame = pd.read_csv(file_path)
            column = _first_matching_column(frame, ("answer", "response"))
            if column is not None:
                return _series_to_str_list(frame[column])
            return _series_to_str_list(frame.iloc[:, 0])

        if path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as handle:
                data: object = json.load(handle)
            if isinstance(data, list):
                return [
                    str(item.get("answer", item)) if isinstance(item, dict) else str(item)
                    for item in data
                ]

        log.warning("No answers loaded from path=%s", file_path)
        return []

    @staticmethod
    def load_qa_pairs(questions_file: str, answers_file: str) -> list[tuple[str, str]]:
        questions = QALoader.load_questions(questions_file)
        answers = QALoader.load_answers(answers_file)

        min_len = min(len(questions), len(answers))
        if min_len == 0:
            log.warning(
                "Empty QA pair load questions=%s answers=%s",
                len(questions),
                len(answers),
            )
        return list(zip(questions[:min_len], answers[:min_len], strict=False))

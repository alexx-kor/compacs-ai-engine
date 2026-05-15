"""Load prompts and few-shot examples, then route to the best system prompt."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import config

log = logging.getLogger(__name__)

_few_shot_cache: list[dict[str, Any]] | None = None


def load_prompt(filename: str) -> str:
    """Load a prompt template from the ``prompts`` directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / filename
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as handle:
            return handle.read()
    return ""


class FewShotLoader:
    """Load CSV/TXT few-shot pairs from ``config.few_shot_folder``."""

    @staticmethod
    def load_examples() -> list[dict[str, Any]]:
        """Return all few-shot examples found under the configured folder."""
        examples: list[dict[str, Any]] = []
        folder_path = Path(config.few_shot_folder)

        if not folder_path.exists():
            return []

        for file_path in folder_path.rglob("*"):
            if file_path.suffix == ".csv":
                try:
                    frame = pd.read_csv(file_path)
                    if "question" in frame.columns and "answer" in frame.columns:
                        for _, row in frame.iterrows():
                            examples.append(
                                {
                                    "question": str(row["question"]),
                                    "answer": str(row["answer"]),
                                    "source": str(row.get("source", "example")),
                                }
                            )
                except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                    log.warning("Skipping few-shot CSV path=%s: %s", file_path, exc)
            elif file_path.suffix == ".txt":
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        lines = [line.strip() for line in handle if line.strip()]
                        for index in range(0, len(lines), 2):
                            if index + 1 < len(lines):
                                examples.append(
                                    {
                                        "question": lines[index],
                                        "answer": lines[index + 1],
                                        "source": "txt",
                                    }
                                )
                except OSError as exc:
                    log.warning("Skipping few-shot TXT path=%s: %s", file_path, exc)

        return examples

    @staticmethod
    def format_examples(examples: list[dict[str, Any]], max_examples: int = 3) -> str:
        """Render selected examples as markdown for injection into prompts."""
        if not examples:
            return ""

        trimmed = examples[:max_examples]
        formatted = "\n\n##  EXAMPLES OF GOOD ANSWERS:\n\n"

        for index, example in enumerate(trimmed, 1):
            formatted += "**Example %s:**\n" % index
            formatted += "Question: %s\n" % example["question"]
            formatted += "Answer: %s\n" % example["answer"]
            if example.get("source"):
                formatted += "Source: %s\n" % example["source"]
            formatted += "\n"

        return formatted


def _few_shot_examples() -> list[dict[str, Any]]:
    """Return cached few-shot rows, loading from disk on first access."""
    global _few_shot_cache
    if _few_shot_cache is None:
        _few_shot_cache = FewShotLoader.load_examples()
        log.debug("Loaded few-shot examples count=%s", len(_few_shot_cache))
    return _few_shot_cache


RAG_API_PROMPT = load_prompt("rag_api_en.txt")
RAG_API_PARAMETER_PROMPT = load_prompt("rag_api_parameter_en.txt")
RAG_API_PARAMETERS_LIST_PROMPT = load_prompt("rag_api_parameters_list_en.txt")

DEFAULT_PROMPT = """You are a technical documentation expert. Answer based ONLY on the provided context.

FORMAT:
ANSWER: [clear, specific answer]
SOURCE: [document name, page X]

If not found: "NOT FOUND"
"""


def get_relevant_examples(question: str, max_examples: int = 3) -> list[dict[str, Any]]:
    """Pick few-shot rows whose questions overlap the query keywords."""
    examples = _few_shot_examples()
    if not examples:
        return []

    q_lower = question.lower()
    q_words = set(re.findall(r"\b\w{4,}\b", q_lower))

    scored_examples: list[tuple[float, dict[str, Any]]] = []
    for example in examples:
        ex_lower = example["question"].lower()
        ex_words = set(re.findall(r"\b\w{4,}\b", ex_lower))

        if q_words and ex_words:
            score = len(q_words & ex_words) / len(q_words)
        else:
            score = 0.0

        scored_examples.append((score, example))

    scored_examples.sort(key=lambda item: item[0], reverse=True)
    return [example for score, example in scored_examples[:max_examples] if score > 0]


def enhance_prompt_with_examples(base_prompt: str, question: str) -> str:
    """Append top matching few-shot examples to the base system prompt."""
    if not _few_shot_examples():
        return base_prompt

    relevant_examples = get_relevant_examples(question, max_examples=3)
    if not relevant_examples:
        return base_prompt

    examples_text = FewShotLoader.format_examples(relevant_examples)
    return base_prompt + examples_text


def select_prompt(question: str) -> tuple[str, int, float]:
    """Return system prompt text, suggested output budget, and temperature."""
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["list of parameters", "all parameters", "parameter list"]):
        base_prompt = RAG_API_PARAMETERS_LIST_PROMPT if RAG_API_PARAMETERS_LIST_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 1200, 0.05
    elif any(kw in q_lower for kw in ["parameter", "param", "field", "difference"]):
        base_prompt = RAG_API_PARAMETER_PROMPT if RAG_API_PARAMETER_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 800, 0.05
    else:
        base_prompt = RAG_API_PROMPT if RAG_API_PROMPT else DEFAULT_PROMPT
        num_predict, temperature = 1000, 0.1

    enhanced_prompt = enhance_prompt_with_examples(base_prompt, question)
    return enhanced_prompt, num_predict, temperature


class SmartPromptRouter:
    """Compatibility wrapper used by ``run.py`` and other entrypoints."""

    @staticmethod
    def format_examples(examples: list[dict[str, Any]], max_examples: int = 3) -> str:
        """Delegate to :meth:`FewShotLoader.format_examples`."""
        return FewShotLoader.format_examples(examples, max_examples=max_examples)

    @staticmethod
    def select(question: str) -> tuple[str, int, float]:
        """Delegate to :func:`select_prompt`."""
        return select_prompt(question)

    @staticmethod
    def get_examples_count() -> int:
        """Return how many few-shot rows are loaded (lazy on first call)."""
        return len(_few_shot_examples())

    @staticmethod
    def reload_examples() -> int:
        """Reload few-shot files from disk and return the new count."""
        global _few_shot_cache
        _few_shot_cache = FewShotLoader.load_examples()
        log.info("Reloaded few-shot examples count=%s", len(_few_shot_cache))
        return len(_few_shot_cache)

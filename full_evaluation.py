#!/usr/bin/env python3
"""Run full RAG evaluation against the golden Q&A set and local metric scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_VECTOR_STORE_DIR = ROOT_DIR / "data" / "vectors"
DEFAULT_GOLDEN_PATH = ROOT_DIR / "instructions" / "golden" / "golden_set.json"


def _apply_bootstrap_env() -> None:
    """Use the same local JSON store as load_graph_chunks.py before config import."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vector-store-dir",
        default=str(DEFAULT_VECTOR_STORE_DIR),
    )
    bootstrap, _ = parser.parse_known_args()
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["LOCAL_VECTOR_STORE_DIR"] = str(Path(bootstrap.vector_store_dir).resolve())
    # Default for fresh indexes; overridden in main() from chunks.json metadata.
    os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")


_apply_bootstrap_env()
sys.path.insert(0, str(ROOT_DIR))

from config import config
from core.database import db
from core.datasets import DatasetScanner, GoldenItem
from core.embedding_alignment import configure_embeddings_for_index
from core.embeddings import embedder
from core.evaluation_utils import (
    is_not_found_answer,
    is_unanswerable_expected,
    tokenize_overlap,
)
from core.reranker import reranker
from router.smart_router import select_prompt

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

log = logging.getLogger(__name__)

class GradeLevel(Enum):
    """Discrete grade buckets."""

    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"


@dataclass
class MetricScores:
    """Metric values in 0-10 range."""

    relevance: float = 0.0
    factuality: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    helpfulness: float = 0.0
    toxicity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to rounded dictionary."""
        return {key: round(value, 2) for key, value in self.__dict__.items()}


@dataclass
class EvaluationResult:
    """Single question evaluation record."""

    id: int
    question: str
    expected_answer: str
    answer: str
    sources: list[tuple[str, int]]
    selected_prompt: str
    scores: MetricScores
    final_score: float
    grade: GradeLevel
    tokens_used: int
    time_seconds: float
    gpt_time_seconds: float
    explanation: str = ""


class RagWithGpt:
    """Answer questions via local vector store retrieval and OpenAI completion."""

    def __init__(self) -> None:
        self.client: Any | None = None
        self.gpt_model = config.openai_model
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                log.info("openai client initialized model=%s", self.gpt_model)
            else:
                log.warning("OPENAI_API_KEY not found")

    def ask(self, question: str) -> dict[str, Any]:
        """Generate one answer using retrieval + GPT."""
        started_at = time.time()
        query_embedding = list(embedder.embed_cached(question))
        retrieved = db.search(query_embedding, query_text=question)
        if not retrieved:
            return {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "selected_prompt": "none",
                "time_total": round(time.time() - started_at, 2),
                "gpt_time": 0.0,
                "tokens": 0,
            }

        reranked = reranker.rerank(question, retrieved)
        context_parts: list[str] = []
        sources: list[tuple[str, int]] = []
        for chunk, source, page, _distance in reranked[: config.rerank_top_k]:
            context_parts.append(f"[{source}, p.{page}]\n{chunk[:800]}")
            sources.append((source, page))
        context = "\n\n".join(context_parts)

        system_prompt, _num_predict, temperature = select_prompt(question)
        if "parameter" in system_prompt.lower() and "list" not in system_prompt.lower():
            prompt_name = "API Parameter Prompt"
        elif "list of parameters" in system_prompt.lower():
            prompt_name = "API Parameters List Prompt"
        else:
            prompt_name = "API Info Prompt"

        answer = ""
        tokens = 0
        gpt_started_at = time.time()
        if self.client is not None:
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
                    ],
                    temperature=temperature,
                    max_tokens=config.openai_max_tokens,
                )
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
            except Exception as error:
                answer = f"ERROR: {error}"
                log.error("gpt request failed: %s", error)
        else:
            answer = "ERROR: OpenAI not available"

        gpt_time = time.time() - gpt_started_at
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "selected_prompt": prompt_name,
            "time_total": round(time.time() - started_at, 2),
            "gpt_time": round(gpt_time, 2),
            "tokens": tokens,
        }


class MetricsCalculator:
    """Compute metric values for an answer."""

    @staticmethod
    def calculate_relevance(question: str, answer: str) -> float:
        if not question or not answer or answer.startswith("ERROR") or is_not_found_answer(answer):
            return 0.0
        question_words = tokenize_overlap(question)
        answer_words = tokenize_overlap(answer)
        if not question_words:
            return 5.0
        return min(10.0, (len(question_words & answer_words) / len(question_words)) * 12)

    @staticmethod
    def calculate_factuality(answer: str) -> float:
        if not answer or answer.startswith("ERROR"):
            return 0.0
        uncertain_markers = ("probably", "maybe", "perhaps", "might", "could", "возможно", "вероятно", "наверное")
        sentences = re.split(r"[.!?]+", answer)
        uncertain = sum(1 for sentence in sentences if any(marker in sentence.lower() for marker in uncertain_markers))
        return min(10.0, max(0.0, 10 - uncertain * 2))

    @staticmethod
    def calculate_completeness(question: str, answer: str) -> float:
        if not answer or answer.startswith("ERROR") or is_not_found_answer(answer):
            return 0.0
        question_words = tokenize_overlap(question)
        answer_words = tokenize_overlap(answer)
        if not question_words:
            return 7.0
        return min(10.0, (len(question_words & answer_words) / len(question_words)) * 10)

    @staticmethod
    def calculate_coherence(answer: str) -> float:
        if not answer or answer.startswith("ERROR"):
            return 0.0

        # Структура
        has_numbers = bool(re.search(r"\d+\.", answer))
        has_bullets = bool(re.search(r"[-*•]", answer))
        has_paragraphs = answer.count("\n\n") > 0

        structure: float = 0
        if has_numbers:
            structure += 0.4
        if has_bullets:
            structure += 0.3
        if has_paragraphs:
            structure += 0.3

        # Логические связки
        connectors = [
            "поэтому",
            "следовательно",
            "во-первых",
            "например",
            "therefore",
            "thus",
            "consequently",
            "first",
            "for example",
        ]
        connector_count = sum(1 for c in connectors if c in answer.lower())
        logic = min(0.5, connector_count * 0.1)

        coherence = (structure + logic) * 10
        return min(10, coherence)

    @staticmethod
    def calculate_helpfulness(answer: str) -> float:
        if not answer or answer.startswith("ERROR") or is_not_found_answer(answer):
            return 0.0
        length_score = min(1.0, len(answer) / 500)
        has_instructions = any(
            word in answer.lower()
            for word in (
                "как",
                "следуйте",
                "выполните",
                "используйте",
                "подключ",
                "запуст",
                "how to",
                "follow",
                "use",
                "step",
            )
        )
        has_example = bool(re.search(r"(example|например|sample|пример|```)", answer.lower()))
        has_structure = bool(re.search(r"^\s*\d+\.", answer, re.MULTILINE))
        score = (
            length_score * 0.25
            + (0.35 if has_instructions else 0.0)
            + (0.2 if has_example else 0.0)
            + (0.2 if has_structure else 0.0)
        )
        return score * 10

    @staticmethod
    def calculate_toxicity(answer: str) -> float:
        toxic_words = ("дурак", "идиот", "урод", "stupid", "idiot")
        return min(10.0, sum(1 for word in toxic_words if word in answer.lower()) * 2)


def load_golden_cases(golden_path: Path) -> list[GoldenItem]:
    """Load evaluation cases from the golden dataset."""
    items = DatasetScanner(config.instructions_dir).load_golden_set(golden_path)
    if not items:
        raise FileNotFoundError(f"no golden cases found at {golden_path}")
    return items


def configure_logging(log_file: Path, is_verbose: bool) -> None:
    """Configure logger handlers for file and console."""
    level = logging.DEBUG if is_verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def evaluate_response(
    golden_item: GoldenItem,
    response: dict[str, Any],
    calculator: MetricsCalculator,
) -> EvaluationResult:
    """Build evaluation record from raw response."""
    question = golden_item.question
    answer = str(response["answer"])

    if is_unanswerable_expected(golden_item.expected_answer):
        if is_not_found_answer(answer):
            scores = MetricScores(
                relevance=10.0,
                factuality=10.0,
                completeness=10.0,
                coherence=10.0,
                helpfulness=10.0,
                toxicity=0.0,
            )
            final_score = 10.0
            grade = GradeLevel.EXCELLENT
            explanation = "correct not-found (docs have no answer)"
        else:
            scores = MetricScores(
                relevance=calculator.calculate_relevance(question, answer),
                factuality=2.0,
                completeness=calculator.calculate_completeness(question, answer),
                coherence=calculator.calculate_coherence(answer),
                helpfulness=calculator.calculate_helpfulness(answer),
                toxicity=calculator.calculate_toxicity(answer),
            )
            final_score = min(
                4.0,
                scores.relevance * 0.25
                + scores.factuality * 0.25
                + scores.completeness * 0.20
                + scores.coherence * 0.15
                + scores.helpfulness * 0.15,
            )
            grade = GradeLevel.POOR
            explanation = "expected not-found but model hallucinated an answer"
    else:
        scores = MetricScores(
            relevance=calculator.calculate_relevance(question, answer),
            factuality=calculator.calculate_factuality(answer),
            completeness=calculator.calculate_completeness(question, answer),
            coherence=calculator.calculate_coherence(answer),
            helpfulness=calculator.calculate_helpfulness(answer),
            toxicity=calculator.calculate_toxicity(answer),
        )
        final_score = (
            scores.relevance * 0.25
            + scores.factuality * 0.25
            + scores.completeness * 0.20
            + scores.coherence * 0.15
            + scores.helpfulness * 0.15
        )
        if is_not_found_answer(answer):
            final_score = min(final_score, 3.0)
        if scores.toxicity > 7:
            final_score *= 0.5
        grade = (
            GradeLevel.EXCELLENT
            if final_score >= 9
            else GradeLevel.GOOD
            if final_score >= 7
            else GradeLevel.SATISFACTORY
            if final_score >= 5
            else GradeLevel.POOR
        )
        explanation = f"{grade.value} quality"
    return EvaluationResult(
        id=golden_item.id,
        question=question,
        expected_answer=golden_item.expected_answer,
        answer=response["answer"],
        sources=response["sources"],
        selected_prompt=response["selected_prompt"],
        scores=scores,
        final_score=round(final_score, 2),
        grade=grade,
        tokens_used=response.get("tokens", 0),
        time_seconds=response["time_total"],
        gpt_time_seconds=response["gpt_time"],
        explanation=explanation,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Full RAG evaluation using instructions/golden/golden_set.json."
    )
    parser.add_argument(
        "--golden",
        "-g",
        default=str(DEFAULT_GOLDEN_PATH),
        help="Path to golden_set.json or instructions/golden/ directory.",
    )
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument(
        "--vector-store-dir",
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="Directory with chunks.json (same as load_graph_chunks.py --output-dir).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run full evaluation pipeline."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("logs") / f"full_evaluation_{timestamp}.log"
    log_file.parent.mkdir(exist_ok=True)
    configure_logging(log_file=log_file, is_verbose=args.verbose)
    vector_store_dir = Path(args.vector_store_dir).resolve()
    chunks_path = vector_store_dir / "chunks.json"
    db.reload_store()
    log.info("vector store backend=%s path=%s", db.backend_name, chunks_path)
    chunk_count = db.get_chunk_count()
    log.info("chunks loaded=%s", chunk_count)
    if chunk_count == 0:
        log.error("no chunks in %s, run load_graph_chunks.py first", chunks_path)
        return 1

    try:
        embedding_provider = configure_embeddings_for_index(vector_store_dir)
    except ValueError as error:
        log.error("%s", error)
        log.error(
            "re-index with one provider, e.g. "
            "EMBEDDING_PROVIDER=ollama python load_graph_chunks.py --force-recreate"
        )
        return 1
    log.info("query embeddings will use provider=%s", embedding_provider)

    golden_path = Path(args.golden).resolve()
    try:
        golden_cases = load_golden_cases(golden_path)
    except FileNotFoundError as error:
        log.error("%s", error)
        return 1

    log.info("golden cases loaded=%s path=%s", len(golden_cases), golden_path)

    rag = RagWithGpt()
    metrics = MetricsCalculator()

    results: list[EvaluationResult] = []
    for golden_item in golden_cases:
        response = rag.ask(golden_item.question)
        evaluation = evaluate_response(golden_item, response, metrics)
        results.append(evaluation)
        log.info(
            "qid=%s score=%.2f grade=%s time=%s tokens=%s",
            evaluation.id,
            evaluation.final_score,
            evaluation.grade.value,
            evaluation.time_seconds,
            evaluation.tokens_used,
        )

    total = len(results)
    avg_score = sum(item.final_score for item in results) / total
    avg_time = sum(item.time_seconds for item in results) / total
    avg_tokens = sum(item.tokens_used for item in results) / total
    excellent = sum(1 for item in results if item.grade == GradeLevel.EXCELLENT)
    good = sum(1 for item in results if item.grade == GradeLevel.GOOD)
    satisfactory = sum(1 for item in results if item.grade == GradeLevel.SATISFACTORY)
    poor = sum(1 for item in results if item.grade == GradeLevel.POOR)

    output_file = Path(args.output) if args.output else Path(f"evaluation_results_{timestamp}.json")
    output_payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": rag.gpt_model,
            "chunks": chunk_count,
            "questions": total,
            "golden_path": str(golden_path),
        },
        "statistics": {
            "avg_score": round(avg_score, 2),
            "avg_time": round(avg_time, 2),
            "avg_tokens": round(avg_tokens, 0),
            "excellent": excellent,
            "good": good,
            "satisfactory": satisfactory,
            "poor": poor,
        },
        "results": [
            {
                "id": item.id,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "answer": item.answer,
                "sources": [(source, page) for source, page in item.sources],
                "selected_prompt": item.selected_prompt,
                "scores": item.scores.to_dict(),
                "final_score": item.final_score,
                "grade": item.grade.value,
                "tokens": item.tokens_used,
                "time": item.time_seconds,
                "explanation": item.explanation,
            }
            for item in results
        ],
    }
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(output_payload, file_handle, ensure_ascii=False, indent=2)

    log.info("results saved to=%s", output_file)
    log.info("log file=%s", log_file)
    log.info("")
    log.info("Grade distribution:")
    log.info("  Excellent: %s (%.1f%%)", excellent, excellent / total * 100)
    log.info("  Good: %s (%.1f%%)", good, good / total * 100)
    log.info("  Satisfactory: %s (%.1f%%)", satisfactory, satisfactory / total * 100)
    log.info("  Poor: %s (%.1f%%)", poor, poor / total * 100)
    log.info(
        "summary total=%s avg_score=%.2f avg_time=%.2f avg_tokens=%.0f excellent=%s good=%s satisfactory=%s poor=%s",
        total,
        avg_score,
        avg_time,
        avg_tokens,
        excellent,
        good,
        satisfactory,
        poor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
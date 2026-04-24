#!/usr/bin/env python3
"""Run full RAG evaluation with OpenAI answers and local metric scoring."""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
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
    """Answer questions via ClickHouse retrieval and OpenAI completion."""

    def __init__(self) -> None:
        self.client: Any | None = None
        self.gpt_model = "gpt-4o-mini"
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
        query_embedding = list(embedder.generate_cached(question))
        retrieved = db.search(query_embedding)
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
                    max_tokens=800,
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
        if not question or not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0
        question_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
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
        if not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0
        question_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        if not question_words:
            return 7.0
        return min(10.0, (len(question_words & answer_words) / len(question_words)) * 10)

    @staticmethod
    def calculate_coherence(answer: str) -> float:
        if not answer or answer.startswith("ERROR"):
            return 0.0
        has_numbers = bool(re.search(r"\d+\.", answer))
        has_bullets = bool(re.search(r"[-*•]", answer))
        has_paragraphs = answer.count("\n\n") > 0
        structure = (0.4 if has_numbers else 0.0) + (0.3 if has_bullets else 0.0) + (0.3 if has_paragraphs else 0.0)
        connectors = ("поэтому", "следовательно", "во-первых", "например", "therefore", "thus", "consequently", "first")
        connector_count = sum(1 for connector in connectors if connector in answer.lower())
        return min(10.0, (structure + min(0.5, connector_count * 0.1)) * 10)

    @staticmethod
    def calculate_helpfulness(answer: str) -> float:
        if not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0
        length_score = min(1.0, len(answer) / 500)
        has_instructions = any(word in answer.lower() for word in ("как", "следуйте", "выполните", "используйте", "how to", "follow", "use"))
        has_example = bool(re.search(r"(example|например|sample|пример)", answer.lower()))
        score = length_score * 0.3 + (0.4 if has_instructions else 0.0) + (0.3 if has_example else 0.0)
        return score * 10

    @staticmethod
    def calculate_toxicity(answer: str) -> float:
        toxic_words = ("дурак", "идиот", "урод", "stupid", "idiot")
        return min(10.0, sum(1 for word in toxic_words if word in answer.lower()) * 2)


QUESTIONS = {
    3: "Create a step by step guide how to integrate sale form",
    9: "What is a Connecting Party?",
    13: "What is a merchant control key? Is it included in request?",
    16: "Do I need private key for v4/transfer?",
    19: "What is the difference between v2/sale and v2/sale-form?",
    22: "Should I implement both status and callback handling?",
    25: "How to calculate control parameter for v2/sale?",
    30: "How to make a reversal?",
    49: "What is the difference between RPI and card number?",
    52: "Do I need PCI for v2/sale?",
}


def configure_logging(log_file: Path, is_verbose: bool) -> None:
    """Configure logger handlers for file and console."""
    level = logging.DEBUG if is_verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def evaluate_response(question_id: int, question: str, response: dict[str, Any], calculator: MetricsCalculator) -> EvaluationResult:
    """Build evaluation record from raw response."""
    scores = MetricScores(
        relevance=calculator.calculate_relevance(question, response["answer"]),
        factuality=calculator.calculate_factuality(response["answer"]),
        completeness=calculator.calculate_completeness(question, response["answer"]),
        coherence=calculator.calculate_coherence(response["answer"]),
        helpfulness=calculator.calculate_helpfulness(response["answer"]),
        toxicity=calculator.calculate_toxicity(response["answer"]),
    )
    final_score = (
        scores.relevance * 0.25
        + scores.factuality * 0.25
        + scores.completeness * 0.20
        + scores.coherence * 0.15
        + scores.helpfulness * 0.15
    )
    if scores.toxicity > 7:
        final_score *= 0.5
    grade = GradeLevel.EXCELLENT if final_score >= 9 else GradeLevel.GOOD if final_score >= 7 else GradeLevel.SATISFACTORY if final_score >= 5 else GradeLevel.POOR
    return EvaluationResult(
        id=question_id,
        question=question,
        answer=response["answer"],
        sources=response["sources"],
        selected_prompt=response["selected_prompt"],
        scores=scores,
        final_score=round(final_score, 2),
        grade=grade,
        tokens_used=response.get("tokens", 0),
        time_seconds=response["time_total"],
        gpt_time_seconds=response["gpt_time"],
        explanation=f"{grade.value} quality",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Full RAG evaluation.")
    parser.add_argument("--questions", "-q", help="JSON file with questions")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run full evaluation pipeline."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("logs") / f"full_evaluation_{timestamp}.log"
    log_file.parent.mkdir(exist_ok=True)
    configure_logging(log_file=log_file, is_verbose=args.verbose)

    chunk_count = db.get_chunk_count()
    log.info("database chunks=%s", chunk_count)
    if chunk_count == 0:
        log.error("no chunks in database, run load_graph_chunks.py first")
        return 1

    rag = RagWithGpt()
    metrics = MetricsCalculator()
    questions_to_ask = QUESTIONS.copy()
    if args.questions and Path(args.questions).exists():
        with Path(args.questions).open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, list):
            questions_to_ask = {i + 1: item.get("question", str(item)) for i, item in enumerate(payload)}
        elif isinstance(payload, dict) and "questions" in payload:
            questions_to_ask = {i + 1: question for i, question in enumerate(payload["questions"])}

    results: list[EvaluationResult] = []
    for question_id, question in questions_to_ask.items():
        response = rag.ask(question)
        evaluation = evaluate_response(question_id, question, response, metrics)
        results.append(evaluation)
        log.info("qid=%s score=%.2f grade=%s time=%s tokens=%s", question_id, evaluation.final_score, evaluation.grade.value, evaluation.time_seconds, evaluation.tokens_used)

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
        "config": {"model": rag.gpt_model, "chunks": chunk_count, "questions": total},
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
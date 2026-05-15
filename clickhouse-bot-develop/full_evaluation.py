"""Full RAG evaluation: local vector retrieval, GPT answers, heuristic scoring, JSON export."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import config  # noqa: E402
from core.database import db  # noqa: E402
from core.embeddings import embedder  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from core.openai_client import get_openai_client  # noqa: E402
from core.reranker import reranker  # noqa: E402
from router.smart_router import select_prompt  # noqa: E402

log = logging.getLogger(__name__)

try:
    import openai as _openai  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

__all__ = [
    "AnswerEvaluator",
    "EvaluationResult",
    "GradeLevel",
    "MetricScores",
    "MetricsCalculator",
    "QUESTIONS",
    "RAGWithGPT",
    "main",
    "metrics",
]


class GradeLevel(Enum):
    """Discrete quality bucket for an evaluated answer."""

    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"


@dataclass(frozen=True)
class MetricScores:
    """Heuristic sub-scores on a 0–10 scale."""

    relevance: float = 0.0
    factuality: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    helpfulness: float = 0.0
    toxicity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize scores with two decimal places.

        Returns:
            Mapping of metric name to rounded score.
        """
        return {key: round(value, 2) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class EvaluationResult:
    """One evaluated question/answer row for export and logging."""

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


class RAGWithGPT:
    """Retrieve context from the local vector store, rerank, then answer with OpenAI GPT."""

    def __init__(self) -> None:
        """Resolve OpenAI availability; the client is lazily created on first call."""
        self.gpt_model = "gpt-4o-mini"
        self._available = OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))

        if not OPENAI_AVAILABLE:
            log.warning("OpenAI not installed; install the openai package to enable GPT answers.")
        elif not os.getenv("OPENAI_API_KEY"):
            log.warning("OPENAI_API_KEY not found")
        else:
            log.info("OpenAI available, model=%s", self.gpt_model)

    def ask(self, question: str) -> dict[str, Any]:
        """Run retrieval, reranking, prompt selection, and optional GPT completion.

        Args:
            question: User question text.

        Returns:
            Dict with keys ``question``, ``answer``, ``sources``, ``selected_prompt``,
            ``time_total``, ``gpt_time``, and ``tokens``.
        """
        t_start = time.time()

        log.debug("Generating embedding for: %s...", question[:50])
        q_emb = list(embedder.generate_cached(question))

        log.debug("Searching local vector store...")
        results = db.search(q_emb)

        if not results:
            return {
                "question": question,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "selected_prompt": "none",
                "time_total": round(time.time() - t_start, 2),
                "gpt_time": 0,
                "tokens": 0,
            }

        reranked = reranker.rerank(question, results)

        context_parts: list[str] = []
        sources: list[tuple[str, int]] = []
        for row in reranked[: config.rerank_top_k]:
            chunk, source, page = row[0], row[1], row[2]
            context_parts.append("[%s, p.%s]\n%s" % (source, page, chunk[:800]))
            sources.append((source, page))

        context = "\n\n".join(context_parts)
        log.debug("Context length: %s chars", len(context))

        system_prompt, _num_predict, temperature = select_prompt(question)

        if "parameter" in system_prompt.lower() and "list" not in system_prompt.lower():
            prompt_name = "API Parameter Prompt"
        elif "list of parameters" in system_prompt.lower():
            prompt_name = "API Parameters List Prompt"
        else:
            prompt_name = "API Info Prompt"

        log.info("Selected prompt: %s", prompt_name)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "CONTEXT:\n%s\n\nQUESTION: %s" % (context, question)},
        ]

        gpt_start = time.time()
        answer = ""
        tokens = 0

        if self._available:
            try:
                response = get_openai_client().chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800,
                )
                content = response.choices[0].message.content
                answer = content if content is not None else ""
                usage = response.usage
                tokens = usage.total_tokens if usage is not None else 0
                log.info(
                    "GPT response: %s tokens, %.2fs",
                    tokens,
                    time.time() - gpt_start,
                )
            except Exception as exc:
                answer = "ERROR: %s" % exc
                log.error("GPT error: %s", exc)
        else:
            log.error("OpenAI not available; set OPENAI_API_KEY")
            answer = "ERROR: OpenAI not available"

        gpt_time = time.time() - gpt_start
        total_time = time.time() - t_start

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "selected_prompt": prompt_name,
            "time_total": round(total_time, 2),
            "gpt_time": round(gpt_time, 2),
            "tokens": tokens,
        }


class MetricsCalculator:
    """Heuristic 0–10 metrics derived from question/answer text."""

    @staticmethod
    def calculate_relevance(question: str, answer: str) -> float:
        """Lexical overlap–based relevance (0–10)."""
        if not question or not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0

        q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        a_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))

        if not q_words:
            return 5.0

        intersection = len(q_words & a_words)
        similarity = intersection / len(q_words)
        return min(10.0, similarity * 12)

    @staticmethod
    def calculate_factuality(answer: str) -> float:
        """Penalize hedging language (0–10)."""
        if not answer or answer.startswith("ERROR"):
            return 0.0

        uncertain_markers = [
            "probably",
            "maybe",
            "perhaps",
            "might",
            "could",
            "возможно",
            "вероятно",
            "наверное",
        ]

        sentences = re.split(r"[.!?]+", answer)
        if not sentences:
            return 5.0

        uncertain_count = sum(
            1 for sentence in sentences if any(marker in sentence.lower() for marker in uncertain_markers)
        )

        factuality = max(0, 10 - uncertain_count * 2)
        return min(10, factuality)

    @staticmethod
    def calculate_completeness(question: str, answer: str) -> float:
        """Keyword coverage of the question in the answer (0–10)."""
        if not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0

        q_keywords = set(re.findall(r"\b\w{4,}\b", question.lower()))
        a_keywords = set(re.findall(r"\b\w{4,}\b", answer.lower()))

        if not q_keywords:
            return 7.0

        covered = len(q_keywords & a_keywords)
        completeness = (covered / len(q_keywords)) * 10
        return min(10, completeness)

    @staticmethod
    def calculate_coherence(answer: str) -> float:
        """Structure and connector–based coherence (0–10)."""
        if not answer or answer.startswith("ERROR"):
            return 0.0

        has_numbers = bool(re.search(r"\d+\.", answer))
        has_bullets = bool(re.search(r"[-*•]", answer))
        has_paragraphs = answer.count("\n\n") > 0

        structure = 0.0
        if has_numbers:
            structure += 0.4
        if has_bullets:
            structure += 0.3
        if has_paragraphs:
            structure += 0.3

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
        connector_count = sum(1 for connector in connectors if connector in answer.lower())
        logic = min(0.5, connector_count * 0.1)

        coherence = (structure + logic) * 10
        return min(10, coherence)

    @staticmethod
    def calculate_helpfulness(answer: str) -> float:
        """Length and instructional cues (0–10)."""
        if not answer or answer.startswith("ERROR"):
            return 0.0

        if answer == "NOT FOUND in documentation":
            return 0.0

        length_score = min(1.0, len(answer) / 500)

        has_instructions = any(
            word in answer.lower()
            for word in [
                "как",
                "следуйте",
                "выполните",
                "используйте",
                "how to",
                "follow",
                "use",
                "write",
            ]
        )

        has_example = bool(re.search(r"(example|например|sample|пример)", answer.lower()))

        helpfulness = length_score * 0.3
        if has_instructions:
            helpfulness += 0.4
        if has_example:
            helpfulness += 0.3

        return helpfulness * 10

    @staticmethod
    def calculate_toxicity(answer: str) -> float:
        """Simple lexicon-based toxicity score (0–10)."""
        toxic_words = ["дурак", "идиот", "урод", "stupid", "idiot"]
        toxic_count = sum(1 for word in toxic_words if word in answer.lower())
        return min(10, toxic_count * 2)


metrics = MetricsCalculator()


class AnswerEvaluator:
    """Combine sub-metrics into a final score and grade."""

    @staticmethod
    def evaluate(question: str, answer: str, time_seconds: float, tokens: int) -> EvaluationResult:
        """Score one answer using static heuristics.

        Args:
            question: Original question.
            answer: Model answer text.
            time_seconds: Wall-clock latency for the turn.
            tokens: Token usage when available.

        Returns:
            Populated :class:`EvaluationResult` (``id`` and ``sources`` are filled later).
        """
        scores = MetricScores(
            relevance=metrics.calculate_relevance(question, answer),
            factuality=metrics.calculate_factuality(answer),
            completeness=metrics.calculate_completeness(question, answer),
            coherence=metrics.calculate_coherence(answer),
            helpfulness=metrics.calculate_helpfulness(answer),
            toxicity=metrics.calculate_toxicity(answer),
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

        if final_score >= 9.0:
            grade = GradeLevel.EXCELLENT
        elif final_score >= 7.0:
            grade = GradeLevel.GOOD
        elif final_score >= 5.0:
            grade = GradeLevel.SATISFACTORY
        else:
            grade = GradeLevel.POOR

        return EvaluationResult(
            id=0,
            question=question,
            answer=answer,
            sources=[],
            selected_prompt="",
            scores=scores,
            final_score=round(final_score, 2),
            grade=grade,
            tokens_used=tokens,
            time_seconds=time_seconds,
            gpt_time_seconds=0,
            explanation="%s quality" % grade.value,
        )


QUESTIONS: dict[int, str] = {
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full evaluation runner.

    Returns:
        Parsed arguments including optional questions file and output path.
    """
    parser = argparse.ArgumentParser(description="Full RAG Evaluation")
    parser.add_argument(
        "--questions",
        "-q",
        type=str,
        help="JSON file with questions (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation loop, write JSON, and log a summary table."""
    arguments = parse_args()

    setup_logging(logging.DEBUG if arguments.verbose else logging.INFO)

    # Also emit to a timestamped file for this run.
    _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / ("full_evaluation_%s.log" % _ts)
    _fh = logging.FileHandler(log_file, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(_fh)

    db.set_active_table("hypothesis")
    log.info("[INFO] Using table: %s", db.load_active_table())
    log.info("[INFO] Chunks in database: %s", db.load_chunk_count())

    log.info("=" * 80)
    log.info("FULL RAG EVALUATION STARTED")
    log.info("Log file: %s", log_file)
    log.info("=" * 80)

    chunk_count = db.load_chunk_count()
    log.info("Database chunks: %s", chunk_count)

    if chunk_count == 0:
        log.error("No chunks in database! Run load_graph_chunks.py first")
        return

    rag = RAGWithGPT()
    evaluator = AnswerEvaluator()

    questions_to_ask: dict[int, str] = dict(QUESTIONS)

    if arguments.questions and os.path.exists(arguments.questions):
        with open(arguments.questions, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                questions_to_ask = {
                    index + 1: item.get("question", str(item)) for index, item in enumerate(data)
                }
            elif isinstance(data, dict) and "questions" in data:
                questions_to_ask = {index + 1: question for index, question in enumerate(data["questions"])}

    log.info("Processing %s questions", len(questions_to_ask))

    results: list[EvaluationResult] = []

    for qid, question in questions_to_ask.items():
        log.info("")
        log.info("=" * 60)
        log.info("Q%s: %s...", qid, question[:80])
        log.info("=" * 60)

        response = rag.ask(question)

        evaluation = evaluator.evaluate(
            question=question,
            answer=response["answer"],
            time_seconds=response["time_total"],
            tokens=response.get("tokens", 0),
        )
        evaluation = replace(
            evaluation,
            id=qid,
            sources=response["sources"],
            selected_prompt=response["selected_prompt"],
            gpt_time_seconds=response.get("gpt_time", 0),
        )

        results.append(evaluation)

        log.info("Answer: %s...", response["answer"][:200])
        log.info(
            "Scores: R=%.1f, F=%.1f, C=%.1f",
            evaluation.scores.relevance,
            evaluation.scores.factuality,
            evaluation.scores.completeness,
        )
        log.info("Final score: %.1f/10 (%s)", evaluation.final_score, evaluation.grade.value)
        log.info("Time: %ss, Tokens: %s", response["time_total"], response.get("tokens", 0))

    total = len(results)
    avg_score = sum(row.final_score for row in results) / total
    avg_time = sum(row.time_seconds for row in results) / total
    avg_tokens = sum(row.tokens_used for row in results) / total

    excellent = sum(1 for row in results if row.grade == GradeLevel.EXCELLENT)
    good = sum(1 for row in results if row.grade == GradeLevel.GOOD)
    satisfactory = sum(1 for row in results if row.grade == GradeLevel.SATISFACTORY)
    poor = sum(1 for row in results if row.grade == GradeLevel.POOR)

    log.info("")
    log.info("=" * 80)
    log.info("FINAL STATISTICS")
    log.info("=" * 80)
    log.info("Total questions: %s", total)
    log.info("Average score: %.1f/10", avg_score)
    log.info("Average time: %.1fs", avg_time)
    log.info("Average tokens: %.0f", avg_tokens)
    log.info("")
    log.info("Grade distribution:")
    log.info("  Excellent: %s (%.1f%%)", excellent, excellent / total * 100)
    log.info("  Good: %s (%.1f%%)", good, good / total * 100)
    log.info("  Satisfactory: %s (%.1f%%)", satisfactory, satisfactory / total * 100)
    log.info("  Poor: %s (%.1f%%)", poor, poor / total * 100)

    output_file = arguments.output if arguments.output else "evaluation_results_%s.json" % _ts

    output_data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": rag.gpt_model,
            "chunks": chunk_count,
            "questions": total,
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
                "id": row.id,
                "question": row.question,
                "answer": row.answer,
                "sources": [(source, page) for source, page in row.sources],
                "selected_prompt": row.selected_prompt,
                "scores": row.scores.to_dict(),
                "final_score": row.final_score,
                "grade": row.grade.value,
                "tokens": row.tokens_used,
                "time": row.time_seconds,
                "explanation": row.explanation,
            }
            for row in results
        ],
    }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, ensure_ascii=False, indent=2)

    log.info("")
    log.info("Results saved to: %s", output_file)
    log.info("Log saved to: %s", log_file)

    log.info("")
    log.info("=" * 80)
    log.info("RESULTS SUMMARY")
    log.info("=" * 80)
    log.info("%-4s %-8s %-14s %-8s %-8s", "ID", "SCORE", "GRADE", "TIME", "TOKENS")
    log.info("-" * 80)

    for row in results:
        grade_icon = (
            "Excellent"
            if row.grade == GradeLevel.EXCELLENT
            else "good"
            if row.grade == GradeLevel.GOOD
            else "SATISFACTORY"
            if row.grade == GradeLevel.SATISFACTORY
            else "POOR"
        )
        log.info(
            "%-4s %-8.1f %s %-12s %-8.1f %-8s",
            row.id,
            row.final_score,
            grade_icon,
            row.grade.value,
            row.time_seconds,
            row.tokens_used,
        )

    log.info("-" * 80)
    log.info("")
    log.info("Average: %.1f/10", avg_score)

    log.info("=" * 80)
    log.info("EVALUATION COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

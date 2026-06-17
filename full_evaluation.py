#!/usr/bin/env python3
"""Run full RAG evaluation against the golden Q&A set and local metric scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
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

from config import Config, config
from core.cost_guard import CostGuard
from core.database import db
from core.datasets import DatasetScanner, GoldenItem
from core.embedding_alignment import configure_embeddings_for_index
from core.embeddings import embedder
from core.evaluation_utils import (
    is_not_found_answer,
    is_unanswerable_expected,
    tokenize_overlap,
)
from core.llm.usage import CompletionUsage
from core.query_filter import filter_query
from core.rag_metrics import (
    estimate_tokens,
    faithfulness_score,
    hit_at_k,
    merge_usage_dicts,
)
from core.llm.chain import LLMChain
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
    llm_provider: str = ""
    llm_judge: dict[str, Any] | None = None
    rag_metrics: dict[str, Any] | None = None


class RagRunner:
    """Answer questions via local vector store retrieval and configured LLM chain."""

    def __init__(self, llm_provider: str | None = None) -> None:
        self._requested_provider = llm_provider
        if llm_provider:
            os.environ["LLM_PROVIDER"] = llm_provider
        if llm_provider == "ollama":
            os.environ["LLM_FALLBACK_ENABLED"] = "false"
        elif llm_provider == "openai":
            os.environ["LLM_FALLBACK_ENABLED"] = "false"
        runtime_config = Config.from_env()
        self._llm = LLMChain(runtime_config, CostGuard(runtime_config))
        self.llm_provider = llm_provider or self._llm._primary.name
        self.model_name = (
            runtime_config.ollama_model
            if self.llm_provider == "ollama"
            else runtime_config.openai_model
        )
        log.info("llm provider=%s model=%s", self.llm_provider, self.model_name)

    def ask(self, question: str) -> dict[str, Any]:
        """Generate one answer using retrieval + LLM."""
        started_at = time.time()
        filter_result = filter_query(question)
        filtered_question = filter_result.filtered if filter_result.filtered else question
        query_embedding = list(embedder.embed_cached(filtered_question))
        embed_tokens = estimate_tokens(
            filtered_question,
            provider="ollama" if self.llm_provider == "ollama" else "openai",
        )
        retrieval_started = time.time()
        retrieved = db.search(query_embedding)
        retrieval_time = time.time() - retrieval_started
        if not retrieved:
            return {
                "question": question,
                "filtered_question": filtered_question,
                "query_filter": filter_result.__dict__,
                "answer": "NOT FOUND in documentation",
                "sources": [],
                "context_chunks": [],
                "selected_prompt": "none",
                "time_total": round(time.time() - started_at, 2),
                "retrieval_time": round(retrieval_time, 2),
                "gpt_time": 0.0,
                "tokens": 0,
                "usage": merge_usage_dicts(
                    {"prompt_tokens": embed_tokens, "completion_tokens": 0, "cost_usd": 0.0}
                ),
            }

        reranked = reranker.rerank(question, retrieved)
        use_ollama = (self._requested_provider or self.llm_provider) == "ollama"
        chunk_limit = 350 if use_ollama else 800
        context_k = min(3, config.rerank_top_k) if use_ollama else config.rerank_top_k
        context_parts: list[str] = []
        context_chunks: list[str] = []
        sources: list[tuple[str, int]] = []
        for chunk, source, page, _distance in reranked[:context_k]:
            snippet = chunk[:chunk_limit]
            context_parts.append(f"[{source}, p.{page}]\n{snippet}")
            context_chunks.append(chunk)
            sources.append((source, page))
        context = "\n\n".join(context_parts)

        system_prompt, num_predict, temperature = select_prompt(filtered_question)
        if "parameter" in system_prompt.lower() and "list" not in system_prompt.lower():
            prompt_name = "API Parameter Prompt"
        elif "list of parameters" in system_prompt.lower():
            prompt_name = "API Parameters List Prompt"
        else:
            prompt_name = "API Info Prompt"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {filtered_question}"},
        ]
        answer = ""
        llm_usage = CompletionUsage()
        llm_started_at = time.time()
        max_tokens = min(num_predict, 600) if self.llm_provider == "ollama" else num_predict
        try:
            completion = self._llm.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if len(completion) == 3:
                answer, provider_used, llm_usage = completion
            else:
                answer, provider_used = completion
                llm_usage = CompletionUsage()
            self.llm_provider = provider_used
        except Exception as error:
            answer = f"ERROR: {error}"
            log.error("llm request failed: %s", error)
        if not (answer or "").strip() and self.llm_provider == "ollama":
            log.warning("ollama returned empty answer; try NUM_CTX=8192 or a larger model")

        llm_time = time.time() - llm_started_at
        total_usage = merge_usage_dicts(
            {
                "prompt_tokens": embed_tokens,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
            llm_usage.to_dict(),
        )
        return {
            "question": question,
            "filtered_question": filtered_question,
            "query_filter": filter_result.__dict__,
            "answer": answer,
            "sources": sources,
            "context_chunks": context_chunks,
            "selected_prompt": prompt_name,
            "time_total": round(time.time() - started_at, 2),
            "retrieval_time": round(retrieval_time, 2),
            "gpt_time": round(llm_time, 2),
            "tokens": total_usage["total_tokens"],
            "usage": total_usage,
            "llm_provider": self.llm_provider,
        }


# Backward-compatible alias
RagWithGpt = RagRunner


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

    context_chunks = list(response.get("context_chunks") or [])
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    rag_metrics = {
        "hit_at_3": round(
            hit_at_k(golden_item.expected_answer, context_chunks, k=3),
            3,
        ),
        "faithfulness": round(
            faithfulness_score(answer, context_chunks),
            3,
        ),
        "retrieval_time_seconds": response.get("retrieval_time", 0.0),
        "llm_time_seconds": response.get("gpt_time", 0.0),
        "latency_seconds": response.get("time_total", 0.0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", response.get("tokens", 0)) or 0),
        "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
    }
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
        tokens_used=int(rag_metrics["total_tokens"]),
        time_seconds=response["time_total"],
        gpt_time_seconds=response["gpt_time"],
        explanation=explanation,
        llm_provider=str(response.get("llm_provider", "")),
        rag_metrics=rag_metrics,
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
    parser.add_argument(
        "--llm-provider",
        choices=("ollama", "openai", "auto"),
        default="ollama",
        help="LLM for answer generation (default: ollama / llama3.2:3b).",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Score each answer with GPT judge (see --judge-model).",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM judge when --llm-judge is set (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("openai", "ollama"),
        default="openai",
        help="Backend for LLM judge (default: openai).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only first N golden questions (0 = all).",
    )
    return parser.parse_args()


def main() -> int:
    """Run full evaluation pipeline."""
    from dotenv import load_dotenv

    load_dotenv(".env.rag", override=True)
    os.environ["CACHE_ENABLED"] = "false"
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

    if args.limit and args.limit > 0:
        golden_cases = golden_cases[: args.limit]
    log.info("golden cases loaded=%s path=%s", len(golden_cases), golden_path)

    llm_provider = args.llm_provider
    if llm_provider == "ollama":
        try:
            import ollama

            ollama.list()
        except (RuntimeError, OSError, ValueError) as error:
            log.error("ollama is not running: %s", error)
            return 1

    rag = RagRunner(llm_provider=llm_provider)
    metrics = MetricsCalculator()
    llm_judge_fn = None
    judge_model = args.judge_model
    if args.llm_judge:
        if args.judge_backend == "openai":
            from llm_evaluate import evaluate_answer_percent_openai

            runtime_config = Config.from_env()
            api_key = runtime_config.openai_api_key
            if not api_key or api_key.strip() == "user_provided":
                log.error("OPENAI_API_KEY in .env.rag required for GPT judge")
                return 1
            llm_judge_fn = lambda q, a: evaluate_answer_percent_openai(q, a, model=judge_model)
            log.info("llm judge enabled backend=openai model=%s", judge_model)
        else:
            from llm_evaluate import evaluate_answer_percent

            llm_judge_fn = evaluate_answer_percent
            log.info("llm judge enabled backend=ollama model=%s", config.ollama_model)

    results: list[EvaluationResult] = []
    for golden_item in golden_cases:
        response = rag.ask(golden_item.question)
        evaluation = evaluate_response(golden_item, response, metrics)
        if llm_judge_fn is not None:
            evaluation.llm_provider = str(response.get("llm_provider", llm_provider))
            evaluation.llm_judge = llm_judge_fn(golden_item.question, str(response.get("answer", "")))
            if evaluation.rag_metrics and evaluation.llm_judge:
                evaluation.rag_metrics["judge_tokens"] = int(
                    evaluation.llm_judge.get("judge_tokens", 0) or 0
                )
                evaluation.rag_metrics["judge_cost_usd"] = float(
                    evaluation.llm_judge.get("judge_cost_usd", 0.0) or 0.0
                )
                evaluation.rag_metrics["cost_usd"] = round(
                    float(evaluation.rag_metrics.get("cost_usd", 0.0))
                    + evaluation.rag_metrics["judge_cost_usd"],
                    6,
                )
                evaluation.tokens_used = int(evaluation.rag_metrics.get("total_tokens", 0)) + int(
                    evaluation.rag_metrics["judge_tokens"]
                )
            log.info(
                "qid=%s llm_judge_total=%.1f",
                evaluation.id,
                float((evaluation.llm_judge or {}).get("total", 0)),
            )
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
    score_values = [item.final_score for item in results]
    avg_score = sum(score_values) / total
    avg_time = sum(item.time_seconds for item in results) / total
    avg_tokens = sum(item.tokens_used for item in results) / total
    rag_with_metrics = [item for item in results if item.rag_metrics]
    avg_hit_at_3 = (
        sum(float(item.rag_metrics["hit_at_3"]) for item in rag_with_metrics) / len(rag_with_metrics)
        if rag_with_metrics
        else 0.0
    )
    avg_faithfulness = (
        sum(float(item.rag_metrics["faithfulness"]) for item in rag_with_metrics)
        / len(rag_with_metrics)
        if rag_with_metrics
        else 0.0
    )
    avg_cost_usd = (
        sum(float(item.rag_metrics.get("cost_usd", 0.0)) for item in rag_with_metrics)
        / len(rag_with_metrics)
        if rag_with_metrics
        else 0.0
    )
    avg_retrieval_time = (
        sum(float(item.rag_metrics.get("retrieval_time_seconds", 0.0)) for item in rag_with_metrics)
        / len(rag_with_metrics)
        if rag_with_metrics
        else 0.0
    )
    excellent = sum(1 for item in results if item.grade == GradeLevel.EXCELLENT)
    good = sum(1 for item in results if item.grade == GradeLevel.GOOD)
    satisfactory = sum(1 for item in results if item.grade == GradeLevel.SATISFACTORY)
    poor = sum(1 for item in results if item.grade == GradeLevel.POOR)

    output_file = Path(args.output) if args.output else Path(f"evaluation_results_{timestamp}.json")
    output_payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": rag.model_name,
            "llm_provider": llm_provider,
            "llm_judge": args.llm_judge,
            "judge_backend": args.judge_backend if args.llm_judge else None,
            "judge_model": judge_model if args.llm_judge else None,
            "chunks": chunk_count,
            "questions": total,
            "golden_path": str(golden_path),
        },
        "statistics": {
            "avg_score": round(avg_score, 2),
            "std_score": round(statistics.stdev(score_values), 3) if total > 1 else 0.0,
            "variance_score": round(statistics.variance(score_values), 3) if total > 1 else 0.0,
            "avg_time": round(avg_time, 2),
            "avg_tokens": round(avg_tokens, 0),
            "excellent": excellent,
            "good": good,
            "satisfactory": satisfactory,
            "poor": poor,
            "avg_hit_at_3": round(avg_hit_at_3, 3),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_cost_usd": round(avg_cost_usd, 6),
            "avg_retrieval_time": round(avg_retrieval_time, 2),
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
                "llm_provider": item.llm_provider,
                "llm_judge": item.llm_judge,
                "rag_metrics": item.rag_metrics,
            }
            for item in results
        ],
    }
    if args.llm_judge:
        judge_totals = [
            float((item.llm_judge or {}).get("total", 0))
            for item in results
            if item.llm_judge and "error" not in item.llm_judge
        ]
        if judge_totals:
            output_payload["statistics"]["llm_judge_avg_percent"] = round(
                sum(judge_totals) / len(judge_totals), 1
            )
            if len(judge_totals) > 1:
                output_payload["statistics"]["llm_judge_std_percent"] = round(
                    statistics.stdev(judge_totals), 2
                )
                output_payload["statistics"]["llm_judge_variance_percent"] = round(
                    statistics.variance(judge_totals), 2
                )
            log.info(
                "llm judge avg=%.1f%% (%s %s)",
                output_payload["statistics"]["llm_judge_avg_percent"],
                args.judge_backend,
                judge_model if args.judge_backend == "openai" else config.ollama_model,
            )
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
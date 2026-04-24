#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНАЯ ОЦЕНКА RAG СИСТЕМЫ
- Использует чанки из ClickHouse
- Отвечает через GPT
- Оценивает качество ответов
- Сохраняет логи и результаты
"""

import os
import sys
import json
import time
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
from core.reranker import reranker
from router.smart_router import select_prompt
# В самом начале, после импортов
from core.database import db

# Переключаемся на правильную таблицу
db.set_active_table("hypothesis")
print(f"[INFO] Using table: {db.get_active_table()}")
print(f"[INFO] Chunks in database: {db.get_chunk_count()}")

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. Install with: pip install openai")

# ============================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"full_evaluation_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

class GradeLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"


@dataclass
class MetricScores:
    relevance: float = 0.0
    factuality: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    helpfulness: float = 0.0
    toxicity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class EvaluationResult:
    id: int
    question: str
    answer: str
    sources: List[Tuple[str, int]]
    selected_prompt: str
    scores: MetricScores
    final_score: float
    grade: GradeLevel
    tokens_used: int
    time_seconds: float
    gpt_time_seconds: float
    explanation: str = ""


# ============================================================
# RAG С GPT
# ============================================================

class RAGWithGPT:
    """RAG система с использованием GPT"""
    
    def __init__(self):
        self.client = None
        self.gpt_model = "gpt-4o-mini"
        
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized with model: {self.gpt_model}")
            else:
                logger.warning("OPENAI_API_KEY not found")
    
    def ask(self, question: str) -> Dict:
        """Отвечает на вопрос используя RAG + GPT"""
        
        t_start = time.time()
        
        # 1. Поиск в ClickHouse
        logger.debug(f"Generating embedding for: {question[:50]}...")
        q_emb = list(embedder.generate_cached(question))
        
        logger.debug("Searching in ClickHouse...")
        results = db.search(q_emb)
        
        if not results:
            return {
                'question': question,
                'answer': "NOT FOUND in documentation",
                'sources': [],
                'selected_prompt': 'none',
                'time_total': round(time.time() - t_start, 2),
                'gpt_time': 0,
                'tokens': 0
            }
        
        # 2. Реранжинг
        reranked = reranker.rerank(question, results)
        
        # 3. Контекст
        context_parts = []
        sources = []
        for r in reranked[:config.rerank_top_k]:
            chunk, source, page = r[0], r[1], r[2]
            context_parts.append(f"[{source}, p.{page}]\n{chunk[:800]}")
            sources.append((source, page))
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Context length: {len(context)} chars")
        
        # 4. Выбор промпта
        system_prompt, num_predict, temperature = select_prompt(question)
        
        # Определяем тип промпта
        if "parameter" in system_prompt.lower() and "list" not in system_prompt.lower():
            prompt_name = "API Parameter Prompt"
        elif "list of parameters" in system_prompt.lower():
            prompt_name = "API Parameters List Prompt"
        else:
            prompt_name = "API Info Prompt"
        
        logger.info(f"Selected prompt: {prompt_name}")
        
        # 5. Запрос к GPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
        
        gpt_start = time.time()
        answer = ""
        tokens = 0
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800
                )
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
                logger.info(f"GPT response: {tokens} tokens, {time.time() - gpt_start:.2f}s")
            except Exception as e:
                answer = f"ERROR: {e}"
                logger.error(f"GPT error: {e}")
        else:
            answer = "ERROR: OpenAI not available"
        
        gpt_time = time.time() - gpt_start
        total_time = time.time() - t_start
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'selected_prompt': prompt_name,
            'time_total': round(total_time, 2),
            'gpt_time': round(gpt_time, 2),
            'tokens': tokens
        }


# ============================================================
# КАЛЬКУЛЯТОР МЕТРИК
# ============================================================

class MetricsCalculator:
    """Вычисляет метрики качества ответа"""
    
    @staticmethod
    def calculate_relevance(question: str, answer: str) -> float:
        """Релевантность (0-10)"""
        if not question or not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0
        
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        a_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not q_words:
            return 5.0
        
        intersection = len(q_words & a_words)
        similarity = intersection / len(q_words)
        return min(10.0, similarity * 12)
    
    @staticmethod
    def calculate_factuality(answer: str) -> float:
        """Фактологичность (0-10)"""
        if not answer or answer.startswith("ERROR"):
            return 0.0
        
        uncertain_markers = ['probably', 'maybe', 'perhaps', 'might', 'could',
                            'возможно', 'вероятно', 'наверное']
        
        sentences = re.split(r'[.!?]+', answer)
        if not sentences:
            return 5.0
        
        uncertain_count = sum(1 for s in sentences 
                             if any(m in s.lower() for m in uncertain_markers))
        
        factuality = max(0, 10 - uncertain_count * 2)
        return min(10, factuality)
    
    @staticmethod
    def calculate_completeness(question: str, answer: str) -> float:
        """Полнота (0-10)"""
        if not answer or answer.startswith("ERROR") or answer == "NOT FOUND in documentation":
            return 0.0
        
        q_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
        a_keywords = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not q_keywords:
            return 7.0
        
        covered = len(q_keywords & a_keywords)
        completeness = (covered / len(q_keywords)) * 10
        return min(10, completeness)
    
    @staticmethod
    def calculate_coherence(answer: str) -> float:
        """Связность (0-10)"""
        if not answer or answer.startswith("ERROR"):
            return 0.0
        
        # Структура
        has_numbers = bool(re.search(r'\d+\.', answer))
        has_bullets = bool(re.search(r'[-*•]', answer))
        has_paragraphs = answer.count('\n\n') > 0
        
        structure = 0
        if has_numbers:
            structure += 0.4
        if has_bullets:
            structure += 0.3
        if has_paragraphs:
            structure += 0.3
        
        # Логические связки
        connectors = ['поэтому', 'следовательно', 'во-первых', 'например',
                     'therefore', 'thus', 'consequently', 'first', 'for example']
        connector_count = sum(1 for c in connectors if c in answer.lower())
        logic = min(0.5, connector_count * 0.1)
        
        coherence = (structure + logic) * 10
        return min(10, coherence)
    
    @staticmethod
    def calculate_helpfulness(answer: str) -> float:
        """Полезность (0-10)"""
        if not answer or answer.startswith("ERROR"):
            return 0.0
        
        if answer == "NOT FOUND in documentation":
            return 0.0
        
        # Длина ответа
        length_score = min(1.0, len(answer) / 500)
        
        # Наличие инструкций
        has_instructions = any(w in answer.lower() for w in 
                              ['как', 'следуйте', 'выполните', 'используйте',
                               'how to', 'follow', 'use', 'write'])
        
        # Наличие примеров
        has_example = bool(re.search(r'(example|например|sample|пример)', answer.lower()))
        
        helpfulness = length_score * 0.3
        if has_instructions:
            helpfulness += 0.4
        if has_example:
            helpfulness += 0.3
        
        return helpfulness * 10
    
    @staticmethod
    def calculate_toxicity(answer: str) -> float:
        """Токсичность (0-10)"""
        toxic_words = ['дурак', 'идиот', 'урод', 'stupid', 'idiot']
        toxic_count = sum(1 for w in toxic_words if w in answer.lower())
        return min(10, toxic_count * 2)


metrics = MetricsCalculator()


# ============================================================
# ОЦЕНЩИК
# ============================================================

class AnswerEvaluator:
    """Оценивает ответы по метрикам"""
    
    @staticmethod
    def evaluate(question: str, answer: str, time_seconds: float, tokens: int) -> EvaluationResult:
        """Оценивает один ответ"""
        
        # Вычисляем метрики
        scores = MetricScores(
            relevance=metrics.calculate_relevance(question, answer),
            factuality=metrics.calculate_factuality(answer),
            completeness=metrics.calculate_completeness(question, answer),
            coherence=metrics.calculate_coherence(answer),
            helpfulness=metrics.calculate_helpfulness(answer),
            toxicity=metrics.calculate_toxicity(answer)
        )
        
        # Итоговая оценка
        final_score = (
            scores.relevance * 0.25 +
            scores.factuality * 0.25 +
            scores.completeness * 0.20 +
            scores.coherence * 0.15 +
            scores.helpfulness * 0.15
        )
        
        # Штраф за токсичность
        if scores.toxicity > 7:
            final_score *= 0.5
        
        # Грейд
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
            explanation=f"{grade.value} quality"
        )


# ============================================================
# ОСНОВНЫЕ ВОПРОСЫ
# ============================================================

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


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Full RAG Evaluation')
    parser.add_argument('--questions', '-q', type=str, help='JSON file with questions (optional)')
    parser.add_argument('--output', '-o', type=str, help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FULL RAG EVALUATION STARTED")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Проверка БД
    chunk_count = db.get_chunk_count()
    logger.info(f"Database chunks: {chunk_count}")
    
    if chunk_count == 0:
        logger.error("No chunks in database! Run load_graph_chunks.py first")
        return
    
    # Инициализация RAG
    rag = RAGWithGPT()
    evaluator = AnswerEvaluator()
    
    # Загрузка вопросов
    questions_to_ask = QUESTIONS
    
    if args.questions and os.path.exists(args.questions):
        with open(args.questions, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions_to_ask = {i+1: item.get('question', str(item)) for i, item in enumerate(data)}
            elif isinstance(data, dict) and 'questions' in data:
                questions_to_ask = {i+1: q for i, q in enumerate(data['questions'])}
    
    logger.info(f"Processing {len(questions_to_ask)} questions")
    
    # Обработка
    results = []
    
    for qid, question in questions_to_ask.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Q{qid}: {question[:80]}...")
        logger.info(f"{'='*60}")
        
        # Получаем ответ от RAG
        response = rag.ask(question)
        
        # Оцениваем
        evaluation = evaluator.evaluate(
            question=question,
            answer=response['answer'],
            time_seconds=response['time_total'],
            tokens=response.get('tokens', 0)
        )
        evaluation.id = qid
        evaluation.sources = response['sources']
        evaluation.selected_prompt = response['selected_prompt']
        
        results.append(evaluation)
        
        # Логируем
        logger.info(f"Answer: {response['answer'][:200]}...")
        logger.info(f"Scores: R={evaluation.scores.relevance:.1f}, F={evaluation.scores.factuality:.1f}, C={evaluation.scores.completeness:.1f}")
        logger.info(f"Final score: {evaluation.final_score:.1f}/10 ({evaluation.grade.value})")
        logger.info(f"Time: {response['time_total']}s, Tokens: {response.get('tokens', 0)}")
    
    # Статистика
    total = len(results)
    avg_score = sum(r.final_score for r in results) / total
    avg_time = sum(r.time_seconds for r in results) / total
    avg_tokens = sum(r.tokens_used for r in results) / total
    
    excellent = sum(1 for r in results if r.grade == GradeLevel.EXCELLENT)
    good = sum(1 for r in results if r.grade == GradeLevel.GOOD)
    satisfactory = sum(1 for r in results if r.grade == GradeLevel.SATISFACTORY)
    poor = sum(1 for r in results if r.grade == GradeLevel.POOR)
    
    logger.info("\n" + "="*80)
    logger.info("FINAL STATISTICS")
    logger.info("="*80)
    logger.info(f"Total questions: {total}")
    logger.info(f"Average score: {avg_score:.1f}/10")
    logger.info(f"Average time: {avg_time:.1f}s")
    logger.info(f"Average tokens: {avg_tokens:.0f}")
    logger.info(f"")
    logger.info(f"Grade distribution:")
    logger.info(f"  Excellent: {excellent} ({excellent/total*100:.1f}%)")
    logger.info(f"  Good: {good} ({good/total*100:.1f}%)")
    logger.info(f"  Satisfactory: {satisfactory} ({satisfactory/total*100:.1f}%)")
    logger.info(f"  Poor: {poor} ({poor/total*100:.1f}%)")
    
    # Сохраняем результаты
    output_file = args.output if args.output else f"evaluation_results_{timestamp}.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': rag.gpt_model,
            'chunks': chunk_count,
            'questions': total
        },
        'statistics': {
            'avg_score': round(avg_score, 2),
            'avg_time': round(avg_time, 2),
            'avg_tokens': round(avg_tokens, 0),
            'excellent': excellent,
            'good': good,
            'satisfactory': satisfactory,
            'poor': poor
        },
        'results': [
            {
                'id': r.id,
                'question': r.question,
                'answer': r.answer,
                'sources': [(s[0], s[1]) for s in r.sources],
                'selected_prompt': r.selected_prompt,
                'scores': r.scores.to_dict(),
                'final_score': r.final_score,
                'grade': r.grade.value,
                'tokens': r.tokens_used,
                'time': r.time_seconds,
                'explanation': r.explanation
            }
            for r in results
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n Results saved to: {output_file}")
    logger.info(f" Log saved to: {log_file}")
    
    # Вывод таблицы
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'ID':<4} {'SCORE':<8} {'GRADE':<14} {'TIME':<8} {'TOKENS':<8}")
    print("-"*80)
    
    for r in results:
        grade_icon = "Excellent" if r.grade == GradeLevel.EXCELLENT else "good" if r.grade == GradeLevel.GOOD else "SATISFACTORY" if r.grade == GradeLevel.SATISFACTORY else "POOR"
        print(f"{r.id:<4} {r.final_score:<8.1f} {grade_icon} {r.grade.value:<12} {r.time_seconds:<8.1f} {r.tokens_used:<8}")
    
    print("-"*80)
    print(f"\nAverage: {avg_score:.1f}/10")
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()
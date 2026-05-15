#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer benchmark questions, assess quality, and optionally audit structured logs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.logger import init_logger, setup_logging
from rag_engine.engine import rag

log = logging.getLogger(__name__)

# ============================================================================
# ВОПРОСЫ ДЛЯ ОТВЕТА
# ============================================================================

QUESTIONS = {
    1: {
        "text": "Create a step by step guide how to integrate sale form",
        "type": "API flow",
        "expected_keywords": ["sale-form", "POST", "redirect", "callback", "status"]
    },
    2: {
        "text": "What is a Connecting Party?",
        "type": "concept",
        "expected_keywords": ["merchant", "client", "integrator", "API caller"]
    },
    3: {
        "text": "What is a merchant control key? Is it included in request?",
        "type": "parameter",
        "expected_keywords": ["control", "signature", "parameter", "request", "security"]
    },
    4: {
        "text": "Do I need private key for v4/transfer?",
        "type": "yes/no",
        "expected_keywords": ["private key", "RSA", "signature", "OAuth", "yes", "no"]
    },
    5: {
        "text": "What is the difference between v2/sale and v2/sale-form?",
        "type": "comparison",
        "expected_keywords": ["server-to-server", "form", "PCI", "card data", "redirect"]
    },
    6: {
        "text": "Should I implement both status and callback handling?",
        "type": "integration",
        "expected_keywords": ["callback", "status", "async", "polling", "recommended"]
    },
    7: {
        "text": "How to calculate control parameter for v2/sale?",
        "type": "parameter",
        "expected_keywords": ["SHA1", "signature", "concatenate", "login", "amount"]
    },
    8: {
        "text": "How to make a reversal?",
        "type": "API flow",
        "expected_keywords": ["return", "cancel", "reversal", "refund", "transaction"]
    },
    9: {
        "text": "What is the difference between RPI and card number?",
        "type": "concept",
        "expected_keywords": ["RPI", "recurring", "token", "PAN", "reference"]
    },
    10: {
        "text": "Do I need PCI for v2/sale?",
        "type": "yes/no",
        "expected_keywords": ["PCI", "DSS", "certification", "cardholder data", "yes", "no"]
    }
}

# ============================================================================
# ОЦЕНКА КАЧЕСТВА ОТВЕТА
# ============================================================================


class AnswerQualityAssessor:
    """Оценивает качество ответа"""

    @staticmethod
    def check_keywords(answer: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
        """Проверяет наличие ключевых слов в ответе"""
        answer_lower = answer.lower()
        found = []
        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                found.append(kw)

        score = len(found) / len(expected_keywords) if expected_keywords else 1.0
        return score, found

    @staticmethod
    def check_length(answer: str) -> float:
        """Оценивает полноту ответа по длине"""
        length = len(answer)
        if length < 100:
            return 0.2
        elif length < 300:
            return 0.5
        elif length < 600:
            return 0.7
        else:
            return 1.0

    @staticmethod
    def check_structure(answer: str, qtype: str) -> float:
        """Проверяет структуру ответа в зависимости от типа"""
        score = 0.0

        if qtype == "API flow":
            if re.search(r'\d+\.', answer):
                score += 0.3
            if re.search(r'(POST|GET|PUT|DELETE)', answer, re.IGNORECASE):
                score += 0.3
            if re.search(r'(step|stage|phase)', answer, re.IGNORECASE):
                score += 0.2
            if re.search(r'(callback|redirect|response)', answer, re.IGNORECASE):
                score += 0.2

        elif qtype == "comparison":
            if re.search(r'(vs|versus|difference)', answer, re.IGNORECASE):
                score += 0.3
            if 'first' in answer.lower() and 'second' in answer.lower():
                score += 0.3
            if re.search(r'[0-9]+\.', answer):
                score += 0.2
            if len(answer.split()) > 100:
                score += 0.2

        elif qtype == "yes/no":
            if re.search(r'\b(yes|no)\b', answer, re.IGNORECASE):
                score += 0.4
            if re.search(r'(because|since|due to|reason)', answer, re.IGNORECASE):
                score += 0.3
            if len(answer) > 200:
                score += 0.3

        elif qtype == "concept":
            if re.search(r'(definition|means|refers to|is a)', answer, re.IGNORECASE):
                score += 0.3
            if len(answer) > 200:
                score += 0.3
            if re.search(r'(example|for instance|e\.g\.)', answer, re.IGNORECASE):
                score += 0.2
            if re.search(r'(role|responsibility|function)', answer, re.IGNORECASE):
                score += 0.2

        elif qtype == "parameter":
            if re.search(r'(required|optional|mandatory)', answer, re.IGNORECASE):
                score += 0.3
            if re.search(r'(type|string|integer|boolean)', answer, re.IGNORECASE):
                score += 0.3
            if re.search(r'(example|sample|format)', answer, re.IGNORECASE):
                score += 0.2
            if len(answer) > 200:
                score += 0.2

        elif qtype == "integration":
            if re.search(r'(recommend|should|must|need)', answer, re.IGNORECASE):
                score += 0.3
            if re.search(r'(both|together|combined)', answer, re.IGNORECASE):
                score += 0.3
            if len(answer) > 200:
                score += 0.2
            if re.search(r'(reason|because|why)', answer, re.IGNORECASE):
                score += 0.2

        return min(1.0, score)

    @staticmethod
    def has_sources(answer: str) -> bool:
        """Проверяет наличие источников"""
        return 'source' in answer.lower() or 'document' in answer.lower() or 'page' in answer.lower()

    @staticmethod
    def assess(question: str, answer: str, qtype: str, expected_keywords: list[str]) -> dict[str, Any]:
        """Полная оценка ответа"""

        # 1. Оценка по ключевым словам
        kw_score, found_keywords = AnswerQualityAssessor.check_keywords(answer, expected_keywords)

        # 2. Оценка по длине
        length_score = AnswerQualityAssessor.check_length(answer)

        # 3. Оценка по структуре
        structure_score = AnswerQualityAssessor.check_structure(answer, qtype)

        # 4. Наличие источников
        has_sources = AnswerQualityAssessor.has_sources(answer)
        sources_score = 0.2 if has_sources else 0.0

        # Итоговая оценка (веса)
        total_score = (
            kw_score * 0.35 +
            length_score * 0.15 +
            structure_score * 0.35 +
            sources_score * 0.15
        )

        # Оценка в звёздах
        stars = '+' * int(total_score * 5)

        return {
            'total_score': round(total_score, 2),
            'stars': stars,
            'keyword_score': round(kw_score, 2),
            'length_score': round(length_score, 2),
            'structure_score': round(structure_score, 2),
            'sources_score': round(sources_score, 2),
            'found_keywords': found_keywords,
            'missing_keywords': [kw for kw in expected_keywords if kw not in found_keywords],
            'has_sources': has_sources
        }


# ============================================================================
# АУДИТ ЛОГОВ
# ============================================================================


class LogAuditor:
    """Анализирует логи после выполнения"""

    def __init__(self, log_file: Path | None = None) -> None:
        self.log_file = log_file
        self.events: list[dict[str, str]] = []

    def find_latest_log(self) -> Path | None:
        """Находит последний файл лога"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return None

        log_files = list(log_dir.glob("run-*.log"))
        if not log_files:
            return None

        latest = max(log_files, key=lambda f: f.stat().st_mtime)
        return latest

    def parse_log(self, log_path: Path) -> list[dict[str, str]]:
        """Парсит лог-файл"""
        events: list[dict[str, str]] = []

        # Пробуем разные кодировки
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                attempt: list[dict[str, str]] = []
                with open(log_path, 'r', encoding=encoding) as handle:
                    for line in handle:
                        match = re.match(
                            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([^\]]+)\] \[([^\]]+)\] (.+)',
                            line.strip(),
                        )
                        if match:
                            attempt.append({
                                'timestamp': match.group(1),
                                'model_tag': match.group(2),
                                'stage': match.group(3),
                                'message': match.group(4)
                            })
                events = attempt
                break
            except UnicodeDecodeError:
                log.debug("Log decode failed with encoding=%s path=%s", encoding, log_path)
                continue
            except OSError as exc:
                log.error("Failed to read log path=%s: %s", log_path, exc)
                raise
        else:
            log.warning("Could not decode log with any known encoding path=%s", log_path)
            events = []

        return events

    def analyze(self) -> dict[str, Any]:
        """Анализирует логи"""

        # Находим последний лог
        if not self.log_file:
            self.log_file = self.find_latest_log()

        if not self.log_file:
            return {"error": "No log files found"}

        # Парсим
        self.events = self.parse_log(self.log_file)

        # Статистика
        stages: dict[str, int] = {}
        embed_calls: list[float] = []
        llm_calls: list[float] = []
        slow_operations: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        stats: dict[str, Any] = {
            "log_file": str(self.log_file),
            "total_events": len(self.events),
            "stages": stages,
            "embed_calls": embed_calls,
            "llm_calls": llm_calls,
            "slow_operations": slow_operations,
            "errors": errors,
        }

        for event in self.events:
            # Стадии
            stage = event["stage"]
            stages[stage] = stages.get(stage, 0) + 1

            # Embedding вызовы
            if "embed.call" in event["message"]:
                match = re.search(r"latency_ms=([\d\.]+)", event["message"])
                if match:
                    embed_calls.append(float(match.group(1)))

            # LLM вызовы
            if "llm.generate" in event["message"]:
                match = re.search(r"latency_ms=([\d\.]+)", event["message"])
                if match:
                    llm_calls.append(float(match.group(1)))

            # Медленные операции
            if ".slow" in event["message"]:
                slow_operations.append(event)

            # Ошибки
            if "error" in event["message"].lower() or "ERROR" in event["message"]:
                errors.append(event)

        # Агрегация
        if embed_calls:
            stats["avg_embed_latency"] = sum(embed_calls) / len(embed_calls)
            stats["max_embed_latency"] = max(embed_calls)
        else:
            stats["avg_embed_latency"] = 0

        if llm_calls:
            stats["avg_llm_latency"] = sum(llm_calls) / len(llm_calls)
            stats["max_llm_latency"] = max(llm_calls)
        else:
            stats["avg_llm_latency"] = 0

        return stats


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================


def answer_question(qid: int, question_data: dict[str, Any]) -> dict[str, Any]:
    """Отвечает на один вопрос и оценивает ответ"""

    log.info("%s", "\n" + "=" * 80)
    log.info("QUESTION #%s: %s", qid, question_data['text'])
    log.info("Type: %s", question_data['type'])
    log.info("%s", "=" * 80)

    start_time = time.time()
    result = rag.ask(question_data['text'])
    elapsed = time.time() - start_time

    # Оцениваем качество
    assessment = AnswerQualityAssessor.assess(
        question=question_data['text'],
        answer=result['answer'],
        qtype=question_data['type'],
        expected_keywords=question_data['expected_keywords']
    )

    # Выводим ответ
    log.info("%s", "\n ANSWER:")
    log.info("%s", result['answer'])
    log.info("\n SOURCES: %s", result['sources'])

    log.info("%s", "\n QUALITY ASSESSMENT:")
    log.info(
        "   Total score: %.2f / 1.00 %s",
        assessment['total_score'],
        assessment['stars'],
    )
    log.info("   Keyword match: %.2f", assessment['keyword_score'])
    log.info("   Length adequacy: %.2f", assessment['length_score'])
    log.info("   Structure: %.2f", assessment['structure_score'])
    log.info("   Sources: %s", 'Yes' if assessment['has_sources'] else 'No')

    if assessment['found_keywords']:
        log.info("   Found keywords: %s", ', '.join(assessment['found_keywords']))
    if assessment['missing_keywords']:
        log.info("   Missing keywords: %s", ', '.join(assessment['missing_keywords']))

    return {
        'id': qid,
        'question': question_data['text'],
        'type': question_data['type'],
        'answer': result['answer'],
        'sources': result['sources'],
        'time_seconds': elapsed,
        'assessment': assessment
    }


def run_full_audit() -> list[dict[str, Any]] | None:
    """Запускает полный аудит: ответы на вопросы + анализ логов"""

    log.info(
        "%s",
        """
        ============================================================================
                ANSWER QUESTIONS + QUALITY ASSESSMENT + LOG AUDIT
        ============================================================================

        Отвечает на 10 вопросов, оценивает качество, анализирует логи
        ============================================================================
        """,
    )
    # Проверка базы данных
    chunk_count = db.get_chunk_count()
    log.info("[CHECK] Database has %s chunks", chunk_count)

    if chunk_count == 0:
        log.error("No chunks found; run hypothesis_loader.py first")
        return None

    # Проверка Ollama
    try:
        import ollama  # type: ignore[import-not-found]

        ollama.list()
        log.info("[OK] Ollama is running")
    except ImportError as exc:
        log.error("Ollama Python package not available: %s", exc)
        return None
    except OSError as exc:
        log.error("Ollama connection failed: %s", exc)
        return None
    except Exception as exc:
        log.error("Ollama is not running or not reachable: %s", exc)
        return None

    # Отвечаем на все вопросы
    results = []
    for qid, qdata in QUESTIONS.items():
        result = answer_question(qid, qdata)
        results.append(result)
        log.debug("Pausing between questions (rate limit) qid=%s", qid)
        time.sleep(0.5)  # Небольшая задержка между вопросами

    # Сохраняем результаты
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_audit")
    output_dir.mkdir(exist_ok=True)

    # JSON
    json_path = output_dir / f"answers_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    log.info("[SAVE] JSON: %s", json_path)

    # TXT отчёт
    txt_path = output_dir / f"answers_{timestamp}.txt"
    gen_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with open(txt_path, 'w', encoding='utf-8') as handle:
        handle.write("="*80 + "\n")
        handle.write("ANSWERS WITH QUALITY ASSESSMENT\n")
        handle.write(f"Generated: {gen_time}\n")
        handle.write("="*80 + "\n\n")

        for r in results:
            handle.write(f"\n{'='*80}\n")
            handle.write(f"QUESTION #{r['id']}: {r['question']}\n")
            handle.write(f"Type: {r['type']}\n")
            handle.write(f"{'='*80}\n")
            handle.write(f"\nANSWER:\n{r['answer']}\n")
            handle.write(f"\nSOURCES: {r['sources']}\n")
            handle.write(f"TIME: {r['time_seconds']}s\n")
            handle.write("\nQUALITY ASSESSMENT:\n")
            handle.write(f"  Total score: {r['assessment']['total_score']:.2f} {r['assessment']['stars']}\n")
            handle.write(f"  Keyword match: {r['assessment']['keyword_score']:.2f}\n")
            handle.write(f"  Found keywords: {', '.join(r['assessment']['found_keywords'])}\n")
            if r['assessment']['missing_keywords']:
                handle.write(f"  Missing keywords: {', '.join(r['assessment']['missing_keywords'])}\n")
            handle.write("\n---\n")

    log.info("[SAVE] TXT: %s", txt_path)

    # Статистика по всем вопросам
    avg_score = sum(r['assessment']['total_score'] for r in results) / len(results)
    avg_time = sum(r['time_seconds'] for r in results) / len(results)

    log.info("%s", "\n" + "="*80)
    log.info("OVERALL STATISTICS")
    log.info("%s", "="*80)
    log.info("   Total questions: %s", len(results))
    log.info(
        "   Average quality score: %.2f / 1.00 (%s)",
        avg_score,
        '+' * int(avg_score * 5),
    )
    log.info("   Average response time: %.2fs", avg_time)
    log.info(
        "   Best question: %s",
        max(results, key=lambda x: x['assessment']['total_score'])['id'],
    )
    log.info(
        "   Worst question: %s",
        min(results, key=lambda x: x['assessment']['total_score'])['id'],
    )

    # Аудит логов
    log.info("%s", "\n" + "="*80)
    log.info("LOG AUDIT")
    log.info("%s", "="*80)

    auditor = LogAuditor()
    log_stats = auditor.analyze()

    if 'error' in log_stats:
        log.warning("Log audit: %s", log_stats['error'])
    else:
        log.info("   Log file: %s", log_stats['log_file'])
        log.info("   Total events: %s", log_stats['total_events'])
        log.info("   Stages: %s", ', '.join(log_stats['stages'].keys()))
        log.info("   Embed calls: %s", len(log_stats['embed_calls']))
        if log_stats['avg_embed_latency'] > 0:
            log.info("   Avg embed latency: %.2fms", log_stats['avg_embed_latency'])
        log.info("   LLM calls: %s", len(log_stats['llm_calls']))
        if log_stats['avg_llm_latency'] > 0:
            log.info("   Avg LLM latency: %.2fms", log_stats['avg_llm_latency'])
        log.info("   Slow operations: %s", len(log_stats['slow_operations']))
        log.info("   Errors: %s", len(log_stats['errors']))

        # Сохраняем аудит логов
        audit_path = output_dir / f"log_audit_{timestamp}.json"
        with open(audit_path, 'w', encoding='utf-8') as handle:
            json.dump(log_stats, handle, ensure_ascii=False, indent=2, default=str)
        log.info("\n[SAVE] Log audit: %s", audit_path)

    # Сохраняем итоговый отчёт
    summary = {
        'timestamp': timestamp,
        'total_questions': len(results),
        'avg_quality_score': avg_score,
        'avg_response_time': avg_time,
        'results': results,
        'log_audit': log_stats if 'error' not in log_stats else None
    }

    summary_path = output_dir / f"full_report_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    log.info("[SAVE] Full report: %s", summary_path)

    log.info("%s", "\n" + "="*80)
    log.info(" COMPLETE!")
    log.info(" Results saved to: %s", output_dir)
    log.info("%s", "="*80)

    return results


def answer_single(qid: int) -> dict[str, Any] | None:
    """Ответить на один конкретный вопрос"""
    if qid not in QUESTIONS:
        log.error("Question id=%s not found", qid)
        return None

    result = answer_question(qid, QUESTIONS[qid])

    # Сохраняем
    output_dir = Path("answers_audit/single")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"answer_{qid}_{timestamp}.txt"

    with open(file_path, 'w', encoding='utf-8') as handle:
        handle.write(f"QUESTION #{qid}: {QUESTIONS[qid]['text']}\n")
        handle.write("="*80 + "\n\n")
        handle.write(f"ANSWER:\n{result['answer']}\n\n")
        handle.write(f"SOURCES: {result['sources']}\n")
        handle.write(f"TIME: {result['time_seconds']}s\n\n")
        handle.write(
            f"QUALITY SCORE: {result['assessment']['total_score']:.2f} "
            f"{result['assessment']['stars']}\n"
        )

    log.info("[SAVE] %s", file_path)
    return result


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Answer Questions + Audit")
    # Kept for CLI backward compatibility (default path runs full audit when no flags).
    parser.add_argument("--all", "-a", action="store_true", help="Answer all questions")
    parser.add_argument("--id", "-i", type=int, help="Answer specific question by ID")
    parser.add_argument("--audit-only", action="store_true", help="Run only log audit")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    if args.audit_only:
        auditor = LogAuditor()
        stats = auditor.analyze()
        sys.stdout.write(json.dumps(stats, indent=2, default=str) + "\n")
    elif args.id is not None:
        init_logger(config.llm_model)
        answer_single(args.id)
    else:
        init_logger(config.llm_model)
        run_full_audit()


if __name__ == "__main__":
    _cli_main()

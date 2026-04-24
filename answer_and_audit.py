#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANSWER QUESTIONS + QUALITY ASSESSMENT + LOG AUDIT
Отвечает на вопросы, оценивает качество, анализирует логи
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from rag_engine.engine import rag
from core.logger import init_logger, get_logger
# В самом начале, после импортов
from core.logger import init_logger
init_logger(config.llm_model)
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
    def check_keywords(answer: str, expected_keywords: List[str]) -> Tuple[int, List[str]]:
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
    def assess(question: str, answer: str, qtype: str, expected_keywords: List[str]) -> Dict:
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
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file
        self.events = []
    
    def find_latest_log(self) -> Path:
        """Находит последний файл лога"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return None
        
        log_files = list(log_dir.glob("run-*.log"))
        if not log_files:
            return None
        
        latest = max(log_files, key=lambda f: f.stat().st_mtime)
        return latest
    
    def parse_log(self, log_path: Path) -> List[Dict]:
        """Парсит лог-файл"""
        events = []
        
        # Пробуем разные кодировки
        for encoding in ['utf-8', 'cp1251', 'latin-1']:
            try:
                with open(log_path, 'r', encoding=encoding) as f:
                    for line in f:
                        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([^\]]+)\] \[([^\]]+)\] (.+)', line.strip())
                        if match:
                            events.append({
                                'timestamp': match.group(1),
                                'model_tag': match.group(2),
                                'stage': match.group(3),
                                'message': match.group(4)
                            })
                break  # Если успешно, выходим
            except UnicodeDecodeError:
                continue
    
        return events
        
        
    
    def analyze(self) -> Dict:
        """Анализирует логи"""
        
        # Находим последний лог
        if not self.log_file:
            self.log_file = self.find_latest_log()
        
        if not self.log_file:
            return {'error': 'No log files found'}
        
        # Парсим
        self.events = self.parse_log(self.log_file)
        
        # Статистика
        stats = {
            'log_file': str(self.log_file),
            'total_events': len(self.events),
            'stages': {},
            'embed_calls': [],
            'llm_calls': [],
            'slow_operations': [],
            'errors': []
        }
        
        for event in self.events:
            # Стадии
            stage = event['stage']
            stats['stages'][stage] = stats['stages'].get(stage, 0) + 1
            
            # Embedding вызовы
            if 'embed.call' in event['message']:
                match = re.search(r'latency_ms=([\d\.]+)', event['message'])
                if match:
                    stats['embed_calls'].append(float(match.group(1)))
            
            # LLM вызовы
            if 'llm.generate' in event['message']:
                match = re.search(r'latency_ms=([\d\.]+)', event['message'])
                if match:
                    stats['llm_calls'].append(float(match.group(1)))
            
            # Медленные операции
            if '.slow' in event['message']:
                stats['slow_operations'].append(event)
            
            # Ошибки
            if 'error' in event['message'].lower() or 'ERROR' in event['message']:
                stats['errors'].append(event)
        
        # Агрегация
        if stats['embed_calls']:
            stats['avg_embed_latency'] = sum(stats['embed_calls']) / len(stats['embed_calls'])
            stats['max_embed_latency'] = max(stats['embed_calls'])
        else:
            stats['avg_embed_latency'] = 0
        
        if stats['llm_calls']:
            stats['avg_llm_latency'] = sum(stats['llm_calls']) / len(stats['llm_calls'])
            stats['max_llm_latency'] = max(stats['llm_calls'])
        else:
            stats['avg_llm_latency'] = 0
        
        return stats


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def answer_question(qid: int, question_data: Dict) -> Dict:
    """Отвечает на один вопрос и оценивает ответ"""
    
    print(f"\n{'='*80}")
    print(f"QUESTION #{qid}: {question_data['text']}")
    print(f"Type: {question_data['type']}")
    print(f"{'='*80}")
    
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
    print(f"\n ANSWER:")
    print(f"{result['answer']}")
    print(f"\n SOURCES: {result['sources']}")
    print(f" TIME: {elapsed:.2f}s")
    
    print(f"\n QUALITY ASSESSMENT:")
    print(f"   Total score: {assessment['total_score']:.2f} / 1.00 {assessment['stars']}")
    print(f"   Keyword match: {assessment['keyword_score']:.2f}")
    print(f"   Length adequacy: {assessment['length_score']:.2f}")
    print(f"   Structure: {assessment['structure_score']:.2f}")
    print(f"   Sources: {'Yes' if assessment['has_sources'] else 'No'}")
    
    if assessment['found_keywords']:
        print(f"   Found keywords: {', '.join(assessment['found_keywords'])}")
    if assessment['missing_keywords']:
        print(f"   Missing keywords: {', '.join(assessment['missing_keywords'])}")
    
    return {
        'id': qid,
        'question': question_data['text'],
        'type': question_data['type'],
        'answer': result['answer'],
        'sources': result['sources'],
        'time_seconds': elapsed,
        'assessment': assessment
    }


def run_full_audit():
    """Запускает полный аудит: ответы на вопросы + анализ логов"""
    
    print("""
        ============================================================================
                ANSWER QUESTIONS + QUALITY ASSESSMENT + LOG AUDIT
        ============================================================================
        
        Отвечает на 10 вопросов, оценивает качество, анализирует логи
        ============================================================================
        """)
    # Проверка базы данных
    chunk_count = db.get_chunk_count()
    print(f"\n[CHECK] Database has {chunk_count} chunks")
    
    if chunk_count == 0:
        print("[ERROR] No chunks found! Run hypothesis_loader.py first")
        return
    
    # Проверка Ollama
    try:
        import ollama
        ollama.list()
        print("[OK] Ollama is running")
    except:
        print("[ERROR] Ollama is not running!")
        return
    
    # Отвечаем на все вопросы
    results = []
    for qid, qdata in QUESTIONS.items():
        result = answer_question(qid, qdata)
        results.append(result)
        time.sleep(0.5)  # Небольшая задержка между вопросами
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_audit")
    output_dir.mkdir(exist_ok=True)
    
    # JSON
    json_path = output_dir / f"answers_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] JSON: {json_path}")
    
    # TXT отчёт
    txt_path = output_dir / f"answers_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANSWERS WITH QUALITY ASSESSMENT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"QUESTION #{r['id']}: {r['question']}\n")
            f.write(f"Type: {r['type']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"\nANSWER:\n{r['answer']}\n")
            f.write(f"\nSOURCES: {r['sources']}\n")
            f.write(f"TIME: {r['time_seconds']}s\n")
            f.write(f"\nQUALITY ASSESSMENT:\n")
            f.write(f"  Total score: {r['assessment']['total_score']:.2f} {r['assessment']['stars']}\n")
            f.write(f"  Keyword match: {r['assessment']['keyword_score']:.2f}\n")
            f.write(f"  Found keywords: {', '.join(r['assessment']['found_keywords'])}\n")
            if r['assessment']['missing_keywords']:
                f.write(f"  Missing keywords: {', '.join(r['assessment']['missing_keywords'])}\n")
            f.write(f"\n---\n")
    
    print(f"[SAVE] TXT: {txt_path}")
    
    # Статистика по всем вопросам
    avg_score = sum(r['assessment']['total_score'] for r in results) / len(results)
    avg_time = sum(r['time_seconds'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"   Total questions: {len(results)}")
    print(f"   Average quality score: {avg_score:.2f} / 1.00 ({'+' * int(avg_score * 5)})")
    print(f"   Average response time: {avg_time:.2f}s")
    print(f"   Best question: {max(results, key=lambda x: x['assessment']['total_score'])['id']}")
    print(f"   Worst question: {min(results, key=lambda x: x['assessment']['total_score'])['id']}")
    
    # Аудит логов
    print("\n" + "="*80)
    print("LOG AUDIT")
    print("="*80)
    
    auditor = LogAuditor()
    log_stats = auditor.analyze()
    
    if 'error' in log_stats:
        print(f"   {log_stats['error']}")
    else:
        print(f"   Log file: {log_stats['log_file']}")
        print(f"   Total events: {log_stats['total_events']}")
        print(f"   Stages: {', '.join(log_stats['stages'].keys())}")
        print(f"   Embed calls: {len(log_stats['embed_calls'])}")
        if log_stats['avg_embed_latency'] > 0:
            print(f"   Avg embed latency: {log_stats['avg_embed_latency']:.2f}ms")
        print(f"   LLM calls: {len(log_stats['llm_calls'])}")
        if log_stats['avg_llm_latency'] > 0:
            print(f"   Avg LLM latency: {log_stats['avg_llm_latency']:.2f}ms")
        print(f"   Slow operations: {len(log_stats['slow_operations'])}")
        print(f"   Errors: {len(log_stats['errors'])}")
        
        # Сохраняем аудит логов
        audit_path = output_dir / f"log_audit_{timestamp}.json"
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(log_stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[SAVE] Log audit: {audit_path}")
    
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
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] Full report: {summary_path}")
    
    print("\n" + "="*80)
    print(" COMPLETE!")
    print(f" Results saved to: {output_dir}")
    print("="*80)
    
    return results


def answer_single(qid: int):
    """Ответить на один конкретный вопрос"""
    if qid not in QUESTIONS:
        print(f"[ERROR] Question {qid} not found")
        return
    
    result = answer_question(qid, QUESTIONS[qid])
    
    # Сохраняем
    output_dir = Path("answers_audit/single")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"answer_{qid}_{timestamp}.txt"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"QUESTION #{qid}: {QUESTIONS[qid]['text']}\n")
        f.write("="*80 + "\n\n")
        f.write(f"ANSWER:\n{result['answer']}\n\n")
        f.write(f"SOURCES: {result['sources']}\n")
        f.write(f"TIME: {result['time_seconds']}s\n\n")
        f.write(f"QUALITY SCORE: {result['assessment']['total_score']:.2f} {result['assessment']['stars']}\n")
    
    print(f"\n[SAVE] {file_path}")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Answer Questions + Audit')
    parser.add_argument('--all', '-a', action='store_true', help='Answer all questions')
    parser.add_argument('--id', '-i', type=int, help='Answer specific question by ID')
    parser.add_argument('--audit-only', action='store_true', help='Run only log audit')
    
    args = parser.parse_args()
    
    if args.audit_only:
        auditor = LogAuditor()
        stats = auditor.analyze()
        print(json.dumps(stats, indent=2, default=str))
    elif args.id:
        answer_single(args.id)
    else:
        run_full_audit()
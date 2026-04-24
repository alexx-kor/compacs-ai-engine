#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANSWER SPECIFIC QUESTIONS - использует существующую базу знаний
Без перегенерации чанков и эмбеддингов
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, r"C:\Users\User\Desktop\RAG SYSTEM")

from config import config
from core.database import db
from core.embeddings import embedder
from core.reranker import reranker
from rag_engine.engine import rag
from router.smart_router import select_prompt

# ============================================================================
# ВОПРОСЫ ДЛЯ ОТВЕТА
# ============================================================================

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
    52: "Do I need PCI for v2/sale?"
}

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ВОПРОСЫ (если нужны)
# ============================================================================

EXTRA_QUESTIONS = {
    "q1": "What is payout-by-ref integration?",
    "q2": "What is the difference between payout and payout-by-ref?",
    "q3": "How to integrate recurring sale?",
}

# ============================================================================
# ФУНКЦИЯ ДЛЯ ОТВЕТА НА ОДИН ВОПРОС
# ============================================================================

def ask_question(question: str, question_id: str = None) -> dict:
    """Задает вопрос и возвращает ответ"""
    
    print(f"\n{'='*80}")
    if question_id:
        print(f"QUESTION #{question_id}")
    print(f"{'='*80}")
    print(f"Q: {question}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = rag.ask(question)
        elapsed = time.time() - start_time
        
        answer_data = {
            'id': question_id,
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'time': round(elapsed, 2),
            'status': result.get('status', 'success')
        }
        
        # Выводим ответ
        print(f"\n🤖 ANSWER:\n{result['answer']}")
        print(f"\n📚 SOURCES: {result['sources']}")
        print(f"⏱️ TIME: {elapsed:.2f}s")
        
        return answer_data
        
    except Exception as e:
        error_data = {
            'id': question_id,
            'question': question,
            'answer': f"ERROR: {e}",
            'sources': [],
            'time': round(time.time() - start_time, 2),
            'status': 'error'
        }
        print(f"\n❌ ERROR: {e}")
        return error_data


# ============================================================================
# ФУНКЦИЯ ДЛЯ ПАКЕТНОЙ ОБРАБОТКИ
# ============================================================================

def answer_all_questions(questions_dict: dict, output_dir: str = None):
    """Отвечает на все вопросы из словаря"""
    
    if output_dir is None:
        output_dir = Path(r"C:\Users\User\Desktop\RAG SYSTEM\data\answers")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    print("\n" + "="*80)
    print("ANSWERING ALL QUESTIONS")
    print("="*80)
    print(f"Total questions: {len(questions_dict)}")
    print(f"Output directory: {output_dir}")
    
    for qid, question in questions_dict.items():
        result = ask_question(question, qid)
        results.append(result)
        
        # Небольшая задержка между запросами
        time.sleep(1)
    
    # Сохраняем все ответы
    save_all_answers(results, output_dir, timestamp)
    
    return results


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_all_answers(results: list, output_dir: Path, timestamp: str):
    """Сохраняет все ответы в разных форматах"""
    
    # 1. Сохраняем в JSON
    json_path = output_dir / f"answers_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] JSON: {json_path}")
    
    # 2. Сохраняем в TXT (читаемый формат)
    txt_path = output_dir / f"answers_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANSWERS TO QUESTIONS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"ID: {r['id']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Question: {r['question']}\n")
            f.write(f"Time: {r['time']}s\n")
            f.write(f"Status: {r['status']}\n")
            f.write(f"Sources: {r['sources']}\n")
            f.write(f"\n--- ANSWER ---\n")
            f.write(f"{r['answer']}\n")
            f.write(f"\n--- END OF ANSWER ---\n")
    
    print(f"[SAVE] TXT: {txt_path}")
    
    # 3. Сохраняем в Markdown (удобно для документации)
    md_path = output_dir / f"answers_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Answers to Questions\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        for r in results:
            f.write(f"## {r['id']}. {r['question']}\n\n")
            f.write(f"**Time:** {r['time']}s | **Status:** {r['status']}\n\n")
            f.write(f"**Sources:** {r['sources']}\n\n")
            f.write(f"### Answer\n\n")
            f.write(f"{r['answer']}\n\n")
            f.write("---\n\n")
    
    print(f"[SAVE] Markdown: {md_path}")
    
    # 4. Краткая сводка
    summary_path = output_dir / f"summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANSWERS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            status_icon = "✅" if r['status'] == 'success' else "❌"
            answer_preview = r['answer'][:100].replace('\n', ' ')
            f.write(f"{status_icon} [{r['id']}] {r['question'][:60]}...\n")
            f.write(f"   Time: {r['time']}s\n")
            f.write(f"   Preview: {answer_preview}...\n\n")
    
    print(f"[SAVE] Summary: {summary_path}")


# ============================================================================
# ФУНКЦИЯ ДЛЯ ОТВЕТА НА ОДИН КОНКРЕТНЫЙ ВОПРОС
# ============================================================================

def answer_single_question(question_id: int):
    """Отвечает на один конкретный вопрос по ID"""
    
    if question_id not in QUESTIONS:
        print(f"[ERROR] Question ID {question_id} not found")
        print(f"Available IDs: {list(QUESTIONS.keys())}")
        return None
    
    question = QUESTIONS[question_id]
    result = ask_question(question, question_id)
    
    # Сохраняем отдельно
    output_dir = Path(r"C:\Users\User\Desktop\RAG SYSTEM\data\answers\single")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"answer_{question_id}_{timestamp}.txt"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Question #{question_id}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Q: {question}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"ANSWER:\n{result['answer']}\n\n")
        f.write(f"SOURCES: {result['sources']}\n")
        f.write(f"TIME: {result['time']}s\n")
    
    print(f"\n[SAVE] Saved to: {file_path}")
    
    return result


# ============================================================================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ============================================================================

def interactive_mode():
    """Интерактивный режим - задавайте любые вопросы"""
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Type 'exit' to quit")
    print("Type 'list' to see predefined questions")
    print("Type 'all' to answer all predefined questions")
    print("-"*80)
    
    while True:
        try:
            user_input = input("\n❓ Question: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            elif user_input.lower() == 'list':
                print("\nPredefined questions:")
                for qid, q in QUESTIONS.items():
                    print(f"   [{qid}] {q}")
                continue
            
            elif user_input.lower() == 'all':
                answer_all_questions(QUESTIONS)
                continue
            
            elif user_input.isdigit() and int(user_input) in QUESTIONS:
                answer_single_question(int(user_input))
                continue
            
            elif user_input:
                ask_question(user_input)
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    ANSWER SPECIFIC QUESTIONS                         ║
    ║                                                                      ║
    ║  Использует существующую базу знаний (без перегенерации)            ║
    ║                                                                      ║
    ║  Commands:                                                          ║
    ║    python answer_questions.py --all      - Answer all questions     ║
    ║    python answer_questions.py --id 3     - Answer question #3       ║
    ║    python answer_questions.py --interactive - Interactive mode      ║
    ║    python answer_questions.py --q "text" - Ask custom question      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Проверка подключения к базе
    print("\n[CHECK] Verifying database connection...")
    chunk_count = db.get_chunk_count()
    print(f"[OK] Database has {chunk_count} chunks")
    
    if chunk_count == 0:
        print("[ERROR] No chunks found in database!")
        print("Please run 'python run.py --load-only' first to load documents")
        return
    
    # Проверка Ollama
    try:
        import ollama
        ollama.list()
        print("[OK] Ollama is running")
    except:
        print("[ERROR] Ollama is not running!")
        print("Run: ollama serve")
        return
    
    # Парсинг аргументов командной строки
    import argparse
    parser = argparse.ArgumentParser(description='Answer specific questions')
    parser.add_argument('--all', '-a', action='store_true', help='Answer all questions')
    parser.add_argument('--id', '-i', type=int, help='Answer question by ID')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--q', '--question', type=str, help='Ask custom question')
    
    args = parser.parse_args()
    
    if args.all:
        answer_all_questions(QUESTIONS)
    elif args.id:
        answer_single_question(args.id)
    elif args.interactive:
        interactive_mode()
    elif args.q:
        ask_question(args.q)
    else:
        # По умолчанию - отвечаем на все вопросы
        print("\n[DEFAULT] Answering all predefined questions...")
        answer_all_questions(QUESTIONS)


if __name__ == "__main__":
    main()
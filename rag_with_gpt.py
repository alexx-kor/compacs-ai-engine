#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG WITH GPT - с полным логированием
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
from core.reranker import reranker
from router.smart_router import select_prompt

# ============================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"rag_gpt_{timestamp}.log"

# Настройка логгера
logger = logging.getLogger('RAG_GPT')
logger.setLevel(logging.DEBUG)

# Файловый handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Консольный handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Формат
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
GPT_MODEL = "gpt-4o-mini"

# Переключаемся на правильную таблицу
db.set_active_table("hypothesis")
logger.info(f"Active table: {db.get_active_table()}")
logger.info(f"Chunks in database: {db.get_chunk_count()}")

# 10 вопросов
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


def ask_question(question: str, qid: int) -> dict:
    """RAG запрос с логированием"""
    
    t_start = time.time()
    logger.info(f"Processing Q{qid}: {question[:80]}...")
    
    # 1. Эмбеддинг
    logger.debug("Generating query embedding...")
    q_emb = list(embedder.generate_cached(question))
    logger.debug(f"Embedding dimension: {len(q_emb)}")
    
    # 2. Поиск
    logger.debug("Searching in ClickHouse...")
    results = db.search(q_emb)
    logger.debug(f"Found {len(results)} chunks")
    
    if not results:
        logger.warning(f"No results found for Q{qid}")
        return {
            'id': qid,
            'question': question,
            'answer': "NOT FOUND in documentation",
            'sources': [],
            'time': round(time.time() - t_start, 2),
            'prompt_used': 'none',
            'tokens': 0
        }
    
    # 3. Реранжинг
    logger.debug("Reranking results...")
    reranked = reranker.rerank(question, results)
    logger.debug(f"Kept {len(reranked)} chunks")
    
    # 4. Контекст
    context_parts = []
    sources = []
    for r in reranked[:config.rerank_top_k]:
        chunk, source, page = r[0], r[1], r[2]
        context_parts.append(f"[{source}, p.{page}]\n{chunk[:800]}")
        sources.append((source, page))
    
    context = "\n\n".join(context_parts)
    logger.debug(f"Context length: {len(context)} chars")
    
    # 5. Выбор промпта
    system_prompt, num_predict, temperature = select_prompt(question)
    
    if "parameter" in system_prompt.lower() and "list" not in system_prompt.lower():
        prompt_name = "API Parameter Prompt"
    elif "list of parameters" in system_prompt.lower():
        prompt_name = "API Parameters List Prompt"
    else:
        prompt_name = "API Info Prompt"
    
    logger.info(f"Selected prompt: {prompt_name} (temp={temperature}, max_tokens={num_predict})")
    
    # 6. Запрос к GPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
    ]
    
    try:
        gpt_start = time.time()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=800
        )
        gpt_time = time.time() - gpt_start
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        logger.info(f"GPT response: {gpt_time:.2f}s, {tokens} tokens")
        logger.debug(f"Answer preview: {answer[:200]}...")
        
    except Exception as e:
        logger.error(f"GPT error: {e}")
        answer = f"ERROR: {e}"
        tokens = 0
    
    return {
        'id': qid,
        'question': question,
        'answer': answer,
        'sources': sources,
        'time': round(time.time() - t_start, 2),
        'prompt_used': prompt_name,
        'tokens': tokens,
        'gpt_time': round(gpt_time, 2) if 'gpt_time' in locals() else 0
    }


def main():
    logger.info("="*80)
    logger.info("RAG WITH GPT - START")
    logger.info(f"Model: {GPT_MODEL}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Проверка БД
    chunk_count = db.get_chunk_count()
    logger.info(f"Database chunks: {chunk_count}")
    
    if chunk_count == 0:
        logger.error("No chunks in database!")
        print("Run: python load_graph_chunks.py --force")
        return
    
    # Проверка API ключа
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY not found in .env")
        return
    
    results = []
    for qid, question in QUESTIONS.items():
        result = ask_question(question, qid)
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Q{qid}: {result['prompt_used']}")
        print(f"ANSWER: {result['answer'][:300]}...")
        print(f"SOURCES: {result['sources']}")
        print(f"TIME: {result['time']}s")
        print(f"TOKENS: {result['tokens']}")
        print(f"{'='*70}")
    
    # Сохраняем результаты
    output_dir = Path("answers_rag_gpt")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"answers_rag_gpt_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Статистика
    avg_time = sum(r['time'] for r in results) / len(results)
    total_tokens = sum(r['tokens'] for r in results)
    
    logger.info("="*80)
    logger.info("STATISTICS")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Average time: {avg_time:.2f}s")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Estimated cost: ${total_tokens * 0.0000015:.4f}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log saved to: {log_file}")
    logger.info("="*80)
    
    print(f"\n[SAVE] Results: {output_file}")
    print(f"[SAVE] Log: {log_file}")


if __name__ == "__main__":
    main()
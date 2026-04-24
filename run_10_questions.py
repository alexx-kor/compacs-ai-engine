#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUN 10 QUESTIONS ON MAIN TABLE
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from rag_engine.engine import rag
from core.logger import init_logger
init_logger(config.llm_model)
# Переключаемся на основную таблицу (старые данные)
db.set_active_table("main")
print(f"[INFO] Using table: {db.get_active_table()}")
print(f"[INFO] Chunks in database: {db.get_chunk_count()}")

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

def main():
    print("\n" + "="*70)
    print("RUNNING 10 QUESTIONS ON MAIN TABLE (OLD DATA)")
    print("="*70)
    
    results = []
    for qid, question in QUESTIONS.items():
        print(f"\n{'='*70}")
        print(f"Q{qid}: {question}")
        print(f"{'='*70}")
        
        start = time.time()
        result = rag.ask(question)
        elapsed = time.time() - start
        
        print(f"\nANSWER:\n{result['answer'][:500]}")
        print(f"\nSOURCES: {result['sources']}")
        print(f"TIME: {elapsed:.2f}s")
        
        results.append({
            'id': qid,
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'time': elapsed
        })
    
    # Сохраняем результаты
    from pathlib import Path
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_main")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"answers_main_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] Results saved to: {output_file}")
    print("\n[DONE]")

if __name__ == "__main__":
    main()
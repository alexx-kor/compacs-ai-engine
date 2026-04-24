#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUN 10 QUESTIONS WITH GPT (выбор модели)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Загружает переменные из .env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Доступные модели
AVAILABLE_MODELS = {
    'gpt-4o': {'name': 'GPT-4o', 'speed': 'fast', 'quality': 'excellent', 'cost_per_1m': 2.50},
    'gpt-4o-mini': {'name': 'GPT-4o Mini', 'speed': 'very fast', 'quality': 'good', 'cost_per_1m': 0.15},
    'gpt-3.5-turbo': {'name': 'GPT-3.5 Turbo', 'speed': 'fastest', 'quality': 'medium', 'cost_per_1m': 0.50},
    'gpt-4-turbo': {'name': 'GPT-4 Turbo', 'speed': 'medium', 'quality': 'excellent', 'cost_per_1m': 10.00},
}

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

SYSTEM_PROMPT = """You are a technical documentation expert. Answer based ONLY on the provided context.

If information not found: "NOT FOUND in documentation"

Be concise and accurate."""


def ask_gpt(question: str, model: str) -> dict:
    """Задает вопрос GPT"""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
        elapsed = time.time() - start
        answer = response.choices[0].message.content
        usage = response.usage
        return {
            'answer': answer,
            'time': elapsed,
            'tokens': usage.total_tokens,
            'status': 'success'
        }
    except Exception as e:
        return {
            'answer': f"ERROR: {e}",
            'time': time.time() - start,
            'tokens': 0,
            'status': 'error'
        }


def main():
    parser = argparse.ArgumentParser(description='Run questions with GPT')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini',
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='GPT model to use')
    args = parser.parse_args()
    
    model = args.model
    model_info = AVAILABLE_MODELS[model]
    
    print("\n" + "="*70)
    print(f"RUNNING 10 QUESTIONS WITH {model_info['name'].upper()}")
    print(f"  Speed: {model_info['speed']}")
    print(f"  Quality: {model_info['quality']}")
    print(f"  Cost: ${model_info['cost_per_1m']}/1M tokens")
    print("="*70)
    
    # Проверка API ключа
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in .env")
        return
    
    print(f"\n[CHECK] OpenAI API Key: {api_key[:10]}...")
    
    results = []
    for qid, question in QUESTIONS.items():
        print(f"\n{'='*70}")
        print(f"Q{qid}: {question[:70]}...")
        print(f"{'='*70}")
        
        result = ask_gpt(question, model)
        
        print(f"\nANSWER:\n{result['answer'][:500]}")
        print(f"\nTIME: {result['time']:.2f}s")
        print(f"TOKENS: {result['tokens']}")
        
        results.append({
            'id': qid,
            'question': question,
            'answer': result['answer'],
            'time': result['time'],
            'tokens': result['tokens'],
            'status': result['status'],
            'model': model
        })
    
    # Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("answers_gpt")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"answers_gpt_{model}_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] Results saved to: {output_file}")
    
    # Статистика
    avg_time = sum(r['time'] for r in results) / len(results)
    total_tokens = sum(r['tokens'] for r in results)
    estimated_cost = total_tokens * model_info['cost_per_1m'] / 1_000_000
    
    print(f"\nSTATISTICS:")
    print(f"  Model: {model_info['name']}")
    print(f"  Average time: {avg_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Estimated cost: ${estimated_cost:.4f}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
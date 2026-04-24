#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARE RESULTS - сравнение старой и новой оценки
Поддерживает CSV и JSON форматы
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def load_old_results(file_path: str):
    """Загружает старые результаты из CSV или JSON"""
    if not os.path.exists(file_path):
        # Пробуем найти файл по маске
        files = glob.glob(file_path)
        if files:
            file_path = files[0]
        else:
            print(f"[ERROR] Old results not found: {file_path}")
            return None
    
    print(f"[LOAD] Loading from: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        print(f"[LOAD] Old results: {len(df)} records from CSV")
        return df
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"[LOAD] Old results: {len(df)} records from JSON")
        return df
    else:
        print(f"[ERROR] Unknown format: {file_path}")
        return None

def load_new_results(file_path: str):
    """Загружает новые результаты из JSON"""
    if not os.path.exists(file_path):
        files = glob.glob(file_path)
        if files:
            file_path = files[0]
        else:
            print(f"[ERROR] New results not found: {file_path}")
            return None
    
    print(f"[LOAD] Loading from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[LOAD] New results: {len(data)} records")
    return data

def compare_scores(old_df, new_data):
    """Сравнивает оценки"""
    
    # Определяем колонку с ID
    id_col = 'id' if 'id' in old_df.columns else 'Id' if 'Id' in old_df.columns else old_df.columns[0]
    score_col = 'similarity_score' if 'similarity_score' in old_df.columns else 'score' if 'score' in old_df.columns else 'total_score'
    
    # Создаём словарь новых результатов по ID вопроса
    new_scores = {}
    for item in new_data:
        qid = item.get('id')
        if qid:
            new_scores[qid] = {
                'score': item.get('score', item.get('total_score', 0)),
                'time': item.get('time', item.get('time_seconds', 0)),
                'question': item.get('question', '')
            }
    
    # Сравниваем
    results = []
    for _, row in old_df.iterrows():
        qid = row.get(id_col)
        if qid and qid in new_scores:
            old_score = row.get(score_col, 0)
            new_score = new_scores[qid]['score']
            diff = new_score - old_score
            
            results.append({
                'id': qid,
                'question': str(row.get('question', ''))[:80],
                'old_score': float(old_score) if old_score else 0,
                'new_score': float(new_score) if new_score else 0,
                'difference': diff,
                'improved': diff > 0,
                'old_time': row.get('time_seconds', row.get('time', 0)),
                'new_time': new_scores[qid]['time']
            })
    
    return results

def print_comparison(results):
    """Печатает сравнение"""
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if not results:
        print("[ERROR] No results to compare!")
        return
    
    # Статистика
    improved = [r for r in results if r['improved']]
    degraded = [r for r in results if not r['improved']]
    
    avg_old = sum(r['old_score'] for r in results) / len(results)
    avg_new = sum(r['new_score'] for r in results) / len(results)
    
    print(f"\nSUMMARY:")
    print(f"  Total questions: {len(results)}")
    print(f"  Improved: {len(improved)}")
    print(f"  Degraded: {len(degraded)}")
    print(f"  Average old score: {avg_old:.3f}")
    print(f"  Average new score: {avg_new:.3f}")
    print(f"  Difference: {avg_new - avg_old:+.3f}")
    
    if improved:
        print(f"\nTOP IMPROVED:")
        for r in sorted(improved, key=lambda x: x['difference'], reverse=True)[:5]:
            print(f"  Q{r['id']}: {r['old_score']:.3f} -> {r['new_score']:.3f} (+{r['difference']:.3f})")
            print(f"    {r['question'][:60]}...")
    
    if degraded:
        print(f"\nTOP DEGRADED:")
        for r in sorted(degraded, key=lambda x: x['difference'])[:5]:
            print(f"  Q{r['id']}: {r['old_score']:.3f} -> {r['new_score']:.3f} ({r['difference']:.3f})")
            print(f"    {r['question'][:60]}...")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("comparison")
    output_dir.mkdir(exist_ok=True)
    
    # CSV
    df_out = pd.DataFrame(results)
    csv_path = output_dir / f"comparison_{timestamp}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n[SAVE] {csv_path}")
    
    # JSON
    json_path = output_dir / f"comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {json_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare old and new results')
    parser.add_argument('--old', '-o', type=str, required=True, help='Old results file (CSV or JSON)')
    parser.add_argument('--new', '-n', type=str, required=True, help='New results JSON file')
    
    args = parser.parse_args()
    
    # Загружаем данные
    old_df = load_old_results(args.old)
    new_data = load_new_results(args.new)
    
    if old_df is None or new_data is None:
        return
    
    # Сравниваем
    results = compare_scores(old_df, new_data)
    
    if not results:
        print("[ERROR] No matching questions found!")
        return
    
    # Печатаем
    print_comparison(results)

if __name__ == "__main__":
    main()
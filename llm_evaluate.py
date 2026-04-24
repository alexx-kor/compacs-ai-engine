#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM EVALUATION - процентное сравнение ответов (0-100%)
Без эмодзи для Windows консоли
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
import ollama


def load_json(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def llm_evaluate_percent(question: str, answer: str) -> dict:
    """
    Оценивает ответ в процентах (0-100%)
    """
    
    answer = answer[:2000] if answer else ""
    
    prompt = f"""You are an expert evaluator of RAG system answers. Rate the answer on a scale of 0-100%.

QUESTION: {question}

ANSWER: {answer}

Rate on these criteria (0-100%, where 0=terrible, 100=perfect):
1. RELEVANCE: Does the answer directly address the question?
2. ACCURACY: Is the information factually correct?
3. COMPLETENESS: Does it provide sufficient detail?
4. CLARITY: Is it well-structured and easy to understand?

CRITICAL: Reply ONLY in JSON format with numbers between 0 and 100.
Example: {{"relevance": 85, "accuracy": 90, "completeness": 75, "clarity": 80, "total": 82.5}}

Do NOT output any text outside the JSON.
Do NOT use numbers below 0 or above 100.
"""

    try:
        response = ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 300
            }
        )
        
        content = response['message']['content']
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Ограничиваем значения 0-100
            for key in ['relevance', 'accuracy', 'completeness', 'clarity', 'total']:
                if key in result:
                    result[key] = max(0, min(100, float(result[key])))
            return result
        else:
            return {"error": "Failed to parse", "total": 0}
    except Exception as e:
        return {"error": str(e), "total": 0}


def compare_two_answers_percent(question: str, answer_a: str, answer_b: str) -> dict:
    """
    Сравнивает два ответа и возвращает процентное преимущество
    """
    
    answer_a = answer_a[:2000] if answer_a else ""
    answer_b = answer_b[:2000] if answer_b else ""
    
    prompt = f"""You are an expert evaluator comparing two RAG answers.

QUESTION: {question}

ANSWER A: {answer_a}

ANSWER B: {answer_b}

Compare these answers and decide:

1. Which answer is BETTER overall? (A or B or TIE)
2. By what PERCENTAGE is the better answer better? (0-100%)
3. Why?

Reply ONLY in JSON format:
{{"winner": "A" or "B" or "TIE", "winner_percent": 0-100, "reason": "brief explanation", "a_score": 0-100, "b_score": 0-100}}
"""

    try:
        response = ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "num_predict": 300
            }
        )
        
        content = response['message']['content']
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Ограничиваем процент
            if 'winner_percent' in result:
                result['winner_percent'] = max(0, min(100, float(result['winner_percent'])))
            return result
        else:
            return {"winner": "UNKNOWN", "winner_percent": 0}
    except Exception as e:
        return {"winner": "ERROR", "winner_percent": 0, "error": str(e)}


def evaluate_all(answers_main: list, answers_hypothesis: list) -> list:
    """
    Оценивает все ответы и возвращает статистику
    """
    
    results = []
    
    main_dict = {item['id']: item for item in answers_main}
    hyp_dict = {item['id']: item for item in answers_hypothesis}
    
    for qid in main_dict:
        if qid not in hyp_dict:
            continue
        
        question = main_dict[qid]['question']
        answer_main = main_dict[qid]['answer']
        answer_hyp = hyp_dict[qid]['answer']
        
        print(f"\n{'='*70}")
        print(f"Q{qid}: {question[:70]}...")
        print(f"{'='*70}")
        
        # Оцениваем MAIN
        print("  [1/3] Evaluating MAIN answer...", end=' ')
        eval_main = llm_evaluate_percent(question, answer_main)
        print(f"done ({eval_main.get('total', 0):.0f}%)")
        
        # Оцениваем HYPOTHESIS
        print("  [2/3] Evaluating HYPOTHESIS answer...", end=' ')
        eval_hyp = llm_evaluate_percent(question, answer_hyp)
        print(f"done ({eval_hyp.get('total', 0):.0f}%)")
        
        # Сравниваем
        print("  [3/3] Comparing answers...", end=' ')
        comparison = compare_two_answers_percent(question, answer_main, answer_hyp)
        print(f"done (Winner: {comparison.get('winner', '?')})")
        
        results.append({
            'id': qid,
            'question': question,
            'main': {
                'answer': answer_main[:500],
                'sources': main_dict[qid].get('sources', []),
                'time': main_dict[qid].get('time', 0),
                'evaluation': eval_main
            },
            'hypothesis': {
                'answer': answer_hyp[:500],
                'sources': hyp_dict[qid].get('sources', []),
                'time': hyp_dict[qid].get('time', 0),
                'evaluation': eval_hyp
            },
            'comparison': comparison
        })
        
        # Выводим результат
        winner = comparison.get('winner', '?')
        winner_pct = comparison.get('winner_percent', 0)
        if winner == 'A':
            winner_icon = "[MAIN]"
        elif winner == 'B':
            winner_icon = "[HYP]"
        else:
            winner_icon = "[TIE]"
        
        print(f"\n  RESULT:")
        print(f"     MAIN: {eval_main.get('total', 0):.1f}%")
        print(f"     HYP:  {eval_hyp.get('total', 0):.1f}%")
        print(f"     WINNER: {winner_icon} (+{winner_pct:.0f}%)")
        print(f"     Reason: {comparison.get('reason', '')[:80]}...")
    
    return results


def calculate_statistics(results: list) -> dict:
    """Вычисляет статистику в процентах"""
    
    main_scores = [r['main']['evaluation'].get('total', 0) for r in results]
    hyp_scores = [r['hypothesis']['evaluation'].get('total', 0) for r in results]
    
    winners = [r['comparison'].get('winner', 'UNKNOWN') for r in results]
    
    wins_main = winners.count('A')
    wins_hyp = winners.count('B')
    ties = winners.count('TIE')
    
    avg_main = sum(main_scores) / len(main_scores) if main_scores else 0
    avg_hyp = sum(hyp_scores) / len(hyp_scores) if hyp_scores else 0
    
    # Процентное преимущество
    if avg_main > avg_hyp:
        advantage = avg_main - avg_hyp
        advantage_pct = (advantage / avg_main) * 100 if avg_main > 0 else 0
        better = "MAIN"
    else:
        advantage = avg_hyp - avg_main
        advantage_pct = (advantage / avg_hyp) * 100 if avg_hyp > 0 else 0
        better = "HYPOTHESIS"
    
    return {
        'total_questions': len(results),
        'main_avg_score': avg_main,
        'hyp_avg_score': avg_hyp,
        'wins_main': wins_main,
        'wins_hyp': wins_hyp,
        'ties': ties,
        'win_rate_main': (wins_main / len(results)) * 100 if results else 0,
        'win_rate_hyp': (wins_hyp / len(results)) * 100 if results else 0,
        'tie_rate': (ties / len(results)) * 100 if results else 0,
        'advantage_percent': advantage_pct,
        'better_system': better,
        'improvement': advantage_pct if better == "HYPOTHESIS" else -advantage_pct
    }


def print_summary(results: list):
    """Печатает сводку в процентах (без эмодзи)"""
    
    stats = calculate_statistics(results)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY - PERCENTAGE BASED")
    print("="*80)
    
    print(f"\nOVERALL SCORES:")
    print(f"   MAIN system:       {stats['main_avg_score']:.1f}%")
    print(f"   HYPOTHESIS system: {stats['hyp_avg_score']:.1f}%")
    
    # Визуализация без эмодзи
    bar_main = '#' * int(stats['main_avg_score'] / 2)
    bar_hyp = '#' * int(stats['hyp_avg_score'] / 2)
    print(f"   MAIN:       [{bar_main:<50}] {stats['main_avg_score']:.1f}%")
    print(f"   HYPOTHESIS: [{bar_hyp:<50}] {stats['hyp_avg_score']:.1f}%")
    
    print(f"\nWIN/LOSS STATISTICS:")
    print(f"   MAIN wins:       {stats['wins_main']} ({stats['win_rate_main']:.1f}%)")
    print(f"   HYPOTHESIS wins: {stats['wins_hyp']} ({stats['win_rate_hyp']:.1f}%)")
    print(f"   Ties:            {stats['ties']} ({stats['tie_rate']:.1f}%)")
    
    print(f"\nIMPROVEMENT:")
    if stats['better_system'] == "HYPOTHESIS":
        print(f"   HYPOTHESIS is better by {stats['advantage_percent']:.1f}%")
    else:
        print(f"   MAIN is better by {stats['advantage_percent']:.1f}%")
    
    # Детали по вопросам
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    print(f"{'ID':<4} {'MAIN %':<10} {'HYP %':<10} {'WINNER':<12} {'ADVANTAGE'}")
    print("-" * 80)
    
    for r in results:
        main_score = r['main']['evaluation'].get('total', 0)
        hyp_score = r['hypothesis']['evaluation'].get('total', 0)
        winner = r['comparison'].get('winner', '?')
        winner_pct = r['comparison'].get('winner_percent', 0)
        
        if winner == 'A':
            winner_text = "[MAIN]"
        elif winner == 'B':
            winner_text = "[HYP]"
        else:
            winner_text = "[TIE]"
        
        adv = f"+{winner_pct:.0f}%" if winner_pct > 0 else "0%"
        print(f"{r['id']:<4} {main_score:<10.1f} {hyp_score:<10.1f} {winner_text:<12} {adv}")
    
    print("-" * 80)


def save_results(results: list, stats: dict):
    """Сохраняет результаты"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("llm_evaluation")
    output_dir.mkdir(exist_ok=True)
    
    # Сохраняем детальные результаты
    output_file = output_dir / f"llm_evaluation_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Сохраняем статистику
    stats_file = output_dir / f"statistics_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Сохраняем читаемый отчёт
    report_file = output_dir / f"report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM EVALUATION REPORT (PERCENTAGE BASED)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL SCORES:\n")
        f.write(f"  MAIN system:       {stats['main_avg_score']:.1f}%\n")
        f.write(f"  HYPOTHESIS system: {stats['hyp_avg_score']:.1f}%\n\n")
        
        f.write("WIN/LOSS:\n")
        f.write(f"  MAIN wins:       {stats['wins_main']} ({stats['win_rate_main']:.1f}%)\n")
        f.write(f"  HYPOTHESIS wins: {stats['wins_hyp']} ({stats['win_rate_hyp']:.1f}%)\n")
        f.write(f"  Ties:            {stats['ties']} ({stats['tie_rate']:.1f}%)\n\n")
        
        f.write(f"CONCLUSION: {stats['better_system']} is better by {stats['advantage_percent']:.1f}%\n\n")
        
        f.write("DETAILS BY QUESTION:\n")
        for r in results:
            f.write(f"\nQ{r['id']}: {r['question'][:80]}\n")
            f.write(f"  MAIN score: {r['main']['evaluation'].get('total', 0):.1f}%\n")
            f.write(f"  HYP score:  {r['hypothesis']['evaluation'].get('total', 0):.1f}%\n")
            f.write(f"  Winner: {r['comparison'].get('winner', '?')}\n")
            f.write(f"  Advantage: +{r['comparison'].get('winner_percent', 0):.0f}%\n")
    
    print(f"\n[SAVE] Results saved to: {output_dir}")
    print(f"   - {output_file.name}")
    print(f"   - {stats_file.name}")
    print(f"   - {report_file.name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Evaluation - Percentage Based')
    parser.add_argument('--main', '-m', type=str, required=True, help='Main answers JSON file')
    parser.add_argument('--hypothesis', '-hyp', type=str, required=True, help='Hypothesis answers JSON file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LLM EVALUATION - PERCENTAGE BASED (0-100%)")
    print("="*80)
    
    # Загружаем ответы
    print(f"\n[LOAD] Loading MAIN answers: {args.main}")
    answers_main = load_json(args.main)
    
    print(f"[LOAD] Loading HYPOTHESIS answers: {args.hypothesis}")
    answers_hypothesis = load_json(args.hypothesis)
    
    print(f"\n[INFO] MAIN: {len(answers_main)} answers")
    print(f"[INFO] HYPOTHESIS: {len(answers_hypothesis)} answers")
    
    # Проверяем Ollama
    try:
        ollama.list()
        print("[OK] Ollama is running\n")
    except:
        print("[ERROR] Ollama is not running!")
        return
    
    # Оцениваем
    results = evaluate_all(answers_main, answers_hypothesis)
    
    # Статистика
    stats = calculate_statistics(results)
    
    # Сохраняем
    save_results(results, stats)
    
    # Печатаем сводку
    print_summary(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
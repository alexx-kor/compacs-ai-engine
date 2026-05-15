#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM EVALUATION - OpenAI GPT version
Оценивает ответы через GPT в облаке
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logger import setup_logging
from core.openai_client import get_openai_client

log = logging.getLogger(__name__)
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_json(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        raw: object = json.load(handle)
    if isinstance(raw, list):
        return cast(list[dict[str, Any]], raw)
    return []


def llm_evaluate_percent(question: str, answer: str) -> dict[str, Any]:
    """Оценивает ответ через GPT"""
    
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
        response = get_openai_client().chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content or ""
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            parsed: object = json.loads(json_match.group())
            if not isinstance(parsed, dict):
                return {"error": "Failed to parse", "total": 0}
            result: dict[str, Any] = cast(dict[str, Any], parsed)
            for key in ["relevance", "accuracy", "completeness", "clarity", "total"]:
                if key in result:
                    result[key] = max(0, min(100, float(result[key])))
            return result
        return {"error": "Failed to parse", "total": 0}
    except Exception as e:
        return {"error": str(e), "total": 0}


def compare_two_answers_percent(question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
    """Сравнивает два ответа через GPT"""
    
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
        response = get_openai_client().chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content or ""
        json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if json_match:
            parsed: object = json.loads(json_match.group())
            if not isinstance(parsed, dict):
                return {"winner": "UNKNOWN", "winner_percent": 0}
            result = cast(dict[str, Any], parsed)
            if "winner_percent" in result:
                result["winner_percent"] = max(0, min(100, float(result["winner_percent"])))
            return result
        return {"winner": "UNKNOWN", "winner_percent": 0}
    except Exception as e:
        return {"winner": "ERROR", "winner_percent": 0, "error": str(e)}


def evaluate_all(
    answers_main: list[dict[str, Any]],
    answers_hypothesis: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Оценивает все ответы"""
    
    results = []
    
    main_dict = {item['id']: item for item in answers_main}
    hyp_dict = {item['id']: item for item in answers_hypothesis}
    
    log.info("%s", "\n" + "=" * 70)
    log.info("GPT MODEL: %s", GPT_MODEL)
    log.info("%s", "=" * 70)

    for qid in main_dict:
        if qid not in hyp_dict:
            continue

        question = main_dict[qid]['question']
        answer_main = main_dict[qid]['answer']
        answer_hyp = hyp_dict[qid]['answer']

        log.info("%s", "\n" + "=" * 70)
        log.info("Q%s: %s...", qid, question[:70])
        log.info("%s", "=" * 70)

        log.info("  [1/3] Evaluating MAIN answer...")
        eval_main = llm_evaluate_percent(question, answer_main)
        log.info("  MAIN done score=%.0f%%", eval_main.get('total', 0))

        log.info("  [2/3] Evaluating HYPOTHESIS answer...")
        eval_hyp = llm_evaluate_percent(question, answer_hyp)
        log.info("  HYP done score=%.0f%%", eval_hyp.get('total', 0))

        log.info("  [3/3] Comparing answers...")
        comparison = compare_two_answers_percent(question, answer_main, answer_hyp)
        log.info("  Compare done winner=%s", comparison.get('winner', '?'))
        
        results.append({
            'id': qid,
            'question': question,
            'main': {
                'answer': answer_main[:500],
                'evaluation': eval_main
            },
            'hypothesis': {
                'answer': answer_hyp[:500],
                'evaluation': eval_hyp
            },
            'comparison': comparison
        })
        
        winner = comparison.get('winner', '?')
        winner_pct = comparison.get('winner_percent', 0)
        winner_icon = "[MAIN]" if winner == 'A' else "[HYP]" if winner == 'B' else "[TIE]"
        
        log.info("%s", "\n  RESULT:")
        log.info("     MAIN: %.1f%%", eval_main.get('total', 0))
        log.info("     HYP:  %.1f%%", eval_hyp.get('total', 0))
        log.info("     WINNER: %s (+%.0f%%)", winner_icon, winner_pct)
    
    return results


def calculate_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Вычисляет статистику"""
    
    main_scores = [r['main']['evaluation'].get('total', 0) for r in results]
    hyp_scores = [r['hypothesis']['evaluation'].get('total', 0) for r in results]
    
    winners = [r['comparison'].get('winner', 'UNKNOWN') for r in results]
    
    wins_main = winners.count('A')
    wins_hyp = winners.count('B')
    ties = winners.count('TIE')
    
    avg_main = sum(main_scores) / len(main_scores) if main_scores else 0
    avg_hyp = sum(hyp_scores) / len(hyp_scores) if hyp_scores else 0
    
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
    }


def print_summary(results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    """Log evaluation summary (name kept for backward compatibility)."""
    _ = results

    log.info("%s", "\n" + "=" * 80)
    log.info("FINAL SUMMARY - OpenAI GPT EVALUATION")
    log.info("%s", "=" * 80)

    log.info("GPT MODEL: %s", GPT_MODEL)
    log.info("Total questions: %s", stats['total_questions'])

    log.info("%s", "\nOVERALL SCORES:")
    log.info("   MAIN system:       %.1f%%", stats['main_avg_score'])
    log.info("   HYPOTHESIS system: %.1f%%", stats['hyp_avg_score'])

    bar_main = '#' * int(stats['main_avg_score'] / 2)
    bar_hyp = '#' * int(stats['hyp_avg_score'] / 2)
    log.info("   MAIN:       [%s] %.1f%%", f"{bar_main:<50}", stats['main_avg_score'])
    log.info("   HYPOTHESIS: [%s] %.1f%%", f"{bar_hyp:<50}", stats['hyp_avg_score'])

    log.info("%s", "\nWIN/LOSS STATISTICS:")
    log.info("   MAIN wins:       %s (%.1f%%)", stats['wins_main'], stats['win_rate_main'])
    log.info("   HYPOTHESIS wins: %s (%.1f%%)", stats['wins_hyp'], stats['win_rate_hyp'])
    log.info("   Ties:            %s (%.1f%%)", stats['ties'], stats['tie_rate'])

    log.info("%s", "\nCONCLUSION:")
    if stats['better_system'] == "HYPOTHESIS":
        log.info("   HYPOTHESIS is better by %.1f%%", stats['advantage_percent'])
    else:
        log.info("   MAIN is better by %.1f%%", stats['advantage_percent'])


def save_results(results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    """Сохраняет результаты"""
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("gpt_evaluation")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"gpt_evaluation_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    stats_file = output_dir / f"statistics_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    log.info("[SAVE] Results saved to dir=%s", output_dir)
    log.info("   - %s", output_file.name)
    log.info("   - %s", stats_file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description='GPT Evaluation - Cloud LLM')
    parser.add_argument('--main', '-m', type=str, required=True, help='Main answers JSON file')
    parser.add_argument('--hypothesis', '-hyp', type=str, required=True, help='Hypothesis answers JSON file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable DEBUG logging')

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("Script started at %s", datetime.now(timezone.utc).isoformat())

    log.info("%s", "\n" + "=" * 80)
    log.info("GPT EVALUATION - Cloud-based LLM (OpenAI)")
    log.info("%s", "=" * 80)

    if not os.getenv('OPENAI_API_KEY'):
        log.error("OPENAI_API_KEY not set in .env file")
        log.error("Please add: OPENAI_API_KEY=your-key-here")
        return

    log.info("Using model: %s", GPT_MODEL)

    log.info("[LOAD] Loading MAIN answers path=%s", args.main)
    answers_main = load_json(args.main)

    log.info("[LOAD] Loading HYPOTHESIS answers path=%s", args.hypothesis)
    answers_hypothesis = load_json(args.hypothesis)

    results = evaluate_all(answers_main, answers_hypothesis)
    stats = calculate_statistics(results)
    save_results(results, stats)
    print_summary(results, stats)

    log.info("%s", "\n" + "=" * 80)
    log.info("GPT EVALUATION COMPLETE")
    log.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
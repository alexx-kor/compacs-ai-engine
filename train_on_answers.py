#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config


def add_to_few_shot(question: str, answer: str, source: str = "training"):
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"
    examples_file.parent.mkdir(parents=True, exist_ok=True)
    
    if examples_file.exists():
        df = pd.read_csv(examples_file)
    else:
        df = pd.DataFrame(columns=['question', 'answer', 'source'])
    
    if not df[df['question'] == question].empty:
        print(f" Example already exists: {question[:50]}...")
        return False
    
    new_row = pd.DataFrame([{
        'question': question,
        'answer': answer,
        'source': source
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(examples_file, index=False, encoding='utf-8')
    print(f" Added to few-shot: {question[:50]}...")
    return True


def train_from_evaluation_results(results_file: str, min_score: float = 0.7):
    if not os.path.exists(results_file):
        print(f" File not found: {results_file}")
        return 0
    
    df = pd.read_csv(results_file)
    good_answers = df[df['similarity_score'] >= min_score]
    
    print(f" Found {len(good_answers)} good answers (score >= {min_score})")
    
    added = 0
    for _, row in good_answers.iterrows():
        if add_to_few_shot(row['question'], row['generated_answer'], source="evaluation"):
            added += 1
    
    print(f" Added {added} new examples to few-shot")
    return added


def train_from_txt(questions_txt: str, answers_txt: str, source: str = "txt"):
    if not os.path.exists(questions_txt) or not os.path.exists(answers_txt):
        print(f" Files not found")
        return 0
    
    with open(questions_txt, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    with open(answers_txt, 'r', encoding='utf-8') as f:
        answers = [line.strip() for line in f if line.strip()]
    
    min_len = min(len(questions), len(answers))
    
    added = 0
    for i in range(min_len):
        if add_to_few_shot(questions[i], answers[i], source=source):
            added += 1
    
    print(f" Added {added} examples from TXT")
    return added


def list_examples():
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"
    
    if not examples_file.exists():
        print(" No examples found")
        return
    
    df = pd.read_csv(examples_file)
    print(f"\n FEW-SHOT EXAMPLES ({len(df)} total):")
    print("="*60)
    for i, row in df.iterrows():
        print(f"\n{i+1}. Q: {row['question'][:80]}...")
        print(f"   A: {row['answer'][:80]}...")


def clear_examples():
    examples_file = Path(config.few_shot_folder) / "training_examples.csv"
    if examples_file.exists():
        examples_file.unlink()
        print(" All examples cleared")
    else:
        print(" No examples to clear")


def main():
    parser = argparse.ArgumentParser(description='Train RAG on answers')
    parser.add_argument('--file', '-f', type=str, help='CSV file with evaluation results')
    parser.add_argument('--questions-txt', type=str, help='TXT file with questions')
    parser.add_argument('--answers-txt', type=str, help='TXT file with answers')
    parser.add_argument('--question', '-q', type=str, help='Single question to add')
    parser.add_argument('--answer', '-a', type=str, help='Answer for the question')
    parser.add_argument('--source', '-s', type=str, default='manual', help='Source of the example')
    parser.add_argument('--min-score', type=float, default=0.7, help='Minimum score to include')
    parser.add_argument('--list', '-l', action='store_true', help='List all examples')
    parser.add_argument('--clear', action='store_true', help='Clear all examples')
    
    args = parser.parse_args()
    
    if args.list:
        list_examples()
    elif args.clear:
        clear_examples()
    elif args.file:
        train_from_evaluation_results(args.file, args.min_score)
    elif args.questions_txt and args.answers_txt:
        train_from_txt(args.questions_txt, args.answers_txt, args.source)
    elif args.question and args.answer:
        add_to_few_shot(args.question, args.answer, args.source)
    else:
        print("""
        
                            TRAIN ON ANSWERS - FEW-SHOT LEARNING              
        
        
        Usage:
            # Train from evaluation results
            python train_on_answers.py --file data/results/evaluation_results.csv
            
            # Train from TXT files
            python train_on_answers.py --questions-txt questions.txt --answers-txt answers.txt
            
            # Add single example
            python train_on_answers.py --question "What is X?" --answer "X is Y"
            
            # List all examples
            python train_on_answers.py --list
            
            # Clear all examples
            python train_on_answers.py --clear
        """)


if __name__ == "__main__":
    main()
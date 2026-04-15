#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
from core.document_processor import doc_processor
from rag_engine.engine import rag
from evaluator.folder_scanner import FolderScanner
from evaluator.qa_loader import QALoader
from evaluator.results import ResultsAnalyzer
from router.smart_router import SmartPromptRouter
from datetime import datetime

def load_documents():
    print("\n" + "="*60)
    print("LOADING KNOWLEDGE BASE")
    print("="*60)
    
    if not config.doc_files:
        print(f"[WARN] No files found in: {config.docs_folder}")
        return 0
    
    # Создаем папку для ошибок загрузки
    error_folder = "./data/errors"
    os.makedirs(error_folder, exist_ok=True)
    
    sources = {}
    for _, source in config.doc_files:
        src = source.split('/')[0] if '/' in source else 'root'
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\nSources:")
    for src, count in sorted(sources.items()):
        print(f"   - {src}: {count} files")
    
    db.init_database()
    chunk_id = 0
    failed_files = []
    
    for file_path, source_name in config.doc_files:
        try:
            chunks = doc_processor.process_document(file_path, source_name, chunk_id)
            if chunks:
                texts = [c['chunk'] for c in chunks]
                print(f"   Generating {len(texts)} embeddings...")
                embeddings = embedder.generate(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk['embedding'] = emb
                db.insert_batch(chunks)
                chunk_id += len(chunks)
        except Exception as e:
            print(f"   [ERROR] FAILED: {source_name} - {str(e)[:50]}")
            failed_files.append({
                'file': file_path,
                'source': source_name,
                'error': str(e)
            })
            # Продолжаем со следующим файлом
            continue
    
    # Сохраняем список упавших файлов
    if failed_files:
        import json
        error_file = os.path.join(error_folder, f"failed_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(failed_files, f, ensure_ascii=False, indent=2)
        print(f"\n[WARN] Failed files saved to: {error_file}")
        print(f"   Total failed: {len(failed_files)} / {len(config.doc_files)}")
    
    print(f"\n[OK] TOTAL CHUNKS: {chunk_id}")
    return chunk_id


def find_all_qa_pairs():
    print("\n" + "="*60)
    print("SEARCHING FOR QUESTIONS & ANSWERS")
    print("="*60)
    
    scanner = FolderScanner(config.docs_folder)
    folders = scanner.scan()
    
    if not folders:
        print("[WARN] No questions/answers files found")
        print("   Expected: questions.txt + answers.txt in any subfolder")
        return []
    
    all_pairs = []
    for folder in folders:
        print(f"\nFolder: {folder['folder_name']}")
        print(f"   Questions: {folder['questions_file']}")
        print(f"   Answers: {folder['answers_file']}")
        
        qa_pairs = QALoader.load_qa_pairs(
            folder['questions_file'],
            folder['answers_file']
        )
        print(f"   Found pairs: {len(qa_pairs)}")
        all_pairs.extend(qa_pairs)
    
    print(f"\nTOTAL QA PAIRS FOUND: {len(all_pairs)}")
    return all_pairs


def run_evaluation(qa_pairs=None):
    print("\n" + "="*60)
    print("QA EVALUATION")
    print("="*60)
    
    if qa_pairs is None:
        qa_pairs = find_all_qa_pairs()
    
    if not qa_pairs:
        print("[ERROR] No QA pairs found for evaluation")
        return None
    
    # Создаем папку для ошибок
    error_folder = "./data/errors"
    os.makedirs(error_folder, exist_ok=True)
    
    examples_count = SmartPromptRouter.get_examples_count()
    if examples_count > 0:
        print(f"\nFew-shot examples: {examples_count}")
    else:
        print(f"\nNo few-shot examples. Add with: python train_on_answers.py --question ...")
    
    all_results = []
    failed_questions = []  # Список упавших вопросов
    
    for i, (question, expected_answer) in enumerate(qa_pairs):
        print(f"   [{i+1}/{len(qa_pairs)}] Processing...", end='\r')
        
        try:
            result = rag.ask(question)
            
            words_q = set(question.lower().split())
            words_a = set(result['answer'].lower().split())
            similarity = len(words_q & words_a) / max(len(words_q), 1) if words_q else 0
            
            all_results.append({
                'question': question[:200],
                'expected_answer': expected_answer[:200],
                'generated_answer': result['answer'][:300],
                'similarity_score': round(similarity, 3),
                'time_seconds': result['time_total'],
                'sources': str(result['sources']),
                'status': 'success'
            })
            
        except Exception as e:
            # Записываем упавший вопрос в исключение
            error_msg = str(e)
            print(f"\n   [WARN] ERROR on question {i+1}: {question[:50]}...")
            print(f"      Error: {error_msg[:100]}")
            
            # Сохраняем в файл ошибок
            failed_questions.append({
                'index': i+1,
                'question': question,
                'expected_answer': expected_answer,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            
            # Добавляем в результаты как ошибку
            all_results.append({
                'question': question[:200],
                'expected_answer': expected_answer[:200],
                'generated_answer': f"ERROR: {error_msg[:100]}",
                'similarity_score': 0,
                'time_seconds': 0,
                'sources': '',
                'status': 'error'
            })
            
            # Продолжаем со следующим вопросом
            continue
    
    print()
    
    # Сохраняем упавшие вопросы в отдельный файл
    if failed_questions:
        import json
        error_file = os.path.join(error_folder, f"failed_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(failed_questions, f, ensure_ascii=False, indent=2)
        print(f"\n[WARN] Failed questions saved to: {error_file}")
        print(f"   Total failed: {len(failed_questions)} / {len(qa_pairs)}")
    
    if all_results:
        df = ResultsAnalyzer.save(all_results)
        avg_score = df[df['status'] == 'success']['similarity_score'].mean() if len(df[df['status'] == 'success']) > 0 else 0
        print(f"\n[STATS] Average score (success only): {avg_score:.3f}")
        print(f"   Success rate: {len([r for r in all_results if r['status'] == 'success'])}/{len(all_results)}")
        print(f"\n[INFO] Train on good answers: python train_on_answers.py --file data/results/evaluation_results.csv")
    
    return all_results


def print_banner():
    print("")
    print("="*60)
    print("RAG SYSTEM - KNOWLEDGE BASE")
    print("="*60)
    print("")
    print("Commands:")
    print("    python run.py                    - Load docs + interactive")
    print("    python run.py --evaluate        - Evaluate on Q&A pairs")
    print("    python run.py --query \"...\"     - Ask a single question")
    print("    python run.py --load-only       - Only load documents")
    print("")


def main():
    parser = argparse.ArgumentParser(description='RAG System')
    parser.add_argument('--query', '-q', type=str, help='Ask a question')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Run QA evaluation')
    parser.add_argument('--load-only', action='store_true', help='Only load documents')
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        import ollama
        ollama.list()
        print("Ollama is running\n")
    except:
        print("[ERROR] Ollama is not running! Run: ollama serve")
        return
    
    if args.load_only:
        load_documents()
        return
    
    if args.evaluate:
        load_documents()
        run_evaluation()
        return
    
    if args.query:
        load_documents()
        result = rag.ask(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {result['sources']}")
        print(f"TIME: {result['time_total']}s")
        return
    
def load_documents():
    print("\n" + "="*60)
    print("LOADING KNOWLEDGE BASE")
    print("="*60)
    
    # Проверяем, есть ли уже данные в базе
    existing_chunks = db.get_chunk_count()
    
    if existing_chunks > 0:
        print(f"[INFO] Database already has {existing_chunks} chunks")
        print("[INFO] Skipping document loading...")
        print("[INFO] Use --force-reload to reload all documents")
        return existing_chunks
    
    if not config.doc_files:
        print(f"[WARN] No files found in: {config.docs_folder}")
        return 0
    
    # Создаём таблицу (без перезаписи)
    db.init_database(force_recreate=False)
    
    # ... остальной код загрузки ...


if __name__ == "__main__":
    main()
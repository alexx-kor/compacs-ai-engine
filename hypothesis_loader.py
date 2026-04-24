#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOTHESIS LOADER - с логированием
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.embeddings import embedder
from core.document_processor import doc_processor
from core.database import db
from core.logger import init_logger, get_logger


def load_hypothesis(
    hypothesis_name: str,
    hypothesis_params: dict = None,
    force_reload: bool = False
):
    """Загружает документы в таблицу гипотез с логированием"""
    
    print("\n" + "="*70)
    print(f" HYPOTHESIS LOADER: {hypothesis_name}")
    print("="*70)
    
    # Инициализируем логгер
    logger = init_logger(config.llm_model)
    start_total = time.time()
    
    # Параметры гипотезы
    params_str = json.dumps(hypothesis_params) if hypothesis_params else ""
    
    if not config.doc_files:
        print(f"[ERROR] No files found")
        return 0
    
    print(f"\n[INFO] Hypothesis: {hypothesis_name}")
    print(f"[INFO] Files: {len(config.doc_files)}")
    
    # Инициализируем таблицу гипотез
    if force_reload:
        db.init_hypothesis_database(force_recreate=True)
    else:
        db.init_hypothesis_database(force_recreate=False)
    
    # Загружаем документы
    chunk_id = 0
    all_chunks = []
    
    print(f"\n[STEP 1] CHUNKING")
    print("-" * 40)
    
    chunk_start = time.time()
    
    for i, (file_path, source_name) in enumerate(config.doc_files):
        file_start = time.time()
        chunks = doc_processor.process_document(file_path, source_name, chunk_id)
        
        if chunks:
            print(f"   [{i+1}/{len(config.doc_files)}] {source_name[:50]}... {len(chunks)} chunks", end='\r')
            
            # Логируем файл
            file_time_ms = (time.time() - file_start) * 1000
            logger.log_ingest_file(
                filename=source_name,
                source_type="TXT",
                chunk_count=len(chunks),
                embedding_time_ms=0,  # Будет позже
                total_time_ms=file_time_ms
            )
            
            all_chunks.extend(chunks)
            chunk_id += len(chunks)
        else:
            # Пропущенный файл
            logger.log_ingest_soft_skip(
                filename=source_name,
                reason="No valid chunks",
                file_size_bytes=os.path.getsize(file_path)
            )
    
    chunk_time = time.time() - chunk_start
    print(f"\n[CHUNKING] Done in {chunk_time:.2f}s")
    print(f"[CHUNKING] Total chunks: {len(all_chunks)}")
    
    # Генерация эмбеддингов
    print(f"\n[STEP 2] EMBEDDING GENERATION")
    print("-" * 40)
    
    embed_start = time.time()
    
    texts = [c['chunk'] for c in all_chunks]
    print(f"   Total texts: {len(texts)}")
    
    embeddings = embedder.generate(texts)
    
    for chunk, emb in zip(all_chunks, embeddings):
        chunk['embedding'] = emb
    
    embed_time = time.time() - embed_start
    embed_time_ms = embed_time * 1000
    
    # Логируем сводку по бэкенду
    logger.log_ingest_backend_summary()
    
    # Вставка в БД гипотез
    print(f"\n[STEP 3] DATABASE INSERT")
    print("-" * 40)
    
    insert_start = time.time()
    
    db.insert_hypothesis_batch(all_chunks, hypothesis_name, params_str)
    
    insert_time = time.time() - insert_start
    
    # Логируем сводку по батчу
    total_time_ms = (time.time() - start_total) * 1000
    logger.log_ingest_batch_summary(total_time_ms)
    
    print(f"\n[EMBEDDING] Done in {embed_time:.2f}s")
    print(f"[INSERT] Done in {insert_time:.2f}s")
    
    # ИТОГО
    total_time = time.time() - start_total
    
    print("\n" + "="*70)
    print(" HYPOTHESIS LOADING COMPLETE!")
    print("="*70)
    print(f"   Hypothesis: {hypothesis_name}")
    print(f"   Files: {len(config.doc_files)}")
    print(f"   Chunks: {len(all_chunks)}")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"\n Log file: {logger.log_file}")
    print("="*70)
    
    return len(all_chunks)


def list_hypotheses():
    """Показывает все гипотезы в базе"""
    
    print("\n" + "="*70)
    print(" HYPOTHESES IN DATABASE")
    print("="*70)
    
    hypotheses = db.get_all_hypotheses()
    
    if not hypotheses:
        print("\n   No hypotheses found")
    else:
        print(f"\n   Found {len(hypotheses)} hypotheses:")
        for h in hypotheses:
            count = db.get_hypothesis_chunk_count(h)
            print(f"      - {h}: {count} chunks")
    
    print("="*70)
    return hypotheses


def main():
    parser = argparse.ArgumentParser(description='Hypothesis Loader for RAG System')
    parser.add_argument('--name', '-n', type=str, help='Hypothesis name')
    parser.add_argument('--params', '-p', type=str, help='Hypothesis parameters (JSON string)')
    parser.add_argument('--force', action='store_true', help='Force reload')
    parser.add_argument('--list', '-l', action='store_true', help='List all hypotheses')
    
    args = parser.parse_args()
    
    if args.list:
        list_hypotheses()
        return
    
    if not args.name:
        print("[ERROR] Please specify --name for hypothesis")
        return
    
    # Парсим параметры
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except:
            print(f"[ERROR] Invalid JSON params: {args.params}")
            return
    
    # Загружаем гипотезу
    load_hypothesis(
        hypothesis_name=args.name,
        hypothesis_params=params,
        force_reload=args.force
    )


if __name__ == "__main__":
    main()
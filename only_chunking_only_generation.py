#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST CHUNKING & EMBEDDING ONLY
Только загрузка документов, чанкирование и генерация эмбеддингов
Никакого RAG, никакой оценки
"""

import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.database import db
from core.embeddings import embedder
from core.document_processor import doc_processor

# ============================================================================
# ТОЛЬКО ЧАНКИРОВАНИЕ И ГЕНЕРАЦИЯ
# ============================================================================

def run_chunking_and_embedding(force_reload: bool = False):
    """Запускает только чанкирование и генерацию эмбеддингов"""
    
    print("\n" + "="*70)
    print(" CHUNKING & EMBEDDING ONLY")
    print("="*70)
    
    start_total = time.time()
    
    # 1. Проверяем существующие данные
    existing_chunks = db.get_chunk_count()
    
    if existing_chunks > 0 and not force_reload:
        print(f"\n[INFO] Database already has {existing_chunks} chunks")
        print("[INFO] Use --force-reload to regenerate")
        print(f"[SKIP] Nothing to do. Exiting.")
        return existing_chunks
    
    # 2. Проверяем наличие файлов
    if not config.doc_files:
        print(f"\n[ERROR] No files found in: {config.docs_folder}")
        return 0
    
    # 3. Показываем статистику
    print(f"\n[SCAN] Found {len(config.doc_files)} files")
    
    total_size = 0
    sources = {}
    for file_path, source_name in config.doc_files:
        src = source_name.split('/')[0] if '/' in source_name else 'root'
        sources[src] = sources.get(src, 0) + 1
        try:
            total_size += os.path.getsize(file_path)
        except:
            pass
    
    print(f"[SIZE] Total: {total_size / 1024 / 1024:.1f} MB")
    print(f"\n[SOURCES]")
    for src, count in sorted(sources.items()):
        print(f"   - {src}: {count} files")
    
    # 4. Инициализируем БД
    print(f"\n[DB] Initializing...")
    db.init_database(force_recreate=force_reload)
    
    # 5. Чанкирование
    print(f"\n[STEP 1] CHUNKING")
    print("-" * 40)
    
    chunk_start = time.time()
    chunk_id = 0
    all_chunks = []
    
    for i, (file_path, source_name) in enumerate(config.doc_files):
        print(f"   [{i+1}/{len(config.doc_files)}] {source_name[:50]}...", end=' ')
        
        file_start = time.time()
        chunks = doc_processor.process_document(file_path, source_name, chunk_id)
        
        if chunks:
            print(f"{len(chunks)} chunks ({time.time()-file_start:.2f}s)")
            all_chunks.extend(chunks)
            chunk_id += len(chunks)
        else:
            print(f"no chunks ({time.time()-file_start:.2f}s)")
    
    chunk_time = time.time() - chunk_start
    print(f"\n[CHUNKING] Done in {chunk_time:.2f}s")
    print(f"[CHUNKING] Total chunks: {len(all_chunks)}")
    
    # 6. Генерация эмбеддингов
    print(f"\n[STEP 2] EMBEDDING GENERATION")
    print("-" * 40)
    
    embed_start = time.time()
    
    texts = [c['chunk'] for c in all_chunks]
    print(f"   Total texts: {len(texts)}")
    print(f"   Batch size: {config.batch_size}")
    
    embeddings = embedder.generate(texts)
    
    # Добавляем эмбеддинги к чанкам
    for chunk, emb in zip(all_chunks, embeddings):
        chunk['embedding'] = emb
    
    embed_time = time.time() - embed_start
    print(f"\n[EMBEDDING] Done in {embed_time:.2f}s")
    print(f"[EMBEDDING] Speed: {len(texts)/embed_time:.1f} chunks/sec")
    
    # 7. Вставка в БД
    print(f"\n[STEP 3] DATABASE INSERT")
    print("-" * 40)
    
    insert_start = time.time()
    
    # Вставляем батчами по 500
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        db.insert_batch(batch)
        percent = (i + len(batch)) / len(all_chunks) * 100
        print(f"   Progress: {i+len(batch)}/{len(all_chunks)} ({percent:.1f}%)", end='\r')
    
    insert_time = time.time() - insert_start
    print(f"\n[INSERT] Done in {insert_time:.2f}s")
    
    # 8. ИТОГО
    total_time = time.time() - start_total
    
    print("\n" + "="*70)
    print(" COMPLETE!")
    print("="*70)
    print(f"   Files processed: {len(config.doc_files)}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Avg chunk size: {sum(len(c['chunk']) for c in all_chunks)/len(all_chunks):.0f} chars")
    print(f"\n   Chunking time: {chunk_time:.2f}s")
    print(f"   Embedding time: {embed_time:.2f}s")
    print(f"   Insert time: {insert_time:.2f}s")
    print(f"   TOTAL TIME: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("="*70)
    
    return len(all_chunks)


def run_chunking_only():
    """Только чанкирование (без эмбеддингов) - для быстрого теста"""
    
    print("\n" + "="*70)
    print(" CHUNKING ONLY (NO EMBEDDINGS)")
    print("="*70)
    
    if not config.doc_files:
        print(f"\n[ERROR] No files found in: {config.docs_folder}")
        return 0
    
    print(f"\n[SCAN] Found {len(config.doc_files)} files")
    
    chunk_start = time.time()
    chunk_id = 0
    total_chunks = 0
    
    for i, (file_path, source_name) in enumerate(config.doc_files):
        print(f"   [{i+1}/{len(config.doc_files)}] {source_name[:50]}...", end=' ')
        
        file_start = time.time()
        chunks = doc_processor.process_document(file_path, source_name, chunk_id)
        
        if chunks:
            print(f"{len(chunks)} chunks ({time.time()-file_start:.2f}s)")
            total_chunks += len(chunks)
            chunk_id += len(chunks)
        else:
            print(f"no chunks ({time.time()-file_start:.2f}s)")
    
    chunk_time = time.time() - chunk_start
    
    print("\n" + "="*70)
    print(" COMPLETE!")
    print("="*70)
    print(f"   Files processed: {len(config.doc_files)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Time: {chunk_time:.2f}s")
    print(f"   Speed: {total_chunks/chunk_time:.1f} chunks/sec")
    print("="*70)
    
    return total_chunks


def check_status():
    """Проверяет текущее состояние базы данных"""
    
    print("\n" + "="*70)
    print(" DATABASE STATUS")
    print("="*70)
    
    chunk_count = db.get_chunk_count()
    print(f"\n   Total chunks: {chunk_count}")
    
    if chunk_count > 0:
        # Получаем список источников
        try:
            client = db.get_client()
            result = client.query("""
                SELECT source, count(*) as cnt 
                FROM default.rag_chunks 
                GROUP BY source 
                ORDER BY cnt DESC 
                LIMIT 10
            """)
            
            print(f"\n   Top sources:")
            for row in result.result_rows:
                print(f"      - {row[0]}: {row[1]} chunks")
        except:
            pass
    
    print("="*70)
    return chunk_count


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    CHUNKING & EMBEDDING SYSTEM                           ║
    ║                                                                          ║
    ║  Только загрузка документов, чанкирование и генерация эмбеддингов       ║
    ║                                                                          ║
    ║  Commands:                                                              ║
    ║    python run.py              - Полный цикл (чанки + эмбеддинги)        ║
    ║    python run.py --chunk-only - Только чанкирование (без эмбеддингов)   ║
    ║    python run.py --status     - Проверить статус БД                     ║
    ║    python run.py --force      - Принудительная перезагрузка             ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description='Chunking & Embedding System')
    parser.add_argument('--chunk-only', '-c', action='store_true', 
                        help='Только чанкирование (без эмбеддингов)')
    parser.add_argument('--status', '-s', action='store_true', 
                        help='Проверить статус базы данных')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Принудительная перезагрузка')
    
    args = parser.parse_args()
    
    # Проверка Ollama (только если нужны эмбеддинги)
    if not args.chunk_only and not args.status:
        try:
            import ollama
            ollama.list()
            print("✅ Ollama is running\n")
        except:
            print("❌ Ollama is not running! Run: ollama serve")
            return
    
    if args.status:
        check_status()
        return
    
    if args.chunk_only:
        run_chunking_only()
        return
    
    # По умолчанию - полный цикл
    run_chunking_and_embedding(force_reload=args.force)


if __name__ == "__main__":
    main()
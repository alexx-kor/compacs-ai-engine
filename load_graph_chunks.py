# load_graph_chunks.py
import json
from core.database import db
from core.embeddings import embedder
import hashlib
def load_graph_chunks(file_path: str = r"C:\Users\User\Desktop\RAG SYSTEM\data\graph_chunks.jsonl"):
    """Загружает графовые чанки из JSONL файла"""
    
    print(f"[LOAD] Loading graph chunks from {file_path}")
    
    # Читаем JSONL
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            chunks.append({
                'id': i,
                'source': data.get('source', 'unknown'),
                'page': data.get('page', 1),
                'chunk': data.get('chunk', ''),
                'answer': data.get('answer', ''),  # Новое поле
                'chunk_hash': hashlib.md5(data.get('chunk', '').encode()).hexdigest(),
                'char_count': len(data.get('chunk', ''))
            })
    
    print(f"[LOAD] Loaded {len(chunks)} chunks")
    
    # Генерируем эмбеддинги
    texts = [c['chunk'] for c in chunks]
    embeddings = embedder.generate(texts)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk['embedding'] = emb
    
    # Вставляем в БД
    db.insert_hypothesis_batch(chunks)  # или db.insert_hypothesis_batch
    
    print(f"[DONE] Loaded {len(chunks)} chunks into database")
    return chunks

if __name__ == "__main__":
    load_graph_chunks()
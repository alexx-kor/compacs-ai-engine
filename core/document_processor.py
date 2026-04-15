import os
import re
import hashlib
from typing import List, Dict, Tuple
from config import config
from core.embeddings import embedder


class DocumentProcessor:
    @staticmethod
    def load_document(file_path: str, source_name: str) -> List[Tuple[int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = []
            if len(content) > config.chunk_size:
                parts = []
                current = []
                current_len = 0
                
                for line in content.split('\n'):
                    if current_len + len(line) > config.chunk_size:
                        parts.append('\n'.join(current))
                        current = [line]
                        current_len = len(line)
                    else:
                        current.append(line)
                        current_len += len(line)
                
                if current:
                    parts.append('\n'.join(current))
                
                for i, part in enumerate(parts):
                    if len(part.strip()) > config.min_chunk_size:
                        chunks.append((i + 1, part.strip()))
            else:
                if len(content.strip()) > config.min_chunk_size:
                    chunks.append((1, content.strip()))
            
            return chunks
        except Exception as e:
            print(f"    Error loading {file_path}: {e}")
            return []
    
    @staticmethod
    def split_chunks(text: str) -> List[str]:
        size = config.chunk_size
        words = text.split()
        chunks = []
        step = size - config.chunk_overlap
        
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i+size])
            if len(chunk) > config.min_chunk_size:
                chunks.append(chunk)
                if len(chunks) >= config.max_chunks_per_doc:
                    break
        return chunks
    
    @staticmethod
    def process_document(file_path: str, source_name: str, start_id: int) -> List[Dict]:
        print(f"\n Processing: {source_name}")
        pages = DocumentProcessor.load_document(file_path, source_name)
        
        if not pages:
            return []
        
        chunks = []
        for page_num, text in pages:
            text_chunks = DocumentProcessor.split_chunks(text)
            
            for chunk in text_chunks:
                if len(chunk) > config.max_text_length:
                    chunk = chunk[:config.max_text_length]
                chunks.append({
                    'id': start_id + len(chunks),
                    'source': source_name,
                    'page': page_num,
                    'chunk': chunk,
                    'chunk_hash': hashlib.md5(chunk.encode()).hexdigest(),
                    'char_count': len(chunk)
                })
                if len(chunks) >= config.max_chunks_per_doc:
                    break
            if len(chunks) >= config.max_chunks_per_doc:
                break
        
        print(f"    Created {len(chunks)} chunks from {source_name}")
        return chunks


doc_processor = DocumentProcessor()
import re
import hashlib
from typing import List, Dict, Tuple
from pypdf import PdfReader
from config import config
from core.embeddings import embedder


class PDFProcessor:
    @staticmethod
    def extract_pdf(pdf_path: str, source_name: str) -> List[Tuple[int, str]]:
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"    Total pages: {total_pages}")
            
            pages = []
            for i in range(total_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text and len(text.strip()) > config.min_chunk_size:
                        text = re.sub(r'\n+', ' ', text)
                        pages.append((i + 1, text.strip()))
                except:
                    pass
            return pages
        except Exception as e:
            print(f"    Error: {e}")
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
    def process_document(pdf_path: str, source_name: str, start_id: int) -> List[Dict]:
        print(f"\n Processing: {source_name}")
        pages = PDFProcessor.extract_pdf(pdf_path, source_name)
        
        if not pages:
            return []
        
        chunks = []
        for page_num, text in pages:
            for chunk in PDFProcessor.split_chunks(text):
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
        
        print(f"    Created {len(chunks)} chunks")
        return chunks


pdf_processor = PDFProcessor()
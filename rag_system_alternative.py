#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG SYSTEM WITH CLICKHOUSE + OLLAMA
Retrieval-Augmented Generation system for document querying.
Optimized for large documents (1000+ pages).
Target: 15-25 seconds per response, Accuracy: 3.5-4.0/5.
"""

__version__ = "2.0.0"
__author__ = "RAG System"
__description__ = "Production RAG system with ClickHouse and Ollama"

# ==============================================================================
# DEPENDENCY CHECK
# ==============================================================================

def check_dependencies():
    """Check if all required packages are installed"""
    required = {
        'clickhouse_connect': 'ClickHouse',
        'ollama': 'Ollama',
        'pypdf': 'PDF reader',
        'pandas': 'Data processing',
        'numpy': 'Numerical operations'
    }
    
    optional = {
        'IPython': 'Jupyter widgets',
        'tqdm': 'Progress bars',
        'sklearn': 'Scikit-learn (reranking)',
        'pyarrow': 'Parquet export',
        'openpyxl': 'Excel export'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, name in required.items():
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
            print(f"[ERROR] {name} - MISSING")
    
    for package, name in optional.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n[WARN] Missing required packages: {', '.join(missing_required)}")
        print(f"Install: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n[INFO] Optional packages not installed: {', '.join(missing_optional)}")
        print(f"Install for full features: pip install {' '.join(missing_optional)}")
    
    print("[OK] All required dependencies are installed!")
    return True

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import json
import time
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Core imports
import ollama
import clickhouse_connect
import pandas as pd
import numpy as np
from pypdf import PdfReader

# Optional imports with fallbacks
try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

try:
    from tqdm.notebook import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs):
            return iterable

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Configuration class for RAG system"""
    
    # ClickHouse connection
    ch_host: str = field(default_factory=lambda: os.getenv('CLICKHOUSE_HOST', 'ug1o26imbr.eu-central-1.aws.clickhouse.cloud'))
    ch_user: str = field(default_factory=lambda: os.getenv('CLICKHOUSE_USER', 'default'))
    ch_password: str = field(default_factory=lambda: os.getenv('CLICKHOUSE_PASSWORD', '~MlK_g7KdbqYH'))
    ch_secure: bool = field(default_factory=lambda: os.getenv('CLICKHOUSE_SECURE', 'true').lower() == 'true')
    
    # Ollama models
    embed_model: str = field(default_factory=lambda: os.getenv('EMBED_MODEL', 'nomic-embed-text'))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'llama3.2:3b'))
    
    # RAG parameters
    chunk_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '1000')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '150')))
    top_k: int = field(default_factory=lambda: int(os.getenv('TOP_K', '8')))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv('RERANK_TOP_K', '3')))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv('SIMILARITY_THRESHOLD', '0.35')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('BATCH_SIZE', '32')))
    
    # Generation parameters
    num_ctx: int = field(default_factory=lambda: int(os.getenv('NUM_CTX', '4096')))
    num_predict: int = field(default_factory=lambda: int(os.getenv('NUM_PREDICT', '400')))
    temperature: float = field(default_factory=lambda: float(os.getenv('TEMPERATURE', '0.1')))
    top_p: float = field(default_factory=lambda: float(os.getenv('TOP_P', '0.9')))
    repeat_penalty: float = field(default_factory=lambda: float(os.getenv('REPEAT_PENALTY', '1.1')))
    
    # Limits
    max_text_length: int = field(default_factory=lambda: int(os.getenv('MAX_TEXT_LENGTH', '3072')))
    min_chunk_size: int = field(default_factory=lambda: int(os.getenv('MIN_CHUNK_SIZE', '100')))
    max_chunks_per_doc: int = field(default_factory=lambda: int(os.getenv('MAX_CHUNKS_PER_DOC', '2000')))
    
    # Cache
    cache_enabled: bool = field(default_factory=lambda: os.getenv('CACHE_ENABLED', 'true').lower() == 'true')
    cache_ttl: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '3600')))
    
    # File paths
    pdf_files: List[Tuple[str, str]] = field(default_factory=lambda: [
        (r'C:\Users\User\Desktop\Folder_vs_documents\integration.pdf', 'Integration'),
        (r'C:\Users\User\Desktop\Folder_vs_documents\manager.pdf', 'Manager'),
        (r'C:\Users\User\Desktop\Folder_vs_documents\merchant.pdf', 'Merchant'),
    ])
    questions_csv: str = field(default_factory=lambda: os.getenv('QUESTIONS_CSV', r'C:\Users\User\Desktop\Folder_vs_documents\questions.csv'))
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.ch_password and self.ch_host != 'localhost':
            print("[ERROR] Password required for remote ClickHouse")
            return False
        if self.chunk_size <= self.chunk_overlap:
            print("[ERROR] chunk_size must be > chunk_overlap")
            return False
        if not 0 < self.similarity_threshold < 1:
            print("[ERROR] similarity_threshold must be between 0 and 1")
            return False
        return True

config = Config()

# ==============================================================================
# DATABASE MANAGER
# ==============================================================================

class DatabaseManager:
    """Manages ClickHouse database operations"""
    
    def __init__(self):
        self._client = None
        self._cache = {}
        self._cache_time = {}
    
    def get_client(self):
        """Get or create ClickHouse client"""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=config.ch_host,
                username=config.ch_user,
                password=config.ch_password,
                secure=config.ch_secure,
                compress=True,
                connect_timeout=30
            )
            print("[OK] Connected to ClickHouse")
        return self._client
    
    def init_database(self):
        """Initialize database schema"""
        client = self.get_client()
        client.command("DROP TABLE IF EXISTS default.rag_chunks")
        client.command("""
            CREATE TABLE default.rag_chunks (
                id UInt64,
                source String,
                page UInt32,
                chunk String,
                embedding Array(Float32),
                chunk_hash String,
                char_count UInt32,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY source
            ORDER BY id
        """)
        print("[OK] Database initialized")
    
    def insert_batch(self, chunks: List[Dict]):
        """Insert chunks in batches"""
        if not chunks:
            return
        client = self.get_client()
        rows = [[c['id'], c['source'], c['page'], c['chunk'], 
                 c['embedding'], c['chunk_hash'], c['char_count']] for c in chunks]
        client.insert('default.rag_chunks', rows,
                     column_names=['id', 'source', 'page', 'chunk', 'embedding', 'chunk_hash', 'char_count'])
        print(f"   [OK] Inserted {len(chunks)} chunks")
    
    def search(self, embedding: List[float]) -> List[tuple]:
        """Search for similar chunks"""
        client = self.get_client()
        query = """
            SELECT chunk, source, page, cosineDistance(embedding, %(emb)s) AS distance
            FROM default.rag_chunks
            WHERE distance < %(threshold)s
            ORDER BY distance ASC
            LIMIT %(top_k)s
        """
        result = client.query(query, parameters={
            'emb': embedding,
            'threshold': config.similarity_threshold,
            'top_k': config.top_k
        })
        return result.result_rows
    
    def get_cache(self, key: str):
        """Get cached value"""
        if not config.cache_enabled:
            return None
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < config.cache_ttl:
                return self._cache[key]
        return None
    
    def set_cache(self, key: str, value: str):
        """Set cached value"""
        if config.cache_enabled:
            self._cache[key] = value
            self._cache_time[key] = time.time()

db = DatabaseManager()

# ==============================================================================
# EMBEDDING GENERATOR
# ==============================================================================

class EmbeddingGenerator:
    """Generates embeddings for text chunks"""
    
    def __init__(self):
        self.model = config.embed_model
        self.batch_size = config.batch_size
        self.max_length = config.max_text_length
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to safe length"""
        if len(text) <= self.max_length:
            return text
        truncated = text[:self.max_length]
        last_period = truncated.rfind('.')
        if last_period > self.max_length // 2:
            truncated = truncated[:last_period + 1]
        return truncated.strip()
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []
        safe_texts = [self._truncate_text(t) for t in texts]
        
        try:
            response = ollama.embed(model=self.model, input=safe_texts)
            return response['embeddings']
        except Exception as e:
            print(f"   [WARN] Embedding error: {e}")
            return [[0.0] * 768 for _ in safe_texts]
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for all texts with batching"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            embeddings = self.generate_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    @lru_cache(maxsize=256)
    def generate_cached(self, text: str) -> tuple:
        """Generate embedding with caching"""
        embedding = self.generate_batch([text])[0]
        return tuple(embedding)

embedder = EmbeddingGenerator()

# ==============================================================================
# RERANKER
# ==============================================================================

class Reranker:
    """Reranks search results for better accuracy"""
    
    @staticmethod
    def rerank(question: str, results: List[tuple]) -> List[tuple]:
        """Rerank results by keyword overlap and similarity"""
        if not results:
            return results
        
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        
        scored = []
        for idx, result in enumerate(results):
            chunk, source, page, distance = result
            c_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            similarity = 1 - distance
            final_score = similarity * 0.6 + overlap * 0.4
            scored.append((final_score, result))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:config.rerank_top_k]]

reranker = Reranker()

# ==============================================================================
# PDF PROCESSOR
# ==============================================================================

class PDFProcessor:
    """Processes PDF documents into chunks"""
    
    @staticmethod
    def extract_pdf(pdf_path: str, source_name: str) -> List[Tuple[int, str]]:
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            print(f"   Total pages: {total_pages}")
            
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
            print(f"   [ERROR] Error: {e}")
            return []
    
    @staticmethod
    def split_chunks(text: str) -> List[str]:
        """Split text into overlapping chunks"""
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
        """Process entire document into chunks"""
        print(f"\nProcessing: {source_name}")
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
        
        print(f"   [OK] Created {len(chunks)} chunks")
        return chunks

pdf_processor = PDFProcessor()

# ==============================================================================
# RAG ENGINE
# ==============================================================================

SYSTEM_PROMPT = """You are a technical documentation expert. Answer based ONLY on the provided context.

FORMAT:
ANSWER: [clear, specific answer]
SOURCE: [document name, page X]
EVIDENCE: [exact quote from documentation]

If information not found: "NOT FOUND in documentation"

Be concise, accurate, and always cite sources."""

class RAGEngine:
    """Main RAG engine for question answering"""
    
    @staticmethod
    def ask(question: str) -> Dict:
        """Ask a question and get answer"""
        t_start = time.time()
        
        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = db.get_cache(cache_key)
        if cached:
            result = json.loads(cached)
            result['cached'] = True
            result['time_total'] = 0.5
            return result
        
        # Search
        q_emb = list(embedder.generate_cached(question))
        results = db.search(q_emb)
        
        if not results:
            return {
                'question': question,
                'answer': "NOT FOUND in documentation",
                'sources': [],
                'time_total': round(time.time() - t_start, 2)
            }
        
        # Rerank
        reranked = reranker.rerank(question, results)
        
        # Prepare context
        context_parts = []
        sources = []
        for r in reranked:
            chunk, source, page = r[0], r[1], r[2]
            context_parts.append(f"[{source}, p.{page}]\n{chunk[:800]}\n[/]")
            sources.append((source, page))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
        
        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=messages,
                options={
                    "num_predict": config.num_predict,
                    "temperature": config.temperature,
                    "top_k": 40,
                    "top_p": config.top_p,
                    "num_ctx": config.num_ctx,
                    "repeat_penalty": config.repeat_penalty
                }
            )
            answer = response.message.content
        except Exception as e:
            answer = f"ERROR: {e}"
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'time_total': round(time.time() - t_start, 2),
            'cached': False
        }
        
        db.set_cache(cache_key, json.dumps(result))
        return result

rag = RAGEngine()

# ==============================================================================
# BENCHMARK
# ==============================================================================

class Benchmark:
    """Benchmarking utilities"""
    
    @staticmethod
    def quick_judge(question: str, answer: str) -> int:
        """Quick quality judgement"""
        prompt = f"""Rate answer quality 1-5. Question: {question[:100]} Answer: {answer[:200]}
Reply ONLY a number 1-5."""
        
        try:
            resp = ollama.chat(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 5, "temperature": 0}
            )
            match = re.search(r'[1-5]', resp.message.content)
            return int(match.group()) if match else 0
        except:
            return 0
    
    @staticmethod
    def run(questions: List[str]) -> pd.DataFrame:
        """Run benchmark on questions"""
        results = []
        for i, q in enumerate(tqdm(questions, desc="   Benchmarking")):
            try:
                res = rag.ask(q)
                score = Benchmark.quick_judge(q, res['answer'])
                results.append({
                    'id': i+1,
                    'question': q,
                    'answer': res['answer'][:300],
                    'score': score,
                    'time': res['time_total'],
                    'sources': len(res['sources']),
                    'status': 'ok'
                })
            except Exception as e:
                results.append({
                    'id': i+1,
                    'question': q,
                    'answer': f'ERROR: {e}',
                    'score': 0,
                    'time': 0,
                    'sources': 0,
                    'status': 'error'
                })
        return pd.DataFrame(results)

# ==============================================================================
# DATA EXPORT
# ==============================================================================

class DataExporter:
    """Export data to various formats"""
    
    @staticmethod
    def to_csv(filename: str = "rag_chunks.csv") -> pd.DataFrame:
        """Export chunks to CSV"""
        print(f"\nExporting to {filename}...")
        client = db.get_client()
        result = client.query("SELECT id, source, page, chunk, char_count FROM default.rag_chunks ORDER BY id")
        df = pd.DataFrame(result.result_rows, columns=['id', 'source', 'page', 'chunk', 'char_count'])
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"   [OK] Exported {len(df)} chunks")
        return df
    
    @staticmethod
    def to_json(filename: str = "rag_results.json", results: List[Dict] = None):
        """Export results to JSON"""
        if results:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"   [OK] Exported to {filename}")
    
    @staticmethod
    def to_parquet(filename: str = "rag_data.parquet"):
        """Export to Parquet (requires pyarrow)"""
        try:
            import pyarrow
            df = DataExporter.to_csv(filename.replace('.parquet', '.csv'))
            df.to_parquet(filename)
            print(f"   [OK] Exported to {filename}")
        except ImportError:
            print("   [WARN] pyarrow not installed, skipping parquet export")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def load_documents():
    """Load all documents into database"""
    print("\n" + "="*60)
    print("LOADING DOCUMENTS")
    print("="*60)
    
    for pdf_path, _ in config.pdf_files:
        if not os.path.exists(pdf_path):
            print(f"[ERROR] File not found: {pdf_path}")
            return 0
    
    db.init_database()
    chunk_id = 0
    
    for pdf_path, source_name in config.pdf_files:
        chunks = pdf_processor.process_document(pdf_path, source_name, chunk_id)
        if chunks:
            texts = [c['chunk'] for c in chunks]
            print(f"   Generating {len(texts)} embeddings...")
            embeddings = embedder.generate(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk['embedding'] = emb
            db.insert_batch(chunks)
            chunk_id += len(chunks)
    
    print(f"\nTOTAL CHUNKS: {chunk_id}")
    return chunk_id

def print_banner():
    """Print system banner"""
    print("""
    ======================================================================
                        RAG SYSTEM v2.0

      Retrieval-Augmented Generation with ClickHouse + Ollama

      Features:
      - Vector search in ClickHouse
      - Reranking for better accuracy
      - Caching for repeated queries
      - Export to CSV/JSON/Parquet

      Expected: 15-25 seconds/query | Accuracy: 3.5-4.0/5
    ======================================================================
    """)

def main():
    """Main execution function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Please install missing dependencies and try again.")
        return
    
    # Validate config
    if not config.validate():
        return
    
    # Check Ollama
    try:
        ollama.list()
        print("[OK] Ollama is running\n")
    except:
        print("[ERROR] Ollama is not running!")
        print("   Run: ollama serve")
        print("   Then: ollama pull llama3.2:3b")
        print("   And: ollama pull nomic-embed-text")
        return
    
    # Load documents
    total_chunks = load_documents()
    if total_chunks == 0:
        print("[ERROR] No documents loaded!")
        return
    
    # Quick test
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    test_q = "Which requests and responses I need to implement for Sale Form integration?"
    print(f"\nQuestion: {test_q[:80]}...")
    
    start = time.time()
    result = rag.ask(test_q)
    elapsed = time.time() - start
    
    print(f"\nANSWER:\n{result['answer'][:500]}")
    print(f"\nSOURCES: {result['sources']}")
    print(f"TIME: {elapsed:.1f} seconds")
    
    if elapsed < 25:
        print(f"   [OK] Speed target achieved! ({elapsed:.1f}s)")
    else:
        print(f"   [WARN] Still slow ({elapsed:.1f}s). Try: config.num_ctx = 2048")
    
    # Run benchmark if questions exist
    if os.path.exists(config.questions_csv):
        df = pd.read_csv(config.questions_csv)
        questions = df.iloc[:, 0].dropna().tolist()
        print(f"\nRunning benchmark on {len(questions)} questions...")
        
        results_df = Benchmark.run(questions)
        
        avg_score = results_df['score'].mean()
        avg_time = results_df['time'].mean()
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Average Score: {avg_score:.1f} / 5")
        print(f"Average Time: {avg_time:.1f} seconds")
        
        # Save results
        results_df.to_csv('benchmark_results.csv', index=False)
        DataExporter.to_json('benchmark_results.json', results_df.to_dict('records'))
        print(f"\nResults saved to: benchmark_results.csv")
    
    # Ask about export
    print("\n" + "="*60)
    print("DATA EXPORT")
    print("="*60)
    print("Export options:")
    print("  1. Export chunks to CSV")
    print("  2. Export chunks to Parquet")
    print("  3. Skip export")
    
    # Auto-export for non-interactive mode
    DataExporter.to_csv()
    
    print("\n" + "="*60)
    print("[OK] RAG SYSTEM READY!")
    print("="*60)
    print("\nUsage examples:")
    print("   result = rag.ask('your question')")
    print("   print(result['answer'])")
    print("   print(result['sources'])")
    print("\nCached questions respond in < 1 second")

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def cli():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System with ClickHouse + Ollama')
    parser.add_argument('--query', '-q', type=str, help='Question to ask')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmark')
    parser.add_argument('--export', '-e', action='store_true', help='Export data')
    parser.add_argument('--version', '-v', action='version', version=f'RAG System v{__version__}')
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        print_banner()
        if load_documents() > 0:
            result = rag.ask(args.query)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['sources']}")
            print(f"Time: {result['time_total']}s")
    elif args.benchmark:
        # Benchmark mode
        main()
    elif args.export:
        # Export mode
        if load_documents() > 0:
            DataExporter.to_csv()
            DataExporter.to_parquet()
    else:
        # Interactive mode
        main()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Check if running in Jupyter
    if 'get_ipython' in globals():
        main()
    else:
        cli()
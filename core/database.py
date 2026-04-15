import time
import json
from typing import List, Dict, Tuple
import clickhouse_connect
from config import config


class DatabaseManager:
    def __init__(self):
        self._client = None
        self._cache = {}
        self._cache_time = {}
    
    def get_client(self):
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=config.ch_host,
                username=config.ch_user,
                password=config.ch_password,
                secure=config.ch_secure,
                compress=True,
                connect_timeout=30
            )
            print(f"[OK] Connected to ClickHouse")
        return self._client
    
    def init_database(self, force_recreate: bool = False):
        """Инициализирует базу данных
        
        Args:
            force_recreate: Если True - пересоздаёт таблицу (удаляет все данные)
                           Если False - создаёт таблицу только если её нет
        """
        client = self.get_client()
        
        # Проверяем существование таблицы
        try:
            result = client.query("EXISTS TABLE default.rag_chunks")
            table_exists = result.result_rows[0][0] if result.result_rows else False
        except:
            table_exists = False
        
        if table_exists and not force_recreate:
            print("[OK] Database already exists, reusing existing data")
            # Проверяем количество чанков
            try:
                count_result = client.query("SELECT count(*) FROM default.rag_chunks")
                chunk_count = count_result.result_rows[0][0] if count_result.result_rows else 0
                print(f"[OK] Existing chunks in database: {chunk_count}")
            except:
                pass
            return
        
        # Если таблица не существует или force_recreate=True
        if force_recreate:
            print("[WARN] Force recreating database...")
            client.command("DROP TABLE IF EXISTS default.rag_chunks")
        
        client.command("""
            CREATE TABLE IF NOT EXISTS default.rag_chunks (
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
        if not chunks:
            return
        client = self.get_client()
        rows = [[c['id'], c['source'], c['page'], c['chunk'], 
                 c['embedding'], c['chunk_hash'], c['char_count']] for c in chunks]
        client.insert('default.rag_chunks', rows,
                     column_names=['id', 'source', 'page', 'chunk', 'embedding', 'chunk_hash', 'char_count'])
        print(f"   [OK] Inserted {len(chunks)} chunks")
    
    def get_chunk_count(self) -> int:
        """Возвращает количество чанков в базе"""
        try:
            client = self.get_client()
            result = client.query("SELECT count(*) FROM default.rag_chunks")
            return result.result_rows[0][0] if result.result_rows else 0
        except:
            return 0
    
    def search(self, embedding: List[float]) -> List[tuple]:
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
        if not config.cache_enabled:
            return None
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < config.cache_ttl:
                return self._cache[key]
        return None
    
    def set_cache(self, key: str, value: str):
        if config.cache_enabled:
            self._cache[key] = value
            self._cache_time[key] = time.time()


db = DatabaseManager()
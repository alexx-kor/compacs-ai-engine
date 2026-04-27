"""ClickHouse storage operations for RAG chunks."""

import logging
import time
from typing import Any

import clickhouse_connect
from config import config

log = logging.getLogger(__name__)


class DatabaseManager:
    """Manage ClickHouse operations and in-memory cache."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._cache: dict[str, str] = {}
        self._cache_time: dict[str, float] = {}

    def get_client(self) -> Any:
        """Return lazily initialized ClickHouse client."""
        if self._client is None:
            assert config.ch_password is not None, "CLICKHOUSE_PASSWORD must be set"
            self._client = clickhouse_connect.get_client(
                host=config.ch_host,
                username=config.ch_user,
                password=config.ch_password,
                secure=config.ch_secure,
                compress=True,
                connect_timeout=30,
            )
            log.info("connected to clickhouse host=%s secure=%s", config.ch_host, config.ch_secure)
        return self._client

    def init_database(self, force_recreate: bool = False) -> None:
        """Initialize chunk table in ClickHouse.

        Args:
            force_recreate: Drop and recreate table when True.
        """
        client = self.get_client()

        try:
            result = client.query("EXISTS TABLE default.rag_chunks")
            table_exists = result.result_rows[0][0] if result.result_rows else False
        except (RuntimeError, OSError, ValueError) as error:
            log.warning("table existence check failed: %s", error)
            table_exists = False

        if table_exists and not force_recreate:
            log.info("database already exists, reusing existing data")
            try:
                count_result = client.query("SELECT count(*) FROM default.rag_chunks")
                chunk_count = count_result.result_rows[0][0] if count_result.result_rows else 0
                log.info("existing chunks count=%s", chunk_count)
            except (RuntimeError, OSError, ValueError) as error:
                log.warning("chunk count query failed: %s", error)
            return

        if force_recreate:
            log.warning("force recreate enabled: dropping old table")
            client.command("DROP TABLE IF EXISTS default.rag_chunks")

        client.command(
            """
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
        """
        )
        log.info("database initialized")

    def insert_batch(self, chunks: list[dict]) -> None:
        """Insert chunk batch into ClickHouse."""
        if not chunks:
            return
        client = self.get_client()
        rows = [
            [c["id"], c["source"], c["page"], c["chunk"], c["embedding"], c["chunk_hash"], c["char_count"]]
            for c in chunks
        ]
        client.insert(
            "default.rag_chunks",
            rows,
            column_names=["id", "source", "page", "chunk", "embedding", "chunk_hash", "char_count"],
        )
        log.info("inserted chunks count=%s", len(chunks))

    def get_chunk_count(self) -> int:
        """Return chunk count from ClickHouse table."""
        try:
            client = self.get_client()
            result = client.query("SELECT count(*) FROM default.rag_chunks")
            return result.result_rows[0][0] if result.result_rows else 0
        except (RuntimeError, OSError, ValueError) as error:
            log.warning("chunk count failed: %s", error)
            return 0

    def search(self, embedding: list[float]) -> list[tuple]:
        """Search similar chunks in ClickHouse."""
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
        """Return cached answer when valid."""
        if not config.cache_enabled:
            return None
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < config.cache_ttl:
                return self._cache[key]
        return None

    def set_cache(self, key: str, value: str) -> None:
        """Store value in in-memory cache."""
        if config.cache_enabled:
            self._cache[key] = value
            self._cache_time[key] = time.time()


db = DatabaseManager()
"""ClickHouse-backed vector store."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import clickhouse_connect
import numpy as np

from config import Config
from core.storage.protocol import ChunkRecord, VectorStore, metadata_from_json, metadata_to_json

log = logging.getLogger(__name__)

_TABLE = "default.rag_chunks"


class ClickHouseVectorStore(VectorStore):
    """Production vector storage using ClickHouse cosine distance."""

    def __init__(self, config: Config) -> None:
        if not config.clickhouse_password:
            raise ValueError("CLICKHOUSE_PASSWORD is required for ClickHouse backend")
        self._config = config
        try:
            self._client = clickhouse_connect.get_client(
                host=config.clickhouse_host,
                port=config.clickhouse_port,
                username=config.clickhouse_user,
                password=config.clickhouse_password,
                secure=config.clickhouse_secure,
                compress=True,
                connect_timeout=30,
            )
            self._ensure_table()
            log.info(
                "ClickHouse vector store initialized host=%s port=%s",
                config.clickhouse_host,
                config.clickhouse_port,
            )
        except Exception as error:
            log.error("ClickHouse initialization failed: %s", error)
            raise ConnectionError(f"Cannot connect to ClickHouse: {error}") from error

    @property
    def backend_name(self) -> str:
        return "clickhouse"

    def init_store(self, force_recreate: bool = False) -> None:
        if force_recreate:
            self._client.command(f"DROP TABLE IF EXISTS {_TABLE}")
        self._ensure_table()

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int,
        similarity_threshold: float,
        source_prefixes: list[str] | None = None,
    ) -> list[ChunkRecord]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        prefix_filter = ""
        parameters: dict[str, Any] = {
            "embedding": [float(value) for value in query_embedding.tolist()],
            "threshold": similarity_threshold,
            "limit": limit,
        }
        if source_prefixes:
            prefix_filter = " AND (" + " OR ".join(
                f"startsWith(source, %(prefix{i})s)" for i in range(len(source_prefixes))
            ) + ")"
            for index, prefix in enumerate(source_prefixes):
                parameters[f"prefix{index}"] = prefix
        query = f"""
            SELECT
                toString(id),
                source,
                page,
                chunk,
                embedding,
                metadata,
                dataset_kind,
                created_at
            FROM {_TABLE}
            WHERE cosineDistance(embedding, %(embedding)s) < %(threshold)s
            {prefix_filter}
            ORDER BY cosineDistance(embedding, %(embedding)s) ASC
            LIMIT %(limit)s
        """
        result = self._client.query(query, parameters=parameters)
        return [self._row_to_record(row) for row in result.result_rows]

    def insert_batch(self, chunks: list[ChunkRecord]) -> int:
        if not chunks:
            raise ValueError("chunks list must not be empty")
        rows = [self._record_to_insert_row(chunk) for chunk in chunks]
        self._client.insert(
            _TABLE,
            rows,
            column_names=[
                "id",
                "source",
                "page",
                "chunk",
                "embedding",
                "chunk_hash",
                "char_count",
                "metadata",
                "dataset_kind",
                "created_at",
            ],
        )
        log.info("Inserted chunks count=%s backend=clickhouse", len(chunks))
        return len(chunks)

    def delete_by_source(self, source: str) -> int:
        before = self.chunk_count()
        self._client.command(
            f"ALTER TABLE {_TABLE} DELETE WHERE source = %(source)s",
            parameters={"source": source},
        )
        after = self.chunk_count()
        return max(before - after, 0)

    def delete_by_source_prefix(self, prefix: str) -> int:
        before = self.chunk_count()
        self._client.command(
            f"ALTER TABLE {_TABLE} DELETE WHERE startsWith(source, %(prefix)s)",
            parameters={"prefix": prefix},
        )
        after = self.chunk_count()
        return max(before - after, 0)

    def list_sources(self) -> list[str]:
        result = self._client.query(f"SELECT DISTINCT source FROM {_TABLE} ORDER BY source")
        return [str(row[0]) for row in result.result_rows]

    def chunk_count(self) -> int:
        result = self._client.query(f"SELECT count() FROM {_TABLE}")
        return int(result.result_rows[0][0]) if result.result_rows else 0

    def load_all_records(self) -> list[ChunkRecord]:
        result = self._client.query(
            f"""
            SELECT
                toString(id),
                source,
                page,
                chunk,
                embedding,
                metadata,
                dataset_kind,
                created_at
            FROM {_TABLE}
            """
        )
        return [self._row_to_record(row) for row in result.result_rows]

    def _ensure_table(self) -> None:
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id UInt64,
                source String,
                page UInt32,
                chunk String,
                embedding Array(Float32),
                chunk_hash String,
                char_count UInt32,
                metadata String DEFAULT '{{}}',
                dataset_kind String DEFAULT 'raw',
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY source
            ORDER BY id
            """
        )

    @staticmethod
    def _record_to_insert_row(record: ChunkRecord) -> list[Any]:
        chunk_hash = str(record.metadata.get("chunk_hash", ""))
        char_count = int(record.metadata.get("char_count", len(record.chunk)))
        return [
            int(record.id) if str(record.id).isdigit() else abs(hash(record.id)) % (10**12),
            record.source,
            record.page,
            record.chunk,
            [float(value) for value in record.embedding.tolist()],
            chunk_hash,
            char_count,
            metadata_to_json(record.metadata),
            record.dataset_kind,
            record.created_at.replace(tzinfo=None),
        ]

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> ChunkRecord:
        created_raw = row[7]
        if isinstance(created_raw, datetime):
            created_at = created_raw.replace(tzinfo=timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)
        return ChunkRecord(
            id=str(row[0]),
            source=str(row[1]),
            page=int(row[2]),
            chunk=str(row[3]),
            embedding=np.asarray(row[4], dtype=np.float64),
            metadata=metadata_from_json(str(row[5])),
            dataset_kind=str(row[6]),
            created_at=created_at,
        )

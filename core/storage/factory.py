"""Vector store factory."""

from __future__ import annotations

import logging

from config import Config
from core.storage.clickhouse_store import ClickHouseVectorStore
from core.storage.json_store import JsonVectorStore
from core.storage.protocol import VectorStore

log = logging.getLogger(__name__)


def create_vector_store(config: Config) -> VectorStore:
    """Create a vector store from configuration.

    Args:
        config: Application configuration.

    Returns:
        Initialized vector store.

    Raises:
        ValueError: When backend name is unknown.
        ConnectionError: When ClickHouse initialization fails in non-auto mode.
    """
    backend = config.storage_backend
    if backend == "clickhouse":
        return ClickHouseVectorStore(config)
    if backend == "json":
        return JsonVectorStore(config)
    if backend == "auto":
        try:
            store = ClickHouseVectorStore(config)
            log.info("Auto-selected storage backend=clickhouse")
            return store
        except (ConnectionError, ValueError) as error:
            log.warning("ClickHouse unavailable, falling back to JSON: %s", error)
            store = JsonVectorStore(config)
            log.info("Auto-selected storage backend=json")
            return store
    raise ValueError(f"Unknown storage backend: {backend}")

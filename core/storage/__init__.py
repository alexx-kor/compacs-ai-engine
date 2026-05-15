"""Vector storage abstractions with ClickHouse and JSON backends."""

from core.storage.factory import create_vector_store
from core.storage.protocol import ChunkRecord, VectorStore

__all__ = ["ChunkRecord", "VectorStore", "create_vector_store"]

"""Align query embedding provider with an existing vector index."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

OPENAI_EMBED_DIM = 1536
OLLAMA_EMBED_DIM = 768


def reset_embedder() -> None:
    """Drop cached EmbeddingChain and reload config from environment."""
    import config as config_module
    from config import Config
    from core.embeddings.chain import EmbeddingChain, _LazyEmbeddingChain

    config_module.config = Config.from_env()
    _LazyEmbeddingChain._instance = None
    EmbeddingChain.embed_cached.cache_clear()


def _read_index_dim(vector_store_dir: Path) -> int | None:
    chunks_path = vector_store_dir / "chunks.json"
    if not chunks_path.is_file():
        return None
    try:
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    for row in rows[:32]:
        if not isinstance(row, dict):
            continue
        embedding = row.get("embedding")
        if isinstance(embedding, list) and embedding:
            return len(embedding)
    return None


def configure_embeddings_for_index(vector_store_dir: Path) -> str:
    """Pick embedding provider matching stored vector dimension."""
    dim = _read_index_dim(vector_store_dir)
    if dim is None:
        raise ValueError(f"cannot detect embedding dimension in {vector_store_dir}")

    if dim == OLLAMA_EMBED_DIM:
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"
        provider = "ollama"
    elif dim == OPENAI_EMBED_DIM:
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"
        provider = "openai"
    else:
        log.warning("unknown embedding dim=%s in %s, using ollama", dim, vector_store_dir)
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"
        provider = "ollama"

    reset_embedder()
    log.info("configured embeddings provider=%s dim=%s store=%s", provider, dim, vector_store_dir)
    return provider

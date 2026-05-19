"""Align query embeddings with the vector index (Ollama 768 vs OpenAI 1536)."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

OLLAMA_DIM = 768
OPENAI_DIM = 1536


def index_dimension_counts(store_path: Path) -> Counter[int]:
    """Count embedding dimensions across all rows in ``chunks.json``."""
    counts: Counter[int] = Counter()
    if not store_path.exists():
        return counts
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return counts
    if not isinstance(payload, dict):
        return counts
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if isinstance(row, dict):
            embedding = row.get("embedding")
            if isinstance(embedding, list) and embedding:
                counts[len(embedding)] += 1
    return counts


def read_index_embedding_meta(store_path: Path) -> tuple[int | None, str | None]:
    """Read dominant embedding dimension from index header or row majority."""
    if not store_path.exists():
        return None, None
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        log.warning("could not read index meta path=%s error=%s", store_path, error)
        return None, None
    if not isinstance(payload, dict):
        return None, None

    provider = payload.get("embedding_provider")
    provider_str = str(provider) if provider else None

    counts = index_dimension_counts(store_path)
    if counts:
        dimension = counts.most_common(1)[0][0]
        if len(counts) > 1:
            log.warning(
                "mixed embedding dimensions in index %s — using majority dim=%s",
                dict(counts),
                dimension,
            )
        return dimension, provider_str

    dim = payload.get("embedding_dim")
    if isinstance(dim, int) and dim > 0:
        return dim, provider_str
    return None, None


def provider_for_dimension(dimension: int) -> str:
    if dimension == OLLAMA_DIM:
        return "ollama"
    if dimension == OPENAI_DIM:
        return "openai"
    raise ValueError(
        f"unsupported embedding dimension in index: {dimension}. "
        f"Expected {OLLAMA_DIM} (Ollama/{'nomic-embed-text'}) or "
        f"{OPENAI_DIM} (OpenAI/text-embedding-3-small)."
    )


def reset_embedder() -> None:
    """Drop cached embedding chain so the next call picks up new env settings."""
    import core.embeddings.chain as chain_module

    lazy = chain_module.embedder
    if hasattr(lazy, "_instance"):
        lazy._instance = None


def configure_embeddings_for_index(
    vector_store_dir: Path,
    *,
    preferred_provider: str | None = None,
) -> str:
    """
    Set ``EMBEDDING_PROVIDER`` to match ``chunks.json`` and reset the embedder.

    Returns the provider name applied.
    """
    store_path = vector_store_dir / "chunks.json"
    dimension, stored_provider = read_index_embedding_meta(store_path)

    if dimension is None:
        provider = preferred_provider or os.getenv("EMBEDDING_PROVIDER", "ollama")
        log.warning("index embedding dimension unknown; using provider=%s", provider)
    else:
        provider = preferred_provider or (
            stored_provider if stored_provider in ("ollama", "openai") else None
        )
        if provider is None:
            provider = provider_for_dimension(dimension)
        elif provider == "ollama" and dimension != OLLAMA_DIM:
            provider = provider_for_dimension(dimension)
        elif provider == "openai" and dimension != OPENAI_DIM:
            provider = provider_for_dimension(dimension)

    os.environ["EMBEDDING_PROVIDER"] = provider
    os.environ["EMBEDDING_FALLBACK_ENABLED"] = "false"
    reset_embedder()
    log.info(
        "embeddings aligned: provider=%s index_dim=%s path=%s",
        provider,
        dimension,
        store_path,
    )
    return provider

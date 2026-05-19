from __future__ import annotations

from config import config
from core.embeddings.providers import (
    OLLAMA_EMBEDDING_DIM,
    truncate_for_embedding,
)


def test_truncate_for_ollama_uses_lower_limit() -> None:
    long_text = "x" * 5000
    truncated = truncate_for_embedding(long_text, config, provider="ollama")
    assert len(truncated) <= config.ollama_embed_max_chars


def test_ollama_embedding_dim_constant() -> None:
    assert OLLAMA_EMBEDDING_DIM == 768

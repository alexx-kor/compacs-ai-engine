"""Embedding fallback chain."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import lru_cache

from config import Config
from core.embeddings.providers import (
    EmbeddingProvider,
    FALLBACK_EMBEDDING_SIZE,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)

log = logging.getLogger(__name__)


class EmbeddingChain:
    """Generate embeddings with provider fallback."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._primary = build_embedding_provider(config)
        self._fallback: EmbeddingProvider | None = None
        if config.embedding_fallback_enabled and self._primary.name != "ollama":
            self._fallback = OllamaEmbeddingProvider(config)

    @property
    def active_provider(self) -> str:
        return self._primary.name

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._primary.embed_batch(texts)
        except Exception as error:
            if self._fallback is None:
                log.warning("Embedding provider failed, returning zero vectors: %s", error)
                return [[0.0] * FALLBACK_EMBEDDING_SIZE for _ in texts]
            log.warning(
                "Primary embedding provider=%s failed, falling back to ollama: %s",
                self._primary.name,
                error,
            )
            return self._fallback.embed_batch(texts)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts using configured batch size."""
        all_embeddings: list[list[float]] = []
        total = len(texts)
        started_at = time.time()
        for offset in range(0, total, self._config.batch_size):
            batch = texts[offset : offset + self._config.batch_size]
            all_embeddings.extend(self.embed_batch(batch))
            completed = offset + len(batch)
            elapsed = time.time() - started_at
            speed = completed / elapsed if elapsed > 0 else 0.0
            log.info("Embedding progress=%.1f%% speed=%.1f/s", (completed / total) * 100, speed)
        return all_embeddings

    @lru_cache(maxsize=1024)
    def embed_cached(self, text: str) -> tuple[float, ...]:
        """Return cached embedding for a single text."""
        return tuple(self.embed_batch([text])[0])


class _LazyEmbeddingChain:
    """Defer chain construction until first use."""

    _instance: EmbeddingChain | None = None

    def _get(self) -> EmbeddingChain:
        if self._instance is None:
            from config import config as app_config

            self._instance = EmbeddingChain(app_config)
        return self._instance

    def __getattr__(self, name: str) -> object:
        return getattr(self._get(), name)


embedder: EmbeddingChain = _LazyEmbeddingChain()  # type: ignore[assignment]

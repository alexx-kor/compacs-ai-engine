"""Embedding generator backed by OpenAI API."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from functools import lru_cache

from config import config
from core.openai_client import get_openai_client


log = logging.getLogger(__name__)
FALLBACK_EMBEDDING_SIZE = 1536

__all__ = ["EmbeddingGenerator", "FALLBACK_EMBEDDING_SIZE", "embedder"]


class EmbeddingGenerator:
    """Generate embeddings through the OpenAI API with batching and LRU cache."""

    def __init__(self) -> None:
        """Build the client and verify API reachability with a minimal embedding call."""
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.batch_size = config.batch_size
        self.max_length = config.max_text_length
        self.client = get_openai_client()
        self._api_checked = False

    def _check_api(self) -> None:
        """Validate embedding API connectivity on first use (non-fatal on failure)."""
        if self._api_checked:
            return
        self._api_checked = True
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"],
                encoding_format="float",
            )
            log.info("OpenAI API is available")
            log.info("Embedding model: %s", self.model)
            log.info("Embedding dimensions: %s", len(response.data[0].embedding))
        except Exception as error:
            log.error("OpenAI API error: %s", error)
            log.error("Please check your OPENAI_API_KEY")

    def _truncate_text(self, text: str) -> str:
        """Truncate input text to a safe embedding length.

        Args:
            text: Source text.

        Returns:
            Truncated text, preferring a sentence boundary when it falls past
            the midpoint of the truncated window.
        """
        if len(text) <= self.max_length:
            return text
        truncated = text[: self.max_length]
        last_period_position = truncated.rfind(".")
        if last_period_position > self.max_length // 2:
            truncated = truncated[: last_period_position + 1]
        return truncated.strip()

    def generate_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a single batch.

        Args:
            texts: Batch of input texts (may be empty).

        Returns:
            Embeddings in the same order as ``texts``, or an empty list when
            ``texts`` is empty. On API failure, returns zero vectors of fixed
            dimension for each input text (same length as ``texts``).
        """
        if not texts:
            return []

        self._check_api()
        safe_texts = [self._truncate_text(text) for text in texts]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=safe_texts,
                encoding_format="float",
            )
            return [list(item.embedding) for item in response.data]
        except Exception as error:
            log.warning("Embedding error: %s", error)
            return [[0.0] * FALLBACK_EMBEDDING_SIZE for _ in safe_texts]

    def generate(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for all provided texts using configured batch size.

        Args:
            texts: Input texts (order preserved across batches).

        Returns:
            Flat list of embedding vectors aligned with ``texts``.
        """
        total_count = len(texts)
        all_embeddings: list[list[float]] = []
        log.info("Generating %s embeddings via OpenAI", total_count)
        log.info("Batch size: %s", self.batch_size)
        log.info("Model: %s", self.model)

        started_at = time.time()
        for offset in range(0, total_count, self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            batch_embeddings = self.generate_batch(batch)
            all_embeddings.extend(batch_embeddings)

            completed = offset + len(batch)
            progress_percent = (completed / total_count * 100) if total_count else 0.0
            elapsed_seconds = time.time() - started_at
            speed = (completed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            log.info("Progress: %.1f%% (%.1f chunks/sec)", progress_percent, speed)

        elapsed_seconds = time.time() - started_at
        total_speed = (total_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        log.info("Done in %.2fs (%.1f chunks/sec)", elapsed_seconds, total_speed)
        log.info("Total cost: ~$%.4f (est.)", total_count * 0.00000013)
        return all_embeddings

    @lru_cache(maxsize=1024)
    def generate_cached(self, text: str) -> tuple[float, ...]:
        """Return a cached embedding for a single text string.

        Args:
            text: Input text used as the cache key.

        Returns:
            Embedding as an immutable tuple of floats.
        """
        embedding = self.generate_batch([text])[0]
        return tuple(embedding)


class _LazyEmbedder:
    """Defer ``EmbeddingGenerator`` construction until first attribute access."""

    _instance: EmbeddingGenerator | None = None

    def _get(self) -> EmbeddingGenerator:
        if self._instance is None:
            self._instance = EmbeddingGenerator()
        return self._instance

    def __getattr__(self, name: str) -> object:
        return getattr(self._get(), name)


embedder: EmbeddingGenerator = _LazyEmbedder()  # type: ignore[assignment]

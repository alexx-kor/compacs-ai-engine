"""Embedding generation helpers for RAG pipelines."""

from functools import lru_cache
import logging

import ollama

from config import config

log = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings via Ollama model."""

    def __init__(self):
        self.model = config.embed_model
        self.batch_size = config.batch_size
        self.max_length = config.max_text_length

    def _truncate_text(self, text: str) -> str:
        """Trim long text to model-safe length."""
        if len(text) <= self.max_length:
            return text
        truncated = text[: self.max_length]
        last_period = truncated.rfind(".")
        if last_period > self.max_length // 2:
            truncated = truncated[:last_period + 1]
        return truncated.strip()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate a single embedding batch.

        Args:
            texts: Input texts.

        Returns:
            Embedding vectors. Returns zero-vectors on error to preserve behavior.
        """
        if not texts:
            return []
        safe_texts = [self._truncate_text(t) for t in texts]
        try:
            response = ollama.embed(
                model=self.model,
                input=safe_texts,
                options={"num_gpu": config.ollama_num_gpu},
            )
            return response["embeddings"]
        except (RuntimeError, OSError, ValueError) as error:
            log.error("embedding batch failed: %s", error)
            return [[0.0] * 768 for _ in safe_texts]

    def generate(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for many texts with configured batching."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.generate_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @lru_cache(maxsize=256)
    def generate_cached(self, text: str) -> tuple:
        """Generate and cache embedding for a single text."""
        embedding = self.generate_batch([text])[0]
        return tuple(embedding)


embedder = EmbeddingGenerator()
"""Embedding provider implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

import ollama

from config import Config
from core.openai_client import get_openai_client

log = logging.getLogger(__name__)

FALLBACK_EMBEDDING_SIZE = 1536


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding API provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = get_openai_client()
        self._model = config.openai_embedding_model

    @property
    def name(self) -> str:
        return "openai"

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=[self._truncate(text) for text in texts],
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]

    def _truncate(self, text: str) -> str:
        if len(text) <= self._config.max_text_length:
            return text
        truncated = text[: self._config.max_text_length]
        last_period = truncated.rfind(".")
        if last_period > self._config.max_text_length // 2:
            truncated = truncated[: last_period + 1]
        return truncated.strip()


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_host)
        self._model = config.embed_model

    @property
    def name(self) -> str:
        return "ollama"

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for text in texts:
            response = self._client.embeddings(model=self._model, prompt=self._truncate(text))
            vector = response.get("embedding", [])
            embeddings.append([float(value) for value in vector])
        return embeddings

    def _truncate(self, text: str) -> str:
        # nomic-embed-text context is ~8192 tokens; stay conservative in characters.
        limit = min(self._config.max_text_length, 2048)
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        last_period = truncated.rfind(".")
        if last_period > limit // 2:
            truncated = truncated[: last_period + 1]
        return truncated.strip()


def build_embedding_provider(config: Config) -> EmbeddingProvider:
    """Create primary embedding provider from configuration."""
    if config.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(config)
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(config)
    if _openai_key_configured(config.openai_api_key):
        return OpenAIEmbeddingProvider(config)
    return OllamaEmbeddingProvider(config)


def _openai_key_configured(key: str | None) -> bool:
    if not key or not str(key).strip():
        return False
    normalized = str(key).strip().lower()
    return normalized not in {"user_provided", "changeme", "none", "null"}

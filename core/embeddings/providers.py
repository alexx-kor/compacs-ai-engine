"""Embedding provider implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import ollama

from config import Config
from core.openai_client import get_openai_client

log = logging.getLogger(__name__)

OPENAI_EMBEDDING_DIM = 1536
OLLAMA_EMBEDDING_DIM = 768
FALLBACK_EMBEDDING_SIZE = OPENAI_EMBEDDING_DIM


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Vector dimension produced by this provider."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts."""


def truncate_for_embedding(text: str, config: Config, *, provider: str) -> str:
    """Truncate text to provider-specific limits before embedding."""
    if provider == "ollama":
        limit = min(config.max_text_length, config.ollama_embed_max_chars)
    else:
        limit = config.max_text_length
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    if provider != "ollama":
        last_period = truncated.rfind(".")
        if last_period > limit // 2:
            truncated = truncated[: last_period + 1]
    return truncated.strip()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding API provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = get_openai_client()
        self._model = config.openai_embedding_model

    @property
    def name(self) -> str:
        return "openai"

    @property
    def embedding_dim(self) -> int:
        return OPENAI_EMBEDDING_DIM

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=[truncate_for_embedding(text, self._config, provider="openai") for text in texts],
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider with per-item truncation and retries."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_host)
        self._model = config.embed_model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def embedding_dim(self) -> int:
        return OLLAMA_EMBEDDING_DIM

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        limits = (
            self._config.ollama_embed_max_chars,
            self._config.ollama_embed_max_chars // 2,
            800,
        )
        last_error: Exception | None = None
        for limit in limits:
            prompt = truncate_for_embedding(text, self._config, provider="ollama")
            if len(prompt) > limit:
                prompt = prompt[:limit].strip()
            try:
                response = self._client.embeddings(model=self._model, prompt=prompt)
                vector = response.get("embedding", [])
                return [float(value) for value in vector]
            except Exception as error:
                last_error = error
                if "context length" not in str(error).lower():
                    raise
                log.debug(
                    "ollama embed retry shorter text limit=%s chars=%s error=%s",
                    limit,
                    len(prompt),
                    error,
                )
        raise RuntimeError(
            f"ollama embedding failed after truncation (chars={len(text)}): {last_error}"
        ) from last_error


def build_embedding_provider(config: Config) -> EmbeddingProvider:
    """Create primary embedding provider from configuration."""
    if config.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(config)
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(config)
    if config.openai_api_key:
        return OpenAIEmbeddingProvider(config)
    return OllamaEmbeddingProvider(config)

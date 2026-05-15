"""Embedding providers with OpenAI primary and Ollama fallback."""

from core.embeddings.chain import EmbeddingChain, embedder

# Backward-compatible alias for legacy imports.
EmbeddingGenerator = EmbeddingChain

__all__ = ["EmbeddingChain", "EmbeddingGenerator", "embedder"]

"""Shared vector similarity helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_distance(query: Sequence[float], vector: Sequence[float]) -> float:
    """Return cosine distance ``1 - cosine_similarity``."""
    query_array = np.asarray(query, dtype=np.float64)
    vector_array = np.asarray(vector, dtype=np.float64)
    if query_array.shape != vector_array.shape:
        raise ValueError(
            f"embedding dimension mismatch: query={query_array.shape[0]} "
            f"stored={vector_array.shape[0]}. "
            "Index and search must use the same EMBEDDING_PROVIDER/EMBED_MODEL "
            "(e.g. both Ollama nomic-embed-text or both OpenAI)."
        )
    query_norm = float(np.linalg.norm(query_array))
    vector_norm = float(np.linalg.norm(vector_array))
    if query_norm == 0.0 or vector_norm == 0.0:
        return 1.0
    similarity = float(np.dot(query_array, vector_array) / (query_norm * vector_norm))
    return 1.0 - similarity

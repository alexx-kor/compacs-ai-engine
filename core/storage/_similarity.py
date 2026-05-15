"""Shared vector similarity helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_distance(query: Sequence[float], vector: Sequence[float]) -> float:
    """Return cosine distance ``1 - cosine_similarity``."""
    query_array = np.asarray(query, dtype=np.float64)
    vector_array = np.asarray(vector, dtype=np.float64)
    query_norm = float(np.linalg.norm(query_array))
    vector_norm = float(np.linalg.norm(vector_array))
    if query_norm == 0.0 or vector_norm == 0.0:
        return 1.0
    similarity = float(np.dot(query_array, vector_array) / (query_norm * vector_norm))
    return 1.0 - similarity

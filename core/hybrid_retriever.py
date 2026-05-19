"""Merge dense vector search with BM25 sparse retrieval."""

from __future__ import annotations

import logging

import numpy as np

from config import Config
from core.bm25_index import Bm25Index
from core.storage.protocol import ChunkRecord
from core.text_processing import expand_query_tokens

log = logging.getLogger(__name__)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], *, k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def merge_dense_bm25(
    records: list[ChunkRecord],
    query: str,
    query_embedding: np.ndarray,
    bm25_index: Bm25Index,
    config: Config,
    catalog: dict[str, ChunkRecord],
) -> list[ChunkRecord]:
    """Fuse dense vector hits with BM25 scores; return reordered ChunkRecords."""
    pool_limit = max(config.top_k * 3, config.top_k)
    query_dim = int(query_embedding.shape[0])
    compatible = [
        record for record in records if int(record.embedding.shape[0]) == query_dim
    ]
    dense_ranked = sorted(
        compatible,
        key=lambda record: _cosine_distance(query_embedding, record.embedding),
    )[:pool_limit]
    dense_ids = [record.id for record in dense_ranked]

    bm25_hits = bm25_index.search(expand_query_tokens(query), limit=pool_limit)
    bm25_ids = [hit.chunk_id for hit in bm25_hits]
    bm25_scores = {hit.chunk_id: hit.score for hit in bm25_hits}
    max_bm25 = max(bm25_scores.values(), default=1.0) or 1.0

    by_id = {record.id: record for record in records}
    for chunk_id in bm25_ids:
        if chunk_id not in by_id and chunk_id in catalog:
            by_id[chunk_id] = catalog[chunk_id]

    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])
    candidate_ids = list(dict.fromkeys(dense_ids + bm25_ids))

    scored: list[tuple[float, str]] = []
    for chunk_id in candidate_ids:
        record = by_id.get(chunk_id)
        if record is None:
            continue
        if int(record.embedding.shape[0]) != query_dim:
            continue
        dense_sim = 1.0 - _cosine_distance(query_embedding, record.embedding)
        bm25_norm = bm25_scores.get(chunk_id, 0.0) / max_bm25
        combined = (
            dense_sim * config.hybrid_dense_weight
            + bm25_norm * config.hybrid_bm25_weight
            + fused.get(chunk_id, 0.0) * 0.15
        )
        scored.append((combined, chunk_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [by_id[chunk_id] for _, chunk_id in scored[: config.top_k] if chunk_id in by_id]


def _cosine_distance(query: np.ndarray, vector: np.ndarray) -> float:
    if int(query.shape[0]) != int(vector.shape[0]):
        return 1.0
    q_norm = float(np.linalg.norm(query))
    v_norm = float(np.linalg.norm(vector))
    if q_norm == 0.0 or v_norm == 0.0:
        return 1.0
    similarity = float(np.dot(query, vector) / (q_norm * v_norm))
    return 1.0 - similarity

from __future__ import annotations

from core.bm25_index import Bm25Index
from core.hybrid_retriever import reciprocal_rank_fusion


def test_bm25_prefers_matching_document() -> None:
    index = Bm25Index(
        ["1", "2"],
        [
            ["sftp", "compacs", "5.32.101.214", "raw_data"],
            ["настройка", "qt", "creator", "proxy"],
        ],
    )
    hits = index.search(["sftp", "compacs"], limit=2)
    assert hits
    assert hits[0].chunk_id == "1"


def test_rrf_merges_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]

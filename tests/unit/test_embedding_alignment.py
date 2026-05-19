from __future__ import annotations

import pytest

from core.embedding_alignment import provider_for_dimension, read_index_embedding_meta


def test_provider_for_known_dimensions() -> None:
    assert provider_for_dimension(768) == "ollama"
    assert provider_for_dimension(1536) == "openai"


def test_provider_for_unknown_dimension() -> None:
    with pytest.raises(ValueError):
        provider_for_dimension(512)


def test_read_meta_from_rows(tmp_path) -> None:
    store = tmp_path / "chunks.json"
    store.write_text(
        '{"version": 1, "rows": [{"id": "0", "embedding": [0.1, 0.2]}]}',
        encoding="utf-8",
    )
    dim, provider = read_index_embedding_meta(store)
    assert dim == 2
    assert provider is None

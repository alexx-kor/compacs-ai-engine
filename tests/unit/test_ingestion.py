from __future__ import annotations

from pathlib import Path

from core.ingestion import ingestion_service


def test_collect_chunks_mixed_formats(tmp_path: Path) -> None:
    long_text = "alpha document " * 20
    (tmp_path / "a.txt").write_text(long_text, encoding="utf-8")
    (tmp_path / "b.md").write_text("# heading\n\n" + long_text, encoding="utf-8")
    chunks, report = ingestion_service.collect_chunks(str(tmp_path))
    assert report.files_processed == 2
    assert report.chunks_created == len(chunks)
    assert chunks


def test_collect_chunks_respects_extension_filter(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("keep " * 30, encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    chunks, report = ingestion_service.collect_chunks(
        str(tmp_path),
        allowed_extensions={".txt"},
    )
    assert report.files_processed == 1
    assert len(chunks) >= 1

from __future__ import annotations

from pathlib import Path

from core.ingestion import ingestion_service


def test_collect_chunks_mixed_formats(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha " * 80, encoding="utf-8")
    (docs / "b.rst").write_text("beta " * 80, encoding="utf-8")
    (docs / "c.md").write_text("gamma " * 80, encoding="utf-8")
    (docs / "skip.csv").write_text("x,y\n1,2", encoding="utf-8")

    chunks, report = ingestion_service.collect_chunks(str(docs), start_id=10)

    assert report.files_discovered == 4
    assert report.files_processed >= 3
    assert report.files_skipped >= 1
    assert report.chunks_created == len(chunks)
    assert chunks
    assert chunks[0]["id"] >= 10


def test_collect_chunks_respects_extension_filter(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.rst").write_text("rst " * 80, encoding="utf-8")
    (docs / "drop.txt").write_text("txt " * 80, encoding="utf-8")

    chunks, report = ingestion_service.collect_chunks(str(docs), allowed_extensions={".rst"})

    assert report.files_discovered == 2
    assert report.files_processed == 1
    assert chunks
    assert all(str(chunk["source"]).endswith(".rst") for chunk in chunks)


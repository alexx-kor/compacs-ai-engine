from __future__ import annotations

from pathlib import Path

from core.chunk_pipeline import build_all_chunks


def test_build_all_chunks_from_txt(tmp_path: Path) -> None:
    instr = tmp_path / "ui_extension"
    instr.mkdir()
    doc = instr / "manual.txt"
    doc.write_text(
        "Кнопка «Новый документ» — формирует новый документ без CDPL-процедур.\n\n"
        "1. Общие сведения\n"
        "Раздел описывает интерфейс оператора.\n" * 20,
        encoding="utf-8",
    )
    chunks = build_all_chunks([doc], instr, tmp_path / "graph", strategies=("sliding", "section"))
    assert len(chunks) >= 1
    assert all("chunk" in item and item.get("source") == "manual.txt" for item in chunks)

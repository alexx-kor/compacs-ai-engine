from __future__ import annotations

from pathlib import Path

from core.knowledge_graph import KnowledgeGraphExtractor, triples_to_graph_chunks


def test_extracts_sftp_and_addresses() -> None:
    text = """
4.1 Адреса:
– 192.168.5.13 – для подключения из внутренней сети (LAN);
– 5.32.101.214 – для подключения из внешней сети (WAN).
5.1.2 Команда: sftp compacs@5.32.101.214
raw_data/ Сырые данные
WT_MAX_FILES_PER_RUN Максимальное количество файлов за один запуск (5000)
"""
    triples = KnowledgeGraphExtractor().extract_from_text(text, "demo.txt")
    preds = {(t.predicate, t.object) for t in triples}
    assert ("hasLanAddress", "192.168.5.13") in preds
    assert ("hasWanAddress", "5.32.101.214") in preds
    assert ("hasSftpEndpoint", "compacs@5.32.101.214") in preds
    assert ("hasDirectory", "raw_data/") in preds
    assert ("hasEnvironmentVariable", "WT_MAX_FILES_PER_RUN") in preds


def test_graph_chunks_tagged_for_rag() -> None:
    triples = KnowledgeGraphExtractor().extract_from_text(
        "sftp compacs@5.32.101.214", "ai.txt"
    )
    chunks = triples_to_graph_chunks(triples, start_id=0)
    assert chunks
    assert chunks[0]["source"].startswith("graph/")
    assert chunks[0]["chunk_type"] == "graph"


def test_extract_from_real_instruction_sample() -> None:
    root = Path(__file__).resolve().parents[2]
    sample = root / "instructions" / "Инструкция_программная_№0002_Инстр_прог_2026_от_31_03_2026_Порядок.extract.txt"
    if not sample.exists():
        return
    triples = KnowledgeGraphExtractor().extract_from_file(sample)
    objects = {t.object for t in triples}
    assert "compacs@5.32.101.214" in objects
    assert "warehouse/catalog.db" in objects

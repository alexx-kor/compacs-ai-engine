#!/usr/bin/env python3
"""Unified CLI entrypoint for the RAG system."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

import typer
import uvicorn

from config import config
from core.datasets import DatasetScanner
from core.database import db
from core.embeddings.chain import EmbeddingChain
from core.ingestion import ingestion_service
from core.logger import setup_logging
from core.storage.protocol import ChunkRecord
from rag_service import rag_service

app = typer.Typer(help="Unified RAG system CLI")
log = logging.getLogger(__name__)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command("serve-api")
def serve_api(
    port: int = typer.Option(8080, help="Stable API port"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
) -> None:
    """Serve production RAG API (engine)."""
    uvicorn.run("api.stable:app_stable", host=host, port=port, reload=False)


@app.command("serve-gateway")
def serve_gateway(port: int = typer.Option(3080, help="Gateway port")) -> None:
    """Serve external gateway UI and API proxy."""
    os.environ["GATEWAY_PORT"] = str(port)
    uvicorn.run("api.gateway:app_gateway", host="0.0.0.0", port=port, reload=False)


@app.command("serve")
def serve(
    gateway_port: int = typer.Option(3080, help="External gateway port"),
    engine_port: int = typer.Option(8080, help="Internal engine port"),
) -> None:
    """Serve gateway (:3080) and RAG engine (:8080, localhost only)."""
    os.environ["RAG_ENGINE_URL"] = f"http://127.0.0.1:{engine_port}"
    os.environ["GATEWAY_PORT"] = str(gateway_port)
    engine = multiprocessing.Process(
        target=serve_api,
        kwargs={"port": engine_port, "host": "127.0.0.1"},
    )
    gateway = multiprocessing.Process(
        target=serve_gateway,
        kwargs={"port": gateway_port},
    )
    engine.start()
    gateway.start()
    log.info("RAG engine http://127.0.0.1:%s", engine_port)
    log.info("Gateway http://0.0.0.0:%s", gateway_port)
    try:
        engine.join()
        gateway.join()
    except KeyboardInterrupt:
        engine.terminate()
        gateway.terminate()


@app.command("serve-dev")
def serve_dev(port: int = typer.Option(8090, help="Development API port")) -> None:
    """Serve development/debug API."""
    uvicorn.run("api.dev:app_dev", host="0.0.0.0", port=port, reload=False)


@app.command("serve-all")
def serve_all() -> None:
    """Serve stable (:8080) and dev (:8090) APIs."""
    stable = multiprocessing.Process(target=serve_api, kwargs={"port": 8080})
    dev = multiprocessing.Process(target=serve_dev, kwargs={"port": 8090})
    stable.start()
    dev.start()
    stable.join()
    dev.join()


@app.command("ingest")
def ingest(
    source: str = typer.Option("instructions/raw", "--source", help="Input directory"),
    force_reload: bool = typer.Option(False, "--force-reload", help="Recreate vector store"),
) -> None:
    """Ingest documents into the active vector store."""
    if force_reload:
        db.init_database(force_recreate=True)
    chunks, report = ingestion_service.collect_chunks(source)
    if not chunks:
        log.warning("No chunks collected from source=%s", source)
        return
    texts = [str(chunk["chunk"]) for chunk in chunks]
    embeddings = EmbeddingChain(config).embed(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    db.insert_batch(chunks)
    log.info(
        "Ingest complete files_processed=%s chunks=%s backend=%s",
        report.files_processed,
        report.chunks_created,
        db.backend_name,
    )


@app.command("query")
def query(question: str = typer.Argument(..., help="Question to ask")) -> None:
    """Run a single RAG query."""
    result = rag_service.ask(question)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("migrate")
def migrate(
    direction: str = typer.Argument(..., help="export or import"),
    from_backend: str = typer.Option(..., "--from", help="Source backend"),
    to_backend: str = typer.Option(..., "--to", help="Target backend or jsonl"),
    input_path: Path | None = typer.Option(None, "--input", help="Input jsonl for import"),
    output_path: Path = typer.Option(
        Path("data/chunks.jsonl"),
        "--output",
        help="Output jsonl for export",
    ),
) -> None:
    """Export or import chunks between storage backends."""
    if direction == "export":
        _migrate_export(from_backend, output_path)
        return
    if direction == "import":
        if input_path is None:
            raise typer.BadParameter("--input is required for import")
        _migrate_import(to_backend, input_path)
        return
    raise typer.BadParameter("direction must be export or import")


def _migrate_export(from_backend: str, output_path: Path) -> None:
    if from_backend not in {"clickhouse", "json"}:
        raise typer.BadParameter("export --from must be clickhouse or json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = _create_store_for_backend(from_backend)
    records = store.load_all_records()
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = record.to_legacy_dict()
            row["dataset_kind"] = record.dataset_kind
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo("Exported chunks count=%s path=%s" % (len(records), output_path))


def _create_store_for_backend(backend: str) -> Any:
    import os

    from config import Config
    from core.storage.factory import create_vector_store

    previous = os.environ.get("STORAGE_BACKEND")
    os.environ["STORAGE_BACKEND"] = backend
    try:
        return create_vector_store(Config.from_env())
    finally:
        if previous is None:
            os.environ.pop("STORAGE_BACKEND", None)
        else:
            os.environ["STORAGE_BACKEND"] = previous


def _migrate_import(to_backend: str, input_path: Path) -> None:
    if to_backend not in {"clickhouse", "json", "jsonl"}:
        raise typer.BadParameter("import --to must be clickhouse, json, or jsonl")
    if not input_path.exists():
        raise typer.BadParameter(f"input file not found: {input_path}")
    records: list[ChunkRecord] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(ChunkRecord.from_legacy_dict(payload))
    if to_backend == "jsonl":
        typer.echo("jsonl target is the file itself; no import performed")
        return
    target_backend = "json" if to_backend == "json" else "clickhouse"
    store = _create_store_for_backend(target_backend)
    store.init_store()
    store.insert_batch(records)
    typer.echo("Imported chunks count=%s backend=%s" % (len(records), target_backend))


def cli_main() -> None:
    app()


if __name__ == "__main__":
    cli_main()

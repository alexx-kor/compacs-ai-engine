"""Development RAG HTTP API (port 8090)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import config
from core.cost_guard import CostGuard
from core.database import db
from core.datasets import DatasetScanner
from core.ingestion import ingestion_service
from core.embeddings.chain import EmbeddingChain
from rag_service import rag_service

app_dev = FastAPI(title="RAG Dev API", version="dev")


class DebugQueryRequest(BaseModel):
    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    source: str = Field(default="instructions/raw")
    force_reload: bool = False


@app_dev.post("/debug/query")
async def debug_query(request: DebugQueryRequest) -> dict[str, Any]:
    """Return detailed RAG response payload for debugging."""
    result = rag_service.ask(request.question)
    result["debug"] = {
        "storage_stats": db.store_stats(),
        "config_storage_backend": config.storage_backend,
    }
    return result


@app_dev.post("/ingest/trigger")
async def trigger_ingest(request: IngestRequest) -> dict[str, Any]:
    """Trigger ingestion from a folder under instructions."""
    if request.force_reload:
        db.init_database(force_recreate=True)
    chunks, report = ingestion_service.collect_chunks(request.source)
    if not chunks:
        return {"inserted": 0, "report": report.__dict__}
    texts = [chunk["chunk"] for chunk in chunks]
    embeddings = EmbeddingChain(config).embed(texts)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk["embedding"] = embedding
    db.insert_batch(chunks)
    return {"inserted": len(chunks), "report": report.__dict__}


@app_dev.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Return storage and OpenAI usage metrics."""
    cost_guard = CostGuard(config)
    scanner = DatasetScanner(config.instructions_dir)
    scan = scanner.scan()
    return {
        "storage": db.store_stats(),
        "openai_usage_today": cost_guard.load_daily_summary(),
        "datasets": {
            "raw_files": len(scan.raw_files),
            "graph_pairs": len(scan.graph_pairs),
            "golden_files": len(scan.golden_files),
        },
    }

"""Production RAG HTTP API (port 8080)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import config
from core.database import db
from rag_service import rag_service

app_stable = FastAPI(title="RAG API", version="1.0")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    provider_used: str
    storage_backend: str
    response_time_ms: int


@app_stable.post("/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Run a RAG query against the configured vector store."""
    result = rag_service.ask(request.question)
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="empty answer from RAG pipeline")
    sources = [
        {"source": source, "page": page}
        for source, page in result.get("sources", [])
    ]
    return QueryResponse(
        answer=str(result["answer"]),
        sources=sources,
        provider_used=str(result.get("provider_used", "unknown")),
        storage_backend=str(result.get("storage_backend", db.backend_name)),
        response_time_ms=int(float(result.get("time_total", 0)) * 1000),
    )


@app_stable.get("/health")
async def health() -> dict[str, str]:
    """Return service health and active backends."""
    stats = db.store_stats()
    return {
        "status": "healthy",
        "storage": str(stats.get("backend", db.backend_name)),
        "llm_provider": config.llm_provider,
        "embedding_provider": config.embedding_provider,
    }

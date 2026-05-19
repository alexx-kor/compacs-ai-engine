"""Production RAG HTTP API (port 8080)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.openai_compat import router as openai_router
from config import config
from core.database import db
from core.embedding_alignment import configure_embeddings_for_index
from core.embeddings.chain import EmbeddingChain
from rag_service import rag_service

app_stable = FastAPI(title="RAG API", version="1.0")


@app_stable.on_event("startup")
def align_embeddings_on_startup() -> None:
    """Match query embeddings to index before first request (768 Ollama vs 1536 OpenAI)."""
    if db.get_chunk_count() == 0:
        return
    configure_embeddings_for_index(config.local_vector_store_dir)
    rag_service._embeddings = EmbeddingChain(config)
app_stable.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app_stable.include_router(openai_router)


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

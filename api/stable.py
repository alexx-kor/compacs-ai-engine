"""Production RAG HTTP API (port 8080)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Union

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.sse import format_sse, openai_chat_stream, rag_event_stream

from config import config
from core.collections import collection_service
from core.cost_guard import CostGuard
from core.database import db
from core.datasets import DatasetScanner
from core.drift_report import _index_chunk_lengths, collect_quality_metrics
from core.export_index import export_chunks_jsonl, export_filename
from rag_service import rag_service

from api.collections import router as collections_router
from api.jobs import router as jobs_router
from api.sources import router as sources_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from core.embedding_alignment import configure_embeddings_for_index

    if db.get_chunk_count() > 0:
        try:
            provider = configure_embeddings_for_index(config.local_vector_store_dir)
            log.info("engine startup: query embeddings provider=%s", provider)
        except ValueError as error:
            log.warning("engine startup: embedding alignment skipped: %s", error)
    yield


app_stable = FastAPI(title="RAG Engine", version="2.0", lifespan=_lifespan)
app_stable.include_router(collections_router)
app_stable.include_router(jobs_router)
app_stable.include_router(sources_router)


def _verify_api_key(authorization: Optional[str]) -> None:
    expected = os.getenv("COMPACS_API_KEY", "").strip()
    if not expected or expected == "user_provided":
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing authorization")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _format_answer(result: dict[str, Any]) -> str:
    answer = str(result.get("answer", ""))
    sources = result.get("sources", [])
    if sources:
        lines = [f"- {source}, p.{page}" for source, page in sources[:5]]
        answer += "\n\n**Sources:**\n" + "\n".join(lines)
    return answer


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    stream: bool = False
    collection_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional folder scope; defaults to PUT /v1/collections/selection",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    provider_used: str
    storage_backend: str
    response_time_ms: int


@app_stable.post("/v1/query", response_model=None)
async def query_rag(request: QueryRequest) -> Union[QueryResponse, StreamingResponse]:
    """Run a RAG query against the configured vector store."""
    if request.stream:
        return StreamingResponse(
            rag_event_stream(
                rag_service.ask_stream(request.question, collection_ids=request.collection_ids)
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = rag_service.ask(request.question, collection_ids=request.collection_ids)
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "compacs-rag"
    messages: list[ChatMessage]
    stream: bool = False
    collection_ids: Optional[list[str]] = None


@app_stable.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """OpenAI-compatible model list for LibreChat custom endpoint."""
    _verify_api_key(authorization)
    models = [
        model.strip()
        for model in os.getenv("COMPACS_MODELS", "compacs-rag").split(",")
        if model.strip()
    ]
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "compacs"}
            for model_id in models
        ],
    }


@app_stable.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
) -> Union[dict[str, Any], StreamingResponse]:
    """OpenAI-compatible chat route used by LibreChat custom endpoint."""
    _verify_api_key(authorization)

    user_messages = [message for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="no user message in messages")

    question = user_messages[-1].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty user message")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        def token_stream():
            for event in rag_service.ask_stream(question, collection_ids=request.collection_ids):
                if event.get("event") == "token":
                    text = str(event.get("data", {}).get("text", ""))
                    if text:
                        yield text

        return StreamingResponse(
            openai_chat_stream(
                completion_id=completion_id,
                model=request.model,
                created=created,
                text_chunks=token_stream(),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = rag_service.ask(question, collection_ids=request.collection_ids)
    if not result.get("answer"):
        raise HTTPException(status_code=500, detail="empty answer from RAG pipeline")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _format_answer(result)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


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


@app_stable.get("/v1/metrics")
async def service_metrics() -> dict[str, Any]:
    """Return storage, collections, usage, and PSI/drift metrics for monitoring."""
    cost_guard = CostGuard(config)
    scanner = DatasetScanner(config.instructions_dir)
    scan = scanner.scan()
    collections = collection_service.list_collections()
    records = db.vector_store.load_all_records()
    chunk_lengths = _index_chunk_lengths(records)
    sources = {record.source for record in records}
    return {
        "storage": db.store_stats(),
        "collections": {
            "count": len(collections),
            "selected_ids": collection_service.get_selection(),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "document_count": len(item.documents),
                }
                for item in collections
            ],
        },
        "datasets": {
            "raw_files": len(scan.raw_files),
            "graph_pairs": len(scan.graph_pairs),
            "golden_files": len(scan.golden_files),
        },
        "openai_usage_today": cost_guard.load_daily_summary(),
        "quality": collect_quality_metrics(
            config.project_root,
            instructions_dir=config.instructions_dir,
            chunk_lengths=chunk_lengths,
            chunk_count=len(records),
            source_count=len(sources),
        ),
    }


@app_stable.get("/v1/export")
async def export_vector_index(format: str = "jsonl") -> Response:
    """Download vector index for offline desktop import."""
    if format != "jsonl":
        raise HTTPException(status_code=400, detail="only format=jsonl is supported")
    payload = export_chunks_jsonl()
    if not payload:
        raise HTTPException(status_code=404, detail="vector index is empty")
    return Response(
        content=payload,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{export_filename()}"'},
    )

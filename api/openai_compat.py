"""OpenAI-compatible HTTP API for LibreChat and other clients."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import config
from rag_service import rag_service

router = APIRouter(prefix="/v1", tags=["openai-compat"])


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="compacs-rag")
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _extract_text_content(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts).strip()


def extract_user_question(messages: list[ChatMessage]) -> str:
    """Return the latest user message text (LibreChat sends full history)."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _extract_text_content(message.content)
        if text:
            return text
    raise HTTPException(status_code=400, detail="no user message found in messages")


def format_answer_with_sources(answer: str, sources: list[Any]) -> str:
    """Append source list for LibreChat fileCitations-style display."""
    if not sources:
        return answer
    lines = ["\n\n---\n**Источники:**"]
    seen: set[tuple[str, Any]] = set()
    for item in sources:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            source, page = item[0], item[1]
        elif isinstance(item, dict):
            source = item.get("source", "")
            page = item.get("page", "")
        else:
            continue
        key = (str(source), page)
        if key in seen:
            continue
        seen.add(key)
        lines.append("- `%s` (стр. %s)" % (source, page))
    return answer + "\n".join(lines)


def _verify_api_key(request: Request) -> None:
    expected = config.compacs_api_key
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    else:
        token = auth.strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


def _completion_id() -> str:
    return "chatcmpl-%s" % uuid.uuid4().hex[:24]


def _usage_estimate(text: str) -> dict[str, int]:
    tokens = max(1, len(text) // 4)
    return {"prompt_tokens": tokens, "completion_tokens": tokens, "total_tokens": tokens * 2}


def _build_completion_payload(
    *,
    completion_id: str,
    model: str,
    content: str,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_estimate(content),
    }


def _stream_chunks(answer: str, completion_id: str, model: str) -> AsyncIterator[str]:
    created = int(time.time())

    async def generate() -> AsyncIterator[str]:
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield "data: %s\n\n" % json.dumps(first, ensure_ascii=False)
        step = 48
        for offset in range(0, len(answer), step):
            piece = answer[offset : offset + step]
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield "data: %s\n\n" % json.dumps(chunk, ensure_ascii=False)
        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield "data: %s\n\n" % json.dumps(done, ensure_ascii=False)
        yield "data: [DONE]\n\n"

    return generate()


@router.get("/models")
async def list_models(_: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """OpenAI-compatible model list for LibreChat custom endpoints."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "compacs",
            }
            for model_id in config.compacs_models
        ],
    }


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    _: None = Depends(_verify_api_key),
) -> Any:
    """Run RAG pipeline and return OpenAI-shaped chat completion."""
    if body.model not in config.compacs_models:
        raise HTTPException(
            status_code=404,
            detail="model not found: %s (available: %s)"
            % (body.model, ", ".join(config.compacs_models)),
        )
    question = extract_user_question(body.messages)
    result = rag_service.ask(question)
    answer = format_answer_with_sources(
        str(result.get("answer", "")),
        result.get("sources", []),
    )
    completion_id = _completion_id()
    if body.stream:
        return StreamingResponse(
            _stream_chunks(answer, completion_id, body.model),
            media_type="text/event-stream",
        )
    return _build_completion_payload(
        completion_id=completion_id,
        model=body.model,
        content=answer,
    )

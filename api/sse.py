"""Server-Sent Events helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def rag_event_stream(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for item in events:
        yield format_sse(str(item.get("event", "message")), item.get("data", {}))


def openai_chat_stream(
    *,
    completion_id: str,
    model: str,
    created: int,
    text_chunks: Iterator[str],
) -> Iterator[str]:
    for chunk in text_chunks:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

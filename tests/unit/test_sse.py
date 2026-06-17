from __future__ import annotations

from api.sse import format_sse, openai_chat_stream, rag_event_stream


def test_format_sse() -> None:
    block = format_sse("token", {"text": "hello"})
    assert block.startswith("event: token\n")
    assert "hello" in block


def test_rag_event_stream() -> None:
    events = [
        {"event": "status", "data": {"phase": "retrieval"}},
        {"event": "token", "data": {"text": "Hi"}},
        {"event": "done", "data": {"answer": "Hi", "sources": []}},
    ]
    chunks = list(rag_event_stream(iter(events)))
    assert len(chunks) == 3
    assert "retrieval" in chunks[0]


def test_openai_chat_stream() -> None:
    chunks = list(
        openai_chat_stream(
            completion_id="chatcmpl-test",
            model="compacs-rag",
            created=1,
            text_chunks=iter(["A", "B"]),
        )
    )
    assert any("A" in chunk for chunk in chunks)
    assert chunks[-1].strip() == "data: [DONE]"

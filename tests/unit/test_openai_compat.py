from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.openai_compat import (
    ChatMessage,
    extract_user_question,
    format_answer_with_sources,
)
from api.stable import app_stable


def test_extract_user_question_last_user_wins() -> None:
    messages = [
        ChatMessage(role="user", content="первый"),
        ChatMessage(role="assistant", content="ответ"),
        ChatMessage(role="user", content="второй вопрос"),
    ]
    assert extract_user_question(messages) == "второй вопрос"


def test_extract_user_question_multimodal() -> None:
    messages = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "как настроить SFTP?"}],
        ),
    ]
    assert extract_user_question(messages) == "как настроить SFTP?"


def test_format_answer_with_sources() -> None:
    text = format_answer_with_sources(
        "ответ",
        [("instructions/foo.txt", 3), ("instructions/foo.txt", 3)],
    )
    assert "Источники" in text
    assert "instructions/foo.txt" in text
    assert text.count("instructions/foo.txt") == 1


@patch("api.openai_compat.rag_service")
def test_chat_completions_json(mock_rag: MagicMock) -> None:
    mock_rag.ask.return_value = {
        "answer": "тестовый ответ",
        "sources": [("doc.txt", 1)],
    }
    client = TestClient(app_stable)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "compacs-rag",
            "messages": [{"role": "user", "content": "вопрос"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert "тестовый ответ" in payload["choices"][0]["message"]["content"]
    mock_rag.ask.assert_called_once_with("вопрос")


@patch("api.openai_compat.rag_service")
def test_chat_completions_unknown_model(mock_rag: MagicMock) -> None:
    client = TestClient(app_stable)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )
    assert response.status_code == 404
    mock_rag.ask.assert_not_called()


@patch("api.openai_compat.config")
@patch("api.openai_compat.rag_service")
def test_chat_completions_requires_api_key(mock_rag: MagicMock, mock_config: MagicMock) -> None:
    mock_config.compacs_api_key = "secret-key"
    mock_config.compacs_models = ("compacs-rag",)
    mock_rag.ask.return_value = {"answer": "ok", "sources": []}
    client = TestClient(app_stable)
    unauthorized = client.post(
        "/v1/chat/completions",
        json={
            "model": "compacs-rag",
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )
    assert unauthorized.status_code == 401
    authorized = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret-key"},
        json={
            "model": "compacs-rag",
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )
    assert authorized.status_code == 200

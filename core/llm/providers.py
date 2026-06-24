"""LLM provider implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, cast

import ollama

from config import Config
from core.cost_guard import CostGuard
from core.openai_client import get_openai_client

log = logging.getLogger(__name__)


def _ollama_chat_content(message: Any) -> str:
    """Extract assistant text from ollama dict or pydantic Message."""
    if message is None:
        return ""
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _ollama_response_message(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("message")
    return getattr(response, "message", None)


class CompletionProvider(ABC):
    """Abstract chat completion provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate a chat completion."""

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """Stream completion tokens; default falls back to blocking complete."""
        yield self.complete(messages, temperature, max_tokens)


class OpenAICompletionProvider(CompletionProvider):
    """OpenAI chat completion provider."""

    def __init__(self, config: Config, cost_guard: CostGuard) -> None:
        self._config = config
        self._cost_guard = cost_guard
        self._client = get_openai_client()

    @property
    def name(self) -> str:
        return "openai"

    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        self._cost_guard.check_limits()
        response = self._client.chat.completions.create(
            model=self._config.openai_model,
            messages=cast(Any, messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self._config.top_p,
        )
        self._cost_guard.track_usage(response.usage, self._config.openai_model)
        content = response.choices[0].message.content
        return content if content is not None else ""

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        self._cost_guard.check_limits()
        stream = self._client.chat.completions.create(
            model=self._config.openai_model,
            messages=cast(Any, messages),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self._config.top_p,
            stream=True,
        )
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        if usage is not None:
            self._cost_guard.track_usage(usage, self._config.openai_model)


class OllamaCompletionProvider(CompletionProvider):
    """Ollama chat completion provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = ollama.Client(
            host=config.ollama_host,
            timeout=config.ollama_client_timeout,
        )

    @property
    def name(self) -> str:
        return "ollama"

    def _chat_options(self, temperature: float, max_tokens: int) -> dict[str, Any]:
        return {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self._config.num_ctx,
            "top_p": self._config.top_p,
            "repeat_penalty": self._config.repeat_penalty,
        }

    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        response = self._client.chat(
            model=self._config.ollama_model,
            messages=messages,
            keep_alive=self._config.ollama_keep_alive,
            options=self._chat_options(temperature, max_tokens),
        )
        return _ollama_chat_content(_ollama_response_message(response))

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        stream = self._client.chat(
            model=self._config.ollama_model,
            messages=messages,
            stream=True,
            keep_alive=self._config.ollama_keep_alive,
            options=self._chat_options(temperature, max_tokens),
        )
        for chunk in stream:
            content = _ollama_chat_content(_ollama_response_message(chunk))
            if content:
                yield content


def build_completion_provider(config: Config, cost_guard: CostGuard) -> CompletionProvider:
    """Create primary completion provider from configuration."""
    if config.llm_provider == "ollama":
        return OllamaCompletionProvider(config)
    if config.llm_provider == "openai":
        return OpenAICompletionProvider(config, cost_guard)
    if _openai_key_configured(config.openai_api_key):
        return OpenAICompletionProvider(config, cost_guard)
    return OllamaCompletionProvider(config)


def _openai_key_configured(key: str | None) -> bool:
    if not key or not str(key).strip():
        return False
    normalized = str(key).strip().lower()
    return normalized not in {"user_provided", "changeme", "none", "null"}

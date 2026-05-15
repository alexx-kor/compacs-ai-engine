"""LLM provider implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import ollama

from config import Config
from core.cost_guard import CostGuard
from core.openai_client import get_openai_client

log = logging.getLogger(__name__)


class CompletionProvider(ABC):
    """Abstract chat completion provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Generate a chat completion."""


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


class OllamaCompletionProvider(CompletionProvider):
    """Ollama chat completion provider."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = ollama.Client(host=config.ollama_host)

    @property
    def name(self) -> str:
        return "ollama"

    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        response = self._client.chat(
            model=self._config.ollama_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": self._config.top_p,
            },
        )
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return ""


def build_completion_provider(config: Config, cost_guard: CostGuard) -> CompletionProvider:
    """Create primary completion provider from configuration."""
    if config.llm_provider == "ollama":
        return OllamaCompletionProvider(config)
    if config.llm_provider == "openai":
        return OpenAICompletionProvider(config, cost_guard)
    if config.openai_api_key:
        return OpenAICompletionProvider(config, cost_guard)
    return OllamaCompletionProvider(config)

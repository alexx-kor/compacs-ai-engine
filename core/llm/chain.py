"""LLM fallback chain."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from config import Config
from core.cost_guard import CostGuard
from core.llm.providers import (
    CompletionProvider,
    OllamaCompletionProvider,
    OpenAICompletionProvider,
    _openai_key_configured,
    build_completion_provider,
)

log = logging.getLogger(__name__)


class LLMChain:
    """Try primary LLM provider and optionally fall back to Ollama."""

    def __init__(self, config: Config, cost_guard: CostGuard) -> None:
        self._config = config
        self._primary = build_completion_provider(config, cost_guard)
        self._fallback: CompletionProvider | None = None
        if config.llm_fallback_enabled and self._primary.name != "ollama":
            self._fallback = OllamaCompletionProvider(config)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Generate a completion and return ``(text, provider_name)``."""
        try:
            if isinstance(self._primary, OpenAICompletionProvider) and not _openai_key_configured(
                self._config.openai_api_key
            ):
                raise ValueError("OPENAI_API_KEY is not configured")
            answer = self._primary.complete(messages, temperature, max_tokens)
            return answer, self._primary.name
        except Exception as error:
            if self._fallback is None:
                raise
            log.warning(
                "Primary LLM provider=%s failed, falling back to ollama: %s",
                self._primary.name,
                error,
            )
            answer = self._fallback.complete(messages, temperature, max_tokens)
            return answer, self._fallback.name

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[tuple[str, str]]:
        """Stream tokens as ``(text, provider_name)`` tuples."""
        try:
            if isinstance(self._primary, OpenAICompletionProvider) and not _openai_key_configured(
                self._config.openai_api_key
            ):
                raise ValueError("OPENAI_API_KEY is not configured")
            for token in self._primary.stream_complete(messages, temperature, max_tokens):
                yield token, self._primary.name
            return
        except Exception as error:
            if self._fallback is None:
                raise
            log.warning(
                "Primary LLM provider=%s stream failed, falling back to ollama: %s",
                self._primary.name,
                error,
            )
            for token in self._fallback.stream_complete(messages, temperature, max_tokens):
                yield token, self._fallback.name

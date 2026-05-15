"""Lazy OpenAI client factory."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import config


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a shared OpenAI client, created on first use."""
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=config.openai_api_key)

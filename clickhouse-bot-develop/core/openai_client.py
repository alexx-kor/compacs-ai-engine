"""Lazy OpenAI client factory (no network or API key required at import time)."""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a shared OpenAI client, created on first use."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

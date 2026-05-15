"""HTTP client for optional external preprocessing service."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, cast


def preprocess_via_service(text: str, service_url: str, timeout_seconds: float = 10.0) -> str:
    """Send text to preprocessing HTTP service and return cleaned text."""
    payload = {"text": text}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=service_url.rstrip("/") + "/clean",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return text

    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        cleaned_text = cast(str, parsed["text"])
        return cleaned_text
    return text


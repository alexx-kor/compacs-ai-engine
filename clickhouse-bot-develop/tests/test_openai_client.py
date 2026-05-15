from __future__ import annotations

import importlib

import pytest


def test_openai_client_import_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the lazy client module must not require OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = importlib.import_module("core.openai_client")
    importlib.reload(module)

    assert hasattr(module, "get_openai_client")


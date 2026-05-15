from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("STORAGE_BACKEND", "json")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("LOCAL_VECTOR_STORE_DIR", str(PROJECT_ROOT / "data" / "vectors_test"))


@pytest.fixture(autouse=True)
def _reset_config_module() -> None:
    import importlib

    import config as config_module

    importlib.reload(config_module)

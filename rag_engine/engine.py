"""RAG engine entrypoint delegating to unified rag service."""

from __future__ import annotations

from typing import Any

from rag_service import rag_service


class RAGEngine:
    """Compatibility wrapper around :class:`RagService`."""

    @staticmethod
    def ask(question: str) -> dict[str, Any]:
        return rag_service.ask(question)


rag = RAGEngine()

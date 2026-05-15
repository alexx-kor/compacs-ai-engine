"""Re-rank retrieval rows by lexical overlap and distance-derived similarity."""

from __future__ import annotations

import re
from typing import Any

from config import config


class Reranker:
    """Score and sort vector search rows before context assembly."""

    @staticmethod
    def rerank(question: str, results: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
        """Combine distance and keyword overlap into a single ranking score.

        Args:
            question: User query text.
            results: Rows with at least four columns (chunk, source, page, distance).

        Returns:
            Top ``config.rerank_top_k`` rows in descending score order, or the
            original slice when scoring cannot be applied.
        """
        if not results:
            return results

        q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))

        scored: list[tuple[float, tuple[Any, ...]]] = []
        for result in results:
            if len(result) < 4:
                continue

            chunk_raw = result[0]
            chunk = str(chunk_raw) if chunk_raw is not None else ""
            distance = result[3]

            try:
                if isinstance(distance, str):
                    distance = float(distance)
                distance = float(distance)
            except (ValueError, TypeError):
                distance = 1.0

            similarity = max(0.0, min(1.0, 1.0 - distance))

            c_words = set(re.findall(r"\b\w{4,}\b", chunk.lower()))
            overlap = len(q_words & c_words) / max(len(q_words), 1)

            final_score = similarity * 0.6 + overlap * 0.4
            scored.append((final_score, result))

        if not scored:
            return list(results[: config.rerank_top_k])

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: config.rerank_top_k]]


reranker = Reranker()

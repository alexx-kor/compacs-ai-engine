"""BM25 sparse index persisted alongside the vector store."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.text_processing import tokenize_for_search

log = logging.getLogger(__name__)


@dataclass
class Bm25Hit:
    chunk_id: str
    score: float


class Bm25Index:
    """Okapi BM25 over pre-tokenized chunk documents."""

    def __init__(
        self,
        chunk_ids: list[str],
        doc_tokens: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(chunk_ids) != len(doc_tokens):
            raise ValueError("chunk_ids and doc_tokens length mismatch")
        self._chunk_ids = chunk_ids
        self._docs = doc_tokens
        self._k1 = k1
        self._b = b
        self._n = len(doc_tokens)
        self._avgdl = sum(len(doc) for doc in doc_tokens) / max(self._n, 1)
        self._df: dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    @classmethod
    def from_chunks(
        cls,
        chunks: list[dict[str, Any]],
        *,
        lemmatize: bool = True,
    ) -> Bm25Index:
        ids: list[str] = []
        docs: list[list[str]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("id", ""))
            text = str(chunk.get("chunk", ""))
            ids.append(chunk_id)
            docs.append(tokenize_for_search(text, lemmatize=lemmatize))
        return cls(ids, docs)

    def search(self, query_tokens: list[str], limit: int) -> list[Bm25Hit]:
        if not query_tokens or self._n == 0 or limit <= 0:
            return []
        scores: list[tuple[float, str]] = []
        for doc_index, doc_tokens in enumerate(self._docs):
            if not doc_tokens:
                continue
            score = self._score_document(query_tokens, doc_tokens)
            if score > 0:
                scores.append((score, self._chunk_ids[doc_index]))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [Bm25Hit(chunk_id=chunk_id, score=score) for score, chunk_id in scores[:limit]]

    def _score_document(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        doc_len = len(doc_tokens)
        term_freq: dict[str, int] = {}
        for token in doc_tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        total = 0.0
        for term in query_tokens:
            if term not in term_freq:
                continue
            df = self._df.get(term, 0)
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            tf = term_freq[term]
            denom = tf + self._k1 * (1.0 - self._b + self._b * doc_len / max(self._avgdl, 1))
            total += idf * (tf * (self._k1 + 1.0)) / max(denom, 1e-9)
        return total

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self._k1,
            "b": self._b,
            "chunk_ids": self._chunk_ids,
            "doc_tokens": self._docs,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        log.info("BM25 index saved path=%s docs=%s", path, self._n)

    @classmethod
    def load(cls, path: Path) -> Bm25Index | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            log.warning("BM25 index load failed path=%s error=%s", path, error)
            return None
        if not isinstance(payload, dict):
            return None
        chunk_ids = payload.get("chunk_ids", [])
        doc_tokens = payload.get("doc_tokens", [])
        if not isinstance(chunk_ids, list) or not isinstance(doc_tokens, list):
            return None
        index = cls(
            [str(item) for item in chunk_ids],
            [[str(token) for token in doc] for doc in doc_tokens],
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )
        log.info("BM25 index loaded path=%s docs=%s", path, index._n)
        return index

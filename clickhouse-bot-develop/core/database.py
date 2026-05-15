"""Local vector store: main, hypothesis, and graph chunk datasets on disk."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import config


log = logging.getLogger(__name__)

__all__ = ["DatabaseManager", "db"]


def _cosine_distance(query: Sequence[float], vector: Sequence[float]) -> float:
    """Cosine distance aligned with typical ``1 - cos_sim`` semantics."""
    q = np.asarray(query, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    qn = float(np.linalg.norm(q))
    vn = float(np.linalg.norm(v))
    if qn == 0.0 or vn == 0.0:
        return 1.0
    return 1.0 - float(np.dot(q, v) / (qn * vn))


class DatabaseManager:
    """Persist chunk embeddings under ``config.local_vector_store_dir``."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._cache_time: dict[str, float] = {}
        self.main_table = "main"
        self.hypothesis_table = "hypothesis"
        self._active_table = self.main_table
        self._store_dir = Path(config.local_vector_store_dir)
        self._main_path = self._store_dir / "main_chunks.json"
        self._hypothesis_path = self._store_dir / "hypothesis_chunks.json"
        self._graph_path = self._store_dir / "graph_chunks.json"
        self._main_rows: list[dict[str, Any]] = []
        self._hypothesis_rows: list[dict[str, Any]] = []
        self._graph_rows: list[dict[str, Any]] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load JSON stores if present."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if self._main_path.exists():
            self._main_rows = self._read_rows(self._main_path)
        if self._hypothesis_path.exists():
            self._hypothesis_rows = self._read_rows(self._hypothesis_path)
        if self._graph_path.exists():
            self._graph_rows = self._read_rows(self._graph_path)

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload: Any = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read store path=%s: %s", path, exc)
            return []
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return [r for r in payload["rows"] if isinstance(r, dict)]
        return []

    def _atomic_write(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        body = json.dumps({"version": 1, "rows": rows}, ensure_ascii=False)
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)

    def set_active_table(self, table: str) -> None:
        """Switch active dataset to ``main`` or ``hypothesis``."""
        if table == "main":
            self._active_table = self.main_table
            log.info("Switched to MAIN table")
            return
        if table == "hypothesis":
            self._active_table = self.hypothesis_table
            log.info("Switched to HYPOTHESIS table")
            return
        log.warning("Unknown table=%s, keeping current table=%s", table, self._active_table)

    def load_active_table(self) -> str:
        """Return the active logical table name."""
        return self._active_table

    def get_active_table(self) -> str:
        """Return the active logical table name (alias for :meth:`load_active_table`)."""
        return self.load_active_table()

    def init_main_database(self, force_recreate: bool = False) -> None:
        """Create or reset the main chunk store."""
        if self._main_rows and not force_recreate:
            log.info("Main table already exists")
            log.info("Existing chunks in main: %s", len(self._main_rows))
            return
        if force_recreate:
            log.warning("Force recreating main table")
        self._main_rows = []
        self._atomic_write(self._main_path, self._main_rows)
        log.info("Main table initialized")

    def insert_main_batch(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Append normalized chunk rows to the main store."""
        if not chunks:
            return
        for raw in chunks:
            row = dict(raw)
            row["embedding"] = [float(x) for x in row["embedding"]]
            self._main_rows.append(row)
        self._atomic_write(self._main_path, self._main_rows)
        log.info("Inserted %s chunks into MAIN table", len(chunks))

    def search_main(self, embedding: Sequence[float]) -> list[tuple[Any, ...]]:
        """Vector search over main chunks; returns (chunk, source, page, distance)."""
        return self._vector_search(
            self._main_rows,
            embedding,
            row_to_tuple=lambda r: (r["chunk"], r["source"], int(r["page"]), 0.0),
        )

    def load_main_chunk_count(self) -> int:
        """Return number of rows in the main store."""
        return len(self._main_rows)

    def get_main_chunk_count(self) -> int:
        """Return number of rows in the main store (alias for :meth:`load_main_chunk_count`)."""
        return self.load_main_chunk_count()

    def init_hypothesis_database(self, force_recreate: bool = False) -> None:
        """Create or reset the hypothesis chunk store."""
        if self._hypothesis_rows and not force_recreate:
            log.info("Hypothesis table already exists")
            log.info("Existing chunks in hypothesis: %s", len(self._hypothesis_rows))
            return
        if force_recreate:
            log.warning("Force recreating hypothesis table")
        self._hypothesis_rows = []
        self._atomic_write(self._hypothesis_path, self._hypothesis_rows)
        log.info("Hypothesis table initialized")

    def insert_hypothesis_batch(
        self,
        chunks: Sequence[Mapping[str, Any]],
        hypothesis_name: str = "default",
        hypothesis_params: str = "",
    ) -> None:
        """Append chunk rows tagged with hypothesis metadata."""
        if not chunks:
            return
        for raw in chunks:
            row = dict(raw)
            row["embedding"] = [float(x) for x in row["embedding"]]
            row["answer"] = str(row.get("answer", ""))
            row["hypothesis_name"] = hypothesis_name
            row["hypothesis_params"] = hypothesis_params
            self._hypothesis_rows.append(row)
        self._atomic_write(self._hypothesis_path, self._hypothesis_rows)
        log.info("Inserted %s chunks into HYPOTHESIS table (%s)", len(chunks), hypothesis_name)

    def search_hypothesis(
        self,
        embedding: Sequence[float],
        hypothesis_name: str | None = None,
    ) -> list[tuple[Any, ...]]:
        """Search hypothesis rows; returns (chunk, source, page, hypothesis_name, distance)."""
        rows = self._hypothesis_rows
        if hypothesis_name is not None:
            rows = [r for r in rows if r.get("hypothesis_name") == hypothesis_name]
        return self._vector_search(
            rows,
            embedding,
            row_to_tuple=lambda r: (
                r["chunk"],
                r["source"],
                int(r["page"]),
                str(r.get("hypothesis_name", "")),
                0.0,
            ),
        )

    def load_hypothesis_chunk_count(self, hypothesis_name: str | None = None) -> int:
        """Count hypothesis rows, optionally filtered by name."""
        if hypothesis_name is None:
            return len(self._hypothesis_rows)
        return sum(1 for r in self._hypothesis_rows if r.get("hypothesis_name") == hypothesis_name)

    def get_hypothesis_chunk_count(self, hypothesis_name: str | None = None) -> int:
        """Count hypothesis rows (alias for :meth:`load_hypothesis_chunk_count`)."""
        return self.load_hypothesis_chunk_count(hypothesis_name)

    def load_all_hypotheses(self) -> list[str]:
        """Distinct hypothesis names."""
        names = {str(r.get("hypothesis_name", "")) for r in self._hypothesis_rows}
        names.discard("")
        return sorted(names)

    def get_all_hypotheses(self) -> list[str]:
        """Distinct hypothesis names (alias for :meth:`load_all_hypotheses`)."""
        return self.load_all_hypotheses()

    def delete_hypothesis(self, hypothesis_name: str) -> None:
        """Remove all rows for a hypothesis name."""
        before = len(self._hypothesis_rows)
        self._hypothesis_rows = [
            r for r in self._hypothesis_rows if r.get("hypothesis_name") != hypothesis_name
        ]
        removed = before - len(self._hypothesis_rows)
        if removed:
            self._atomic_write(self._hypothesis_path, self._hypothesis_rows)
            log.info("Deleted hypothesis: %s (%s rows)", hypothesis_name, removed)
        else:
            log.info("No rows for hypothesis: %s", hypothesis_name)

    def init_database(self, force_recreate: bool = False) -> None:
        """Initialize the store backing the active table."""
        if self._active_table == self.main_table:
            self.init_main_database(force_recreate)
            return
        self.init_hypothesis_database(force_recreate)

    def insert_batch(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Insert into whichever store is active."""
        if self._active_table == self.main_table:
            self.insert_main_batch(chunks)
            return
        self.insert_hypothesis_batch(chunks, "default", "")

    def search(self, embedding: Sequence[float]) -> list[tuple[Any, ...]]:
        """Search active store."""
        if self._active_table == self.main_table:
            return self.search_main(embedding)
        return self.search_hypothesis(embedding)

    def load_chunk_count(self) -> int:
        """Row count for the active store."""
        if self._active_table == self.main_table:
            return self.load_main_chunk_count()
        return self.load_hypothesis_chunk_count()

    def get_chunk_count(self) -> int:
        """Row count for the active store (alias for :meth:`load_chunk_count`)."""
        return self.load_chunk_count()

    def resolve_cache(self, key: str) -> Any | None:
        """Resolve a cached RAG answer by key (in-process TTL cache)."""
        if not config.cache_enabled:
            return None
        if key not in self._cache:
            return None
        cached_at = self._cache_time.get(key, 0.0)
        if time.time() - cached_at >= config.cache_ttl:
            return None
        return self._cache[key]

    def get_cache(self, key: str) -> Any | None:
        """In-process TTL cache for serialized RAG answers (alias for :meth:`resolve_cache`)."""
        return self.resolve_cache(key)

    def set_cache(self, key: str, value: Any) -> None:
        """Store a cache entry when caching is enabled."""
        if not config.cache_enabled:
            return
        self._cache[key] = value
        self._cache_time[key] = time.time()

    def insert_graph_batch(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Persist graph-style rows (used for optional graph index workflows)."""
        if not chunks:
            return
        for raw in chunks:
            row = dict(raw)
            row["embedding"] = [float(x) for x in row["embedding"]]
            row["answer"] = str(row.get("answer", ""))
            self._graph_rows.append(row)
        self._atomic_write(self._graph_path, self._graph_rows)
        log.info("Inserted %s chunks into graph table", len(chunks))

    def store_stats(self) -> dict[str, Any]:
        """Diagnostics for local stores (replaces raw SQL introspection)."""
        return {
            "store_dir": str(self._store_dir),
            "main_chunks": len(self._main_rows),
            "hypothesis_chunks": len(self._hypothesis_rows),
            "graph_chunks": len(self._graph_rows),
            "active": self._active_table,
        }

    def embedding_dim_summary(self) -> dict[str, dict[int, int]]:
        """Map ``{store: {dim: count}}`` for quick health checks."""
        out: dict[str, dict[int, int]] = {"main": {}, "hypothesis": {}, "graph": {}}
        for label, rows in (
            ("main", self._main_rows),
            ("hypothesis", self._hypothesis_rows),
            ("graph", self._graph_rows),
        ):
            for row in rows:
                emb = row.get("embedding")
                if not isinstance(emb, list):
                    continue
                dim = len(emb)
                out[label][dim] = out[label].get(dim, 0) + 1
        return out

    def _vector_search(
        self,
        rows: list[dict[str, Any]],
        embedding: Sequence[float],
        row_to_tuple: Callable[[dict[str, Any]], tuple[Any, ...]],
    ) -> list[tuple[Any, ...]]:
        """Filter by distance threshold and return top_k rows."""
        threshold = config.similarity_threshold
        top_k = config.top_k
        scored: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            emb = row.get("embedding")
            if not isinstance(emb, (list, tuple)):
                continue
            dist = _cosine_distance(embedding, emb)
            if dist >= threshold:
                continue
            base = row_to_tuple(row)
            # replace trailing distance placeholder
            parts = list(base[:-1]) + [dist]
            scored.append((dist, tuple(parts)))

        scored.sort(key=lambda item: item[0])
        return [item[1] for item in scored[:top_k]]


db = DatabaseManager()

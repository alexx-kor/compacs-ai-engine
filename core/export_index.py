"""Export vector index for desktop clients."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

from core.database import db
from core.storage.protocol import ChunkRecord


def export_chunks_jsonl(records: list[ChunkRecord] | None = None) -> bytes:
    """Serialize all chunk records to JSONL bytes."""
    rows = records if records is not None else db.vector_store.load_all_records()
    buffer = io.StringIO()
    for record in rows:
        row = record.to_legacy_dict()
        row["dataset_kind"] = record.dataset_kind
        row["metadata"] = record.metadata
        buffer.write(json.dumps(row, ensure_ascii=False) + "\n")
    return buffer.getvalue().encode("utf-8")


def export_filename(suffix: str = "jsonl") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"compacs-vectors-{stamp}.{suffix}"

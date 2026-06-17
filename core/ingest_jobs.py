"""In-process background jobs for document ingestion."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestJob:
    id: str
    collection_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "collection_id": self.collection_id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestJobService:
    """Thread-pool backed ingestion jobs (single-process; use Redis/Celery at scale)."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")

    def submit(self, collection_id: str, filename: str, content: bytes) -> IngestJob:
        job_id = uuid.uuid4().hex[:12]
        job = IngestJob(id=job_id, collection_id=collection_id, filename=filename)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, collection_id, filename, content)
        log.info("ingest job queued id=%s collection=%s file=%s", job_id, collection_id, filename)
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, collection_id: str, filename: str, content: bytes) -> None:
        from core.collections import CollectionError, collection_service

        self._update(job_id, status=JobStatus.RUNNING)
        try:
            doc = collection_service.ingest_document(collection_id, filename, content)
            self._update(
                job_id,
                status=JobStatus.COMPLETED,
                result={
                    "collection_id": collection_id,
                    "document": {
                        "filename": doc.filename,
                        "source": doc.source,
                        "uploaded_at": doc.uploaded_at,
                        "chunk_count": doc.chunk_count,
                        "size_bytes": doc.size_bytes,
                    },
                },
            )
            log.info("ingest job completed id=%s", job_id)
        except CollectionError as error:
            self._update(job_id, status=JobStatus.FAILED, error=str(error))
            log.warning("ingest job failed id=%s error=%s", job_id, error)
        except Exception as error:  # noqa: BLE001
            self._update(job_id, status=JobStatus.FAILED, error=str(error))
            log.exception("ingest job failed id=%s", job_id)

    def _update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = _utc_now()
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error


import os

ingest_jobs = IngestJobService(max_workers=int(os.getenv("INGEST_JOB_WORKERS", "2")))

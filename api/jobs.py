"""HTTP routes for background ingestion jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.ingest_jobs import ingest_jobs

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    """Poll ingestion job status (POST /load?background=true)."""
    job = ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return job.to_dict()

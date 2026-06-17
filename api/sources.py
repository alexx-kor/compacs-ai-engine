"""HTTP API for indexed document sources (v2 /sources routes)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.collections import collection_service
from core.license import activate_pro, current_license
from core.sources import SourceError, source_service

router = APIRouter(tags=["sources"])


class UpgradeRequest(BaseModel):
    license_key: str = Field(min_length=1)


def _source_payload(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "source": item.source,
        "collection_id": item.collection_id,
        "filename": item.filename,
        "uploaded_at": item.uploaded_at,
        "chunk_count": item.chunk_count,
        "size_bytes": item.size_bytes,
        "kind": item.kind,
    }


@router.get("/sources")
async def list_sources() -> dict[str, Any]:
    """List indexed sources with metadata."""
    items = source_service.list_sources()
    return {
        "count": len(items),
        "selected_collection_ids": collection_service.get_selection(),
        "sources": [_source_payload(item) for item in items],
    }


@router.get("/sources/{source_id}/download")
async def download_source(source_id: str) -> FileResponse:
    """Download the original uploaded file for a collection source."""
    try:
        item = source_service.get_source(source_id)
        path = source_service.resolve_file_path(item.source)
    except SourceError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(
        path=path,
        filename=item.filename or path.name,
        media_type="application/octet-stream",
    )


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str) -> dict[str, Any]:
    """Delete a source and remove its chunks from the vector index."""
    try:
        return source_service.delete_source(source_id)
    except SourceError as error:
        status = 404 if "not found" in str(error).lower() else 400
        raise HTTPException(status_code=status, detail=str(error)) from error


@router.post("/upgrade")
async def upgrade_license(request: UpgradeRequest) -> dict[str, Any]:
    """Activate pro tier when a valid license key is supplied."""
    try:
        state = activate_pro(request.license_key)
    except ValueError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    return {"tier": state.tier, "pro_enabled": state.pro_enabled}


@router.get("/license")
async def license_status() -> dict[str, Any]:
    """Return current license tier."""
    state = current_license()
    return {"tier": state.tier, "pro_enabled": state.pro_enabled}

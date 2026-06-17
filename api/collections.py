"""HTTP API for thematic document collections (folders)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from core.collections import CollectionError, collection_service
from core.ingest_jobs import ingest_jobs

router = APIRouter(prefix="/v1/collections", tags=["collections"])


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    id: Optional[str] = Field(default=None, max_length=64)
    description: str = Field(default="", max_length=512)


class SelectionRequest(BaseModel):
    collection_ids: list[str] = Field(default_factory=list)


def _document_payload(doc: Any) -> dict[str, Any]:
    return {
        "filename": doc.filename,
        "source": doc.source,
        "uploaded_at": doc.uploaded_at,
        "chunk_count": doc.chunk_count,
        "size_bytes": doc.size_bytes,
    }


def _collection_payload(info: Any) -> dict[str, Any]:
    return {
        "id": info.id,
        "name": info.name,
        "description": info.description,
        "created_at": info.created_at,
        "document_count": len(info.documents),
        "documents": [_document_payload(doc) for doc in info.documents],
    }


@router.post("")
async def create_collection(request: CreateCollectionRequest) -> dict[str, Any]:
    """Create a thematic folder for RAG documents."""
    try:
        info = collection_service.create_collection(
            request.name,
            collection_id=request.id,
            description=request.description,
        )
    except CollectionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _collection_payload(info)


@router.get("")
async def list_collections() -> dict[str, Any]:
    """List all thematic folders."""
    items = collection_service.list_collections()
    selection = collection_service.get_selection()
    return {
        "selected_collection_ids": selection,
        "collections": [_collection_payload(item) for item in items],
    }


@router.get("/selection")
async def get_selection() -> dict[str, Any]:
    """Return folder ids currently used to scope RAG search."""
    return {"collection_ids": collection_service.get_selection()}


@router.put("/selection")
async def set_selection(request: SelectionRequest) -> dict[str, Any]:
    """Set which folders RAG searches (empty list = search all indexed data)."""
    try:
        ids = collection_service.set_selection(request.collection_ids)
    except CollectionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"collection_ids": ids}


@router.get("/{collection_id}")
async def get_collection(collection_id: str) -> dict[str, Any]:
    """Get one folder and its documents."""
    try:
        info = collection_service.get_collection(collection_id)
    except CollectionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return _collection_payload(info)


@router.delete("/{collection_id}")
async def delete_collection(collection_id: str) -> dict[str, Any]:
    """Delete a folder, its files, and all related vector chunks."""
    try:
        collection_service.delete_collection(collection_id)
    except CollectionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"deleted": collection_id}


@router.get("/{collection_id}/documents")
async def list_documents(collection_id: str) -> dict[str, Any]:
    """List documents in a folder."""
    try:
        documents = collection_service.list_documents(collection_id)
    except CollectionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        "collection_id": collection_id,
        "documents": [_document_payload(doc) for doc in documents],
    }


@router.post("/{collection_id}/documents")
async def upload_document(
    collection_id: str,
    file: UploadFile = File(...),
    background: bool = Query(default=False, description="Queue ingestion; poll GET /v1/jobs/{job_id}"),
) -> dict[str, Any]:
    """Upload a document, chunk it, embed, and index into the folder."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    if background:
        job = ingest_jobs.submit(collection_id, file.filename, content)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "collection_id": collection_id,
            "filename": file.filename,
            "poll_url": f"/v1/jobs/{job.id}",
        }

    try:
        doc = collection_service.ingest_document(collection_id, file.filename, content)
    except CollectionError as error:
        status = 404 if "not found" in str(error) else 400
        raise HTTPException(status_code=status, detail=str(error)) from error
    return {"collection_id": collection_id, "document": _document_payload(doc)}


@router.delete("/{collection_id}/documents/{filename}")
async def delete_document(collection_id: str, filename: str) -> dict[str, Any]:
    """Remove a document from a folder and delete its vector chunks."""
    try:
        collection_service.delete_document(collection_id, filename)
    except CollectionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"collection_id": collection_id, "deleted": filename}

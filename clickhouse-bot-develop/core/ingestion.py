"""Folder-based ingestion for mixed document formats."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.document_processor import doc_processor
from core.pdf_processor import pdf_processor


log = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


@dataclass(frozen=True)
class IngestionReport:
    """Aggregate ingestion stats for one folder run."""

    input_dir: str
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0


class IngestionService:
    """Discover and process supported files from an input directory."""

    @staticmethod
    def collect_chunks(
        input_dir: str,
        start_id: int = 0,
        allowed_extensions: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], IngestionReport]:
        """Collect chunk dicts from ``input_dir`` recursively.

        Args:
            input_dir: Root directory with documents.
            start_id: Base numeric id used for generated chunks.
            allowed_extensions: Optional extension allowlist (lowercase with dot).

        Returns:
            Tuple of ``(chunks, report)``.
        """
        root = Path(input_dir).resolve()
        chunks: list[dict[str, Any]] = []

        if not root.exists():
            log.warning("Input directory does not exist: %s", root)
            return chunks, IngestionReport(input_dir=str(root))

        paths = sorted(path for path in root.rglob("*") if path.is_file())
        files_discovered = len(paths)
        files_processed = 0
        files_skipped = 0

        extensions = allowed_extensions if allowed_extensions is not None else SUPPORTED_EXTENSIONS
        for file_path in paths:
            extension = file_path.suffix.lower()
            if extension not in extensions:
                files_skipped += 1
                continue

            source_name = str(file_path.relative_to(root)).replace("\\", "/")
            next_id = start_id + len(chunks)
            if extension in PDF_EXTENSIONS:
                new_chunks = pdf_processor.process_document(str(file_path), source_name, next_id)
            else:
                new_chunks = doc_processor.process_document(str(file_path), source_name, next_id)

            if not new_chunks:
                files_skipped += 1
                continue

            chunks.extend(new_chunks)
            files_processed += 1

        report = IngestionReport(
            input_dir=str(root),
            files_discovered=files_discovered,
            files_processed=files_processed,
            files_skipped=files_skipped,
            chunks_created=len(chunks),
        )
        return chunks, report


ingestion_service = IngestionService()


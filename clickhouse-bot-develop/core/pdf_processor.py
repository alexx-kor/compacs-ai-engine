"""Extract text from PDFs and chunk for the same schema as plain-text ingestion."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from pypdf import PdfReader

from config import config


log = logging.getLogger(__name__)


class PDFProcessor:
    """PDF-specific extraction and chunking."""

    @staticmethod
    def extract_pdf(pdf_path: str, source_name: str) -> list[tuple[int, str]]:
        """Extract per-page text from a PDF when pages exceed minimum size."""
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            log.info("Total pages=%s source=%s", total_pages, source_name)

            pages: list[tuple[int, str]] = []
            for index in range(total_pages):
                try:
                    page = reader.pages[index]
                    text = page.extract_text()
                    if text and len(text.strip()) > config.min_chunk_size:
                        text = re.sub(r"\n+", " ", text)
                        pages.append((index + 1, text.strip()))
                except (IndexError, KeyError, AttributeError, TypeError) as exc:
                    log.warning(
                        "PDF page extract failed source=%s page=%s: %s",
                        source_name,
                        index + 1,
                        exc,
                    )
            return pages
        except Exception as exc:
            log.warning("PDF extract error path=%s error=%s", pdf_path, exc)
            return []

    @staticmethod
    def split_chunks(text: str) -> list[str]:
        """Split text into overlapping word windows (same policy as documents)."""
        size = config.chunk_size
        words = text.split()
        chunks: list[str] = []
        step = size - config.chunk_overlap

        for offset in range(0, len(words), step):
            chunk = " ".join(words[offset : offset + size])
            if len(chunk) > config.min_chunk_size:
                chunks.append(chunk)
                if len(chunks) >= config.max_chunks_per_doc:
                    break
        return chunks

    @staticmethod
    def process_document(pdf_path: str, source_name: str, start_id: int) -> list[dict[str, Any]]:
        """Build chunk dicts from a PDF path."""
        log.info("Processing PDF source=%s", source_name)
        pages = PDFProcessor.extract_pdf(pdf_path, source_name)

        if not pages:
            return []

        chunks: list[dict[str, Any]] = []
        for page_num, text in pages:
            for chunk in PDFProcessor.split_chunks(text):
                if len(chunk) > config.max_text_length:
                    chunk = chunk[: config.max_text_length]
                chunks.append(
                    {
                        "id": start_id + len(chunks),
                        "source": source_name,
                        "page": page_num,
                        "chunk": chunk,
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest(),
                        "char_count": len(chunk),
                    }
                )
                if len(chunks) >= config.max_chunks_per_doc:
                    break
            if len(chunks) >= config.max_chunks_per_doc:
                break

        log.info("Created %s chunks from PDF source=%s", len(chunks), source_name)
        return chunks


pdf_processor = PDFProcessor()

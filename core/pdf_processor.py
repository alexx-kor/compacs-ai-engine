"""PDF document extraction and chunking utilities."""

import hashlib
import logging
import re

from pypdf import PdfReader

from config import config

log = logging.getLogger(__name__)


class PDFProcessor:
    @staticmethod
    def extract_pdf(pdf_path: str, source_name: str) -> list[tuple[int, str]]:
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            log.info("pdf total pages=%s", total_pages)

            pages = []
            for i in range(total_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text and len(text.strip()) > config.min_chunk_size:
                        text = re.sub(r'\n+', ' ', text)
                        pages.append((i + 1, text.strip()))
                except Exception as exc:
                    log.exception("failed extracting pdf page index=%s source=%s: %s", i, source_name, exc)
                    raise
            return pages
        except Exception as e:
            log.error("pdf extraction failed path=%s error=%s", pdf_path, e)
            return []

    @staticmethod
    def split_chunks(text: str) -> list[str]:
        size = config.chunk_size
        words = text.split()
        chunks = []
        step = size - config.chunk_overlap

        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i+size])
            if len(chunk) > config.min_chunk_size:
                chunks.append(chunk)
                if len(chunks) >= config.max_chunks_per_doc:
                    break
        return chunks

    @staticmethod
    def process_document(pdf_path: str, source_name: str, start_id: int) -> list[dict]:
        log.info("processing pdf source=%s", source_name)
        pages = PDFProcessor.extract_pdf(pdf_path, source_name)

        if not pages:
            return []

        chunks: list[dict] = []
        for page_num, text in pages:
            for chunk in PDFProcessor.split_chunks(text):
                if len(chunk) > config.max_text_length:
                    chunk = chunk[:config.max_text_length]
                chunks.append({
                    'id': start_id + len(chunks),
                    'source': source_name,
                    'page': page_num,
                    'chunk': chunk,
                    'chunk_hash': hashlib.md5(chunk.encode()).hexdigest(),
                    'char_count': len(chunk)
                })
                if len(chunks) >= config.max_chunks_per_doc:
                    break
            if len(chunks) >= config.max_chunks_per_doc:
                break

        log.info("created chunks source=%s count=%s", source_name, len(chunks))
        return chunks


pdf_processor = PDFProcessor()

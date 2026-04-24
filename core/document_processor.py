"""Document chunking utilities for instruction sources."""

import hashlib
import logging

from config import config

log = logging.getLogger(__name__)


class DocumentProcessor:
    """Transform source documents into chunk dictionaries."""

    @staticmethod
    def load_document(file_path: str, source_name: str) -> list[tuple[int, str]]:
        """Load a text document and split by character size.

        Args:
            file_path: Absolute path to source file.
            source_name: Logical source name for logging.

        Returns:
            List of tuples: (page_number, page_text). Returns empty list on failure.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                content = file_handle.read()

            chunks: list[tuple[int, str]] = []
            if len(content) > config.chunk_size:
                parts: list[str] = []
                current: list[str] = []
                current_len = 0

                for line in content.split("\n"):
                    if current_len + len(line) > config.chunk_size:
                        parts.append("\n".join(current))
                        current = [line]
                        current_len = len(line)
                    else:
                        current.append(line)
                        current_len += len(line)
                
                if current:
                    parts.append("\n".join(current))

                for i, part in enumerate(parts):
                    if len(part.strip()) > config.min_chunk_size:
                        chunks.append((i + 1, part.strip()))
            else:
                if len(content.strip()) > config.min_chunk_size:
                    chunks.append((1, content.strip()))

            return chunks
        except (OSError, UnicodeDecodeError, ValueError) as error:
            log.error("failed loading document path=%s source=%s error=%s", file_path, source_name, error)
            return []

    @staticmethod
    def split_chunks(text: str) -> list[str]:
        """Split text into overlapping word chunks.

        Args:
            text: Input page text.

        Returns:
            Chunk list.
        """
        size = config.chunk_size
        words = text.split()
        chunks: list[str] = []
        step = size - config.chunk_overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + size])
            if len(chunk) > config.min_chunk_size:
                chunks.append(chunk)
                if len(chunks) >= config.max_chunks_per_doc:
                    break
        return chunks

    @staticmethod
    def process_document(file_path: str, source_name: str, start_id: int) -> list[dict]:
        """Process one source document into chunk dictionaries.

        Args:
            file_path: Source file path.
            source_name: Source name to store in chunk metadata.
            start_id: First chunk id offset.

        Returns:
            Chunk dictionaries for downstream embedding/indexing.
        """
        log.info("processing source=%s", source_name)
        pages = DocumentProcessor.load_document(file_path, source_name)

        if not pages:
            return []

        chunks: list[dict] = []
        for page_num, text in pages:
            text_chunks = DocumentProcessor.split_chunks(text)

            for chunk in text_chunks:
                if len(chunk) > config.max_text_length:
                    chunk = chunk[:config.max_text_length]
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

        log.info("created chunks source=%s count=%s", source_name, len(chunks))
        return chunks


doc_processor = DocumentProcessor()
#!/usr/bin/env python3
"""Run baseline RAG pipeline with ClickHouse and Ollama synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

import ollama

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTIONS_DIR = ROOT_DIR / "instructions"
DEFAULT_QUESTIONS_FILE = ROOT_DIR / "baseline" / "questions"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "baseline" / "rag_answers_gpu.json"
DEFAULT_DOCKER_CANDIDATES = (
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\Docker\resources\bin\docker.exe"),
)
DEFAULT_LOCAL_CLICKHOUSE_HOST = "127.0.0.1"
DEFAULT_LOCAL_CLICKHOUSE_PORT = 8123
DEFAULT_CLICKHOUSE_CONTAINER_NAME = "clickhouse-server"
DEFAULT_CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server"
DEFAULT_TOP_K = 8
DEFAULT_EMBED_BATCH = 16
DEFAULT_TOKEN_MIN_SCORE = 0.02
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_CLIENT_TIMEOUT", "120"))
NOT_FOUND = "Information not found in the current documentation index."
SYNTHESIS_SYSTEM_PROMPT = (
    "Ты — технический эксперт. На основе предоставленного контекста дай прямой, краткий ответ. "
    "Убери мета-данные аудита (Confidence, API involved). Оставь только суть и команды в Markdown. "
    "Не цитируй дословно большие блоки документа — переформулируй и сожми. "
    "Не включай в ответ списки источников, имена файлов, номера страниц и служебные метки. "
    "Если в контексте нет ответа, напиши ровно: Information not found in the current documentation index."
)

os.environ.setdefault("DOCS_FOLDER", str(DEFAULT_INSTRUCTIONS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from config import config
from core.document_processor import doc_processor
from core.embeddings import embedder

log = logging.getLogger(__name__)
_ollama_client: ollama.Client | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate baseline RAG answers.")
    parser.add_argument("--instructions-dir", default=str(DEFAULT_INSTRUCTIONS_DIR))
    parser.add_argument("--questions-file", default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embed-batch", type=int, default=DEFAULT_EMBED_BATCH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def configure_logging(is_debug: bool) -> None:
    """Configure module logging.

    Args:
        is_debug: Enables debug level when True.
    """
    level = logging.DEBUG if is_debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_text(file_path: Path) -> str:
    """Load UTF-8 text file.

    Args:
        file_path: Source text file path.

    Returns:
        File content.
    """
    with file_path.open("r", encoding="utf-8") as file_handle:
        return file_handle.read()


def load_questions(questions_file: Path) -> list[str]:
    """Load non-empty questions from file.

    Args:
        questions_file: Path to questions file.

    Returns:
        Question list.
    """
    with questions_file.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def resolve_docker_executable() -> Path | None:
    """Resolve docker executable location.

    Returns:
        Docker executable path or None.
    """
    system_path = shutil.which("docker")
    if system_path:
        return Path(system_path)
    for docker_candidate in DEFAULT_DOCKER_CANDIDATES:
        if docker_candidate.is_file():
            return docker_candidate
    return None


def is_tcp_port_open(host: str, port: int) -> bool:
    """Check whether TCP endpoint accepts connections.

    Args:
        host: Target host.
        port: Target port.

    Returns:
        True when endpoint is accepting connections.
    """
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


def log_port_diagnostics() -> None:
    """Log diagnostics for local ClickHouse port."""
    if not is_tcp_port_open(DEFAULT_LOCAL_CLICKHOUSE_HOST, DEFAULT_LOCAL_CLICKHOUSE_PORT):
        log.info("[docker] port 8123 is not accepting connections")
        return

    log.warning("[docker] port 8123 is occupied")
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [
                line
                for line in (result.stdout or "").splitlines()
                if ":8123" in line and "LISTENING" in line.upper()
            ]
            for line in lines[:12]:
                log.warning("[docker] netstat=%s", line.strip())
            if not lines:
                log.warning("[docker] netstat has no LISTENING row for :8123")
        else:
            result = subprocess.run(
                ["ss", "-lntp"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in (result.stdout or "").splitlines():
                if ":8123" in line:
                    log.warning("[docker] ss=%s", line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        log.warning("[docker] diagnostics command failed: %s", error)


def inspect_docker_container_status(docker_executable: Path, container_name: str) -> str | None:
    """Inspect Docker container status.

    Args:
        docker_executable: Docker executable path.
        container_name: Container name.

    Returns:
        Container state string or None if container does not exist.
    """
    result = subprocess.run(
        [str(docker_executable), "inspect", "-f", "{{.State.Status}}", container_name],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip().lower() or None


def start_docker_container(docker_executable: Path, container_name: str) -> bool:
    """Start existing Docker container.

    Args:
        docker_executable: Docker executable path.
        container_name: Container name.

    Returns:
        True when start succeeded.
    """
    result = subprocess.run(
        [str(docker_executable), "start", container_name],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        log.error("[docker] docker start failed code=%s details=%s", result.returncode, error_text[:1500])
        return False
    log.info("[docker] started existing container")
    return True


def create_docker_container(docker_executable: Path, container_name: str, image_name: str) -> bool:
    """Create ClickHouse Docker container.

    Args:
        docker_executable: Docker executable path.
        container_name: Container name.
        image_name: Docker image.

    Returns:
        True when create succeeded.
    """
    result = subprocess.run(
        [
            str(docker_executable),
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            "8123:8123",
            "-p",
            "9000:9000",
            "--ulimit",
            "nofile=262144:262144",
            image_name,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "").strip()
        log.error("[docker] docker run failed code=%s details=%s", result.returncode, error_text[:2000])
        log_port_diagnostics()
        return False
    log.info("[docker] created new container")
    return True


def wait_for_clickhouse_ping(host: str, port: int, secure: bool, max_wait_sec: float = 60.0) -> bool:
    """Wait until ClickHouse ping endpoint is healthy.

    Args:
        host: ClickHouse host.
        port: ClickHouse HTTP port.
        secure: Uses HTTPS when True.
        max_wait_sec: Maximum waiting time.

    Returns:
        True when endpoint becomes healthy.
    """
    scheme = "https" if secure else "http"
    url = f"{scheme}://{host}:{port}/ping"
    deadline = time.monotonic() + max_wait_sec
    log.info("[clickhouse] waiting for ping max_wait_sec=%.0f", max_wait_sec)
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode("utf-8", errors="ignore").strip()
                if body == "Ok." or "Ok" in body:
                    log.info("[clickhouse] ping successful")
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        log.info("[clickhouse] ping retry")
        time.sleep(2.0)
    log.error("[clickhouse] ping timeout")
    return False


def is_clickhouse_http_alive(host: str, port: int, secure: bool) -> bool:
    """Check ClickHouse HTTP ping endpoint.

    Args:
        host: ClickHouse host.
        port: ClickHouse HTTP port.
        secure: Uses HTTPS when True.

    Returns:
        True when ping endpoint is healthy.
    """
    scheme = "https" if secure else "http"
    url = f"{scheme}://{host}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            body = response.read().decode("utf-8", errors="ignore").strip()
            return body == "Ok." or "Ok" in body
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def create_clickhouse_client(host: str, user: str, password: str, secure: bool) -> Any:
    """Create ClickHouse client instance.

    Args:
        host: ClickHouse host.
        user: ClickHouse user.
        password: ClickHouse password.
        secure: Uses HTTPS when True.

    Returns:
        ClickHouse client.
    """
    import clickhouse_connect

    port = 8443 if secure else 8123
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        secure=secure,
        compress=True,
        connect_timeout=15,
    )


def can_query_clickhouse(client: Any) -> bool:
    """Check whether ClickHouse client can execute SQL.

    Args:
        client: ClickHouse client.

    Returns:
        True when basic query succeeds.
    """
    try:
        client.query("SELECT 1")
        return True
    except RuntimeError:
        return False
    except OSError:
        return False


def ensure_local_clickhouse_container() -> bool:
    """Ensure local ClickHouse docker container is running.

    Returns:
        True when container is up and healthy.
    """
    docker_executable = resolve_docker_executable()
    if docker_executable is None:
        log.error("[clickhouse] docker executable not found")
        return False

    status = inspect_docker_container_status(docker_executable, DEFAULT_CLICKHOUSE_CONTAINER_NAME)
    if status == "running":
        log.info("[docker] container already running")
        return wait_for_clickhouse_ping(DEFAULT_LOCAL_CLICKHOUSE_HOST, DEFAULT_LOCAL_CLICKHOUSE_PORT, False)

    if status in {"exited", "created", "paused", "restarting", "dead"}:
        if not start_docker_container(docker_executable, DEFAULT_CLICKHOUSE_CONTAINER_NAME):
            return False
        return wait_for_clickhouse_ping(DEFAULT_LOCAL_CLICKHOUSE_HOST, DEFAULT_LOCAL_CLICKHOUSE_PORT, False)

    if status is not None:
        log.warning("[docker] container state=%s", status)
        if start_docker_container(docker_executable, DEFAULT_CLICKHOUSE_CONTAINER_NAME):
            return wait_for_clickhouse_ping(DEFAULT_LOCAL_CLICKHOUSE_HOST, DEFAULT_LOCAL_CLICKHOUSE_PORT, False)
        log.warning("[docker] start failed, attempting docker run")

    if not create_docker_container(
        docker_executable,
        DEFAULT_CLICKHOUSE_CONTAINER_NAME,
        DEFAULT_CLICKHOUSE_IMAGE,
    ):
        return False
    return wait_for_clickhouse_ping(DEFAULT_LOCAL_CLICKHOUSE_HOST, DEFAULT_LOCAL_CLICKHOUSE_PORT, False)


def ensure_clickhouse() -> tuple[Any | None, bool]:
    """Resolve ClickHouse connection with local Docker fallback.

    Returns:
        Tuple of (client, is_clickhouse_available).
    """
    port = 8443 if config.ch_secure else 8123
    if not is_clickhouse_http_alive(config.ch_host, port, config.ch_secure):
        log.warning("[clickhouse] ping failed host=%s port=%s", config.ch_host, port)
    try:
        configured_client = create_clickhouse_client(config.ch_host, config.ch_user, config.ch_password, config.ch_secure)
        if can_query_clickhouse(configured_client):
            log.info("[clickhouse] connected to configured endpoint")
            return configured_client, True
    except (RuntimeError, OSError, ValueError) as error:
        log.warning("[clickhouse] configured endpoint failed: %s", error)

    if ensure_local_clickhouse_container():
        try:
            local_client = create_clickhouse_client(DEFAULT_LOCAL_CLICKHOUSE_HOST, "default", "", False)
            if can_query_clickhouse(local_client):
                log.info("[clickhouse] connected to local Docker endpoint")
                return local_client, True
        except (RuntimeError, OSError, ValueError) as error:
            log.warning("[clickhouse] local endpoint failed: %s", error)

    log.warning("[clickhouse] unavailable, using token fallback")
    return None, False


def initialize_rag_table(client: Any) -> None:
    """Recreate target ClickHouse table for chunks.

    Args:
        client: ClickHouse client.
    """
    client.command("DROP TABLE IF EXISTS default.rag_chunks")
    client.command(
        """
        CREATE TABLE IF NOT EXISTS default.rag_chunks (
            id UInt64,
            source String,
            page UInt32,
            chunk String,
            embedding Array(Float32),
            chunk_hash String,
            char_count UInt32,
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY source
        ORDER BY id
        """
    )
    log.info("[clickhouse] table default.rag_chunks ready")


def insert_clickhouse_chunks(client: Any, chunks: Sequence[dict[str, Any]]) -> None:
    """Insert chunk batch to ClickHouse.

    Args:
        client: ClickHouse client.
        chunks: Prepared chunk rows.
    """
    if not chunks:
        return
    rows = [
        [chunk["id"], chunk["source"], chunk["page"], chunk["chunk"], chunk["embedding"], chunk["chunk_hash"], chunk["char_count"]]
        for chunk in chunks
    ]
    client.insert(
        "default.rag_chunks",
        rows,
        column_names=["id", "source", "page", "chunk", "embedding", "chunk_hash", "char_count"],
    )
    log.info("[clickhouse] inserted rows=%s", len(chunks))


def search_clickhouse_chunks(client: Any, embedding: Sequence[float], top_k: int) -> list[tuple[str, str, int, float]]:
    """Search nearest chunks in ClickHouse.

    Args:
        client: ClickHouse client.
        embedding: Query embedding.
        top_k: Max results.

    Returns:
        Search rows.
    """
    result = client.query(
        """
        SELECT chunk, source, page, cosineDistance(embedding, %(emb)s) AS distance
        FROM default.rag_chunks
        ORDER BY distance ASC
        LIMIT %(top_k)s
        """,
        parameters={"emb": list(embedding), "top_k": top_k},
    )
    return list(result.result_rows)


def try_embed_texts(texts: Sequence[str]) -> list[list[float]] | None:
    """Generate embeddings with error handling.

    Args:
        texts: Input texts.

    Returns:
        Embeddings or None on failure.
    """
    if not texts:
        return []
    try:
        return embedder.generate(list(texts))
    except (RuntimeError, OSError, ValueError) as error:
        log.error("[embedder] batch failed: %s", error)
        return None


def try_embed_query_text(query_text: str) -> list[float] | None:
    """Generate query embedding with error handling.

    Args:
        query_text: Query text.

    Returns:
        Query embedding or None on failure.
    """
    try:
        return list(embedder.generate_cached(query_text))
    except (RuntimeError, OSError, ValueError) as error:
        log.error("[embedder] query failed: %s", error)
        return None


def load_instruction_chunks_to_clickhouse(
    client: Any,
    instructions_dir: Path,
    embed_batch: int,
) -> tuple[int, bool]:
    """Load instruction chunks to ClickHouse with embeddings.

    Args:
        client: ClickHouse client.
        instructions_dir: Instruction files directory.
        embed_batch: Embedding batch size.

    Returns:
        Tuple of (loaded_chunks_count, embeddings_success).
    """
    initialize_rag_table(client)
    all_chunks: list[dict[str, Any]] = []
    next_chunk_id = 0
    for source_file in sorted(instructions_dir.glob("*.txt")):
        chunk_part = doc_processor.process_document(str(source_file), source_file.name, next_chunk_id)
        if not chunk_part:
            continue
        all_chunks.extend(chunk_part)
        next_chunk_id += len(chunk_part)

    for offset in range(0, len(all_chunks), embed_batch):
        chunk_batch = all_chunks[offset : offset + embed_batch]
        batch_embeddings = try_embed_texts([chunk["chunk"] for chunk in chunk_batch])
        if batch_embeddings is None:
            log.error("[embedder] disabled ClickHouse path for current run")
            return 0, False
        for chunk, embedding in zip(chunk_batch, batch_embeddings):
            chunk["embedding"] = embedding
        insert_clickhouse_chunks(client, chunk_batch)
        loaded = min(offset + embed_batch, len(all_chunks))
        log.info("[embedder] loaded=%s/%s", loaded, len(all_chunks))

    return len(all_chunks), True


def build_token_overlap_index(instructions_dir: Path) -> list[dict[str, Any]]:
    """Build fallback token overlap index.

    Args:
        instructions_dir: Instruction files directory.

    Returns:
        Indexed token rows.
    """
    token_rows: list[dict[str, Any]] = []
    for source_file in sorted(instructions_dir.glob("*.txt")):
        source_text = load_text(source_file)
        for page_number, paragraph in enumerate(source_text.split("\n\n"), start=1):
            paragraph = paragraph.strip()
            if len(paragraph) < 80:
                continue
            token_rows.append(
                {
                    "source": source_file.name,
                    "page": page_number,
                    "chunk": paragraph[: config.max_text_length],
                    "tokens": set(re.findall(r"[\w\d]{3,}", paragraph.lower())),
                }
            )
    log.info("[clickhouse] token overlap rows=%s", len(token_rows))
    return token_rows


def retrieve_by_token_overlap(
    token_index: Sequence[dict[str, Any]],
    question: str,
    top_k: int,
) -> list[tuple[str, str, int, float]]:
    """Retrieve fallback rows by token overlap.

    Args:
        token_index: Token index rows.
        question: User question.
        top_k: Number of rows.

    Returns:
        Fallback retrieval rows.
    """
    question_tokens = set(re.findall(r"[\w\d]{3,}", question.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []
    for token_item in token_index:
        overlap_score = (
            len(question_tokens & token_item["tokens"]) / max(len(question_tokens), 1)
            if question_tokens
            else 0.0
        )
        scored.append((overlap_score, token_item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        (item["chunk"], item["source"], item["page"], 1.0 - score)
        for score, item in scored[:top_k]
    ]


def format_retrieval_context(rows: Sequence[tuple[str, str, int, float]]) -> str:
    """Format retrieval context for LLM prompt.

    Args:
        rows: Retrieval rows.

    Returns:
        Formatted context string.
    """
    return "\n\n---\n\n".join(f"[{source} p.{page}]\n{chunk[:2500]}" for chunk, source, page, _ in rows)


def normalize_answer_text(answer_text: str) -> str:
    """Normalize model answer for not-found responses.

    Args:
        answer_text: Raw model answer.

    Returns:
        Normalized answer.
    """
    normalized = answer_text.strip()
    if not normalized:
        return NOT_FOUND
    if normalized.split("\n", 1)[0].strip().lower().startswith(NOT_FOUND.lower()):
        return NOT_FOUND
    return normalized


def build_ollama_client() -> ollama.Client:
    """Create or reuse Ollama client.

    Returns:
        Ollama client.
    """
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(timeout=DEFAULT_OLLAMA_TIMEOUT)
    return _ollama_client


def generate_llm_answer(question: str, context: str) -> tuple[str, str | None]:
    """Generate answer via Ollama synthesis.

    Args:
        question: User question.
        context: Retrieved context.

    Returns:
        Tuple of (answer, error_message).
    """
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    try:
        response = build_ollama_client().chat(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={
                "num_gpu": config.ollama_num_gpu,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 1400,
            },
        )
    except (RuntimeError, OSError, ValueError) as error:
        log.error("[llm] chat failed: %s", error)
        return NOT_FOUND, str(error)
    content = (response.get("message") or {}).get("content") or ""
    return normalize_answer_text(content), None


def build_retriever(
    clickhouse_client: Any | None,
    token_index: Sequence[dict[str, Any]],
    top_k: int,
) -> Callable[[str], tuple[list[tuple[str, str, int, float]], list[str], str, float | None]]:
    """Build retrieval function with ClickHouse and fallback.

    Args:
        clickhouse_client: ClickHouse client or None.
        token_index: Fallback token index.
        top_k: Max retrieval rows.

    Returns:
        Retrieval callable.
    """
    def retrieve(question: str) -> tuple[list[tuple[str, str, int, float]], list[str], str, float | None]:
        if clickhouse_client is not None:
            query_embedding = try_embed_query_text(question)
            if query_embedding is not None:
                vector_rows = search_clickhouse_chunks(clickhouse_client, query_embedding, top_k)
                vector_distance = float(vector_rows[0][3]) if vector_rows else None
                if vector_rows and vector_distance is not None and vector_distance <= config.similarity_threshold:
                    source_list = sorted({row[1] for row in vector_rows})
                    return vector_rows, source_list, "vector", vector_distance
                token_rows = retrieve_by_token_overlap(token_index, question, top_k)
                source_list = sorted({row[1] for row in token_rows})
                return token_rows, source_list, "token", vector_distance
        token_rows = retrieve_by_token_overlap(token_index, question, top_k)
        source_list = sorted({row[1] for row in token_rows})
        return token_rows, source_list, "token", None

    return retrieve


def run_pipeline(
    instructions_dir: Path,
    questions_file: Path,
    output_file: Path,
    top_k: int,
    embed_batch: int,
) -> None:
    """Execute baseline RAG pipeline.

    Args:
        instructions_dir: Instruction files directory.
        questions_file: Questions file path.
        output_file: Output JSON path.
        top_k: Retrieval top-k.
        embed_batch: Embedding batch size.
    """
    questions = load_questions(questions_file)
    log.info("[setup] questions=%s", len(questions))

    clickhouse_client, has_clickhouse = ensure_clickhouse()
    token_index = build_token_overlap_index(instructions_dir)

    if has_clickhouse and clickhouse_client is not None:
        chunks_count, embeddings_ok = load_instruction_chunks_to_clickhouse(clickhouse_client, instructions_dir, embed_batch)
        if embeddings_ok:
            log.info("[clickhouse] indexed chunks=%s", chunks_count)
        else:
            clickhouse_client = None

    retrieve = build_retriever(clickhouse_client, token_index, top_k)
    results: list[dict[str, Any]] = []
    total_questions = len(questions)

    for index, question in enumerate(questions, start=1):
        rows, sources, retrieval_kind, best_distance = retrieve(question)
        if not rows:
            answer = NOT_FOUND
            llm_error = None
            log.info("[llm] %s/%s no context", index, total_questions)
        elif retrieval_kind == "token" and (1.0 - float(rows[0][3])) < DEFAULT_TOKEN_MIN_SCORE:
            answer = NOT_FOUND
            llm_error = None
            log.info("[llm] %s/%s token score too low", index, total_questions)
        else:
            context = format_retrieval_context(rows)
            log.info("[llm] %s/%s synthesize kind=%s", index, total_questions, retrieval_kind)
            answer, llm_error = generate_llm_answer(question, context)

        metadata: dict[str, Any] = {
            "sources": sources,
            "retrieval_kind": retrieval_kind,
            "best_vector_distance": best_distance,
        }
        if llm_error is not None:
            metadata["llm_error"] = llm_error

        results.append(
            {
                "id": index,
                "question": question,
                "answer": answer,
                "_meta": metadata,
            }
        )
        log.info("[llm] %s/%s done", index, total_questions)

    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, ensure_ascii=False, indent=2)

    output_rel = output_file.relative_to(ROOT_DIR).as_posix()
    log.info("[batch] Processed %s/%s questions. Results saved to %s", len(results), total_questions, output_rel)


def main() -> int:
    """Program entry point.

    Returns:
        Process exit code.
    """
    args = parse_args()
    configure_logging(is_debug=args.debug)
    run_pipeline(
        instructions_dir=Path(args.instructions_dir),
        questions_file=Path(args.questions_file),
        output_file=Path(args.output_file),
        top_k=max(1, args.top_k),
        embed_batch=max(1, args.embed_batch),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Full RAG baseline: ClickHouse vector store (with Docker bootstrap) + Ollama LLM synthesis.
Fallback: in-memory token overlap retrieval if ClickHouse is unavailable; answers still via LLM.
"""
from __future__ import annotations

import json
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
from typing import Any, Callable, Optional

import ollama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DOCS_FOLDER", str(ROOT / "instructions"))

from config import config
from core.document_processor import doc_processor
from core.embeddings import embedder

INSTRUCTIONS_DIR = ROOT / "instructions"
QUESTIONS_FILE = ROOT / "baseline" / "questions"
OUTPUT_FILE = ROOT / "baseline" / "rag_answers_gpu.json"

LLM_MODEL = "llama3.2:3b"
TOP_K = 8
EMBED_BATCH = 16
CH_CONTAINER = "clickhouse-server"
CH_IMAGE = "clickhouse/clickhouse-server"
LOCAL_CH_HOST = "127.0.0.1"
LOCAL_CH_PORT = 8123

# Hard ceiling for LLM HTTP client (seconds); override only if needed.
OLLAMA_HARD_TIMEOUT = float(os.getenv("OLLAMA_CLIENT_TIMEOUT", "120"))

DOCKER_CANDIDATES = [
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\Docker\resources\bin\docker.exe"),
]

SYNTHESIS_SYSTEM = (
    "Ты — технический эксперт. На основе предоставленного контекста дай прямой, краткий ответ. "
    "Убери мета-данные аудита (Confidence, API involved). Оставь только суть и команды в Markdown. "
    "Не цитируй дословно большие блоки документа — переформулируй и сожми. "
    "Не включай в ответ списки источников, имена файлов, номера страниц и служебные метки. "
    "Если в контексте нет ответа, напиши ровно: Information not found in the current documentation index."
)

NOT_FOUND = "Information not found in the current documentation index."

_ollama_client: Optional[ollama.Client] = None


def get_ollama_client() -> ollama.Client:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(timeout=OLLAMA_HARD_TIMEOUT)
    return _ollama_client


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_questions() -> list[str]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_docker_exe() -> Optional[Path]:
    which = shutil.which("docker")
    if which:
        return Path(which)
    for p in DOCKER_CANDIDATES:
        if p.is_file():
            return p
    return None


def tcp_port_accepting(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.8):
            return True
    except OSError:
        return False


def log_port_8123_diagnostics() -> None:
    if not tcp_port_accepting(LOCAL_CH_HOST, LOCAL_CH_PORT):
        print("[docker] port 8123: nothing accepting connections (or unreachable)", flush=True)
        return
    print("[docker] port 8123: a process is accepting TCP connections (possible ClickHouse or conflict)", flush=True)
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [ln for ln in (r.stdout or "").splitlines() if ":8123" in ln and "LISTENING" in ln.upper()]
            for ln in lines[:12]:
                print(f"[docker] netstat: {ln.strip()}", flush=True)
            if not lines:
                print("[docker] netstat: no LISTENING line for :8123 (parse may differ on this OS)", flush=True)
        else:
            r = subprocess.run(
                ["ss", "-lntp"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for ln in (r.stdout or "").splitlines():
                if ":8123" in ln:
                    print(f"[docker] ss: {ln.strip()}", flush=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[docker] could not run netstat/ss: {e}", flush=True)


def docker_inspect_status(docker: Path) -> Optional[str]:
    r = subprocess.run(
        [str(docker), "inspect", "-f", "{{.State.Status}}", CH_CONTAINER],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        return None
    return (r.stdout or "").strip().lower() or None


def docker_start_container(docker: Path) -> bool:
    r = subprocess.run(
        [str(docker), "start", CH_CONTAINER],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        print(f"[docker] docker start failed (exit {r.returncode}): {err[:1500]}", flush=True)
        return False
    print("[docker] started existing container", flush=True)
    return True


def docker_run_new_container(docker: Path) -> bool:
    r = subprocess.run(
        [
            str(docker),
            "run",
            "-d",
            "--name",
            CH_CONTAINER,
            "-p",
            "8123:8123",
            "-p",
            "9000:9000",
            "--ulimit",
            "nofile=262144:262144",
            CH_IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        print(f"[docker] docker run failed (exit {r.returncode}): {err[:2000]}", flush=True)
        log_port_8123_diagnostics()
        return False
    print("[docker] created new container", flush=True)
    return True


def wait_for_clickhouse_ping(
    host: str,
    port: int,
    secure: bool,
    max_wait_sec: float = 60.0,
    interval_sec: float = 2.0,
) -> bool:
    scheme = "https" if secure else "http"
    url = f"{scheme}://{host}:{port}/ping"
    deadline = time.monotonic() + max_wait_sec
    sys.stdout.write("[clickhouse] waiting for /ping (max {:.0f}s)".format(max_wait_sec))
    sys.stdout.flush()
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
                if body == "Ok." or "Ok" in body:
                    sys.stdout.write(" ok\n")
                    sys.stdout.flush()
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(interval_sec)
    sys.stdout.write(" timeout\n")
    sys.stdout.flush()
    return False


def http_ping_clickhouse(host: str, port: int, secure: bool) -> bool:
    scheme = "https" if secure else "http"
    url = f"{scheme}://{host}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            body = r.read().decode("utf-8", errors="ignore").strip()
            return body == "Ok." or "Ok" in body
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def try_clickhouse_client(
    host: str,
    user: str,
    password: str,
    secure: bool,
) -> Any:
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


def probe_clickhouse_client(client: Any) -> bool:
    try:
        client.query("SELECT 1")
        return True
    except Exception:
        return False


def start_clickhouse_docker() -> bool:
    docker = find_docker_exe()
    if not docker:
        print("[clickhouse] docker.exe not found; cannot auto-start container", flush=True)
        return False

    status = docker_inspect_status(docker)
    if status == "running":
        print("[docker] container already running", flush=True)
        return wait_for_clickhouse_ping(LOCAL_CH_HOST, LOCAL_CH_PORT, False, max_wait_sec=60.0)
    if status in ("exited", "created", "paused", "restarting", "dead"):
        if not docker_start_container(docker):
            return False
        return wait_for_clickhouse_ping(LOCAL_CH_HOST, LOCAL_CH_PORT, False, max_wait_sec=60.0)

    if status is not None:
        print(f"[docker] container in state '{status}', attempting start", flush=True)
        if docker_start_container(docker):
            return wait_for_clickhouse_ping(LOCAL_CH_HOST, LOCAL_CH_PORT, False, max_wait_sec=60.0)
        print("[docker] start failed; will try docker run (may fail if name exists)", flush=True)

    if not docker_run_new_container(docker):
        return False
    return wait_for_clickhouse_ping(LOCAL_CH_HOST, LOCAL_CH_PORT, False, max_wait_sec=60.0)


def ensure_clickhouse() -> tuple[Optional[Any], bool]:
    port = 8443 if config.ch_secure else 8123
    if not http_ping_clickhouse(config.ch_host, port, config.ch_secure):
        print(f"[clickhouse] ping failed for {config.ch_host}:{port}", flush=True)
    try:
        client = try_clickhouse_client(
            config.ch_host, config.ch_user, config.ch_password, config.ch_secure
        )
        if probe_clickhouse_client(client):
            print("[clickhouse] connected using config.py credentials", flush=True)
            return client, True
    except Exception as e:
        print(f"[clickhouse] config client failed: {e}", flush=True)

    if start_clickhouse_docker():
        try:
            client = try_clickhouse_client("127.0.0.1", "default", "", False)
            if probe_clickhouse_client(client):
                print("[clickhouse] connected to local Docker instance", flush=True)
                return client, True
        except Exception as e:
            print(f"[clickhouse] local client failed: {e}", flush=True)

    print("[clickhouse] unavailable; using in-memory fallback retrieval", flush=True)
    return None, False


def init_rag_table(client: Any, force_recreate: bool = True) -> None:
    if force_recreate:
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
    print("[clickhouse] table default.rag_chunks ready", flush=True)


def insert_chunks_clickhouse(client: Any, chunks: list[dict]) -> None:
    if not chunks:
        return
    rows = [
        [c["id"], c["source"], c["page"], c["chunk"], c["embedding"], c["chunk_hash"], c["char_count"]]
        for c in chunks
    ]
    client.insert(
        "default.rag_chunks",
        rows,
        column_names=["id", "source", "page", "chunk", "embedding", "chunk_hash", "char_count"],
    )
    print(f"[clickhouse] inserted {len(chunks)} rows", flush=True)


def search_clickhouse(client: Any, embedding: list[float], top_k: int) -> list[tuple]:
    q = """
        SELECT chunk, source, page, cosineDistance(embedding, %(emb)s) AS distance
        FROM default.rag_chunks
        ORDER BY distance ASC
        LIMIT %(top_k)s
    """
    result = client.query(q, parameters={"emb": embedding, "top_k": top_k})
    return result.result_rows


def safe_embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    if not texts:
        return []
    try:
        return embedder.generate(texts)
    except Exception as e:
        print(f"[embedder] batch failed: {e}", flush=True)
        return None


def safe_embed_query(text: str) -> Optional[list[float]]:
    try:
        return list(embedder.generate_cached(text))
    except Exception as e:
        print(f"[embedder] query embed failed: {e}", flush=True)
        return None


def load_instructions_to_clickhouse(client: Any) -> tuple[int, bool]:
    init_rag_table(client, force_recreate=True)
    chunk_id = 0
    all_chunks: list[dict] = []
    for file_path in sorted(INSTRUCTIONS_DIR.glob("*.txt")):
        source_name = file_path.name
        part = doc_processor.process_document(str(file_path), source_name, chunk_id)
        if not part:
            continue
        all_chunks.extend(part)
        chunk_id += len(part)

    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i : i + EMBED_BATCH]
        texts = [c["chunk"] for c in batch]
        embs = safe_embed_texts(texts)
        if embs is None:
            print("[embedder] disabling ClickHouse path for this run (embedding unavailable)", flush=True)
            return 0, False
        for c, e in zip(batch, embs):
            c["embedding"] = e
        insert_chunks_clickhouse(client, batch)
        print(
            f"[embedder] embedded+inserted {min(i + EMBED_BATCH, len(all_chunks))}/{len(all_chunks)}",
            flush=True,
        )
    return len(all_chunks), True


def build_token_index() -> list[dict]:
    files = sorted(INSTRUCTIONS_DIR.glob("*.txt"))
    items: list[dict] = []
    for file_path in files:
        text = read_text(file_path)
        for i, para in enumerate(text.split("\n\n")):
            para = para.strip()
            if len(para) < 80:
                continue
            items.append(
                {
                    "source": file_path.name,
                    "page": i + 1,
                    "chunk": para[: config.max_text_length],
                    "tokens": set(re.findall(r"[\w\d]{3,}", para.lower())),
                }
            )
    print(f"[clickhouse] token overlap index entries: {len(items)}", flush=True)
    return items


def retrieve_token(index: list[dict], question: str, top_k: int) -> list[tuple[str, str, int, float]]:
    q_tokens = set(re.findall(r"[\w\d]{3,}", question.lower()))
    scored: list[tuple[float, dict]] = []
    for item in index:
        if not q_tokens:
            score = 0.0
        else:
            score = len(q_tokens & item["tokens"]) / max(len(q_tokens), 1)
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[tuple[str, str, int, float]] = []
    for score, item in scored[:top_k]:
        out.append((item["chunk"], item["source"], item["page"], 1.0 - score))
    return out


def format_context(rows: list[tuple[str, str, int, float]]) -> str:
    blocks = []
    for chunk, source, page, dist in rows:
        blocks.append(f"[{source} p.{page}]\n{chunk[:2500]}")
    return "\n\n---\n\n".join(blocks)


def normalize_llm_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return NOT_FOUND
    first = t.split("\n", 1)[0].strip()
    if first.lower().startswith(NOT_FOUND.lower()):
        return NOT_FOUND
    return t


def generate_llm_answer(question: str, context: str) -> tuple[str, Optional[str]]:
    """
    Returns (answer_text, error). answer_text is ONLY model output for JSON field `answer`.
    """
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    try:
        resp = get_ollama_client().chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            options={
                "num_gpu": config.ollama_num_gpu,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 1400,
            },
        )
    except Exception as e:
        print(f"[llm] chat failed: {e}", flush=True)
        return NOT_FOUND, str(e)
    text = (resp.get("message") or {}).get("content") or ""
    text = normalize_llm_answer(text.strip())
    return text, None


def build_retriever(
    ch_client: Optional[Any],
    token_index: list[dict],
) -> Callable[[str], tuple[list[tuple[str, str, int, float]], list[str], str, Optional[float]]]:
    def retrieve(question: str) -> tuple[list[tuple[str, str, int, float]], list[str], str, Optional[float]]:
        if ch_client is not None:
            q_emb = safe_embed_query(question)
            if q_emb is None:
                rows = retrieve_token(token_index, question, TOP_K)
                return rows, sorted({r[1] for r in rows}), "token", None
            rows = search_clickhouse(ch_client, q_emb, TOP_K)
            best_dist: Optional[float] = float(rows[0][3]) if rows else None
            if rows and best_dist is not None and best_dist <= config.similarity_threshold:
                return rows, sorted({r[1] for r in rows}), "vector", best_dist
            rows = retrieve_token(token_index, question, TOP_K)
            return rows, sorted({r[1] for r in rows}), "token", best_dist
        rows = retrieve_token(token_index, question, TOP_K)
        return rows, sorted({r[1] for r in rows}), "token", None

    return retrieve


def main() -> None:
    questions = load_questions()
    nq = len(questions)
    print(f"[setup] questions: {nq}", flush=True)

    ch_client, ch_ok = ensure_clickhouse()
    token_index = build_token_index()
    if ch_ok and ch_client is not None:
        n, emb_ok = load_instructions_to_clickhouse(ch_client)
        if not emb_ok:
            ch_client = None
        else:
            print(f"[clickhouse] indexed chunks: {n}", flush=True)

    retrieve = build_retriever(ch_client, token_index)

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        rows, sources, kind, best_dist = retrieve(q)
        preview = (format_context(rows)[:400] + "…") if rows else ""

        if not rows:
            answer = NOT_FOUND
            llm_err = None
            print(f"[llm] {i}/{nq} no retrieval context; answer=NOT_FOUND (no model call)", flush=True)
        elif kind == "token":
            top_sim = 1.0 - float(rows[0][3])
            if top_sim < 0.02:
                answer = NOT_FOUND
                llm_err = None
                print(f"[llm] {i}/{nq} token score too low; answer=NOT_FOUND (no model call)", flush=True)
            else:
                ctx = format_context(rows)
                print(f"[llm] {i}/{nq} synthesize (kind={kind}) ctx_preview={preview!r}", flush=True)
                answer, llm_err = generate_llm_answer(q, ctx)
        else:
            ctx = format_context(rows)
            print(
                f"[llm] {i}/{nq} synthesize (kind={kind}, best_distance={best_dist}) "
                f"ctx_preview={preview!r}",
                flush=True,
            )
            answer, llm_err = generate_llm_answer(q, ctx)

        meta: dict[str, Any] = {
            "sources": sources,
            "retrieval_kind": kind,
            "best_vector_distance": best_dist,
        }
        if llm_err:
            meta["llm_error"] = llm_err

        results.append(
            {
                "id": i,
                "question": q,
                "answer": answer,
                "_meta": meta,
            }
        )
        print(f"[llm] {i}/{nq} done", flush=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    errors = [r for r in results if str(r.get("answer", "")).startswith("ERROR:")]
    rel_out = OUTPUT_FILE.relative_to(ROOT).as_posix()
    print(f"[batch] Processed {len(results)}/{nq} questions. Results saved to {rel_out}", flush=True)
    if errors:
        print(f"[llm] ERROR-style answers: {len(errors)}", flush=True)


if __name__ == "__main__":
    main()

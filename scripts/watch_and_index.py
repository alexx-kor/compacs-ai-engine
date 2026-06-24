#!/usr/bin/env python3
"""Watch a folder and POST changed documents to RAG gateway (background /load)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATEWAY = os.getenv("RAG_GATEWAY_URL", "http://127.0.0.1:3090").rstrip("/")
DEFAULT_WATCH_DIR = ROOT / "data" / "incoming"
DEFAULT_COLLECTION = os.getenv("RAG_WATCH_COLLECTION", "incoming")
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".rst"}
STATE_FILE = ".watcher_state.json"

log = logging.getLogger("watch_and_index")


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _load_state(state_path: Path) -> dict[str, dict[str, Any]]:
    if not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state_path: Path, state: dict[str, dict[str, Any]]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _wait_stable(path: Path, *, settle_sec: float, timeout_sec: float) -> bool:
    """Return True when file size/mtime stop changing."""
    deadline = time.time() + timeout_sec
    last: tuple[int, int] | None = None
    stable_since: float | None = None
    while time.time() < deadline:
        if not path.is_file():
            return False
        stat = path.stat()
        current = (stat.st_mtime_ns, stat.st_size)
        if current == last:
            if stable_since is None:
                stable_since = time.time()
            if time.time() - stable_since >= settle_sec:
                return True
        else:
            last = current
            stable_since = None
        time.sleep(0.25)
    return False


def _ensure_collection(client: httpx.Client, collection_id: str, name: str) -> None:
    response = client.post(
        "/v1/collections",
        json={"id": collection_id, "name": name, "description": "auto-indexed by watch_and_index.py"},
    )
    if response.status_code in (200, 400):
        return
    response.raise_for_status()


def _poll_job(client: httpx.Client, job_id: str, *, timeout_sec: float) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = client.get(f"/load/{job_id}")
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", ""))
        if status in {"completed", "failed"}:
            return payload
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_sec}s")


def _upload_file(
    client: httpx.Client,
    path: Path,
    collection_id: str,
    *,
    job_timeout_sec: float,
) -> dict[str, Any]:
    mime = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"
    with path.open("rb") as handle:
        response = client.post(
            "/load",
            params={"collection_id": collection_id, "background": "true"},
            files={"file": (path.name, handle, mime)},
        )
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        return payload
    job = _poll_job(client, str(job_id), timeout_sec=job_timeout_sec)
    if job.get("status") != "completed":
        raise RuntimeError(f"indexing failed for {path.name}: {job.get('error') or job}")
    return job


def _iter_candidates(watch_dir: Path) -> list[Path]:
    if not watch_dir.is_dir():
        return []
    files = [
        path
        for path in watch_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and path.name != STATE_FILE
    ]
    return sorted(files)


def _scan_once(
    client: httpx.Client,
    watch_dir: Path,
    collection_id: str,
    state_path: Path,
    *,
    settle_sec: float,
    stable_timeout_sec: float,
    job_timeout_sec: float,
) -> int:
    state = _load_state(state_path)
    uploaded = 0
    for path in _iter_candidates(watch_dir):
        rel = str(path.relative_to(watch_dir)).replace("\\", "/")
        signature = _file_signature(path)
        if state.get(rel) == signature:
            continue
        if not _wait_stable(path, settle_sec=settle_sec, timeout_sec=stable_timeout_sec):
            log.warning("skip unstable file: %s", rel)
            continue
        signature = _file_signature(path)
        log.info("indexing %s -> collection=%s", rel, collection_id)
        _upload_file(client, path, collection_id, job_timeout_sec=job_timeout_sec)
        state[rel] = signature
        _save_state(state_path, state)
        uploaded += 1
        log.info("done: %s", rel)
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch folder and index changed files via RAG gateway")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="Gateway base URL")
    parser.add_argument("--watch-dir", type=Path, default=DEFAULT_WATCH_DIR, help="Folder to watch")
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION, help="Target collection id")
    parser.add_argument("--collection-name", default="Incoming", help="Collection display name")
    parser.add_argument("--poll-sec", type=float, default=5.0, help="Polling interval in watch mode")
    parser.add_argument("--settle-sec", type=float, default=1.0, help="Wait for file writes to settle")
    parser.add_argument("--stable-timeout-sec", type=float, default=30.0, help="Max wait for stable file")
    parser.add_argument("--job-timeout-sec", type=float, default=300.0, help="Max wait for background job")
    parser.add_argument("--once", action="store_true", help="Scan once and exit (cron-friendly)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    watch_dir = args.watch_dir.resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    state_path = watch_dir / STATE_FILE

    with httpx.Client(base_url=args.gateway.rstrip("/"), timeout=args.job_timeout_sec + 30) as client:
        health = client.get("/health")
        health.raise_for_status()
        log.info("gateway ok: %s", args.gateway)
        _ensure_collection(client, args.collection_id, args.collection_name)

        if args.once:
            count = _scan_once(
                client,
                watch_dir,
                args.collection_id,
                state_path,
                settle_sec=args.settle_sec,
                stable_timeout_sec=args.stable_timeout_sec,
                job_timeout_sec=args.job_timeout_sec,
            )
            log.info("scan complete, uploaded=%s", count)
            return 0

        log.info("watching %s (collection=%s, poll=%ss)", watch_dir, args.collection_id, args.poll_sec)
        try:
            while True:
                _scan_once(
                    client,
                    watch_dir,
                    args.collection_id,
                    state_path,
                    settle_sec=args.settle_sec,
                    stable_timeout_sec=args.stable_timeout_sec,
                    job_timeout_sec=args.job_timeout_sec,
                )
                time.sleep(args.poll_sec)
        except KeyboardInterrupt:
            log.info("stopped")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

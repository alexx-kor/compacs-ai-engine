#!/usr/bin/env python3
"""Manual verification of all RAG v2 HTTP endpoints (gateway :3080 + engine :8080)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATEWAY = os.getenv("RAG_GATEWAY_URL", "http://127.0.0.1:3080").rstrip("/")
DEFAULT_ENGINE = os.getenv("RAG_ENGINE_URL", "http://127.0.0.1:8080").rstrip("/")
TEST_COLLECTION = "api-manual-check"
TEST_FILE = ROOT / "data" / "demo_upload" / "operator_note.txt"


@dataclass
class CheckResult:
    scope: str
    method: str
    path: str
    status: int | str
    ok: bool
    note: str = ""


@dataclass
class Runner:
    gateway: str
    engine: str
    api_key: str
    pro_key: str
    skip_slow: bool
    gateway_only: bool = False
    engine_direct: bool = True
    results: list[CheckResult] = field(default_factory=list)
    _source_id: str | None = None
    _job_id: str | None = None

    def _probe_engine(self) -> bool:
        try:
            with self._client(self.engine, 3.0) as client:
                return client.get("/health").status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _engine_base(self) -> str:
        return self.engine if self.engine_direct else self.gateway

    def record(
        self,
        scope: str,
        method: str,
        path: str,
        status: int | str,
        ok: bool,
        note: str = "",
    ) -> None:
        self.results.append(CheckResult(scope, method, path, status, ok, note))

    def _client(self, base: str, timeout: float = 120.0) -> httpx.Client:
        return httpx.Client(base_url=base, timeout=timeout)

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def check_health(self) -> None:
        with self._client(self.gateway, 15.0) as client:
            r = client.get("/health")
            body = r.json() if r.status_code == 200 else {}
            ok = r.status_code == 200 and body.get("status") == "healthy"
            self.record("gateway", "GET", "/health", r.status_code, ok, str(body)[:120])
            if self.engine_direct:
                with self._client(self.engine, 15.0) as eng:
                    r2 = eng.get("/health")
                    ok2 = r2.status_code == 200 and r2.json().get("status") == "healthy"
                    self.record("engine", "GET", "/health", r2.status_code, ok2, str(r2.json())[:120])
            else:
                engine = body.get("engine", {})
                ok2 = engine.get("status") == "healthy"
                self.record(
                    "engine",
                    "GET",
                    "/health (via gateway)",
                    r.status_code,
                    ok2,
                    str(engine)[:120],
                )

    def check_ui_pages(self) -> None:
        pages = [
            ("/", "text/html"),
            ("/sources", "text/html"),
            ("/metrics", "text/html"),
        ]
        with self._client(self.gateway, 30.0) as client:
            for path, ctype in pages:
                r = client.get(path)
                ok = r.status_code == 200 and ctype in r.headers.get("content-type", "")
                self.record("gateway", "GET", path, r.status_code, ok, r.headers.get("content-type", ""))

    def check_unknown_get_404(self) -> None:
        with self._client(self.gateway, 10.0) as client:
            r = client.get("/definitely-not-a-route")
            self.record("gateway", "GET", "/{unknown}", r.status_code, r.status_code == 404)

    def check_metrics_export(self) -> None:
        with self._client(self.gateway, 60.0) as client:
            r = client.get("/metrics", params={"format": "json"})
            body = r.json() if r.status_code == 200 else {}
            ok = r.status_code == 200 and "storage" in body and "quality" in body
            self.record(
                "gateway",
                "GET",
                "/metrics?format=json",
                r.status_code,
                ok,
                f"chunks={body.get('storage', {}).get('chunk_count', '?')}",
            )

            r = client.get("/export")
            ok = r.status_code == 200 and (
                "ndjson" in r.headers.get("content-type", "")
                or "octet-stream" in r.headers.get("content-type", "")
                or len(r.content) > 100
            )
            self.record(
                "gateway",
                "GET",
                "/export",
                r.status_code,
                ok,
                f"bytes={len(r.content)}",
            )

    def check_sources(self) -> None:
        with self._client(self.gateway, 30.0) as client:
            r = client.get("/sources", params={"format": "json"})
            ok = r.status_code == 200 and "sources" in r.json()
            data = r.json() if ok else {}
            self.record("gateway", "GET", "/sources?format=json", r.status_code, ok, f"count={data.get('count', 0)}")
            if data.get("sources"):
                self._source_id = data["sources"][0]["id"]
                sid = self._source_id
                r = client.get(f"/sources/{sid}/download")
                self.record(
                    "gateway",
                    "GET",
                    f"/sources/{{id}}/download",
                    r.status_code,
                    r.status_code == 200,
                    f"bytes={len(r.content)}",
                )

    def check_collections_crud(self) -> None:
        with self._client(self.gateway, 180.0) as client:
            r = client.post(
                "/v1/collections",
                json={
                    "id": TEST_COLLECTION,
                    "name": "API manual check",
                    "description": "temporary folder for endpoint verification",
                },
            )
            ok = r.status_code in (200, 400)
            self.record("gateway", "POST", "/v1/collections", r.status_code, ok, r.text[:100])

            r = client.get("/v1/collections")
            ok = r.status_code == 200 and "collections" in r.json()
            self.record("gateway", "GET", "/v1/collections", r.status_code, ok)

            r = client.get(f"/v1/collections/{TEST_COLLECTION}")
            ok = r.status_code == 200
            self.record("gateway", "GET", f"/v1/collections/{TEST_COLLECTION}", r.status_code, ok)

            r = client.get(f"/v1/collections/{TEST_COLLECTION}/documents")
            ok = r.status_code == 200
            self.record(
                "gateway",
                "GET",
                f"/v1/collections/{TEST_COLLECTION}/documents",
                r.status_code,
                ok,
            )

            r = client.get("/v1/collections/selection")
            ok = r.status_code == 200 and "collection_ids" in r.json()
            self.record("gateway", "GET", "/v1/collections/selection", r.status_code, ok)

            r = client.put("/v1/collections/selection", json={"collection_ids": [TEST_COLLECTION]})
            ok = r.status_code == 200
            self.record("gateway", "PUT", "/v1/collections/selection", r.status_code, ok)

    def check_load_and_jobs(self) -> None:
        if not TEST_FILE.is_file():
            self.record("gateway", "POST", "/load", "skip", False, f"missing {TEST_FILE}")
            return

        with self._client(self.gateway, 180.0) as client:
            with TEST_FILE.open("rb") as handle:
                r = client.post(
                    "/load",
                    params={"collection_id": TEST_COLLECTION},
                    files={"file": (TEST_FILE.name, handle, "text/plain")},
                )
            ok = r.status_code == 200
            self.record("gateway", "POST", "/load", r.status_code, ok, r.text[:120])

            with TEST_FILE.open("rb") as handle:
                r = client.post(
                    "/load",
                    params={"collection_id": TEST_COLLECTION, "background": "true"},
                    files={"file": ("bg_" + TEST_FILE.name, handle, "text/plain")},
                )
            ok = r.status_code in (200, 202)
            payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            self._job_id = payload.get("job_id")
            self.record(
                "gateway",
                "POST",
                "/load?background=true",
                r.status_code,
                ok and bool(self._job_id),
                f"job_id={self._job_id}",
            )

            if self._job_id:
                for _ in range(30):
                    r = client.get(f"/load/{self._job_id}")
                    if r.status_code != 200:
                        break
                    status = r.json().get("status", "")
                    if status in {"completed", "failed"}:
                        break
                    time.sleep(1)
                ok = r.status_code == 200 and r.json().get("status") in {"completed", "running", "pending"}
                self.record(
                    "gateway",
                    "GET",
                    "/load/{job_id}",
                    r.status_code,
                    ok,
                    r.json().get("status", "") if r.status_code == 200 else r.text[:80],
                )

    def _read_stream(self, response: httpx.Response) -> tuple[bool, str]:
        """Read SSE stream; tolerate early connection close after some data."""
        ctype = response.headers.get("content-type", "")
        if "event-stream" not in ctype:
            return False, f"content-type={ctype}"
        total = 0
        try:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > 0 and self.skip_slow:
                    break
        except httpx.RemoteProtocolError:
            if total > 0:
                return True, f"bytes={total} (truncated)"
            return False, "connection closed before data"
        return total > 0, f"bytes={total}"

    def check_query_and_chat(self) -> None:
        question = "Что делает кнопка «Новый документ»?"
        with self._client(self.gateway, 180.0) as client:
            r = client.post(
                "/v1/query",
                json={"question": question, "limit": 3, "collection_ids": [TEST_COLLECTION]},
            )
            body = r.json() if r.status_code == 200 else {}
            ok = r.status_code == 200 and bool(body.get("answer"))
            self.record(
                "gateway",
                "POST",
                "/v1/query",
                r.status_code,
                ok,
                (body.get("answer") or "")[:80],
            )

            if not self.skip_slow:
                with client.stream(
                    "POST",
                    "/v1/query",
                    json={"question": question, "stream": True, "collection_ids": [TEST_COLLECTION]},
                ) as r:
                    ok, note = self._read_stream(r)
                    ok = r.status_code == 200 and ok
                    self.record(
                        "gateway",
                        "POST",
                        "/v1/query (stream)",
                        r.status_code,
                        ok,
                        note,
                    )

            r = client.post(
                "/api/chat",
                json={"question": question, "collection_ids": [TEST_COLLECTION]},
            )
            ok = r.status_code == 200 and "answer" in r.json()
            self.record("gateway", "POST", "/api/chat", r.status_code, ok)

            if not self.skip_slow:
                with client.stream(
                    "POST",
                    "/api/chat",
                    json={"question": question, "stream": True, "collection_ids": [TEST_COLLECTION]},
                ) as r:
                    ok, note = self._read_stream(r)
                    ok = r.status_code == 200 and ok
                    self.record(
                        "gateway",
                        "POST",
                        "/api/chat (stream)",
                        r.status_code,
                        ok,
                        note,
                    )

    def check_openai_compat(self) -> None:
        headers = self._auth_headers()
        base = self._engine_base()
        with self._client(base, 180.0) as client:
            r = client.get("/v1/models", headers=headers)
            ok = r.status_code in (200, 401)
            if self.api_key:
                ok = r.status_code == 200 and r.json().get("object") == "list"
            self.record("engine", "GET", "/v1/models", r.status_code, ok)

            r = client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "compacs-rag",
                    "messages": [{"role": "user", "content": "Кратко: что такое NvF?"}],
                    "stream": False,
                },
            )
            ok = r.status_code in (200, 401)
            if self.api_key and r.status_code == 200:
                ok = bool(r.json().get("choices"))
            self.record("engine", "POST", "/v1/chat/completions", r.status_code, ok, r.text[:80])

            if not self.skip_slow and (self.api_key or r.status_code != 401):
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": "compacs-rag",
                        "messages": [{"role": "user", "content": "Кратко: что такое NvF?"}],
                        "stream": True,
                    },
                ) as sr:
                    ok, note = self._read_stream(sr)
                    ok = sr.status_code in (200, 401) and (sr.status_code == 401 or ok)
                    self.record(
                        "engine",
                        "POST",
                        "/v1/chat/completions (stream)",
                        sr.status_code,
                        ok,
                        note,
                    )

    def check_engine_direct(self) -> None:
        """Routes on engine :8080; proxied via gateway /v1/* when engine is internal."""
        base = self._engine_base()
        with self._client(base, 60.0) as client:
            r = client.get("/v1/metrics")
            ok = r.status_code == 200 and "quality" in r.json()
            self.record("engine", "GET", "/v1/metrics", r.status_code, ok)

            r = client.get("/v1/export", params={"format": "jsonl"})
            ok = r.status_code == 200 and len(r.content) > 100
            self.record("engine", "GET", "/v1/export", r.status_code, ok, f"bytes={len(r.content)}")

            sources_path = "/sources?format=json" if not self.engine_direct else "/sources"
            r = client.get("/sources", params={"format": "json"} if not self.engine_direct else None)
            data = r.json() if r.status_code == 200 else {}
            ok = r.status_code == 200 and "sources" in data
            self.record("engine", "GET", sources_path, r.status_code, ok)

            r = client.get("/v1/jobs/nonexistent-job-id")
            self.record("engine", "GET", "/v1/jobs/{id} (404)", r.status_code, r.status_code == 404)

    def check_license_upgrade(self) -> None:
        if self.engine_direct:
            with self._client(self.engine, 15.0) as client:
                r = client.get("/license")
                ok = r.status_code == 200 and "tier" in r.json()
                self.record("engine", "GET", "/license", r.status_code, ok, r.json().get("tier", ""))
        else:
            self.record(
                "engine",
                "GET",
                "/license",
                "skip",
                True,
                "engine internal only (use bare-metal or publish :8080)",
            )

        with self._client(self.gateway, 15.0) as client:
            r = client.post("/upgrade", json={"license_key": "invalid-key"})
            ok = r.status_code in (403, 500)
            self.record("gateway", "POST", "/upgrade (bad key)", r.status_code, ok, r.text[:60])

            if self.pro_key:
                r = client.post("/upgrade", json={"license_key": self.pro_key})
                ok = r.status_code == 200 and r.json().get("pro_enabled") is True
                self.record("gateway", "POST", "/upgrade (valid key)", r.status_code, ok)
            else:
                self.record(
                    "gateway",
                    "POST",
                    "/upgrade (valid key)",
                    "skip",
                    True,
                    "COMPACS_PRO_KEY not set",
                )

    def cleanup(self) -> None:
        with self._client(self.gateway, 60.0) as client:
            r = client.get(f"/v1/collections/{TEST_COLLECTION}/documents")
            if r.status_code == 200:
                for doc in r.json().get("documents", []):
                    name = doc.get("filename", "")
                    if name:
                        dr = client.delete(f"/v1/collections/{TEST_COLLECTION}/documents/{name}")
                        self.record(
                            "gateway",
                            "DELETE",
                            f"/v1/collections/{TEST_COLLECTION}/documents/{{name}}",
                            dr.status_code,
                            dr.status_code == 200,
                            name,
                        )

            dr = client.delete(f"/v1/collections/{TEST_COLLECTION}")
            self.record(
                "gateway",
                "DELETE",
                f"/v1/collections/{TEST_COLLECTION}",
                dr.status_code,
                dr.status_code in (200, 404),
            )

            client.put("/v1/collections/selection", json={"collection_ids": []})

    def run(self) -> int:
        print(f"Gateway: {self.gateway}")
        print(f"Engine:  {self.engine}")
        self.engine_direct = not self.gateway_only and self._probe_engine()
        if not self.engine_direct:
            print("Note: engine not reachable on host; API checks go through gateway.")
        print()

        self.check_health()
        self.check_ui_pages()
        self.check_unknown_get_404()
        self.check_metrics_export()
        self.check_sources()
        self.check_collections_crud()
        self.check_load_and_jobs()
        self.check_query_and_chat()
        self.check_openai_compat()
        self.check_engine_direct()
        self.check_license_upgrade()
        self.cleanup()

        passed = sum(1 for item in self.results if item.ok)
        failed = [item for item in self.results if not item.ok]

        print(f"{'SCOPE':<8} {'METHOD':<6} {'PATH':<45} {'STATUS':<8} {'OK':<4} NOTE")
        print("-" * 110)
        for item in self.results:
            mark = "OK" if item.ok else "FAIL"
            print(
                f"{item.scope:<8} {item.method:<6} {item.path:<45} {str(item.status):<8} {mark:<4} {item.note[:50]}"
            )

        print()
        print(f"Total: {len(self.results)}  Passed: {passed}  Failed: {len(failed)}")
        if failed:
            print("\nFailed checks:")
            for item in failed:
                print(f"  - [{item.scope}] {item.method} {item.path} -> {item.status} ({item.note})")
        return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all RAG v2 API endpoints manually")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--api-key", default=os.getenv("COMPACS_API_KEY", ""))
    parser.add_argument("--pro-key", default=os.getenv("COMPACS_PRO_KEY", ""))
    parser.add_argument(
        "--gateway-only",
        action="store_true",
        default=os.getenv("RAG_SKIP_ENGINE", "").lower() in ("1", "true", "yes"),
        help="Skip direct engine :8080 checks (Docker host-ollama: engine is internal)",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip streaming endpoints (SSE) to save time",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    runner = Runner(
        gateway=args.gateway.rstrip("/"),
        engine=args.engine.rstrip("/"),
        api_key=args.api_key.strip(),
        pro_key=args.pro_key.strip(),
        skip_slow=args.skip_slow,
        gateway_only=args.gateway_only,
    )
    code = runner.run()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "scope": item.scope,
                        "method": item.method,
                        "path": item.path,
                        "status": item.status,
                        "ok": item.ok,
                        "note": item.note,
                    }
                    for item in runner.results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Simple preprocessing HTTP server for baseline/experiment profiles."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from core.preprocessing import preprocess_text


def run_preprocess_server(host: str, port: int, profile: str) -> None:
    """Run HTTP service exposing ``POST /clean``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/clean":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                payload: Any = json.loads(body)
            except json.JSONDecodeError:
                payload = {}

            text = payload.get("text", "") if isinstance(payload, dict) else ""
            if not isinstance(text, str):
                text = str(text)

            cleaned = preprocess_text(text, profile)
            response = json.dumps({"text": cleaned, "profile": profile}, ensure_ascii=False).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                payload = json.dumps({"status": "ok", "profile": profile}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    import logging as _logging
    _log = _logging.getLogger(__name__)
    server = ThreadingHTTPServer((host, port), Handler)
    _log.info("[PREPROCESS] Running profile=%s on http://%s:%s", profile, host, port)
    _log.info("[PREPROCESS] Endpoints: GET /health, POST /clean")
    server.serve_forever()


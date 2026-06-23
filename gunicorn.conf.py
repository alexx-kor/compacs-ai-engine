"""Gunicorn configuration for COMPACS RAG gateway (port 3080)."""

from __future__ import annotations

import multiprocessing
import os

bind = os.getenv("GATEWAY_BIND", "0.0.0.0:3080")
worker_class = "uvicorn.workers.UvicornWorker"
workers = int(os.getenv("GATEWAY_WORKERS", max(2, multiprocessing.cpu_count())))
timeout = int(os.getenv("GATEWAY_TIMEOUT", "300"))
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GATEWAY_LOG_LEVEL", "info")

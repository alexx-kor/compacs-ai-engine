"""Gunicorn entrypoint (spec: gunicorn wsgi:app -k uvicorn.workers.UvicornWorker)."""

from api.gateway import app_gateway as app

__all__ = ["app"]

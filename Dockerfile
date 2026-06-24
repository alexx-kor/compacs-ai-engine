# COMPACS RAG v2 — gateway (:3080) + engine (:8080)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md app.py rag_service.py config.py gunicorn.conf.py wsgi.py ./
COPY api ./api
COPY core ./core
COPY router ./router
COPY prompts ./prompts
COPY scripts ./scripts
COPY docs ./docs

RUN pip install --upgrade pip \
    && pip install -e .

RUN mkdir -p data/vectors data/collections data/results data/graph

EXPOSE 3080 8080

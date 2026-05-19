FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY app.py rag_service.py config.py ./
COPY api ./api
COPY core ./core
COPY router ./router
COPY rag_engine ./rag_engine
COPY prompts ./prompts

RUN pip install --upgrade pip && pip install .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "api.stable:app_stable", "--host", "0.0.0.0", "--port", "8080"]

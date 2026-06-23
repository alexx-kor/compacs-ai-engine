# COMPACS RAG Engine v2

Серверный RAG-стенд КОМПАКС® 7: ingestion, hybrid search, Ollama/OpenAI, единая точка входа `:3080`.

Спецификация: [`docs/TECHNICAL_NOTE_V2.md`](docs/TECHNICAL_NOTE_V2.md) · API: [`docs/RAG_V2_API.md`](docs/RAG_V2_API.md) · Контроллер: [`docs/CONTROLLER_PATTERN.md`](docs/CONTROLLER_PATTERN.md)

---

## 1. Развёртывание (Docker)

### Полный стек (Ollama в контейнере)

```bash
cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.yml up -d --build

# Первый запуск — модели Ollama
docker compose -f rag-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f rag-compose.yml exec ollama ollama pull llama3.2:3b
```

### Ollama на хосте (без ~2 GB pull образа ollama/ollama)

```bash
ollama serve
ollama pull llama3.2:3b && ollama pull nomic-embed-text
ollama run llama3.2:3b "ok"

cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.host-ollama.yml up -d --build
```

Локальный индекс `data/vectors/` монтируется в engine автоматически.

| Сервис | URL |
|--------|-----|
| **RAG Gateway (UI + API)** | http://localhost:3080 |
| Engine (internal) | http://rag-engine:8080 (внутри compose) |
| Ollama | http://localhost:11434 |

Индекс UI Extension (локально, до Docker):
```bash
python scripts/ui_extension_pipeline.py extract
python scripts/ui_extension_pipeline.py index
# том rag_data в compose сохраняет data/vectors
```

**Bare metal:**
```bash
pip install -e .
python -m app serve
# или production:
gunicorn wsgi:app -c gunicorn.conf.py
python -m uvicorn api.stable:app_stable --host 127.0.0.1 --port 8080
```

LibreChat (опционально, порт **3081**): `docker compose -f librechat-compose.yml up -d`

### Ollama: таймауты и «холодная» модель

На CPU/GPU с `llama3.2:3b` первый запрос может тратить **~50 с** на загрузку модели в RAM, если Ollama её выгрузил. Большой RAG-промпт (много чанков) + генерация легко превышают **120 с** — gateway обрывает запрос, источники уже есть, ответ пустой.

**Рекомендуемые переменные** (см. `.env.rag.docker.example`):

| Переменная | Значение | Зачем |
|------------|----------|--------|
| `GATEWAY_TIMEOUT` | `300` | Gunicorn + httpx proxy к engine |
| `OLLAMA_KEEP_ALIVE` | `30m` | не выгружать модель между запросами |
| `OLLAMA_CLIENT_TIMEOUT` | `300` | таймаут Python-клиента к Ollama |
| `NUM_CTX` | `8192` | контекстное окно модели |
| `NUM_PREDICT` / `MAX_TOKENS` | `400` / `600` | лимит генерации |
| `RERANK_TOP_K` | `3` | меньше текста в промпте |
| `OLLAMA_CHUNK_CHARS` | `350` | обрезка чанка для Ollama |

**Прогрев после старта:**
```bash
ollama run llama3.2:3b "ok"
# или в Docker:
docker compose -f rag-compose.yml exec ollama ollama run llama3.2:3b "ok"
```

**Диагностика** (подставьте реальный промпт, не плейсхолдер):
```powershell
$body = @{
  model = "llama3.2:3b"
  prompt = "Кратко: что такое HTTP-прокси?"
  stream = $false
  options = @{ num_ctx = 8192; num_predict = 400 }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method Post -Body $body -ContentType "application/json"
```
Смотрите `load_duration` (холодная загрузка) vs `eval_duration` (генерация).

---

## 2. Вопросы (API / UI)

**UI:** http://localhost:3080 — чат с SSE-стримингом.

**API:**
```bash
curl -X POST http://localhost:3080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что делает кнопка «Новый документ»?"}'

# SSE
curl -N -X POST http://localhost:3080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "stream": true}'
```

Scope по папкам: `"collection_ids": ["demo-upload"]`  
Демо-вопросы: [`docs/DEMO_QUESTIONS.md`](docs/DEMO_QUESTIONS.md)

---

## 3. Загрузка документов (`/load`)

```bash
# Создать папку
curl -X POST http://localhost:3080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"id": "ops-manual", "name": "Операторская"}'

# Загрузить файл (синхронно)
curl -X POST "http://localhost:3080/load?collection_id=ops-manual" \
  -F "file=@document.pdf"

# Фоновая индексация (gateway не блокируется)
curl -X POST "http://localhost:3080/load?collection_id=ops-manual&background=true" \
  -F "file=@document.pdf"
# → {"job_id":"...", "status":"pending", "poll_url":"/v1/jobs/..."}

curl "http://localhost:3080/load/{job_id}"

# Источники
curl "http://localhost:3080/sources?format=json"

# Удалить источник
curl -X DELETE "http://localhost:3080/sources/{id}"
```

Скрипт-пример: `python scripts/demo_upload_http.py`

**Десктоп:** `curl -OJ http://localhost:3080/export` → JSONL vector index.

---

## 4. Тесты

```bash
pip install -e ".[dev]"

# Unit + integration
python -m pytest tests/unit tests/integration -q

# Smoke (нужен индекс + Ollama)
python scripts/smoke_rag.py
python scripts/smoke_rag.py --stream

# Golden 28 (evaluation)
python full_evaluation.py --golden baseline/golden_set.json --llm-provider ollama

# Дрейф / PSI
python scripts/monitor_data_drift.py --preset splits
curl "http://localhost:3080/metrics?format=json"
```

**Git-гигиена:** `git rm --cached <path>` для файлов, попавших в индекс до `.gitignore`.

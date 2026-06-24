# Паттерн контроллера: два стрима (данные + инференс)

Как связать контейнеры/сервисы RAG v2 в стиле «контроллер + фоновая индексация + быстрый query».

См. также: [`RAG_V2_API.md`](./RAG_V2_API.md), [`TECHNICAL_NOTE_V2.md`](./TECHNICAL_NOTE_V2.md).

---

## Роль контроллера

**Gateway (`:3080`, `api/gateway.py`)** — единая точка входа для клиента. Он:

- принимает GET/POST с `localhost:3080`
- проксирует ingest и RAG в **engine** (`:8080`)
- не пересобирает векторный индекс на каждый вопрос
- не вызывает Ollama напрямую (это делает engine)

```
Клиент ──GET/POST──> localhost:3080  (gateway = контроллер)
                          │
                          v
                    rag-engine:8080   (retrieval, jobs, LLM chain)
                          │
              ┌───────────┴───────────┐
              v                       v
      data/vectors/chunks.json   Ollama :11434
      (обновляется /load)        (горячая модель)
```

Внутри Docker имена сервисов — `rag-engine`, `rag-gateway`; с хоста всё проверяется через **localhost**.

---

## Паттерн 1 — фоновый стрим (данные)

> Изменение файлов → (cron/webhook) → обновление БД/векторки **один раз**

| Действие | Endpoint | Когда |
|----------|----------|--------|
| Загрузить документ | `POST /load?collection_id=...` | Файл новый/изменился |
| Фоновая индексация | `POST /load?...&background=true` | Не блокировать gateway |
| Статус job | `GET /load/{job_id}` | Poll после 202 |
| Список источников | `GET /sources?format=json` | Аудит |
| Удалить источник | `DELETE /sources/{id}` | Убрать из индекса |
| Экспорт индекса | `GET /export` | Десктоп / бэкап |

**На `POST /v1/query` индекс не пересобирается** — только чтение.

Cron/webhook (вне репозитория): watcher на папку → `POST /load` при изменении файла.

---

## Паттерн 2 — быстрый стрим (инференс)

> Запрос клиента → горячая модель в RAM → ответ

| Действие | Endpoint |
|----------|----------|
| Вопрос (JSON) | `POST /v1/query` |
| Вопрос (UI) | `POST /api/chat` |
| SSE | `"stream": true` на query/chat |
| LibreChat | `POST /v1/chat/completions` на engine `:8080` |

**Латентность (реалистично для `llama3.2:3b` на CPU):**

| Фаза | Обычно |
|------|--------|
| Retrieval (embed + search) | 0.2–2 с |
| Cold load модели | до ~50 с (если выгружена) |
| Prefill большого промпта | десятки секунд |
| Генерация | 10–90+ с |

«30 мс» относится к **retrieval/маршрутизации**, не к полному ответу LLM на CPU.

Обязательные настройки (`.env.rag`):

```env
GATEWAY_TIMEOUT=300
OLLAMA_KEEP_ALIVE=30m
OLLAMA_CLIENT_TIMEOUT=300
NUM_CTX=8192
NUM_PREDICT=400
RERANK_TOP_K=3
OLLAMA_CHUNK_CHARS=350
```

Прогрев после старта Ollama:

```bash
ollama run llama3.2:3b "ok"
```

---

## Развёртывание: два варианта Docker

### A. Полный стек (Ollama в контейнере, ~2 GB pull)

```bash
cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.yml up -d --build
docker compose -f rag-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f rag-compose.yml exec ollama ollama pull llama3.2:3b
```

### B. Ollama на хосте (без pull `ollama/ollama`)

```bash
# Терминал 1 — на хосте
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama run llama3.2:3b "ok"

# Терминал 2 — только gateway + engine
cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.host-ollama.yml up -d --build
```

Том `./data` монтируется в engine — локальный `data/vectors/chunks.json` сразу виден в контейнере.

### C. Bare metal (без Docker)

```bash
python -m uvicorn api.stable:app_stable --host 127.0.0.1 --port 8080
RAG_ENGINE_URL=http://127.0.0.1:8080 python -m uvicorn api.gateway:app_gateway --host 0.0.0.0 --port 3080
```

---

## Ручная проверка связности (чеклист)

Выполнять **с хоста** после старта стека.

### 1. Health

```powershell
Invoke-RestMethod http://127.0.0.1:3080/health
Invoke-RestMethod http://127.0.0.1:3080/health   # includes engine status
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Ожидание: gateway `healthy`, engine `healthy`, в tags есть `llama3.2:3b`.

### 2. Ollama: cold vs warm

```powershell
$body = @{
  model = "llama3.2:3b"
  prompt = "Столица Франции?"
  stream = $false
  options = @{ num_ctx = 8192; num_predict = 100 }
} | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod -Uri http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType "application/json"
# Повторить сразу — load_duration должен упасть
```

### 3. Фоновый стрим — загрузка

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:3080/v1/collections -Method Post `
  -Body '{"id":"ctrl-test","name":"Controller test"}' -ContentType "application/json"

curl -X POST "http://127.0.0.1:3080/load?collection_id=ctrl-test" `
  -F "file=@data/demo_upload/operator_note.txt"

Invoke-RestMethod "http://127.0.0.1:3080/sources?format=json"
```

### 4. Инференс — ответ не пустой

```powershell
$body = '{"question":"Что делает кнопка «Новый документ»?"}'
$r = Invoke-RestMethod -Uri http://127.0.0.1:3080/v1/query -Method Post `
  -Body $body -ContentType "application/json; charset=utf-8"
$r.answer.Length
$r.sources.Count
```

**Плохо:** `sources > 0` и `answer` пустой → таймаут или cold Ollama.  
**Хорошо:** непустой `answer`, повторный запрос быстрее первого.

### 5. Автопрогон API

```bash
python scripts/manual_api_check.py
```

Ожидание: `Failed: 0`.

---

## Типичные сбои

| Симптом | Причина | Действие |
|---------|---------|----------|
| 5 sources, пустой answer | Gateway timeout 120 с, тяжёлый промпт | `GATEWAY_TIMEOUT=300`, `RERANK_TOP_K=3` |
| `load_duration` ~50 с каждый раз | Модель выгружается | `OLLAMA_KEEP_ALIVE=30m`, прогрев |
| `unexpected EOF` при docker pull | Обрыв сети на 2 GB ollama image | `rag-compose.host-ollama.yml` |
| ClickHouse warning при старте | `STORAGE_BACKEND=auto` | `STORAGE_BACKEND=json` |
| 503 gateway | Engine не запущен | `curl :3080/health` (смотри поле engine) |

---

## Ответственность компонентов

| Компонент | Фоновый стрим | Инференс |
|-----------|---------------|----------|
| Gateway | `POST /load`, proxy | `POST /v1/query`, `/api/chat` |
| Engine | ingest jobs, chunks | `rag_service.ask()` |
| Ollama | embed при `/load` | chat при query |
| `chunks.json` | пишется при load | только читается при query |

Контроллер **связывает** потоки, но **не смешивает** их: query никогда не должен запускать полную переиндексацию.

# RAG v2 — API и архитектура

Единое приложение с **одной точкой входа** (`:3080`), два контура.

## Архитектура

```
Клиент / браузер
       │
       ▼  :3080
┌──────────────────────────────────────┐
│  Gateway — FastAPI + Gunicorn        │
│  /  /load  /sources  /export         │
│  /metrics  /upgrade                  │
└──────────────┬───────────────────────┘
               │ internal HTTP POST/GET
               ▼  :8080 (127.0.0.1)
┌──────────────────────────────────────┐
│  GRA-Hybrid RAG Engine               │
│  parse · Ollama embed/LLM · graph    │
│  hybrid search · vector index        │
└──────────────────────────────────────┘
```

| Контур | Порт | Стек | Роль |
|--------|------|------|------|
| **Gateway** | `3080` | FastAPI, Gunicorn + UvicornWorker | UI, роутинг, async proxy |
| **Engine** | `8080` | FastAPI, uvicorn | Ingestion, RAG, Ollama, индекс |

## Запуск

```bash
# Dev: оба контура одной командой
python -m app serve

# Production gateway
gunicorn -c gunicorn.conf.py api.gateway:app_gateway

# Production engine (только localhost)
python -m app serve-api --host 127.0.0.1 --port 8080
```

Переменные:

| Variable | Default | Описание |
|----------|---------|----------|
| `RAG_ENGINE_URL` | `http://127.0.0.1:8080` | URL engine для gateway |
| `GATEWAY_BIND` | `0.0.0.0:3080` | Bind Gunicorn |
| `LIBRECHAT_PORT` | `3081` | LibreChat UI (Docker Compose) |
| `GATEWAY_WORKERS` | `CPU count` | Workers |
| `COMPACS_PRO_KEY` | — | Ключ для `POST /upgrade` |

---

## Маршруты Gateway (`:3080`)

| Маршрут | Метод | Назначение |
|---------|-------|------------|
| `/` | GET | Чат с AI-агентом (UI) |
| `/load` | POST | Загрузка документов → ingestion → переиндексация |
| `/sources` | GET | Список источников с метаданными |
| `/sources/{id}/download` | GET | Скачать исходный файл |
| `/sources/{id}` | DELETE | Удалить источник + переиндексация |
| `/export` | GET | Выгрузка vector index (JSONL) для десктопа |
| `/metrics` | GET | Мониторинг (storage, collections, usage, **PSI/drift**) |
| `/upgrade` | POST | Переключение на pro (по ключу) |
| `*` | GET | Все прочие GET → **404** |

JSON-варианты UI-страниц: `?format=json` или `Accept: application/json`.

---

## `/sources` — управление источниками

`id` — opaque token (base64url от vector source path), например:
`collections/ui-ext/manual.txt` → `Y29sbGVjdGlvbnMvdWktZXh0L21hbnVhbC50eHQ`

### Список
```http
GET /sources
GET /sources?format=json
```

```json
{
  "count": 2,
  "selected_collection_ids": ["ui-ext"],
  "sources": [
    {
      "id": "...",
      "source": "collections/ui-ext/manual.txt",
      "collection_id": "ui-ext",
      "filename": "manual.txt",
      "uploaded_at": "2026-06-10T12:00:00+00:00",
      "chunk_count": 42,
      "size_bytes": 120000,
      "kind": "collection"
    }
  ]
}
```

### Скачать оригинал
```http
GET /sources/{id}/download
```

### Удалить
```http
DELETE /sources/{id}
```

---

## `/load` — загрузка

```http
POST /load?collection_id=ui-ext
Content-Type: multipart/form-data

file=@document.pdf
```

Проксируется в engine: `POST /v1/collections/{id}/documents`.

---

## `/upgrade` — pro

```http
POST /upgrade
Content-Type: application/json

{ "license_key": "your-pro-key" }
```

Требует `COMPACS_PRO_KEY` на сервере.

---

## Тематические папки (collections)

Дополнительный API engine/gateway: `/v1/collections`

- `POST /v1/collections` — создать папку
- `PUT /v1/collections/selection` — выбрать папки для RAG scope
- `POST /v1/query` — вопрос с `collection_ids`

Подробнее: [COLLECTIONS_API.md](./COLLECTIONS_API.md)

---

## `/export` — десктоп

```http
GET /export
```

Файл `compacs-vectors-{timestamp}.jsonl` — импорт в десктопный клиент (llama.cpp + локальный индекс).

---

## `/metrics` — мониторинг

```http
GET /metrics?format=json
```

PSI / деградация встроены в ответ `quality` (сравнение индекса с golden set, QA splits, eval-файлы, сохранённый offline-отчёт).

---

## Связь с десктопом

Только через **`/export`** — сервер отдаёт vector index, десктоп импортирует локально. Бизнес-логика клиента (роутер, llama.cpp, MCP) — вне этого репозитория.

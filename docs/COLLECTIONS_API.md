# Тематические папки и API коллекций

Управление документами для RAG: создание папок, загрузка, выбор scope поиска, удаление.

## Архитектура

| Контур | Порт | Назначение |
|--------|------|------------|
| **Gateway** | `3080` | UI (`/`, `/load`, `/metrics`, `/export`) + прокси `/v1/*` |
| **Engine** | `8080` | RAG, ingest, embeddings, vector store (localhost) |

Запуск обоих контуров одной командой:

```bash
python -m app serve
# Gateway: http://localhost:3080
# Engine:  http://127.0.0.1:8080 (только internal)
```

Production (Gunicorn + Uvicorn workers):

```bash
gunicorn api.gateway:app_gateway -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:3080 --workers 4 --timeout 300
# Engine отдельно:
python -m app serve-api --host 127.0.0.1 --port 8080
```

---

## UI (gateway :3080)

| URL | Описание |
|-----|----------|
| `/` | Чат с RAG |
| `/load` | Создание папок, выбор scope, загрузка файлов |
| `/metrics` | Storage, коллекции, OpenAI usage |
| `/export` | Скачать vector index (JSONL) для десктопа |

---

## REST API

Базовый URL через gateway: `http://localhost:3080/v1/...`  
Напрямую к engine: `http://127.0.0.1:8080/v1/...`

### Папки (collections)

#### Создать папку
```http
POST /v1/collections
Content-Type: application/json

{
  "name": "Руководство оператора",
  "id": "operator-manual",
  "description": "опционально"
}
```

#### Список папок
```http
GET /v1/collections
```

Ответ:
```json
{
  "selected_collection_ids": ["operator-manual"],
  "collections": [
    {
      "id": "operator-manual",
      "name": "Руководство оператора",
      "document_count": 2,
      "documents": [...]
    }
  ]
}
```

#### Удалить папку
```http
DELETE /v1/collections/{collection_id}
```
Удаляет файлы, чанки в индексе и запись в реестре.

---

### Документы в папке

#### Загрузить файл
```http
POST /v1/collections/{collection_id}/documents
Content-Type: multipart/form-data

file=@manual.txt
```

Поддерживаемые форматы: `.pdf`, `.txt`, `.md`, `.rst`

Через gateway UI:
```http
POST /load?collection_id=operator-manual
Content-Type: multipart/form-data

file=@manual.txt
```

#### Список файлов
```http
GET /v1/collections/{collection_id}/documents
```

#### Удалить файл
```http
DELETE /v1/collections/{collection_id}/documents/{filename}
```

---

### Выбор папок для RAG

RAG ищет только в документах выбранных папок.

#### Установить выбор
```http
PUT /v1/collections/selection
Content-Type: application/json

{ "collection_ids": ["operator-manual", "ui-ext"] }
```

- **`[]` (пустой список)** — поиск по **всем** данным в индексе
- **Непустой список** — только `collections/{id}/...`

#### Получить текущий выбор
```http
GET /v1/collections/selection
```

---

### Вопрос к RAG

```http
POST /v1/query
Content-Type: application/json

{
  "question": "Какие функции выполняет кнопка «Новый документ»?",
  "collection_ids": ["ui-ext"]
}
```

`collection_ids` опционален — если не указан, используется `PUT /v1/collections/selection`.

Через gateway UI:
```http
POST /api/chat
Content-Type: application/json

{ "question": "...", "collection_ids": ["ui-ext"] }
```

---

### Экспорт индекса (десктоп)

```http
GET /v1/export?format=jsonl
```

или через gateway: `GET /export`

Файл: `compacs-vectors-{timestamp}.jsonl` — для импорта в десктопный клиент.

---

### Метрики

```http
GET /v1/metrics
```

```json
{
  "storage": { "backend": "json", "chunk_count": 3470, "sources": [...] },
  "collections": { "count": 2, "selected_ids": ["ui-ext"], "items": [...] },
  "datasets": { "raw_files": 0, "graph_pairs": 0, "golden_files": 1 },
  "openai_usage_today": { ... }
}
```

---

## Хранение на диске

| Путь | Содержимое |
|------|------------|
| `data/collections/registry.json` | реестр папок и selection |
| `data/collections/{id}/files/` | исходные файлы |
| `data/vectors/chunks.json` | vector index |

Source в индексе: `collections/{folder_id}/{filename}`

---

## Типичный сценарий

```bash
# 1. Запуск
python -m app serve

# 2. Создать папку
curl -X POST http://localhost:3080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"UI","id":"ui-ext"}'

# 3. Загрузить документ
curl -X POST "http://localhost:3080/load?collection_id=ui-ext" \
  -F "file=@manual.txt"

# 4. Выбрать папку
curl -X PUT http://localhost:3080/v1/collections/selection \
  -H "Content-Type: application/json" \
  -d '{"collection_ids":["ui-ext"]}'

# 5. Спросить
curl -X POST http://localhost:3080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Что такое логический канал?"}'

# 6. Экспорт для десктопа
curl -OJ http://localhost:3080/export
```

---

## Ограничения v2.0

- Streaming ответов в UI — в следующей итерации
- `/upgrade` (pro) — после выдачи лицензионного ключа
- LibreChat на `:3081` (gateway COMPACS RAG — `:3080`)

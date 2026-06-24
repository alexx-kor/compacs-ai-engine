# Quickstart: серверный стенд → офлайн-десктоп

Краткий сценарий для тестировщика и интегратора десктопного клиента КОМПАКС® 7.

**Граница ответственности репозитория `mcp-layer`:** обучение/наполнение на сервере и выгрузка артефактов.  
**Логика десктопа** (роутер, llama.cpp, MCP, UI) — **вне этого репозитория** (`docs/TECHNICAL_NOTE_V2.md`).

---

## Два контура

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Серверный стенд (этот repo) │         │  Офлайн-десктоп (другой repo) │
│  :3080 gateway               │  export │  импорт JSONL + локальный LLM │
│  ingest → индекс → чат       │ ──────> │  hybrid search без облака     │
└─────────────────────────────┘         └──────────────────────────────┘
```

| Контур | Что делает | Где описано |
|--------|------------|-------------|
| **Сервер** | Загрузка документов, индексация, проверка в чате, экспорт | `README.md`, `docs/COLLECTIONS_API.md` |
| **Десктоп** | Импорт индекса, локальный RAG, MCP-инструменты | *документация клиентской команды* |

---

## Сценарий тестировщика (сервер)

### 1. Поднять стенд

```bash
cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.yml up -d --build
docker compose -f rag-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f rag-compose.yml exec ollama ollama pull llama3.2:3b
```

Проверка: http://localhost:3080/health

Подробнее: `README.md` §1.

### 2. Создать папку и загрузить документы

**UI:** http://localhost:3080 → блок **«Загрузка документов»**

1. Создать папку (`id`, название)
2. Выбрать папку в списке
3. Прикрепить `.pdf` / `.txt` / `.md` / `.rst` (ZIP не поддерживается)
4. Дождаться в логе `status: completed`

**API:**

```bash
curl -X POST http://localhost:3080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"id": "ops-manual", "name": "Операторская"}'

curl -X POST "http://localhost:3080/load?collection_id=ops-manual&background=true" \
  -F "file=@manual.pdf"
```

Проверка: http://localhost:3080/sources — список не пустой.

### 3. Проверить ответы на стенде

**UI:** чат на http://localhost:3080  
**API:**

```bash
curl -X POST http://localhost:3080/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Что делает кнопка «Новый документ»?", "collection_ids": ["ops-manual"]}'
```

Демо-вопросы: `docs/DEMO_QUESTIONS.md`

### 4. Экспортировать индекс для десктопа

```bash
curl -f -OJ http://localhost:3080/export
```

Или в браузере: http://localhost:3080/export

| Параметр | Значение |
|----------|----------|
| Endpoint | `GET /export` (gateway) → `GET /v1/export?format=jsonl` (engine) |
| Формат | **JSONL** (NDJSON), одна строка = один чанк |
| Имя файла | `compacs-vectors-{timestamp}.jsonl` |
| Пустой индекс | **404** `vector index is empty` |

Реализация: `api/stable.py` (`/v1/export`), `core/export_index.py`.

---

## Формат экспорта (для разработчика десктопа)

Каждая строка JSONL — объект чанка:

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | ID чанка |
| `source` | string | Путь источника, напр. `collections/ops-manual/manual.pdf` |
| `page` | int | Номер страницы/блока |
| `chunk` | string | Текст чанка |
| `embedding` | float[] | Вектор (см. ниже) |
| `chunk_hash` | string | Хэш текста |
| `char_count` | int | Длина чанка |
| `dataset_kind` | string | Тип датасета |
| `metadata` | object | Доп. метаданные |

Схема полей: `core/storage/protocol.py` → `ChunkRecord.to_legacy_dict()`.

### Размерность embeddings

| Провайдер на стенде | Размерность |
|---------------------|-------------|
| Ollama `nomic-embed-text` (по умолчанию) | **768** |
| OpenAI `text-embedding-3-small` | **1536** |

Десктоп **обязан** использовать ту же модель эмбеддингов при запросах, что и при построении индекса (`core/embedding_alignment.py`).

### Что **не** входит в export

| Артефакт | Статус |
|----------|--------|
| Веса LLM (GGUF) | **Не экспортируются** — десктоп использует свою локальную модель |
| Оригиналы PDF/HTML | Отдельно: `GET /sources/{id}/download` |
| Модель `compacs-ui-ft` | Локально на сервере (Ollama Modelfile); в десктоп — отдельная поставка |
| Формат `.bin` / `.db` | **TBD** (`docs/TECHNICAL_NOTE_V2.md` §9) |

---

## Сценарий десктопа (чеклист интегратора)

> Шаги ниже — контракт для клиентской команды. Реализация импорта — в репозитории десктопа.

1. **Получить** `compacs-vectors-*.jsonl` с сервера (`GET /export`).
2. **Импортировать** строки в локальное vector store (аналог `data/vectors/chunks.json`).
3. **Настроить** локальный embedder с размерностью **768** (`nomic-embed-text` или эквивалент).
4. **Подключить** локальный LLM (llama.cpp / Ollama) — веса **не** из export.
5. **Проверить** те же вопросы, что на стенде (`docs/DEMO_QUESTIONS.md`).
6. **Убедиться**, что десктоп работает **без сети** после импорта.

### Три экрана UI (по ТЗ v2)

| Экран | Сервер | Десктоп |
|-------|--------|---------|
| **Чат** | http://localhost:3080 | Локальный RAG |
| **Загрузка** | `POST /load` | Импорт JSONL / локальные файлы |
| **Источники** | `GET /sources` | Список импортированных источников |

Тонкий клиент на сервере: gateway UI (`api/gateway.py`). На десктопе — аналогичный UX, но офлайн.

---

## Обучение модели (опционально, только сервер)

Цикл дообучения **не обязателен** для export индекса:

```bash
python scripts/ui_extension_pipeline.py extract
python scripts/ui_extension_pipeline.py index
# generate-qa / finetune — нужен OPENAI_API_KEY или только Ollama Modelfile
python scripts/ui_extension_pipeline.py finetune --ollama-only
```

Подробно: `docs/UI_EXTENSION.md`.

Результат `compacs-ui-ft` остаётся на сервере; в десктоп передаётся **отдельно** (GGUF / Ollama bundle) — **не через** `GET /export`.

---

## Диагностика

| Симптом | Причина | Действие |
|---------|---------|----------|
| `404` на `/export` | Индекс пуст | Загрузить документы, дождаться `completed` |
| Чат без ответа, sources есть | Ollama cold / timeout | `OLLAMA_KEEP_ALIVE=30m`, прогрев (`README.md`) |
| Десктоп «не находит» | Другая размерность embed | Переиндексация или тот же embedder 768 |
| ZIP не грузится | Не поддерживается | Распаковать, грузить `.pdf`/`.txt` |

---

## Карта документации

| Документ | Аудитория |
|----------|-----------|
| **Этот файл** | Тестировщик + интегратор десктопа |
| `README.md` | DevOps, запуск стенда |
| `docs/COLLECTIONS_API.md` | API upload/collections/export |
| `docs/RAG_V2_API.md` | Полная спецификация gateway |
| `docs/CONTROLLER_PATTERN.md` | Архитектура двух стримов |
| `docs/DEMO_QUESTIONS.md` | Вопросы для приёмки |
| `docs/TECHNICAL_NOTE_V2.md` | Production scope, TBD |

---

## Открытые пункты (нужно от клиентской команды)

1. Спецификация импорта JSONL в десктоп (API, путь к файлу, миграции).
2. Формат поставки GGUF / `compacs-ui-ft` на десктоп.
3. Нужен ли `/export` в бинарном формате (`.bin`/`.db`) — сейчас только JSONL.
4. Версионирование индекса (совместимость embedder / chunk schema).

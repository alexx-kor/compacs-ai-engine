# Техническая записка — RAG v2 (production-ready стенд)

**Проект:** КОМПАКС® 7 · AI-агент · серверный контур (стенд обучения)  
**Статус:** v1 работает (28 golden) → **v2 реализован в репозитории**, Docker-стенд — `rag-compose.yml`

---

## 1. Назначение

v1: рабочий RAG-прототип (golden 28 вопросов).  
v2: production-ready — одна точка входа, контейнеры, async/SSE, служебные операции (загрузка, export, metrics, sources CRUD).

Связь с десктопом: только **`GET /export`** (векторный индекс). Бизнес-логика клиента (роутер + llama.cpp + MCP) — вне scope.

---

## 2. Архитектура: два контура

```
Клиент → :3080 Gateway (FastAPI + Gunicorn + UvicornWorker)
              → HTTP → :8080 Engine (GRA-Hybrid RAG + Ollama)
```

| Контур | Порт | Стек | Роль |
|--------|------|------|------|
| **Gateway** | 3080 | Gunicorn, `uvicorn.workers.UvicornWorker` | UI, роутинг, SSE, proxy |
| **Engine** | 8080 | Uvicorn | ingest, embed, search, LLM |
| **Ollama** | 11434 | контейнер | эмбеддинги + генерация |

Код: `api/gateway.py`, `api/stable.py`, `rag_service.py`, `gunicorn.conf.py`, `wsgi.py`.

---

## 3. Маршруты API (gateway :3080)

| Маршрут | Метод | Статус | Назначение |
|---------|-------|--------|------------|
| `/` | GET | ✅ | Чат UI + `#load` |
| `/load` | POST | ✅ | Загрузка → ingestion (`?background=true` → 202 + poll `/load/{job_id}`) |
| `/sources` | GET | ✅ | Реестр источников |
| `/sources/{id}/download` | GET | ✅ | Оригинал файла |
| `/sources/{id}` | DELETE | ✅ | Удаление + переиндексация |
| `/export` | GET | ⚠️ JSONL | Выгрузка индекса (`.bin/.db` — TBD) |
| `/metrics` | GET | ✅ | Storage + PSI/drift (`quality`) |
| `/upgrade` | POST | ⚠️ stub | Pro по `COMPACS_PRO_KEY` |
| `/v1/*` | * | ✅ | Proxy к engine |
| прочие GET | * | ✅ | 404 |

**SSE:** `POST /api/chat` и `POST /v1/query` с `"stream": true`.

---

## 4. Управление источниками (CRUD)

| UI | Endpoint | Реализация |
|----|----------|------------|
| Список | `GET /sources` | `core/sources.py` + collections registry |
| Download src | `GET /sources/{id}/download` | Файл из `data/collections/{id}/files/` |
| Delete | `DELETE /sources/{id}` | Чанки + registry (`core/collections.py`) |

**Различие:** `/sources/{id}/download` — **оригинал документа**; `/export` — **весь vector index** для десктопа.

---

## 5. Развёртывание (Docker)

```bash
cp .env.rag.docker.example .env.rag
docker compose -f rag-compose.yml build
docker compose -f rag-compose.yml up -d

# Модели Ollama (первый запуск)
docker compose -f rag-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f rag-compose.yml exec ollama ollama pull llama3.2:3b
```

**URL:** http://localhost:3080

**Gunicorn (bare metal):**
```bash
gunicorn wsgi:app -c gunicorn.conf.py
python -m uvicorn api.stable:app_stable --host 127.0.0.1 --port 8080
```

`GATEWAY_TIMEOUT=300` — запас под cold-load Ollama + RAG prefill/генерацию; для `llama3.2:3b` на CPU не ставьте 120. `OLLAMA_KEEP_ALIVE=30m` держит модель в RAM. При росте нагрузки — очередь для `/load` (см. §9).

**Импорт существующего индекса:** смонтировать том `rag_data` или скопировать `data/vectors/chunks.json` в volume. Для **host-ollama** compose используется bind-mount `./data`.

**Паттерн контроллера (два стрима):** [`docs/CONTROLLER_PATTERN.md`](./CONTROLLER_PATTERN.md) — фоновый `/load` vs быстрый `/v1/query`, чеклист ручной проверки.

---

## 6. Синхронизация с десктопом

`GET /export` → JSONL (`compacs-vectors-{ts}.jsonl`). Клиент импортирует как локальную vector DB. Веса LLM не синхронизируются.

---

## 7. Мониторинг качества

`GET /metrics` → блок `quality`:
- PSI + KS (`core/drift_report.py`)
- сравнение индекса с golden
- offline: `python scripts/monitor_data_drift.py --preset splits`

**Faithfulness:** в текущем виде **не использовать как gate** (высокая доля нулей). Headline — **LLM-judge** и retrieval-метрики (`full_evaluation.py --llm-judge`).

---

## 8. Миграция с v1

| v1 | v2 |
|----|-----|
| Engine :8080 | Сохранён, расширен (`/v1/collections`, SSE) |
| Прямой доступ | Gateway :3080 — единая точка входа |
| 28 golden | `baseline/golden_set.json` + smoke scripts |
| LibreChat :3080 | Перенесён на **:3081** (нет конфликта) |

**Git:** файлы в индексе до `.gitignore` → `git rm --cached <path>`, commit, push.

---

## 9. Открытые шаги

| # | Задача | Приоритет |
|---|--------|-----------|
| 1 | `/export` в `.bin/.db` для десктопа | Средний |
| 2 | Очередь/фон для `/load` | ✅ in-process thread pool (`INGEST_JOB_WORKERS`); Redis/Celery при масштабе |
| 3 | `/upgrade` pro — полная логика на gateway | После выдачи ключа |
| 4 | Замена faithfulness → judge/NLI в gates | Средний |
| 5 | Пересчёт графа при DELETE source | Частично (vector only) |
| 6 | ClickHouse backend в Docker (опционально) | Низкий |

---

## 10. Проверка стенда

```bash
python scripts/smoke_rag.py
python scripts/demo_upload_http.py
python -m pytest tests/unit tests/integration -q
```

Демо-вопросы: [`DEMO_QUESTIONS.md`](./DEMO_QUESTIONS.md)  
API: [`RAG_V2_API.md`](./RAG_V2_API.md)

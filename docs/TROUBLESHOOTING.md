# Troubleshooting — COMPACS RAG Engine v2

Формат: **симптом → причина → фикс**. Факты из кода и конфигов; см. ссылки на файлы.

---

## Источники есть, ответ пустой или таймаут

**Симптом:** `POST /v1/query` возвращает `sources` (или UI показывает релевантные фрагменты), но `answer` пустой, обрыв по таймауту, HTTP 502/504.

**Причины:**

1. **Холодная загрузка модели** — первый вызов Ollama может занять **~50 с** на загрузку `llama3.2:3b` в RAM (комментарий `.env.rag.docker.example:15`).
2. **Таймаут gateway** — большой RAG-промпт + генерация на CPU превышают дефолт **120 с** у многих прокси; в проекте рекомендуется **300 с** (`gunicorn.conf.py:11`, `rag-compose.host-ollama.yml:42`).
3. **Пустой ответ от LLM** — engine отдаёт HTTP **500** `empty answer from RAG pipeline` (`api/stable.py:103-104`).

**Фикс:**

```bash
# 1. Прогрев модели
ollama run llama3.2:3b "ok"

# 2. В .env.rag (или compose environment)
OLLAMA_KEEP_ALIVE=30m
GATEWAY_TIMEOUT=300
OLLAMA_CLIENT_TIMEOUT=300
NUM_CTX=8192
NUM_PREDICT=400
RERANK_TOP_K=3
OLLAMA_CHUNK_CHARS=350
OLLAMA_CONTEXT_CHUNKS=3

# 3. Перезапуск compose после правки .env.rag
docker compose -f rag-compose.host-ollama.yml up -d --build
```

**Диагностика Ollama** (смотрите `load_duration` vs `eval_duration`):

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Кратко: что такое HTTP-прокси?",
  "stream": false,
  "options": {"num_ctx": 8192, "num_predict": 400}
}'
```

**PowerShell:**

```powershell
$body = @{
  model = "llama3.2:3b"
  prompt = "Кратко: что такое HTTP-прокси?"
  stream = $false
  options = @{ num_ctx = 8192; num_predict = 400 }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method Post -Body $body -ContentType "application/json"
```

**Стриминг:** `"stream": true` на `POST /v1/query` — токены приходят раньше, меньше риск «тихого» таймаута на клиенте.

---

## Смена провайдера эмбеддингов без переиндексации

> **ПРЕДУПРЕЖДЕНИЕ:** нельзя менять `EMBEDDING_PROVIDER` / `EMBED_MODEL` на уже построенном индексе без полной переиндексации. Размерности векторов разные — поиск деградирует или ломается.

| Провайдер | Модель (дефолт) | Размерность |
|-----------|-----------------|-------------|
| Ollama | `nomic-embed-text` | **768** (`core/embedding_alignment.py:13`) |
| OpenAI | `text-embedding-3-small` | **1536** (`core/embedding_alignment.py:12`) |

При старте `configure_embeddings_for_index()` подбирает провайдер по размерности в `chunks.json` (`core/embedding_alignment.py:47-69`).

**Фикс — полная переиндексация:**

```bash
# CLI: пересоздать store и загрузить из instructions/raw
python -m app ingest --force-reload

# Или через HTTP: удалить источники / коллекции и загрузить документы заново
```

После смены провайдера перезапустите engine/gateway.

---

## Английский вопрос к русскому корпусу

**Симптом:** вопрос на EN, корпус на RU — низкий score, `NOT FOUND`, при этом таймаутов нет.

**Причина:** это не баг таймаута. `nomic-embed-text` поддерживает кросс-язычный retrieval, но качество ниже, чем при совпадении языка вопроса и документов.

**Фикс:** задавайте вопрос на языке корпуса; для приёмки используйте `docs/DEMO_QUESTIONS.md`.

---

## `GET /export` → 404 `vector index is empty`

**Причина:** нет чанков в `LOCAL_VECTOR_STORE_DIR` (`api/stable.py:264-265`).

**Фикс:** загрузите документы через `/load` или `python -m app ingest`, дождитесь завершения индексации.

---

## Docker на Windows: engine не видит Ollama

**Симптом:** запросы к RAG падают, Ollama на хосте работает (`curl localhost:11434/api/tags` OK), из контейнера — нет.

**Причина:** Ollama слушает только `127.0.0.1`; compose использует `host.docker.internal` (`rag-compose.host-ollama.yml:34`).

**Фикс:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-ollama-docker-windows.ps1
```

---

## Загрузка: ZIP / неподдерживаемый формат

**Симптом:** upload молча не индексирует или ошибка валидации.

**Причина:** поддерживаются только `.pdf`, `.txt`, `.md`, `.rst` (`core/ingestion.py:15-17`).

**Фикс:** распакуйте архив, загружайте файлы по одному.

---

## `python scripts/smoke_rag.py` → exit 1

**Причины:**

- нет `data/vectors/chunks.json` — нужна индексация;
- ответ содержит `NOT FOUND` (`scripts/smoke_rag.py:84-86`).

**Фикс:** `python scripts/ui_extension_pipeline.py extract && python scripts/ui_extension_pipeline.py index` (если есть UI Extension) или upload + query по своему корпусу.

---

## `full_evaluation.py` требует OpenAI

**Симптом:** ошибка `OPENAI_API_KEY required for GPT judge`.

**Причина:** `--llm-judge` по умолчанию использует `--judge-backend openai` (`full_evaluation.py:517-519`).

**Фикс (офлайн):**

```bash
python full_evaluation.py --golden baseline/golden_set.json --llm-provider ollama --llm-judge --judge-backend ollama
```

Без судьи (только метрики RAG):

```bash
python full_evaluation.py --golden baseline/golden_set.json --llm-provider ollama
```

---

## Порт gateway не тот

**Симптом:** `curl localhost:3080/health` — connection refused.

**Причина:** в `.env.rag.docker.example` задано `RAG_GATEWAY_PORT=3090` (строка 44); compose мапит `${RAG_GATEWAY_PORT:-3080}:3080`.

**Фикс:** проверьте `.env.rag` и используйте тот же порт в URL, либо установите `RAG_GATEWAY_PORT=3080`.

---

## Полный Docker: обрыв pull `ollama/ollama`

**Симптом:** `unexpected EOF` при скачивании образа ~2 GB.

**Фикс:** используйте `rag-compose.host-ollama.yml` (комментарий `rag-compose.host-ollama.yml:1`, `docs/CONTROLLER_PATTERN.md`).

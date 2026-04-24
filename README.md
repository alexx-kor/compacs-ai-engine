# MCP Layer: Build & Run Guide

Короткая инструкция по сборке окружения и запуску пайплайнов.

## 1) Сборка окружения

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Подготовка моделей Ollama:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama serve
```

При необходимости задайте ключ OpenAI:

```bash
set OPENAI_API_KEY=your_key
```

## 2) Порядок запуска

```mermaid
flowchart LR
    A[Build env] --> B[load_graph_chunks.py]
    B --> C[baseline/run_gpu_baseline.py]
    C --> D[llm_evaluate.py / full_evaluation.py]
    D --> E[compare_results.py]
```

## 3) Индексация данных (`instructions` -> ClickHouse)

```bash
python load_graph_chunks.py
```

Без пересоздания таблицы:

```bash
python load_graph_chunks.py --no-force-recreate
```

## 4) Генерация baseline-ответов

```bash
python baseline/run_gpu_baseline.py
```

Результат: `baseline/rag_answers_gpu.json`

## 5) Оценка качества

```bash
python llm_evaluate.py --main <main.json> --hypothesis <hyp.json>
```

или полный прогон:

```bash
python full_evaluation.py
```

## 6) Сравнение прогонов

```bash
python compare_results.py --old <old.csv|old.json> --new <new.json>
```

## 7) Ключевые файлы

- `instructions/` — исходные документы
- `baseline/questions` — вопросы
- `baseline/golden_set.json` — эталон
- `baseline/rag_answers_gpu.json` — сгенерированные ответы
- `load_graph_chunks.py` — предобработка и загрузка чанков
- `baseline/run_gpu_baseline.py` — основной RAG пайплайн

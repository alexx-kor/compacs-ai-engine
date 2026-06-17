# UI Extension: пайплайн дообучения RAG

Документация по подсистеме индексации HTML-руководства оператора КОМПАКС, генерации синтетического QA, fine-tune Ollama и оценке качества.

## Цель

1. Извлечь текст из HTML (`UI extension/`)
2. Проиндексировать чанки в векторное хранилище (merge с основным индексом)
3. Сгенерировать ~150 пар вопрос–ответ для обучения
4. Разбить на train / val / test (seed 42)
5. Собрать Modelfile и создать `compacs-ui-ft` в Ollama
6. Оценить baseline vs fine-tuned на val и test
7. Сформировать Excel-отчёт для руководства

## Скрипты

| Скрипт | Назначение |
|--------|------------|
| `scripts/ui_extension_pipeline.py` | Пошаговый пайплайн: extract → index → generate-qa → finetune → evaluate |
| `scripts/run_ui_extension_automation.py` | Полная автоматизация с split, audit, eval, Excel |
| `scripts/split_ui_qa_dataset.py` | Разбиение QA на train/val/test + JSONL для FT |
| `scripts/audit_qa_pairs.py` | Аудит качества QA (heuristic + GPT sample) |
| `scripts/export_ui_extension_report.py` | Excel: Сравнение, Метрики, Метаданные, Качество QA, Выводы |
| `scripts/summarize_ui_split_eval.py` | Сводка delta judge/score по val и test |
| `export_hybrid_comparison_xlsx.py` | Общие хелперы экспорта метрик в Excel |

## Быстрый старт

### Требования

- Python ≥ 3.10, зависимости из `pyproject.toml`
- Ollama (`llama3.2:3b` + созданная модель `compacs-ui-ft`)
- `OPENAI_API_KEY` в `.env.rag` (генерация QA, GPT-судья, опционально OpenAI FT)
- Проиндексированное хранилище `data/vectors/` (JSON backend)

### Полная автоматизация

```powershell
python scripts/run_ui_extension_automation.py -v
```

Шаги автоматизации:

1. Index merge (если UI-чанков < 500)
2. Генерация QA (если нет `instructions/golden/ui_extension_qa_150.json`)
3. Split → `instructions/golden/splits/` (132 / 16 / 18 при 166 парах, seed 42)
4. QA audit → `data/finetune/qa_audit_report.json`
5. Finetune только на train split (`--train-qa-path`)
6. Eval val: `evaluation_ui_extension_val_{baseline,finetuned}.json`
7. Eval test: `evaluation_ui_extension_test_{baseline,finetuned}.json`
8. Excel → `comparison_ui_extension_finetune.xlsx`

### Пошаговый запуск

```powershell
# 1. HTML → txt
python scripts/ui_extension_pipeline.py extract

# 2. Индексация (merge)
python scripts/ui_extension_pipeline.py index

# 3. Генерация QA
python scripts/ui_extension_pipeline.py generate-qa --count 150 --seed 42

# 4. Split
python scripts/split_ui_qa_dataset.py --seed 42

# 5. Аудит QA
python scripts/audit_qa_pairs.py --judge-sample 50

# 6. Fine-tune (только train)
python scripts/ui_extension_pipeline.py finetune --ollama-only `
  --train-qa-path instructions/golden/splits/ui_extension_qa_train.json

# 7. Оценка
python full_evaluation.py --golden instructions/golden/splits/ui_extension_qa_val.json `
  --output evaluation_ui_extension_val_baseline.json --llm-judge --llm-provider ollama
# (повторить для compacs-ui-ft и test split)

# 8. Excel-отчёт
python scripts/export_ui_extension_report.py `
  --golden instructions/golden/splits/ui_extension_qa_val.json `
  --baseline-json evaluation_ui_extension_val_baseline.json `
  --finetuned-json evaluation_ui_extension_val_finetuned.json
```

## Артефакты

| Путь | Описание |
|------|----------|
| `instructions/golden/ui_extension_qa_150.json` | Полный синтетический QA-набор |
| `instructions/golden/splits/ui_extension_qa_{train,val,test}.json` | Сплиты |
| `instructions/golden/splits/ui_extension_qa_split_meta.json` | Мета split (counts, seed) |
| `data/finetune/ui_extension_train.jsonl` | Chat JSONL для FT |
| `data/finetune/Modelfile.compacs-ui-ft` | Modelfile Ollama |
| `data/finetune/qa_audit_report.json` | Отчёт аудита QA |
| `comparison_ui_extension_finetune.xlsx` | Итоговый Excel |

## Мониторинг дисперсии метрик при росте данных

Среднее по 16 val-вопросам может быть нестабильным. Скрипт `monitor_metric_dispersion.py` строит **кривую устойчивости**: при n = 4, 8, 12, … N считает mean, std, variance (bootstrap).

```powershell
python scripts/monitor_metric_dispersion.py `
  --baseline-json evaluation_ui_extension_val_baseline.json `
  --finetuned-json evaluation_ui_extension_val_finetuned.json `
  --min-n 4 --step 4 --bootstrap-repeats 200 --seed 42
```

Отчёт: `data/finetune/metric_dispersion_report.json`

| Поле | Смысл |
|------|--------|
| `full_sample.std` | Разброс метрики на всей выборке |
| `by_sample_size[].std_of_mean` | Насколько «прыгает» среднее при n вопросах |
| `by_sample_size[].variance_of_sample` | Средняя дисперсия внутри подвыборок |

**Как читать:** при росте n `std_of_mean` обычно **падает** — оценка стабилизируется. Если на n=16 дисперсия всё ещё высокая, val-сет слишком мал или метрика шумная.

В `full_evaluation.py` в `statistics` теперь также пишутся `std_score`, `variance_score`, `llm_judge_std_percent`.

## Дрейф данных: PSI и Колмогоров–Смирнов

Скрипт `monitor_data_drift.py` сравнивает **распределения** reference vs current:

| Тест | Что показывает | Порог |
|------|----------------|-------|
| **PSI** | Сдвиг долей по бинам | <0.1 ок, 0.1–0.25 умеренный, ≥0.25 сильный |
| **KS** | Макс. разница CDF + p-value | p < 0.05 → дрейф |

```powershell
# Сплиты QA: train vs val / test (длины вопросов и ответов)
python scripts/monitor_data_drift.py --preset splits

# Дрейф метрик eval: baseline vs finetuned на val
python scripts/monitor_data_drift.py --preset eval-val

# Произвольная пара JSON (QA, eval, qa_audit)
python scripts/monitor_data_drift.py `
  --reference evaluation_ui_extension_val_baseline.json `
  --current evaluation_ui_extension_val_finetuned.json `
  --metrics judge_percent,final_score,faithfulness
```

Отчёт: `data/finetune/data_drift_report.json`

В `monitor_metric_dispersion.py` при сравнении baseline vs FT в блок `comparison._drift` автоматически добавляются PSI и KS.

## Тестирование

Юнит-тесты UI extension:

```powershell
pytest tests/unit/test_audit_qa_pairs.py tests/unit/test_ui_extension_split.py `
  tests/unit/test_ui_extension_export.py tests/unit/test_ui_extension_pipeline.py -v
```

Полный прогон тестов проекта:

```powershell
pytest
```

## Git (приватный репозиторий)

Локальный репозиторий инициализируется командой `git init`. Для приватного remote на GitHub:

1. Создайте **Private** repo на GitHub (без README)
2. Привяжите remote и запушьте:

```powershell
git remote add origin https://github.com/<org>/compacs-ai-engine.git
git add .
git commit -m "Initial commit: RAG engine + UI extension pipeline"
git push -u origin main
```

**Важно:** `.gitignore` исключает `data/`, `instructions/`, `.env*` — секреты и тяжёлые артефакты не попадут в git. Код скриптов и тестов коммитится.

## Чеклист для сдачи Zhakesh

- [ ] `pytest` — все тесты зелёные
- [ ] Прогон `run_ui_extension_automation.py` завершён без ошибок
- [ ] Excel-отчёт сформирован и проверен
- [ ] Приватный git: remote создан, первый push выполнен
- [ ] Документация актуальна (`docs/UI_EXTENSION.md`, этот чеклист)

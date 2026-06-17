# Demo questions for COMPACS RAG

## UI Extension (после `ui_extension_pipeline.py index`)

| # | Вопрос |
|---|--------|
| 1 | Что делает кнопка «Новый документ»? |
| 2 | Как создать документ без CDPL-процедур? |
| 3 | Как открыть справку по текущему окну? |
| 4 | Что такое дерево объектов в интерфейсе оператора? |

## AI Server / Dagster (`baseline/golden_set.json`)

| # | Вопрос |
|---|--------|
| 1 | Какой WAN-адрес и учётная запись SFTP для загрузки ZIP на AI-сервер? |
| 2 | Какой шаблон имени сырого архива перед загрузкой в raw_data/? |
| 3 | На каком TCP-порту доступен веб-интерфейс Dagster? |

## Быстрая проверка

```bash
python scripts/smoke_rag.py
python scripts/smoke_rag.py -q "Что делает кнопка «Новый документ»?" --stream
python scripts/smoke_rag.py --golden baseline/golden_set.json -n 5
```

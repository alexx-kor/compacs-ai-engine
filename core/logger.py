#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Структурированное логирование для RAG системы
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class StructuredLogger:
    """Структурированное логирование с ключами для grep"""
    
    def __init__(self, model_tag: str = "llama3.2:3b"):
        self.model_tag = model_tag
        self._run_start = datetime.now()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Создаём файл лога для этого запуска
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run-{model_tag.replace(':', '_')}-{timestamp}.log"
        self.log_file = self.log_dir / log_filename
        
        # Настройка логирования
        self.logger = logging.getLogger(f"rag_{model_tag}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        # Не накапливаем хендлеры при повторной инициализации
        if self.logger.handlers:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        
        # Хендлер для файла
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=10_485_760, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Формат: [model_tag] [pipeline_stage] key=value ...
        formatter = logging.Formatter(
            f'%(asctime)s [{model_tag}] [%(stage)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Консольный вывод (только INFO и выше)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Контекст для текущей операции
        self._stage = "init"
        self._context = {}
        
        # Статистика по запуску
        self.stats = {
            'embed_calls_gpu': 0,
            'embed_calls_cpu': 0,
            'total_chunks': 0,
            'total_embed_time_ms': 0,
            'total_ingest_time_ms': 0,
            'files_processed': 0,
            'skipped_files': 0,
            'gpu_queries': 0,
            'cpu_queries': 0
        }
        
        # Логируем старт
        self.log_run_start()
    
    def set_stage(self, stage: str):
        """Устанавливает текущую стадию пайплайна"""
        self._stage = stage
    
    def _log(self, event: str, **kwargs):
        """Внутренний метод для логирования"""
        parts = [event]
        for key, value in kwargs.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            elif isinstance(value, dict):
                parts.append(f"{key}={json.dumps(value)}")
            else:
                parts.append(f"{key}={value}")
        
        extra = {'stage': self._stage}
        self.logger.info(' '.join(parts), extra=extra)
    
    # ========================================================================
    # RUN LOGGING
    # ========================================================================
    
    def log_run_start(self):
        """Логирует начало запуска"""
        self._log(
            "run.start",
            model_tag=self.model_tag,
            log_file=str(self.log_file),
            timestamp=datetime.now().isoformat()
        )
    
    def log_run_end(self):
        """Логирует завершение запуска"""
        self._log(
            "run.end",
            model_tag=self.model_tag,
            duration_seconds=(datetime.now() - self._run_start).total_seconds()
        )
    
    # ========================================================================
    # OLLAMA BACKEND CHECK
    # ========================================================================
    
    def log_ollama_backend(self, compute: str, vram_mb: int):
        """Логирует информацию о бэкенде Ollama"""
        self.set_stage("init")
        self._log(
            "ollama.backend",
            model=self.model_tag,
            compute=compute,
            size_vram=vram_mb
        )
        
        if compute == "CPU" and vram_mb == 0:
            self._log(
                "ollama.backend.warn",
                model=self.model_tag,
                compute=compute,
                message="CPU fallback - GPU not available"
            )
    
    # ========================================================================
    # EMBEDDING LOGGING
    # ========================================================================
    
    def log_embed_call(self, model: str, input_chars: int, latency_ms: float, batching: str = "sequential"):
        """Логирует вызов эмбеддинга"""
        self.set_stage("embed")
        
        self._log(
            "embed.call",
            model_tag=self.model_tag,
            model=model,
            batching=batching,
            input_chars=input_chars,
            latency_ms=latency_ms
        )
        
        if latency_ms > 500:
            self._log(
                "embed.call.slow",
                model_tag=self.model_tag,
                model=model,
                latency_ms=latency_ms,
                threshold=500
            )
        
        # Обновляем статистику
        self.stats['total_embed_time_ms'] += latency_ms
    
    def log_ollama_backend_embed(self, model: str, compute: str, vram_mb: int):
        """Логирует информацию о бэкенде при эмбеддинге"""
        self.set_stage("embed")
        self._log(
            "ollama.embed.backend",
            model=model,
            compute=compute,
            size_vram=vram_mb
        )
        
        if compute == "GPU":
            self.stats['embed_calls_gpu'] += 1
        else:
            self.stats['embed_calls_cpu'] += 1
    
    # ========================================================================
    # UPSTREAM ERRORS
    # ========================================================================
    
    def log_upstream_5xx(self, subsystem: str, status_code: int, message: str = ""):
        """Логирует ошибки 5xx от апстрима"""
        self.set_stage("upstream")
        self._log(
            "upstream.http.5xx",
            subsystem=subsystem,
            status_code=status_code,
            message=message
        )
    
    # ========================================================================
    # INGESTION LOGGING
    # ========================================================================
    
    def log_ingest_file(self, filename: str, source_type: str, chunk_count: int, 
                        embedding_time_ms: float, total_time_ms: float):
        """Логирует обработку одного файла"""
        self.set_stage("ingest")
        
        self._log(
            "ingest.file.summary",
            model_tag=self.model_tag,
            filename=filename,
            source_type=source_type,
            chunk_count=chunk_count,
            total_embedding_time_ms=embedding_time_ms,
            total_ingest_time_ms=total_time_ms
        )
        
        if total_time_ms > 10000:  # >10 секунд
            self._log(
                "ingest.file.slow",
                model_tag=self.model_tag,
                filename=filename,
                total_ingest_time_ms=total_time_ms,
                threshold=10000
            )
        
        self.stats['files_processed'] += 1
        self.stats['total_chunks'] += chunk_count
    
    def log_ingest_soft_skip(self, filename: str, reason: str, file_size_bytes: int):
        """Логирует пропуск файла"""
        self.set_stage("ingest")
        self._log(
            "ingest.soft_skip",
            model_tag=self.model_tag,
            filename=filename,
            reason=reason,
            file_size_bytes=file_size_bytes
        )
        self.stats['skipped_files'] += 1
    
    def log_ingest_backend_summary(self):
        """Логирует сводку по бэкенду"""
        self.set_stage("ingest")
        total_calls = self.stats['embed_calls_gpu'] + self.stats['embed_calls_cpu']
        fallback_rate = (self.stats['embed_calls_cpu'] / total_calls * 100) if total_calls > 0 else 0
        
        self._log(
            "ingest.backend.summary",
            model_tag=self.model_tag,
            embed_calls_gpu=self.stats['embed_calls_gpu'],
            embed_calls_cpu=self.stats['embed_calls_cpu'],
            fallback_rate=f"{fallback_rate:.1f}%"
        )
    
    def log_ingest_batch_summary(self, wall_clock_ms: float):
        """Логирует сводку по батчу"""
        self.set_stage("ingest")
        avg_ms_per_chunk = wall_clock_ms / self.stats['total_chunks'] if self.stats['total_chunks'] > 0 else 0
        
        self._log(
            "ingest.batch.summary",
            model_tag=self.model_tag,
            files_processed=self.stats['files_processed'],
            skipped_count=self.stats['skipped_files'],
            total_chunks=self.stats['total_chunks'],
            wall_clock_ms=wall_clock_ms,
            avg_ms_per_chunk=avg_ms_per_chunk
        )
    
    # ========================================================================
    # CHROMA OPERATIONS
    # ========================================================================
    
    def log_chroma_pre_clean(self, collection: str, items_deleted: int, latency_ms: float):
        """Логирует очистку коллекции Chroma"""
        self.set_stage("chroma")
        self._log(
            "chroma.pre_clean",
            model_tag=self.model_tag,
            collection=collection,
            items_deleted=items_deleted,
            latency_ms=latency_ms
        )
    
    def log_chroma_op(self, collection: str, operation: str, latency_ms: float, result_count: int = 0):
        """Логирует операцию Chroma"""
        self.set_stage("chroma")
        self._log(
            "chroma.op",
            model_tag=self.model_tag,
            collection=collection,
            operation=operation,
            latency_ms=latency_ms,
            result_count=result_count
        )
        
        if operation == "query" and latency_ms > 300:
            self._log(
                "chroma.query.slow",
                model_tag=self.model_tag,
                collection=collection,
                latency_ms=latency_ms,
                threshold=300
            )
    
    # ========================================================================
    # RAG PIPELINE CHECKPOINTS
    # ========================================================================
    
    def log_rag_checkpoint_a(self, retrieved_chunks: list):
        """Логирует pre-rerank чанки"""
        self.set_stage("pre_rerank")
        
        # Форматируем чанки: chunkId:score
        chunks_str = ','.join([f"{c[0]}:{float(c[3]) if isinstance(c[3], (int, float)) else c[3]}" for c in retrieved_chunks[:5]])
        
        self._log(
            "rag.checkpoint.a",
            model_tag=self.model_tag,
            retrieved_chunks=f"[{chunks_str}]"
        )
    
    def log_rag_checkpoint_b(self, reranker: str, context_chars: int):
        """Логирует post-rerank этап"""
        self.set_stage("post_rerank")
        self._log(
            "rag.checkpoint.b",
            model_tag=self.model_tag,
            reranker=reranker if reranker else "absent",
            context_chars=context_chars
        )
    
    def log_rag_checkpoint_c(self, prompt_chars: int, model: str, temperature: float):
        """Логирует LLM prompt этап"""
        self.set_stage("llm_prompt")
        self._log(
            "rag.checkpoint.c",
            model_tag=self.model_tag,
            prompt_chars=prompt_chars,
            model=model,
            temperature=temperature
        )
    
    def log_rag_checkpoint_c_prompt_head(self, prompt_head: str):
        """Логирует начало промпта (DEBUG)"""
        self.set_stage("llm_prompt")
        self._log(
            "rag.checkpoint.c.prompt_head",
            model_tag=self.model_tag,
            prompt_head=prompt_head[:200]
        )
    
    # ========================================================================
    # LLM GENERATION
    # ========================================================================
    
    def log_llm_generate(self, provider: str, model: str, prompt_chars: int, 
                         latency_ms: float, response_chars: int):
        """Логирует генерацию LLM"""
        self.set_stage("llm_generate")
        
        self._log(
            "llm.generate",
            model_tag=self.model_tag,
            provider=provider,
            model=model,
            prompt_chars=prompt_chars,
            latency_ms=latency_ms,
            response_chars=response_chars
        )
        
        if latency_ms > 5000:
            self._log(
                "llm.generate.slow",
                model_tag=self.model_tag,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                threshold=5000
            )
        
        # Обновляем статистику
        if provider == "ollama":
            # Явно учитываем только переданный тип compute, если он есть в model/provider.
            # Без достоверного сигнала считаем как CPU вызов.
            if "gpu" in str(model).lower() or "gpu" in str(provider).lower():
                self.stats['gpu_queries'] += 1
            else:
                self.stats['cpu_queries'] += 1
    
    def log_llm_stream(self, provider: str, model: str, latency_ms: float):
        """Логирует стриминг LLM"""
        self.set_stage("llm_generate")
        self._log(
            "llm.stream",
            model_tag=self.model_tag,
            provider=provider,
            model=model,
            latency_ms=latency_ms
        )
        
        if latency_ms > 5000:
            self._log(
                "llm.stream.slow",
                model_tag=self.model_tag,
                provider=provider,
                model=model,
                latency_ms=latency_ms
            )
    
    def log_openai_error(self, http_status: int, message: str):
        """Логирует ошибки OpenAI API"""
        self.set_stage("llm_generate")
        self._log(
            "openai.api.error",
            model_tag=self.model_tag,
            http_status=http_status,
            message=message
        )
    
    # ========================================================================
    # HTTP LEVEL
    # ========================================================================
    
    def log_query_result(self, latency_ms: float, response_chars: int, prompt_prefix: str = ""):
        """Логирует результат запроса"""
        self.set_stage("query")
        self._log(
            "query.result",
            model_tag=self.model_tag,
            latency_ms=latency_ms,
            response_chars=response_chars,
            prompt_prefix=prompt_prefix[:50] if prompt_prefix else ""
        )
    
    def log_http_response_large(self, body_chars: int, threshold_chars: int = 51200):
        """Логирует большие HTTP ответы"""
        self.set_stage("http")
        self._log(
            "http.response.large",
            model_tag=self.model_tag,
            body_chars=body_chars,
            threshold_chars=threshold_chars
        )
    
    # ========================================================================
    # EVALUATION BATCH
    # ========================================================================
    
    def log_eval_batch_summary(self, questions: int, avg_latency_ms: float, 
                                gpu_calls: int, cpu_calls: int, skipped_files: int):
        """Логирует сводку по батчу оценки"""
        self.set_stage("eval")
        self._log(
            "eval.batch.summary",
            model_tag=self.model_tag,
            questions=questions,
            avg_latency_ms=avg_latency_ms,
            gpu_calls=gpu_calls,
            cpu_calls=cpu_calls,
            skipped_files=skipped_files
        )


# Глобальный экземпляр логгера
logger = None


def init_logger(model_tag: str = "llama3.2:3b") -> StructuredLogger:
    """Инициализирует глобальный логгер"""
    global logger
    logger = StructuredLogger(model_tag)
    return logger


def get_logger() -> StructuredLogger:
    """Возвращает глобальный логгер"""
    return logger
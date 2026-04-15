#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Engine - FIXED VERSION
"""

import time
import json
import hashlib
from typing import Dict, List, Tuple
import ollama

from config import config
from core.database import db
from core.embeddings import embedder
from core.reranker import reranker
from router.smart_router import select_prompt


class RAGEngine:
    """RAG движок с автоматическим выбором промпта"""
    
    @staticmethod
    def ask(question: str) -> Dict:
        t_start = time.time()
        
        # Кэш
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached = db.get_cache(cache_key)
        if cached:
            result = json.loads(cached)
            result['cached'] = True
            return result
        
        # Поиск контекста
        q_emb = list(embedder.generate_cached(question))
        results = db.search(q_emb)
        
        if not results:
            return {
                'question': question,
                'answer': "NOT FOUND in documentation",
                'sources': [],
                'time_total': round(time.time() - t_start, 2)
            }
        
        # Реранжинг
        reranked = reranker.rerank(question, results)
        
        # Контекст
        context_parts = []
        sources = []
        for r in reranked[:config.rerank_top_k]:
            chunk, source, page = r[0], r[1], r[2]
            context_parts.append(f"[{source}, p.{page}]\n{chunk[:800]}")
            sources.append((source, page))
        
        context = "\n\n".join(context_parts)
        
        # Выбор промпта
        system_prompt, num_predict, temperature = select_prompt(question)
        
        # Форматирование
        if '{context}' in system_prompt and '{query}' in system_prompt:
            formatted_prompt = system_prompt.format(context=context, query=question)
        else:
            formatted_prompt = system_prompt
        
        # Генерация
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
        
        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=messages,
                options={
                    "num_predict": num_predict,
                    "temperature": temperature,
                    "top_k": 40,
                    "top_p": config.top_p,
                    "num_ctx": config.num_ctx,
                    "repeat_penalty": config.repeat_penalty,
                    "num_gpu": config.ollama_num_gpu
                }
            )
            # ✅ ИСПРАВЛЕНО: response - это словарь
            if isinstance(response, dict):
                answer = response['message']['content']
            else:
                answer = response.message.content
            status = 'success'
        except Exception as e:
            answer = f"ERROR: {e}"
            status = 'error'
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'time_total': round(time.time() - t_start, 2),
            'cached': False,
            'status': status
        }
        
        db.set_cache(cache_key, json.dumps(result))
        return result


rag = RAGEngine()
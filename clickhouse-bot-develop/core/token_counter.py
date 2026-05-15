#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token counter for OpenAI models
"""

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Подсчитывает количество токенов в тексте"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def estimate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    """Оценивает стоимость запроса"""
    # Цены за 1M токенов (актуальные на 2025)
    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    if model in prices:
        # Примерная оценка (50/50 input/output)
        return tokens * prices[model]["input"] / 1_000_000
    return 0
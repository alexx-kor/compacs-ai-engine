import hashlib
from typing import List
from functools import lru_cache
import ollama
from config import config


class EmbeddingGenerator:
    def __init__(self):
        self.model = config.embed_model
        self.batch_size = config.batch_size
        self.max_length = config.max_text_length
    
    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_length:
            return text
        truncated = text[:self.max_length]
        last_period = truncated.rfind('.')
        if last_period > self.max_length // 2:
            truncated = truncated[:last_period + 1]
        return truncated.strip()
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        safe_texts = [self._truncate_text(t) for t in texts]
        try:
            response = ollama.embed(
                model=self.model,
                input=safe_texts,
                options={"num_gpu": config.ollama_num_gpu}
            )
            return response['embeddings']
        except Exception as e:
            print(f"    Embedding error: {e}")
            return [[0.0] * 768 for _ in safe_texts]
    
    def generate(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            embeddings = self.generate_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    @lru_cache(maxsize=256)
    def generate_cached(self, text: str) -> tuple:
        embedding = self.generate_batch([text])[0]
        return tuple(embedding)


embedder = EmbeddingGenerator()
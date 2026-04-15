import re
from typing import List, Tuple
from config import config


class Reranker:
    @staticmethod
    def rerank(question: str, results: List[tuple]) -> List[tuple]:
        if not results:
            return results
        
        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        
        scored = []
        for result in results:
            chunk, source, page, distance = result
            c_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            similarity = 1 - distance
            final_score = similarity * 0.6 + overlap * 0.4
            scored.append((final_score, result))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:config.rerank_top_k]]


reranker = Reranker()
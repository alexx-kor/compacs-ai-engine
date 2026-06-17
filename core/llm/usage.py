"""Token usage tracking for LLM completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens or (self.prompt_tokens + self.completion_tokens),
            "cost_usd": self.cost_usd,
        }

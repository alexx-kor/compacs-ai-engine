"""OpenAI usage tracking and budget enforcement."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from config import Config

log = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when OpenAI daily budget is exceeded."""


class RateLimitExceededError(RuntimeError):
    """Raised when OpenAI per-minute request limit is exceeded."""


@dataclass(frozen=True)
class UsageSnapshot:
    """Point-in-time OpenAI usage counters."""

    request_count: int
    total_tokens: int
    cost_usd: float


class CostGuard:
    """Track OpenAI usage and enforce configured limits."""

    def __init__(self, config: Config, usage_file: Path | None = None) -> None:
        self._max_requests_per_min = config.openai_max_requests_per_min
        self._daily_budget_usd = config.openai_daily_budget_usd
        self._usage_file = usage_file or Path(config.project_root / "data" / "openai_usage.json")
        self._request_timestamps: deque[float] = deque()

    def check_limits(self) -> None:
        """Validate rate and budget limits before an OpenAI call."""
        self._prune_request_window()
        if len(self._request_timestamps) >= self._max_requests_per_min:
            raise RateLimitExceededError(
                f"OpenAI rate limit exceeded: {self._max_requests_per_min} requests/min"
            )
        daily_cost = self._load_usage().get("cost_usd", 0.0)
        if daily_cost >= self._daily_budget_usd:
            raise BudgetExceededError(
                f"OpenAI daily budget exceeded: ${self._daily_budget_usd:.2f}"
            )

    def record_request(self) -> None:
        """Record one OpenAI request for rate limiting."""
        self._request_timestamps.append(time.time())

    def track_usage(self, usage: Any | None, model: str) -> UsageSnapshot:
        """Persist token usage and estimated cost.

        Args:
            usage: OpenAI usage object from a completion response.
            model: Model name used for cost estimation.

        Returns:
            Snapshot of recorded usage for the current call.
        """
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self._estimate_cost(prompt_tokens, completion_tokens, model)
        self._append_usage(total_tokens=total_tokens, cost_usd=cost_usd)
        self.record_request()
        snapshot = UsageSnapshot(
            request_count=len(self._request_timestamps),
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
        log.info(
            "OpenAI usage tokens=%s cost_usd=%.6f model=%s",
            total_tokens,
            cost_usd,
            model,
        )
        return snapshot

    def load_daily_summary(self) -> dict[str, float]:
        """Return today's usage counters."""
        return self._load_usage()

    def _prune_request_window(self) -> None:
        cutoff = time.time() - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()

    def _load_usage(self) -> dict[str, float]:
        today_key = str(date.today())
        if not self._usage_file.exists():
            return {"cost_usd": 0.0, "total_tokens": 0.0}
        try:
            with self._usage_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            log.warning("Could not read usage file path=%s error=%s", self._usage_file, error)
            return {"cost_usd": 0.0, "total_tokens": 0.0}
        if not isinstance(payload, dict):
            return {"cost_usd": 0.0, "total_tokens": 0.0}
        day_payload = payload.get(today_key, {})
        if not isinstance(day_payload, dict):
            return {"cost_usd": 0.0, "total_tokens": 0.0}
        return {
            "cost_usd": float(day_payload.get("cost_usd", 0.0)),
            "total_tokens": float(day_payload.get("total_tokens", 0.0)),
        }

    def _append_usage(self, total_tokens: int, cost_usd: float) -> None:
        today_key = str(date.today())
        self._usage_file.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if self._usage_file.exists():
            try:
                with self._usage_file.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {}
        day_payload = payload.get(today_key, {})
        if not isinstance(day_payload, dict):
            day_payload = {}
        day_payload["cost_usd"] = float(day_payload.get("cost_usd", 0.0)) + cost_usd
        day_payload["total_tokens"] = int(day_payload.get("total_tokens", 0)) + total_tokens
        day_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload[today_key] = day_payload
        with self._usage_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
        prices = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-3.5-turbo": (0.50, 1.50),
        }
        input_price, output_price = prices.get(model, (0.15, 0.60))
        return (
            prompt_tokens * input_price + completion_tokens * output_price
        ) / 1_000_000

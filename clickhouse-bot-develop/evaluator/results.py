"""Persist evaluation rows to CSV under the configured results folder."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import pandas as pd

from config import config


log = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Write tabular evaluation outputs."""

    @staticmethod
    def save(
        results: Sequence[Mapping[str, Any]],
        filename: str = "evaluation_results.csv",
    ) -> pd.DataFrame:
        """Save evaluation rows to CSV and return the in-memory DataFrame."""
        os.makedirs(config.results_folder, exist_ok=True)
        frame = pd.DataFrame(list(results))
        path = os.path.join(config.results_folder, filename)
        frame.to_csv(path, index=False, encoding="utf-8")
        log.info("Results saved to path=%s", path)
        return frame

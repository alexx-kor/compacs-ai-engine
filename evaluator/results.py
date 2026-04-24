"""Evaluation result persistence helpers."""

import logging
import os

import pandas as pd

from config import config

log = logging.getLogger(__name__)


class ResultsAnalyzer:
    @staticmethod
    def save(results: list, filename: str = "evaluation_results.csv") -> pd.DataFrame:
        os.makedirs(config.results_folder, exist_ok=True)
        df = pd.DataFrame(results)
        path = os.path.join(config.results_folder, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        log.info("results saved path=%s", path)
        return df

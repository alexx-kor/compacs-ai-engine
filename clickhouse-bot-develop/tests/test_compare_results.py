from __future__ import annotations

import pandas as pd
import pytest

from compare_results import compare_scores


def test_compare_scores_aligns_by_id() -> None:
    old_df = pd.DataFrame(
        [
            {"id": 1, "question": "Q1", "similarity_score": 0.5, "time_seconds": 1.0},
            {"id": 2, "question": "Q2", "similarity_score": 0.7, "time_seconds": 2.0},
        ]
    )
    new_data = [
        {"id": 1, "score": 0.8, "time": 1.5, "question": "Q1"},
        {"id": 2, "score": 0.6, "time": 2.5, "question": "Q2"},
    ]

    rows = compare_scores(old_df, new_data)

    assert len(rows) == 2
    by_id = {row["id"]: row for row in rows}
    assert by_id[1]["improved"] is True
    assert by_id[1]["difference"] == pytest.approx(0.3)
    assert by_id[2]["improved"] is False
    assert by_id[2]["difference"] == pytest.approx(-0.1)

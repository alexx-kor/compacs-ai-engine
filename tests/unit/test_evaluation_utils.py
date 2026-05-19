from __future__ import annotations

from core.evaluation_utils import is_not_found_answer, is_unanswerable_expected


def test_detects_not_found_variants() -> None:
    assert is_not_found_answer("Information not found in the current documentation index.")
    assert is_not_found_answer("NOT FOUND in documentation")
    assert not is_not_found_answer("SFTP host is 5.32.101.214")


def test_unanswerable_expected() -> None:
    expected = "Information not found in the current documentation index."
    assert is_unanswerable_expected(expected)


def test_structured_answer_with_partial_not_found_is_not_refusal() -> None:
    answer = (
        "1. Short answer\n- Типичная длительность спринта: две недели.\n\n"
        "2. API / Objects involved\n- Information not found in the current documentation index.\n"
    )
    assert not is_not_found_answer(answer)

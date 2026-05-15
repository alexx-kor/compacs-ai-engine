from __future__ import annotations

import logging

from core.logger import setup_logging


def test_setup_logging_is_idempotent() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    setup_logging(logging.INFO)
    handler_count_after_first = len(root.handlers)

    setup_logging(logging.DEBUG)
    handler_count_after_second = len(root.handlers)

    assert handler_count_after_first >= 1
    assert handler_count_after_second == handler_count_after_first
    assert root.level == logging.DEBUG

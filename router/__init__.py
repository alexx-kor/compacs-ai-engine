"""Prompt routing package."""

import logging

from router.smart_router import SmartPromptRouter, select_prompt

log = logging.getLogger(__name__)

__all__ = ["SmartPromptRouter", "select_prompt"]

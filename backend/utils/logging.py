"""Thin wrapper around loguru with sensible defaults.

Why loguru? Rich formatting, structured logs, no boilerplate, and a single
handler we can reconfigure from `AppConfig`.
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger


_INITIALIZED = False


def get_logger(name: Optional[str] = None, level: str = "INFO"):
    global _INITIALIZED
    if not _INITIALIZED:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            colorize=True,
            backtrace=False,
            diagnose=False,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> "
                "<level>{level: <7}</level> "
                "<cyan>{extra[name]}</cyan> | "
                "<level>{message}</level>"
            ),
        )
        _INITIALIZED = True
    return logger.bind(name=name or "cse")

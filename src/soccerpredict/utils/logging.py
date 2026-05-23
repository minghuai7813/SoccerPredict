"""Centralized logging built on top of `loguru`.

We expose a thin ``get_logger`` wrapper so the rest of the codebase does
not import loguru directly — making it easy to swap implementations later.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger as _logger

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _logger.remove()
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    _logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a configured loguru logger bound to the caller's name."""
    _configure()
    if name:
        return _logger.bind(name=name)
    return _logger

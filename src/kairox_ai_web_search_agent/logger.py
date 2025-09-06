"""Centralised logging configuration for the Kairox.ai agent.

This module provides a ``get_logger`` function returning a
pre-`configured Python ``logging.Logger``.  The logger uses a
human-`friendly formatter that includes timestamps and the log level.
By default, logs are emitted at the INFO level; set the environment
variable ``KAIRAOX_LOG_LEVEL`` to ``DEBUG`` for verbose output.
"""

from __future__ import annotations

import logging
import os
from logging import Logger


_LOGGER_NAME = "kairox_ai_web_search_agent"


def _create_logger() -> Logger:
    """Create and configure the root logger for this package."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # Already configured
        return logger

    # Determine log level from environment variable
    level_str = os.getenv("KAIRAOX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> Logger:
    """Return a child logger of the package logger or the root one.

    If ``name`` is provided, the returned logger will be a child of
    ``kairox_ai_web_search_agent.name``.  This ensures that log lines
    include the module name and respect the configured log level.
    """
    root = _create_logger()
    if name:
        return root.getChild(name)
    return root


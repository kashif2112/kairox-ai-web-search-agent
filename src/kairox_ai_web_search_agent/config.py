"""Configuration helpers for the Kairox.ai web search agent.

This module centralises the reading of environment variables and
exposes configuration values with sensible defaults.  Environment
variables are loaded using ``python-dotenv`` so that developers can
create a ``.env`` file at the root of the project for local
development without polluting the global environment.

If an expected variable is missing the module will raise a
``RuntimeError`` when accessed.  This explicit failure makes it clear
to the caller which secret or API key is missing, rather than
falling back to insecure defaults.

Usage:

>>> from kairox_ai_web_search_agent.config import NVIDIA_API_KEY
>>> print(NVIDIA_API_KEY)
"your-key"

The module also exposes ``FIRECRAWL_API_KEY`` and helper functions for
retrieving arbitrary environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except ImportError:
    # Provide a no-op shim if python-dotenv is not installed.  The shim
    # maintains compatibility when running in environments where
    # python-dotenv is unavailable.  Without it, an ImportError would
    # propagate at import time.
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None
    def find_dotenv(*args, **kwargs):  # type: ignore
        return ""


# Load environment variables from ``.env`` in the project root, if present.
# This call is idempotent and safe to run multiple times.
def _load_env() -> None:
    """Load environment variables from the nearest `.env`.

    Uses python-dotenv's `find_dotenv` to locate a `.env` starting from the
    current working directory and walking up. Falls back to the package
    directory if discovery fails.
    """
    # Prefer discovery from CWD upwards
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)
        return
    # Fallbacks: package dir and its parent
    here = Path(__file__).resolve().parent
    for env_path in (here / ".env", here.parent / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


_load_env()


def get_env(name: str, default: str | None = None) -> str | None:
    """Get an arbitrary environment variable with optional default.

    Ensures values from `.env` are loaded before reading. Returns the
    environment value or `default` if not set.
    """
    _load_env()
    return os.getenv(name, default)


def _require(var_name: str) -> str:
    """Return the value of environment variable ``var_name`` or raise.

    Parameters
    ----------
    var_name:
        The name of the environment variable to look up.

    Raises
    ------
    RuntimeError
        If the variable is not set in the environment.
    """
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {var_name} is not set. "
            "Define it in your shell or in a .env file."
        )
    return value


# Public configuration values
NVIDIA_API_KEY: str = _require("NVIDIA_API_KEY")
"""API key for the NVIDIA NIM ChatNVIDIA model.  Must be set in env."""

FIRECRAWL_API_KEY: str = _require("FIRECRAWL_API_KEY")
"""API key for Firecrawl SSE server.  Must be set in env."""

"""
Optional Tavily integration settings. MCP integration is removed; only the
direct Tavily client tool is supported when enabled.
Set ENABLE_TAVILY_CLIENT=true to expose a built-in `internet_search` tool
backed by Tavily's Python client (requires the `tavily-python` package).
"""

TAVILY_API_KEY: str | None = get_env("TAVILY_API_KEY")
ENABLE_TAVILY_CLIENT: bool = (get_env("ENABLE_TAVILY_CLIENT", "false") or "").lower() == "true"

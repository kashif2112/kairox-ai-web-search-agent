"""Model factory for Kairox.ai.

This module exposes functions to construct the ChatNVIDIA large language
model for use in the agent.  All parameters are configurable via
environment variables or function arguments.
"""

from __future__ import annotations

from typing import Optional

from .config import NVIDIA_API_KEY
from .logger import get_logger

from langchain_nvidia_ai_endpoints import ChatNVIDIA


log = get_logger(__name__)


def create_llm(
    api_key: Optional[str] = None,
    *,
    model_name: str = "moonshotai/kimi-k2-instruct",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_completion_tokens: int = 4096,
    extra_body: dict | None = None,
) -> ChatNVIDIA:
    """Instantiate a ChatNVIDIA model with sensible defaults.

    Parameters
    ----------
    api_key:
        The API key to authenticate with NVIDIA NIM.  If omitted,
        :data:`~kairox_ai_web_search_agent.config.NVIDIA_API_KEY` will
        be used.
    model_name:
        The fully qualified model name (default: kimi‑k2‐instruct).
    temperature:
        Randomness parameter for generation.
    top_p:
        Nucleus sampling parameter.
    max_completion_tokens:
        Maximum number of tokens to generate.
    extra_body:
        Additional fields to include in the API request body, such
        as chat template directives.  A default enabling ``thinking``
        mode is provided if not given.
    """
    api_key = api_key or NVIDIA_API_KEY
    if extra_body is None:
        extra_body = {"chat_template_kwargs": {"thinking": True}}
    log.debug(
        "Creating ChatNVIDIA model with model_name=%s, temperature=%s, top_p=%s",
        model_name,
        temperature,
        top_p,
    )
    return ChatNVIDIA(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        extra_body=extra_body,
    )

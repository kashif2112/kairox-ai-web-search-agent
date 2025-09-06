"""Top‑level package for Kairox.ai Web Search Agent.

This package provides a modular implementation of a DeepAgent powered by
LangChain and Firecrawl MCP tools.  It exposes factory functions for
constructing the agent, as well as utilities for launching a CLI or
embedding the agent into a larger application.

The repository is organised into logical modules:

* :mod:`config` manages environment variable loading
* :mod:`logger` initialises a structured logger
* :mod:`models` constructs the ChatNVIDIA language model
* :mod:`mcp_client` registers the Firecrawl SSE server with
  :class:`MultiServerMCPClient`
* :mod:`subagents` defines prompts and default subagent
  configuration
* :mod:`agent_factory` composes the model, tools and subagents
  into a ready‑to‑use agent
* :mod:`utils` provides helper functions for normalising strings and
  parsing JSON fragments
* :mod:`orchestrator` coordinates the planner → research → critic →
  final pipeline, streaming each subagent's output
* :mod:`cli` exposes a command‑line interface for interacting with the
  agent in a terminal

For an example of how to use this package, see the ``cli`` module or
read the README in the root of the repository.
"""

from importlib.metadata import version as _get_version

# Re-export key factories for convenience
from .agent_factory import create_agent  # noqa: F401
from .orchestrator import run_conversation  # noqa: F401

__all__ = [
    "create_agent",
    "run_conversation",
]

try:
    __version__ = _get_version(__name__)
except Exception:
    # Package not installed. Provide a sensible default.
    __version__ = "0.0.0"

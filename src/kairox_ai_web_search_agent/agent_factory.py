"""Factory functions for constructing the Kairox.ai agent.

This module provides a high‑level ``create_agent`` function that
assembles the ChatNVIDIA model, the MultiServerMCPClient with the
Firecrawl SSE server and the default subagents into a DeepAgent
instance ready for use.  The function is asynchronous because the
construction of the MCP client may involve network calls to fetch
available tools.
"""

from __future__ import annotations

from typing import Optional
import asyncio

from .logger import get_logger
from .config import NVIDIA_API_KEY, FIRECRAWL_API_KEY
from .models import create_llm
from .mcp_client import create_mcp_client
from .subagents import default_subagents
from .tools import build_local_tools

from deepagents import create_deep_agent

log = get_logger(__name__)


async def create_agent(
    *,
    nvidia_api_key: Optional[str] = None,
    firecrawl_api_key: Optional[str] = None,
) -> object:
    """Compose and return a DeepAgent instance with all dependencies.

    Parameters
    ----------
    nvidia_api_key:
        Override the NVIDIA API key.  If None, uses
        :data:`~kairox_ai_web_search_agent.config.NVIDIA_API_KEY`.
    firecrawl_api_key:
        Override the Firecrawl API key.  If None, uses
        :data:`~kairox_ai_web_search_agent.config.FIRECRAWL_API_KEY`.

    Returns
    -------
    object
        A configured DeepAgent with the Firecrawl tools registered.
    """
    # Create LLM
    log.info("Creating LLM model...")
    llm = create_llm(api_key=nvidia_api_key or NVIDIA_API_KEY)

    # Create MCP client (register Firecrawl server)
    log.info("Creating MCP client and registering Firecrawl server...")
    mcp_client = await create_mcp_client(
        firecrawl_api_key or FIRECRAWL_API_KEY,
    )

    # Fetch tools with a timeout so startup doesn't hang indefinitely.
    log.info("Fetching MCP tools. This may take a moment...")
    try:
        tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=25)
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        log.info(
            "Fetched %s MCP tool(s): %s", len(tools), ", ".join(tool_names) if tool_names else "No tools found"
        )

    except asyncio.TimeoutError:
        log.error(
            "Timed out fetching MCP tools. Proceeding without tools. "
            "Check network/firewall access to mcp.firecrawl.dev."
        )
        tools = []
    except Exception as exc:
        log.error(
            "Failed to fetch MCP tools (%s). Proceeding without tools.", exc
        )
        tools = []

    # Add optional built-in tools (e.g., Tavily client)
    local_tools = build_local_tools()
    if local_tools:
        log.info("Adding %s built-in tool(s).", len(local_tools))
        tools.extend(local_tools)

    # Require at least one tool (either MCP or built-in)
    if not tools:
        raise RuntimeError("No tools available. Check API keys, network, or enable built-in tools.")

    subagents = default_subagents()

    log.debug("Composing DeepAgent with %s tools and %s subagents", len(tools), len(subagents))
    agent = create_deep_agent(
        tools=tools,
        model=llm,
        instructions=(
            "You are a researcher/orchestrator with MCP tools (Firecrawl). "
            "When acting as a subagent, return only the requested artifact."
        ),
        subagents=subagents,
    )
    # Attach tool/servers status for UI diagnostics
    try:
        servers_cfg = getattr(mcp_client, "_servers_config", {})
        mcp_tool_names = [getattr(t, "name", str(t)) for t in tools if hasattr(t, "name")]
        builtin_tool_names = []
        for t in tools:
            name = getattr(t, "name", None)
            if not name:
                name = getattr(t, "__name__", None) or str(t)
            # Heuristic: built-in tools are callables without a fully qualified MCP name
            if isinstance(name, str) and not name.startswith("firecrawl"):
                builtin_tool_names.append(name)
        status = {
            "servers": servers_cfg,
            "mcp_tool_names": mcp_tool_names,
            "builtin_tool_names": builtin_tool_names,
        }
        setattr(agent, "_tool_status", status)
    except Exception:
        pass
    return agent

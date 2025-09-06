"""MCP client factory and tool registration.

This module constructs a :class:`~langchain_mcp_adapters.client.MultiServerMCPClient`
with the Firecrawl SSE server registered.  It hides the details of
the transport configuration and ensures that the appropriate API key
is injected into the SSE URL.

If the underlying version of ``langchain_mcp_adapters`` does not
support the ``servers`` argument on the constructor, a clear
``RuntimeError`` will be raised instructing the developer to upgrade.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from .config import FIRECRAWL_API_KEY
from .logger import get_logger

from langchain_mcp_adapters.client import MultiServerMCPClient


log = get_logger(__name__)

# Template for Firecrawl SSE URL.  The API key is inserted into the path.
FIRECRAWL_SSE_TEMPLATE = "https://mcp.firecrawl.dev/{api_key}/v2/sse"


async def create_mcp_client(
    firecrawl_api_key: Optional[str] = None,
) -> MultiServerMCPClient:
    """Create a ``MultiServerMCPClient`` with the Firecrawl server attached.

    Parameters
    ----------
    firecrawl_api_key:
        Override the Firecrawl API key.  If omitted,
        ``FIRECRAWL_API_KEY`` from :mod:`kairox_ai_web_search_agent.config` is used.

    Returns
    -------
    MultiServerMCPClient
        A client instance with the Firecrawl server available under the
        name ``firecrawl``.

    Raises
    ------
    RuntimeError
        If the installed version of ``langchain_mcp_adapters`` does not
        support the required transport registration method.
    """
    servers: Dict[str, Dict[str, Any]] = {}

    fc_key = firecrawl_api_key or FIRECRAWL_API_KEY
    if fc_key:
        fc_url = FIRECRAWL_SSE_TEMPLATE.format(api_key=fc_key)
        log.debug("Registering Firecrawl MCP server with URL %s", fc_url)
        servers["firecrawl"] = {"url": fc_url, "transport": "sse"}

    # Pass servers to constructor; fallback to raising if unsupported
    try:
        client = MultiServerMCPClient(servers)
        # Attach servers config for diagnostics in UI
        try:
            setattr(client, "_servers_config", servers)
        except Exception:
            pass
        # Do not fetch tools here; leave network I/O to the caller so
        # they can add timeouts or alternative handling.
        return client
    except Exception as exc:
        msg = (
            "Failed to register Firecrawl SSE server with MCP client. "
            "Ensure `langchain-mcp-adapters` is up to date and supports the"
            " `servers` constructor argument. Underlying error: %s"
        )
        log.error(msg, exc)
        raise RuntimeError(msg % exc) from exc

from __future__ import annotations

from typing import List

from .logger import get_logger
from .config import ENABLE_TAVILY_CLIENT, TAVILY_API_KEY


log = get_logger(__name__)


def _maybe_import_tool_decorator():
    try:
        from langchain_core.tools import tool  # type: ignore
        return tool
    except Exception:
        try:
            from langchain.tools import tool  # type: ignore
            return tool
        except Exception:
            def identity(x):
                return x
            return identity


def build_local_tools() -> List[object]:
    """Return a list of optional built-in tools (e.g., Tavily).

    Tools are only added if explicitly enabled via environment variables
    and corresponding API keys are present.
    """
    tools: List[object] = []

    if ENABLE_TAVILY_CLIENT and TAVILY_API_KEY:
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception as exc:
            log.warning("Tavily client not available: %s. Skipping built-in tool.", exc)
        else:
            tool = _maybe_import_tool_decorator()
            client = TavilyClient(api_key=TAVILY_API_KEY)

            @tool
            def internet_search(
                query: str,
                max_results: int = 5,
                topic: str = "general",
                include_raw_content: bool = False,
            ) -> dict:
                """Run a web search via Tavily. Returns JSON results.

                topic: one of 'general', 'news', 'finance'
                """
                return client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic,  # type: ignore[arg-type]
                )

            tools.append(internet_search)
            log.info("Enabled built-in Tavily internet_search tool.")

    return tools


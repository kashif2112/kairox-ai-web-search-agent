"""Command‑line interface for the Kairox.ai web search agent.

Run this module as a script to interactively chat with the agent in
your terminal.  The script loads configuration from environment
variables (see :mod:`kairox_ai_web_search_agent.config`), creates the
agent and orchestrates the conversation loop.  Errors during agent
construction or conversation are logged and printed to stderr.

Usage::

    $ python -m kairox_ai_web_search_agent.cli
    DeepAgent ready.  Type a question (or 'quit').
    > What is ARC-AGI?
    [planner-agent] ...
    ... final answer ...
"""

from __future__ import annotations

import asyncio
import sys
from typing import NoReturn

from .logger import get_logger
from .agent_factory import create_agent
from .orchestrator import run_conversation


log = get_logger(__name__)


async def _main() -> None:
    # Create the agent
    log.info("Initialising agent...")
    agent = await create_agent()
    print("Kairox.ai agent ready. Type a question (or 'quit').\n")

    # Conversation loop
    while True:
        try:
            q = input("Enter your question (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if q.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return
        if not q:
            print("Please enter a non-empty question.")
            continue
        try:
            await run_conversation(agent, q)
        except Exception as e:
            log.exception("Error during conversation: %s", e)
            print(f"[ERROR] {e}")


def main() -> NoReturn:
    try:
        asyncio.run(_main())
    except RuntimeError as exc:
        # Running from a running loop (e.g. inside IPython); fallback
        log.warning("Fallback to asyncio.create_task due to runtime error: %s", exc)
        loop = asyncio.get_event_loop()
        loop.create_task(_main())
        loop.run_forever()


if __name__ == "__main__":
    main()

# Kairox.ai Web Search Agent

This repository contains a modular implementation of a DeepAgent powered
by LangChain, the NVIDIA NIM ChatNVIDIA model and the Firecrawl MCP
tools.  The agent orchestrates a multi‑step workflow—planning,
research, critique and final synthesis—to answer user questions with
supporting evidence collected from the web.

## Structure

```
kairox_ai_web_search_agent/
  __init__.py     Package metadata
  config.py       Environment variable loading
  logger.py       Centralised logging configuration
  models.py       ChatNVIDIA model factory
  mcp_client.py   MultiServerMCPClient factory with Firecrawl SSE server
  subagents.py    Prompt definitions and default subagent configs
  agent_factory.pyFactory to assemble the agent
  utils.py        Helper functions for string normalisation and JSON parsing
  orchestrator.py Pipeline for planner→research→critic→final
  cli.py          Command–line entrypoint
  README.md       This file
```

## Usage

Install the required Python dependencies (LangChain MCP adapters and
NIM endpoints) in your own environment.  Then set the following
environment variables:

- `NVIDIA_API_KEY` – your API key for the ChatNVIDIA model.
- `FIRECRAWL_API_KEY` – your API key for the Firecrawl SSE server.
- `TAVILY_API_KEY` – your API key for Tavily client tool (optional)
- `ENABLE_TAVILY_CLIENT` – set to `true` to enable a built-in Tavily search tool (requires `tavily-python`)

Optionally create a `.env` file in the repository root containing these
keys.  The configuration loader will pick them up automatically when
running the CLI.

Run the agent interactively using the installed console script or module:

Option A (recommended, install in editable mode):

```bash
pip install -e .
kairox-agent
```

Option B (module run without install):

```bash
python -m kairox_ai_web_search_agent.cli
```

You will be prompted for a question. The agent streams each subagent’s
output to your terminal and produces a final answer with citations and
an action checklist.

## Streamlit UI

A chat-style UI with live reasoning and shimmer animation is included.

- Install (editable): `pip install -e .`
- Run the UI:
  - `streamlit run src/kairox_ai_web_search_agent/ui_streamlit.py`
  - or, after install: `kairox-agent-ui`

Features:
- Chat-like layout with user/assistant avatars
- Live streaming of planner/research/critic/main stages
- Shimmer effect displayed over reasoning text while streaming
- Sidebar with previous conversations for quick recall

MCP Servers
- Firecrawl (required): `https://mcp.firecrawl.dev/{apiKey}/v2/sse`

Optional built-in Tavily tool (non-MCP)
- Install: `pip install tavily-python`
- Enable: set `ENABLE_TAVILY_CLIENT=true` and `TAVILY_API_KEY=...`

## Extensibility

The design is modular to enable extension and customisation:

- **Subagents** – modify the prompts in `subagents.py` or add new
  subagent configurations.  You can adjust `model_settings` to tune
  temperature or token limits per subagent.
- **MCP client** – register additional tool servers in
  `mcp_client.py` by passing more entries into the `servers` dict.
- **Model** – swap out the NVIDIA NIM model for another provider by
  editing `models.py` and updating the factory in `agent_factory.py`.
- **Interface** – integrate the orchestrator into a web app or other
  UI by importing `create_agent` and `run_conversation`.

## Note

This repository assumes that the necessary external libraries
(`langchain_mcp_adapters` and `langchain_nvidia_ai_endpoints`) are
installed in your environment.  If they are missing, import errors will
occur when running the modules.  Install the packages via pip or
adjust the code to use your own tooling stack.

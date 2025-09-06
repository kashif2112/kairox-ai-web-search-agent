from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import streamlit as st

try:
    # Prefer package-relative imports when running as a module
    from .agent_factory import create_agent
    from .logger import get_logger
    from .orchestrator import run_conversation
except Exception:
    # Fallback for "streamlit run path/to/ui_streamlit.py" execution where
    # the package context isn't established.
    import os
    import sys
    # Ensure the repo's `src` directory is on sys.path so absolute imports work
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # points to `src`
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from kairox_ai_web_search_agent.agent_factory import create_agent  # type: ignore
    from kairox_ai_web_search_agent.logger import get_logger  # type: ignore
    from kairox_ai_web_search_agent.orchestrator import run_conversation  # type: ignore


log = get_logger(__name__)


@dataclass
class ConversationRecord:
    id: str
    question: str
    final: str
    timestamps: Dict[str, float]
    stages: Dict[str, str]


def _init_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "history" not in st.session_state:
        st.session_state.history: List[Dict] = []
    if "current_buffers" not in st.session_state:
        st.session_state.current_buffers: Dict[str, str] = {}
    if "seen_critic" not in st.session_state:
        st.session_state.seen_critic = False
    if "final_buffer" not in st.session_state:
        st.session_state.final_buffer = ""
    if "selected" not in st.session_state:
        st.session_state.selected = None


HISTORY_PATH = ".kairox_ui_history.json"


def _load_history() -> None:
    try:
        import json, os
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
    except Exception:
        pass


def _save_history() -> None:
    try:
        import json
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; }

        /* Ephemeral status chip with shimmer */
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background: #f8fafc;
            font-size: 0.9rem;
            color: #374151;
            position: relative;
            overflow: hidden;
        }
        .chip.shimmer::after {
            content: "";
            position: absolute;
            top: 0; left: -150%;
            width: 150%; height: 100%;
            background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(200,200,200,0.25) 50%, rgba(255,255,255,0) 100%);
            animation: shimmer 1.6s infinite;
        }
        @keyframes shimmer {
            0% { left: -150%; }
            100% { left: 150%; }
        }
        .dot {
            width: 8px; height: 8px; border-radius: 50%; background: #10b981;
            animation: pulse 1.4s ease-in-out infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.8); opacity: .7; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(0.8); opacity: .7; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


async def _ensure_agent() -> object:
    if st.session_state.agent is None:
        log.info("Initialising agent for Streamlit UI...")
        st.toast("Starting agent…", icon="🤖")
        st.session_state.agent = await create_agent()
        st.toast("Agent ready", icon="✅")
    return st.session_state.agent


def _render_sidebar(history: List[Dict]) -> None:
    st.sidebar.header("Conversations")
    if not history:
        st.sidebar.info("No conversations yet.")
        return
    for item in history[::-1]:  # newest first
        label = item.get("question", "Untitled")
        if st.sidebar.button(f"🗂️ {label[:48]}", key=f"hist-{item['id']}"):
            st.session_state.selected = item["id"]
            # Streamlit API change: prefer st.rerun() on newer versions
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()  # type: ignore[attr-defined]
                except Exception:
                    pass


def _render_tools_status() -> None:
    agent = st.session_state.get("agent")
    if not agent:
        return
    status = getattr(agent, "_tool_status", {}) or {}
    servers = status.get("servers", {})
    mcp_tools = status.get("mcp_tool_names", [])
    builtin_tools = status.get("builtin_tool_names", [])
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            if "firecrawl" in servers:
                st.markdown("<span class='chip'>✅ Firecrawl MCP</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='chip'>⚠️ Firecrawl MCP</span>", unsafe_allow_html=True)
        with cols[1]:
            if builtin_tools:
                st.markdown("<span class='chip'>✅ Built-in: " + ", ".join(builtin_tools) + "</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='chip'>— Built-in tools</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(
                f"<span class='chip'>Tools: {len(mcp_tools) + len(builtin_tools)}</span>",
                unsafe_allow_html=True,
            )


def _on_stream(role: str, text: str, status_placeholder: st.delta_generator.DeltaGenerator, labels: Dict[str, str]) -> None:
    # Show a single ephemeral chip for the current stage; it gets replaced by the next stage.
    if role not in st.session_state.current_buffers:
        st.session_state.current_buffers[role] = "seen"
        label = labels.get(role, role.title())
        status_placeholder.markdown(
            f"<span class='chip shimmer'><span class='dot'></span>{label}</span>",
            unsafe_allow_html=True,
        )


def app() -> None:
    st.set_page_config(page_title="Kairox Agent", page_icon="🤖", layout="wide")
    _inject_css()
    _init_state()
    _load_history()

    _render_sidebar(st.session_state.history)

    st.title("Kairox.ai Web Search Agent")
    st.caption("Chat-style interface with live reasoning and citations.")
    _render_tools_status()
    # Controls row
    col1, col2 = st.columns([1, 1])
    with col1:
        deep_research = st.toggle("Deep Research", value=False, help="Use Tavily internet_search tool")
    with col2:
        show_reasoning = st.toggle("Show reasoning", value=True)

    # Conversation area
    with st.container():
        user_prompt = st.chat_input("Ask me anything…")

        # Render selected history (if any)
        selected_id = st.session_state.get("selected")
        if selected_id:
            found = next((h for h in st.session_state.history if h["id"] == selected_id), None)
            if found:
                st.chat_message("user", avatar="🧑").markdown(found.get("question", ""))
                st.chat_message("assistant", avatar="🤖").markdown(found.get("final", ""))

        if user_prompt:
            st.chat_message("user", avatar="🧑").markdown(user_prompt)

            # A single status placeholder to show ephemeral stage labels
            status_ph = st.empty()
            # Optional reasoning panel (DeepSeek-style)
            reasoning_ph = st.empty() if show_reasoning else None
            reasoning_buffer = ""
            # Placeholder for streaming the final assistant answer
            assistant_stream_ph = st.chat_message("assistant", avatar="🤖").empty()
            stage_labels = {
                "planner-agent": "Thinking",
                "research-agent": "Researching",
                "critic-agent": "Reviewing",
                "main-agent": "Editing",
            }

            # Reset streaming buffers and flags
            st.session_state.current_buffers = {}
            st.session_state.seen_critic = False
            st.session_state.final_buffer = ""

            async def _run():
                agent = await _ensure_agent()

                timestamps = {"start": time.time()}

                def cb(role: str, text: str, kind: str) -> None:
                    # Stage chip updates
                    _on_stream(role, text, status_ph, stage_labels)
                    # Track critic stage; next main-agent stream is the final
                    if role == "critic":
                        st.session_state.seen_critic = True
                    # DeepSeek-style reasoning stream from candidate stage or reasoning channel
                    if role in ("reasoning",) and show_reasoning and isinstance(reasoning_ph, st.delta_generator.DeltaGenerator):
                        nonlocal reasoning_buffer
                        reasoning_buffer += text
                        reasoning_ph.markdown(f"<div class='chip shimmer'>🤔 Thinking…</div>\n\n{reasoning_buffer}", unsafe_allow_html=True)
                    # Stream only the final assistant answer tokens after critic
                    if role == "final" and kind == "content" and st.session_state.seen_critic:
                        # Once content begins, clear any reasoning panel
                        if isinstance(reasoning_ph, st.delta_generator.DeltaGenerator):
                            reasoning_ph.empty()
                        st.session_state.final_buffer += text
                        assistant_stream_ph.markdown(st.session_state.final_buffer)

                # Show immediate planner status
                status_ph.markdown(
                    f"<span class='chip shimmer'><span class='dot'></span>{stage_labels['planner-agent']}</span>",
                    unsafe_allow_html=True,
                )

                # Determine research preference based on toggle and tool availability
                prefer = "tavily" if deep_research and "internet_search" in (getattr(agent, "_tool_status", {}).get("builtin_tool_names", [])) else "firecrawl"

                result = await run_conversation(
                    agent, user_prompt, quiet=True, on_text=cb, research_preference=prefer
                )
                timestamps["end"] = time.time()

                # Clear the ephemeral status chip after completion
                status_ph.empty()

                # Show any warnings from the pipeline
                for w in result.get("warnings", []) or []:
                    st.warning(w)

                # Ensure final buffer shows the completed assistant message
                if not st.session_state.final_buffer:
                    st.session_state.final_buffer = result.get("final", "")
                    assistant_stream_ph.markdown(st.session_state.final_buffer)

                # Save history
                rec = ConversationRecord(
                    id=str(uuid.uuid4()),
                    question=user_prompt,
                    final=result.get("final", ""),
                    timestamps=timestamps,
                    stages={},
                )
                st.session_state.history.append(asdict(rec))
                _save_history()

            # Run async flow synchronously in Streamlit
            asyncio.run(_run())


def main() -> None:
    """Console entrypoint that boots a Streamlit server for this app."""
    try:
        from streamlit.web.bootstrap import run as st_run  # type: ignore
    except Exception:
        # Fallback: instruct the user to run via CLI
        print("Please run via: streamlit run src/kairox_ai_web_search_agent/ui_streamlit.py")
        return
    st_run(__file__, "", [], {})


if __name__ == "__main__":
    app()

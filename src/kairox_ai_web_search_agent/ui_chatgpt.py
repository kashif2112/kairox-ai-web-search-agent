from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List

import streamlit as st

try:
    # Prefer package-relative imports when running as a module
    from .agent_factory import create_agent
    from .logger import get_logger
    from .orchestrator import run_conversation
except Exception:
    # Fallback for "streamlit run path/to/ui_chatgpt.py" execution where
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
    messages: List[Dict[str, str]]


HISTORY_PATH = ".kairox_ui_history.json"


def _load_history() -> List[Dict]:
    try:
        import json, os
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(history: List[Dict]) -> None:
    try:
        import json
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; }
        .chip { display:inline-flex;align-items:center;gap:.5rem;padding:.25rem .65rem;border-radius:999px;border:1px solid #e5e7eb;background:#f8fafc;color:#374151;position:relative;overflow:hidden;font-size:.9rem;}
        .chip.shimmer::after{content:"";position:absolute;top:0;left:-150%;width:150%;height:100%;background:linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(200,200,200,0.25) 50%, rgba(255,255,255,0) 100%);animation:shimmer 1.6s infinite;}
        @keyframes shimmer{0%{left:-150%;}100%{left:150%;}}
        .dot{width:8px;height:8px;border-radius:50%;background:#10b981;animation:pulse 1.4s ease-in-out infinite;}
        @keyframes pulse{0%{transform:scale(.8);opacity:.7;}50%{transform:scale(1.1);opacity:1;}100%{transform:scale(.8);opacity:.7;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def app() -> None:
    st.set_page_config(page_title="Kairox Agent", page_icon="🤖", layout="wide")
    _inject_css()

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "history" not in st.session_state:
        st.session_state.history = _load_history()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected" not in st.session_state:
        st.session_state.selected = None
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "final_buffer" not in st.session_state:
        st.session_state.final_buffer = ""
    if "seen_critic" not in st.session_state:
        st.session_state.seen_critic = False

    # Sidebar (ChatGPT-like)
    st.sidebar.header("Conversations")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("New Chat"):
        st.session_state.messages = []
        st.session_state.selected = None
        st.rerun()
    if c2.button("Clear All"):
        st.session_state.history = []
        _save_history([])
        st.session_state.messages = []
        st.session_state.selected = None
        st.rerun()
    if st.session_state.history:
        for item in st.session_state.history[::-1]:
            label = item.get("question", "Untitled")
            if st.sidebar.button(label[:48], key=f"hist-{item['id']}"):
                st.session_state.selected = item["id"]
                st.session_state.messages = item.get("messages", [])
                st.rerun()
    else:
        st.sidebar.caption("No conversations yet.")

    # Header + controls
    st.title("Kairox.ai Web Search Agent")
    st.caption("Chat-style interface with live reasoning and citations.")
    col1, col2 = st.columns([1, 1])
    with col1:
        deep_research = st.toggle("Deep Research", value=False, help="Prefer Tavily internet_search tool")
    with col2:
        show_reasoning = st.toggle("Show reasoning", value=True)

    # Render stacked messages
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        st.chat_message("user" if role == "user" else "assistant", avatar="🧑" if role == "user" else "🤖").markdown(content)

    # Input
    user_prompt = st.chat_input("Ask me anything…")
    if not user_prompt:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user", avatar="🧑").markdown(user_prompt)

    # Holders
    status_ph = st.empty()
    reasoning_ph = st.empty() if show_reasoning else None
    reasoning_buffer = ""
    assistant_stream_ph = st.chat_message("assistant", avatar="🤖").empty()
    stage_labels = {
        "planner-agent": "Thinking",
        "research-agent": "Researching",
        "critic-agent": "Reviewing",
        "main-agent": "Editing",
    }

    # Reset state for run
    st.session_state.final_buffer = ""
    st.session_state.seen_critic = False
    st.session_state.stop_requested = False

    async def _ensure_agent() -> object:
        if st.session_state.agent is None:
            st.toast("Starting agent…", icon="🤖")
            st.session_state.agent = await create_agent()
            st.toast("Agent ready", icon="✅")
        return st.session_state.agent

    async def _run():
        agent = await _ensure_agent()
        timestamps = {"start": time.time()}

        def cb(role: str, text: str, kind: str) -> None:
            status_ph.markdown(
                f"<span class='chip shimmer'><span class='dot'></span>{stage_labels.get(role, role)}</span>",
                unsafe_allow_html=True,
            )
            if role == "critic":
                st.session_state.seen_critic = True
            if role == "reasoning" and show_reasoning and isinstance(reasoning_ph, st.delta_generator.DeltaGenerator):
                nonlocal reasoning_buffer
                reasoning_buffer += text
                reasoning_ph.markdown(f"<div class='chip shimmer'>🤔 Thinking…</div>\n\n{reasoning_buffer}", unsafe_allow_html=True)
            if role == "final" and kind == "content" and st.session_state.seen_critic:
                if isinstance(reasoning_ph, st.delta_generator.DeltaGenerator):
                    reasoning_ph.empty()
                st.session_state.final_buffer += text
                assistant_stream_ph.markdown(st.session_state.final_buffer)

        def should_stop() -> bool:
            return bool(st.session_state.get("stop_requested", False))

        # Status + stop
        status_ph.markdown(
            f"<span class='chip shimmer'><span class='dot'></span>{stage_labels['planner-agent']}</span>",
            unsafe_allow_html=True,
        )
        if st.button("Stop", type="secondary"):
            st.session_state.stop_requested = True

        prefer = "tavily" if deep_research and "internet_search" in (getattr(agent, "_tool_status", {}).get("builtin_tool_names", [])) else "firecrawl"

        try:
            result = await run_conversation(
                agent, user_prompt, quiet=True, on_text=cb, research_preference=prefer, should_stop=should_stop
            )
        except InterruptedError:
            status_ph.empty()
            if isinstance(reasoning_ph, st.delta_generator.DeltaGenerator):
                reasoning_ph.empty()
            st.info("Stopped by user.")
            return
        timestamps["end"] = time.time()

        status_ph.empty()
        for w in result.get("warnings", []) or []:
            st.warning(w)
        if not st.session_state.final_buffer:
            st.session_state.final_buffer = result.get("final", "")
            assistant_stream_ph.markdown(st.session_state.final_buffer)

        st.session_state.messages.append({"role": "assistant", "content": st.session_state.final_buffer})
        rec = ConversationRecord(
            id=str(uuid.uuid4()),
            question=user_prompt,
            final=st.session_state.final_buffer,
            timestamps=timestamps,
            messages=st.session_state.messages.copy(),
        )
        st.session_state.history.append(asdict(rec))
        _save_history(st.session_state.history)

    asyncio.run(_run())


if __name__ == "__main__":
    app()

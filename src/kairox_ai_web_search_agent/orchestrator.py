"""Conversation orchestrator for the Kairox.ai agent.

This module defines the end‑to‑end pipeline for answering a user
question using the DeepAgent.  The pipeline proceeds through
planner → research → candidate → critic → final stages, streaming
each subagent's response to the console (or any provided stream
handler).  The orchestrator also handles parsing JSON results and
gracefully falls back to raw text if parsing fails.

Consumers can import :func:`run_conversation` and call it with a
constructed agent and a user query.  For CLI usage, see
:mod:`kairox_ai_web_search_agent.cli`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Callable

from .logger import get_logger
from .utils import normalize_short, extract_json_substring, extract_first_step_description


log = get_logger(__name__)


async def stream_subagent(
    agent: Any,
    role_name: str,
    payload_text: str,
    *,
    quiet: bool = False,
    on_text: Optional[Callable[[str, str, str], None]] = None,
    tag: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """Stream a message to a subagent and return the combined output.

    This helper constructs a minimal role invocation to avoid leaking
    system prompts.  It filters tool call noise and removes
    self‑echoed instructions.  The return value is the concatenation
    of all streamed chunks.
    """
    role_invocation = (
        f"ROLE: {role_name}\n"
        "INSTRUCTIONS: Return only the requested output (JSON or plain "
        "text per task). DO NOT echo role text."
    )
    user_msg = {"role": "user", "content": role_invocation + "\nTASK:\n" + payload_text}

    normalized_role_invocation = normalize_short(role_invocation)
    # Import SUB_PROMPTS in a way that works both when running the package
    # as an installed module (python -m kairox_ai_web_search_agent.cli)
    # and when running directly from the repo (python cli.py).
    from .subagents import SUB_PROMPTS  # type: ignore
    normalized_registered_prompt = normalize_short(SUB_PROMPTS.get(role_name, ""), 300)

    collected: list[str] = []
    seen_norms: set[str] = set()
    if not quiet and on_text is None:
        print(f"\n[-> {role_name}] {payload_text[:240]}{'...' if len(payload_text) > 240 else ''}\n")
    async for chunk in agent.astream({"messages": [user_msg]}, stream_mode="values"):
        if should_stop and should_stop():
            raise InterruptedError("Stopped by user")
        texts: list[str] = []
        kinds: list[str] = []  # parallel to texts: 'content' or 'reasoning'
        if isinstance(chunk, dict) and "messages" in chunk:
            for m in chunk["messages"]:
                content = getattr(m, "content", None)
                if content:
                    texts.append(content)
                    kinds.append("content")
                rc = getattr(m, "additional_kwargs", {}).get("reasoning_content")
                if rc:
                    texts.append(rc)
                    kinds.append("reasoning")
        else:
            content = getattr(chunk, "content", None)
            if content:
                texts.append(content)
                kinds.append("content")
            rc = getattr(chunk, "additional_kwargs", {}).get("reasoning_content")
            if rc:
                texts.append(rc)
                kinds.append("reasoning")

        for t, kind in zip(texts, kinds or ["content"] * len(texts)):
            if should_stop and should_stop():
                raise InterruptedError("Stopped by user")
            t_norm = normalize_short(t)
            # Filter noise and echoes
            if any(marker in t for marker in ("<|tool_call", "<|tool_calls_section", "functions.write_todos")):
                continue
            if normalized_role_invocation and normalized_role_invocation in t_norm:
                continue
            if normalized_registered_prompt and normalized_registered_prompt in t_norm:
                continue
            if any(key in t.lower() for key in ("\"objective\"", "\"constraints\"", "objective:")):
                continue
            if re.match(r"^\s*i('ll| will) research", t.strip().lower()):
                continue
            if t_norm in seen_norms:
                continue
            if len(t.strip()) < 3:
                continue
            if on_text is not None:
                try:
                    on_text(tag or role_name, t, kind)
                except Exception:
                    pass
            elif not quiet:
                print(f"[{role_name}] {t}", end="", flush=True)
            collected.append(t)
            seen_norms.add(t_norm)
    if not quiet and on_text is None:
        print(f"\n\n[{role_name}] --- END ---\n")
    return "".join(collected)


async def run_conversation(
    agent: Any,
    user_question: str,
    *,
    quiet: bool = False,
    on_text: Optional[Callable[[str, str, str], None]] = None,
    research_preference: str = "firecrawl",  # 'firecrawl' or 'tavily'
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run the full pipeline for a user question.

    Returns a dictionary containing all intermediate and final outputs.  If
    ``quiet`` is True, suppresses streaming prints and only returns
    structured data.
    """
    if not quiet and on_text is None:
        print(f"\n===== NEW CONVERSATION =====\n[User] {user_question}\n")

    # 1) Planner
    planner_payload = json.dumps(
        {
            "objective": user_question,
            "constraints": [
                "include 3 concrete action steps if recommending usage",
            ],
        },
        indent=2,
    )
    planner_out = await stream_subagent(
        agent, "planner-agent", planner_payload, quiet=quiet, on_text=on_text, tag="planner", should_stop=should_stop
    )

    # Parse planner
    plan = None
    try:
        plan = json.loads(planner_out)
    except Exception:
        js = extract_json_substring(planner_out)
        if js:
            try:
                plan = json.loads(js)
            except Exception:
                plan = None

    research_focus: str
    if plan:
        if isinstance(plan, list) and plan:
            first = plan[0]
            research_focus = first.get("description") or json.dumps(first)
        elif isinstance(plan, dict):
            # Search for list
            desc: Optional[str] = None
            for v in plan.values():
                if isinstance(v, list) and v:
                    entry = v[0]
                    if isinstance(entry, dict):
                        desc = entry.get("description") or json.dumps(entry)
                        break
            research_focus = desc or extract_first_step_description(planner_out) or user_question
        else:
            research_focus = extract_first_step_description(planner_out) or user_question
    else:
        research_focus = extract_first_step_description(planner_out) or user_question

    # 2) Research
    pref = "tavily" if str(research_preference).lower().startswith("tav") else "firecrawl"
    research_instruction = (
        f"Research task (brief): {research_focus}\n\n"
        f"Runtime preference: {pref}.\n"
        "Return JSON ONLY as per your role schema. Include method.tools_used listing exact tool names used.\n"
        "Do NOT echo the planner/objective or any tool-call logs."
    )
    research_out_raw = await stream_subagent(
        agent, "research-agent", research_instruction, quiet=quiet, on_text=on_text, tag="research", should_stop=should_stop
    )

    research_json: Dict[str, Any] | None
    try:
        research_json = json.loads(research_out_raw)
    except Exception:
        js = extract_json_substring(research_out_raw)
        if js:
            try:
                research_json = json.loads(js)
            except Exception:
                research_json = None
        else:
            research_json = None
    if not research_json:
        # Fallback minimal structure
        research_json = {
            "answer": normalize_short(research_out_raw, 1000),
            "citations": [],
            "method": {"tools_used": []},
        }
    # Enforce tool usage: prefer Firecrawl; allow deep Tavily when requested
    warnings: list[str] = []
    tools_used = []
    try:
        mu = research_json.get("method", {}).get("tools_used", [])
        if isinstance(mu, list):
            tools_used = [str(x).lower() for x in mu]
    except Exception:
        tools_used = []
    # Basic evidence check: any citations/sources/evidence present
    has_evidence = bool(research_json.get("citations")) or bool(research_json.get("sources")) or bool(research_json.get("evidence")) or bool(research_json.get("evidence_table"))
    if not has_evidence:
        warnings.append("Research returned no citations/sources. Tools may be unavailable or prompts need tuning.")

    if pref == "tavily":
        if not any("internet_search" in t for t in tools_used):
            warnings.append("Deep Research requested, but Tavily internet_search tool was not used.")
    else:
        if not any("firecrawl" in t for t in tools_used):
            warnings.append("Research did not use Firecrawl tools. Check Firecrawl availability or prompts.")

    # 3) Candidate assembly
    assemble_payload = (
        "Using the structured research JSON below, assemble a clear, concise answer to the user's question.\n"
        "Guidelines:\n"
        "- Write a direct explanatory response (not pros/cons/checklists) unless the question explicitly asks for them.\n"
        "- Keep it clear and focused.\n"
        "- Add bracketed citation IDs like [S1], [S2] right after specific claims, using the 'citations' list if present, or map from 'sources'.\n"
        "- No planner JSON, no tool logs, no meta.\n\n"
        f"RESEARCH_JSON:\n{json.dumps(research_json, indent=2)}"
    )
    candidate_out = await stream_subagent(
        agent, "main-agent", assemble_payload, quiet=quiet, on_text=on_text, tag="reasoning", should_stop=should_stop
    )

    # 4) Critic
    critic_payload = json.dumps(
        {
            "candidate_answer": candidate_out,
            "research_summary": research_json,
        },
        indent=2,
    )
    critic_out_raw = await stream_subagent(
        agent, "critic-agent", critic_payload, quiet=quiet, on_text=on_text, tag="critic", should_stop=should_stop
    )

    critic_json: Dict[str, Any] | None
    try:
        critic_json = json.loads(critic_out_raw)
    except Exception:
        js = extract_json_substring(critic_out_raw)
        if js:
            try:
                critic_json = json.loads(js)
            except Exception:
                critic_json = None
        else:
            critic_json = None
    if not critic_json:
        critic_json = {
            "verdict": "REVISE",
            "fixes": ["Critic output unparsable — request a follow-up."],
            "notes": critic_out_raw[:400],
        }

    # 5) Final answer assembly
    final_payload = (
        "Produce the final user-facing answer. Apply critic fixes. Include explicit bracketed citations [Sx] for factual claims.\n"
        "Do NOT reprint planner JSON or tool-call logs. Avoid checklists unless the question asks for steps.\n\n"
        "Candidate:\n" + candidate_out + "\n\n"
        "Critic (parsed):\n" + json.dumps(critic_json, indent=2) + "\n\n"
        "Research (summary):\n" + json.dumps(
            {
                "short_answer": research_json.get("short_answer") or research_json.get("answer"),
                "top_sources": research_json.get("sources", [])[:3] or research_json.get("citations", [])[:3],
            },
            indent=2,
        )
    )
    final_out = await stream_subagent(
        agent, "main-agent", final_payload, quiet=quiet, on_text=on_text, tag="final", should_stop=should_stop
    )

    if not quiet and on_text is None:
        print("\n===== FINAL ANSWER =====\n")
        print(final_out)
        print("\n===== END CONVERSATION =====\n")

    return {
        "user_question": user_question,
        "planner": planner_out,
        "plan_parsed": plan,
        "research_raw": research_out_raw,
        "research_json": research_json,
        "candidate": candidate_out,
        "critic_raw": critic_out_raw,
        "critic_json": critic_json,
        "final": final_out,
        "warnings": warnings,
    }

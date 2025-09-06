"""Utility functions for the Kairox.ai agent.

This module collects helper functions used across the codebase, such
as functions for normalising whitespace and extracting the largest
JSON substring from a piece of text.  Keeping these utilities
together avoids duplication and clarifies their purpose.
"""

from __future__ import annotations

import json
import re
from typing import Optional


def normalize_short(s: str, max_chars: int = 300) -> str:
    """Return a lowercased, space-`normalised string truncated to ``max_chars``.

    This helper is used to compare strings for equality without
    concerning about extra whitespace or differences in capitalisation.
    """
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.split()).lower()[:max_chars]


def extract_json_substring(text: str) -> Optional[str]:
    """Attempt to extract the first JSON object or array from text.

    The function looks for the first ``{`` or ``[``, then tries to
    balance braces and brackets to isolate a valid JSON substring.  If
    such a substring is found it is returned; otherwise ``None`` is
    returned.  The caller should still ``json.loads`` the result to
    validate it.
    """
    if not text:
        return None
    start_idx = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start_idx = i
            break
    if start_idx is None:
        return None
    stack = []
    i = start_idx
    while i < len(text):
        ch = text[i]
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                candidate = text[start_idx : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    pass
        i += 1
    return None


def extract_first_step_description(planner_text: str) -> Optional[str]:
    """Extract the description of the first step from a planner JSON string.

    If ``planner_text`` contains a valid JSON array of step objects, the
    function returns the ``description`` field of the first step.  If
    parsing fails, a regex fallback is used to find a "description"
    field in any JSON object within the string.  As a last resort the
    function returns a truncated and normalised version of the input.
    """
    try:
        parsed = json.loads(planner_text)
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                return first.get("description") or json.dumps(first)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and v:
                    entry = v[0]
                    if isinstance(entry, dict):
                        return entry.get("description") or json.dumps(entry)
    except Exception:
        pass
    m = re.search(r'"description"\s*:\s*"([^"]+)"', planner_text)
    if m:
        return m.group(1)
    t = normalize_short(planner_text, 400)
    return (t[:350] + "...") if t else None


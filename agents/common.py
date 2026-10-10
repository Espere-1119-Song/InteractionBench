"""Helpers shared by the agent-harness runners (run_polling.py, run_grounding.py, run_mcq.py)."""
from __future__ import annotations

import os
import sys


def capability_roundrobin(items, cap=None, iid=None):
    cap = cap or (lambda it: it.get("capability") or "?")
    iid = iid or (lambda it: it.get("item_id") or "")
    groups: dict[str, list] = {}
    for it in sorted(items, key=iid):
        groups.setdefault(cap(it), []).append(it)
    queues = [groups[k] for k in sorted(groups)]
    out, i = [], 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
        i += 1
    return out


def check_agent_setup(harness, agent_home, mcp=True):
    problems = []
    if harness == "claude" and mcp and not os.path.exists(f"{agent_home}/.mcp.json"):
        problems.append(f"{agent_home}/.mcp.json not found "
                        "(copy agents/config/mcp.json to that path)")
    if harness == "gemini" and not os.path.exists(f"{agent_home}/.gemini/settings.json"):
        problems.append(f"{agent_home}/.gemini/settings.json not found "
                        "(copy agents/config/gemini_settings.json to that path)")
    if (mcp or harness != "claude") and not os.environ.get("QWEN_MM_PLUGINS_ROOT"):
        problems.append("QWEN_MM_PLUGINS_ROOT is not set "
                        "(path of the Qwen-MM-Plugins checkout)")
    if harness == "openai" and not os.environ.get("OAI_MODEL"):
        problems.append("OAI_MODEL is not set (model name on the endpoint)")
    if problems:
        sys.exit("agent setup incomplete:\n  " + "\n  ".join(problems)
                 + "\nsee agents/README.md")

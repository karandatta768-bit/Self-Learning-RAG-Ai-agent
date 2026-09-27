"""Decision agent: Jev-based tool/strategy selection."""
from __future__ import annotations

from typing import Any

from app.models.jev import get_jev
from app.tools.registry import get_registry

_ALL_TOOLS = ["rag", "web_search", "calculator", "none"]


def decide_tool(
    task: str,
    task_type: str,
    retrieval_context: str,
    prior_failure: str | None = None,
    *,
    exclude_tools: set[str] | None = None,
) -> dict[str, Any]:
    jev = get_jev()
    registry = get_registry()

    exclude = exclude_tools or set()
    options = [t for t in _ALL_TOOLS if t not in exclude] or ["none"]

    context_parts = [
        f"Task: {task}",
        f"Classified task_type: {task_type}",
        f"Available tools:\n{registry.describe()}",
    ]
    if retrieval_context:
        context_parts.append(f"Retrieved memory:\n{retrieval_context}")
    if prior_failure:
        context_parts.append(
            f"IMPORTANT: a previous attempt failed because: {prior_failure}. "
            f"Do NOT pick the same failing tool again."
        )

    resp = jev.choose(
        "Which single tool should the agent use next?",
        options,
        "\n\n".join(context_parts),
    )
    return {
        "next_tool": resp["choice"],
        "confidence": float(resp.get("confidence", 0.5)),
        "reason": "Jev choice over tool set",
    }


def decide_strategy(task_type: str, retrieval_context: str) -> dict[str, Any]:
    jev = get_jev()
    options = [
        "rag_then_answer", "web_then_verify",
        "calculator_direct", "clarify", "decline",
    ]
    resp = jev.choose(
        "Which strategy best matches this task?",
        options,
        f"task_type={task_type}\n\n{retrieval_context}",
    )
    return {"strategy": resp["choice"], "confidence": float(resp.get("confidence", 0.5))}
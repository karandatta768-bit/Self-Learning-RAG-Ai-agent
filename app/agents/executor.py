"""Executor: the only place tools are invoked."""
from __future__ import annotations

from typing import Any

from app.models.llm import get_llm
from app.tools.registry import get_registry


def execute(
    tool_name: str,
    task: str,
    *,
    expression_hint: str = "",
) -> dict[str, Any]:
    registry = get_registry()
    tool = registry.get(tool_name)

    if tool_name == "calculator":
        expr = expression_hint.strip() or _extract_expression_via_llm(task)
        kwargs: dict[str, Any] = {"expression": expr}
    else:
        kwargs = {"query": task}

    result = tool.run(**kwargs)

    # Hard gate: empty output from a "successful" tool is a failure.
    if result.success and tool_name != "none" and not (result.output or "").strip():
        result.success = False
        result.error = "Empty output."
        result.confidence = 0.0

    return {
        "tool": tool_name,
        "input": kwargs,
        "output": result.output,
        "success": result.success,
        "error": result.error,
        "sources": result.sources,
        "confidence": result.confidence,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
    }


_EXPR_SYS = (
    "Extract a single arithmetic expression from the task. "
    'Return strict JSON: {"expression": "..."}. '
    "Use only numbers and + - * / % ( ) ** and functions sqrt, log, exp, abs, round, min, max."
)


def _extract_expression_via_llm(task: str) -> str:
    try:
        data = get_llm().chat_json(
            [
                {"role": "system", "content": _EXPR_SYS},
                {"role": "user", "content": task},
            ]
        )
        expr = str(data.get("expression") or "").strip()
        return expr or task
    except Exception:
        return task
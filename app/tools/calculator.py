"""Deterministic calculator. AST evaluation, no eval()."""
from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from app.tools.base import Tool, ToolResult

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}

_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        return fn(*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Deterministic arithmetic: numbers, +, -, *, /, %, **, //, "
        "abs, round, min, max, sqrt, log, exp. Use this instead of doing math yourself."
    )
    input_schema = {"expression": "string, e.g. '(120 - 90) / 90 * 100'"}

    def run(self, expression: str, **_: Any) -> ToolResult:
        t0 = time.time()
        try:
            tree = ast.parse(expression, mode="eval")
            value = _safe_eval(tree)
            out = repr(value) if not isinstance(value, float) else f"{value:.10g}"
            return ToolResult(
                output=f"{expression} = {out}",
                confidence=1.0,
                latency_s=time.time() - t0,
            )
        except Exception as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Calculator error: {e}",
                confidence=0.0,
                latency_s=time.time() - t0,
            )
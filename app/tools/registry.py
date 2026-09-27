from __future__ import annotations

from app.tools.base import Tool, ToolResult
from app.tools.calculator import CalculatorTool
from app.tools.rag_tool import RAGTool
from app.tools.web_search import WebSearchTool


class _NullTool(Tool):
    name = "none"
    description = "Use when no tool is needed; the LLM can answer directly."

    def run(self, **_: object) -> ToolResult:
        return ToolResult(output="", confidence=1.0)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {
            "none": _NullTool(),
            "rag": RAGTool(),
            "web_search": WebSearchTool(),
            "calculator": CalculatorTool(),
        }

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def describe(self) -> str:
        return "\n".join(f"- {n}: {t.description}" for n, t in self._tools.items())


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    output: str
    success: bool = True
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...
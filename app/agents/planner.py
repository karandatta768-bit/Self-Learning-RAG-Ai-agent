"""Planner: structured task representation + task_type. Also extracts math expressions."""
from __future__ import annotations

from typing import Any

from app.memory.schemas import safe_task_type
from app.models.llm import get_llm

_SYSTEM = """You are a planning module for an AI agent.
Return strict JSON with this exact shape:
Return strict JSON only — no prose before or after. Do not narrate your reasoning.
{
  "task_type": "research" | "rag" | "calculation" | "ambiguous" | "out_of_scope" | "multi_step",
  "requirements": ["..."],
  "sub_questions": ["..."],
  "expression": "arithmetic expression if task_type is calculation, else empty string",
  "notes": "short rationale"
}
Rules:
- "rag": questions about internal docs, policies, or indexed knowledge.
- "research": questions needing current external information.
- "calculation": numeric/arithmetic.
- "multi_step": genuinely needs more than one of the above.
- "ambiguous": underspecified.
- "out_of_scope": not answerable with available tools.
For "expression" use only: numbers, + - * / % ( ) ** and functions sqrt log exp abs round min max.
"""


def plan(task: str) -> dict[str, Any]:
    llm = get_llm()
    data = llm.chat_json(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Task: {task}"},
        ]
    )
    data["task_type"] = safe_task_type(data.get("task_type"))
    data.setdefault("requirements", [])
    data.setdefault("sub_questions", [])
    data.setdefault("expression", "")
    data.setdefault("notes", "")
    return data

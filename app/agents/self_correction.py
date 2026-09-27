"""Self-correction policy. Chooses ONE lever per failure."""
from __future__ import annotations

from typing import Any

from app.models.jev import get_jev


def choose_correction(
    task: str,
    failed_tool: str,
    failure_reason: str,
    attempt: int,
) -> dict[str, Any]:
    jev = get_jev()
    options = ["rephrase_query", "switch_tool", "give_up"]
    context = (
        f"Task: {task}\n"
        f"Failed tool: {failed_tool}\n"
        f"Failure reason: {failure_reason}\n"
        f"Attempt number: {attempt}\n"
        "Which single correction should the agent try next?"
    )
    resp = jev.choose("Choose the next correction action.", options, context)
    return {
        "action": resp["choice"],
        "confidence": float(resp.get("confidence", 0.5)),
    }


def rephrase(task: str, failure_reason: str) -> str:
    from app.models.llm import get_llm

    llm = get_llm()
    data = llm.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the user task into a better query for a tool. "
                    'Return strict JSON: {"query": "..."}'
                ),
            },
            {
                "role": "user",
                "content": f"Original task: {task}\nPrevious failure: {failure_reason}",
            },
        ]
    )
    return str(data.get("query") or task)
"""Short-term memory = the LangGraph state."""
from __future__ import annotations

import uuid
from typing import Annotated, Any, TypedDict


def _append(old: list, new: list) -> list:
    return (old or []) + (new or [])


class AgentState(TypedDict, total=False):
    # --- input ---
    original_task: str
    task: str
    run_id: str
    task_id: str
    # benchmark-scoped, top-level so node_understand can't clobber them
    heldout_task_ids: list[str]
    isolate_run: bool

    task_type: str
    task_representation: dict[str, Any]

    # --- retrieval ---
    retrieved_experiences: list[dict[str, Any]]
    retrieved_knowledge: list[dict[str, Any]]

    # --- decision ---
    plan: dict[str, Any]
    strategy: str
    next_tool: str
    decision_confidence: float
    decision_reason: str

    # --- execution ---
    tool_results: Annotated[list[dict[str, Any]], _append]
    steps: Annotated[list[dict[str, Any]], _append]

    # --- evaluation / correction ---
    evaluation: dict[str, Any]
    correction_history: Annotated[list[dict[str, Any]], _append]
    attempt: int
    rephrased_task: str

    # --- output ---
    final_answer: str
    trace: Annotated[list[dict[str, Any]], _append]
    error: str

    # --- accounting ---
    total_cost_usd: float
    total_latency_s: float


def new_state(
    task: str,
    *,
    run_id: str | None = None,
    task_id: str = "",
    heldout_task_ids: list[str] | None = None,
    isolate_run: bool = False,
) -> AgentState:
    return AgentState(
        original_task=task,
        task=task,
        run_id=run_id or uuid.uuid4().hex,
        task_id=task_id,
        heldout_task_ids=list(heldout_task_ids or []),
        isolate_run=bool(isolate_run),
        task_type="",
        task_representation={},
        retrieved_experiences=[],
        retrieved_knowledge=[],
        plan={},
        strategy="",
        next_tool="",
        decision_confidence=0.0,
        decision_reason="",
        tool_results=[],
        steps=[],
        evaluation={},
        correction_history=[],
        attempt=1,
        rephrased_task="",
        final_answer="",
        trace=[],
        error="",
        total_cost_usd=0.0,
        total_latency_s=0.0,
    )
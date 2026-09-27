from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph.workflow import run
from app.memory.episodic import get_episodic
from app.memory.retriever import retrieve_for_task
from app.memory.strategy import get_strategy

router = APIRouter()


class RunRequest(BaseModel):
    task: str
    run_id: str | None = None


class RunResponse(BaseModel):
    final_answer: str
    task_type: str
    strategy: str
    tools: list[str]
    evaluation: dict
    attempts: int
    total_cost_usd: float
    total_latency_s: float
    trace: list[dict]


@router.post("/run", response_model=RunResponse)
def run_task(req: RunRequest) -> RunResponse:
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")
    # Production: shared run_id="prod", isolate_run=False → agent learns across calls.
    state = run(req.task, run_id=req.run_id or "prod", isolate_run=False)
    return RunResponse(
        final_answer=state.get("final_answer", ""),
        task_type=state.get("task_type", ""),
        strategy=state.get("strategy", ""),
        tools=[r["tool"] for r in state.get("tool_results", [])],
        evaluation=state.get("evaluation", {}),
        attempts=state.get("attempt", 1),
        total_cost_usd=state.get("total_cost_usd", 0.0),
        total_latency_s=state.get("total_latency_s", 0.0),
        trace=state.get("trace", []),
    )


@router.get("/memory/episodes")
def list_episodes(limit: int = 20) -> list[dict]:
    eps = list(get_episodic().all())[-limit:]
    return [e.model_dump() for e in eps]


@router.get("/memory/strategies")
def list_strategies(run_id: str = "prod") -> list[dict]:
    out: list[dict] = []
    for tt in ["research", "rag", "calculation", "ambiguous", "out_of_scope", "multi_step"]:
        out.extend([s.model_dump() for s in get_strategy().all_for(tt, run_id=run_id)])
    return out


@router.get("/memory/retrieve")
def retrieve(q: str) -> dict:
    bundle = retrieve_for_task(q)
    return {
        "experiences": [e.model_dump() for e in bundle.experiences],
        "knowledge": [k.model_dump() for k in bundle.knowledge],
        "strategies": [s.model_dump() for s in bundle.strategies],
    }


@router.get("/health")
def health() -> dict:
    return {"ok": True}
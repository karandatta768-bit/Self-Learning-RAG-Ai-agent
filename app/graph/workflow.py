"""LangGraph workflow with leakage guards and post-generation verification.

Topology:
    understand -> retrieve -> plan -> decide -> execute -> evaluate
                                                    |
                                          success / failure / exhausted
                                                    v
                                        finalize -- post_evaluate -- END
                                                 ^
                                                 |
                                              correct
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.decision_agent import decide_strategy, decide_tool
from app.agents.executor import execute
from app.agents.planner import plan as plan_task
from app.agents.runtime_evaluator import evaluate_final_answer, evaluate_tool_output
from app.agents.self_correction import choose_correction, rephrase
from app.config import get_settings
from app.graph.state import AgentState, new_state
from app.memory.retriever import retrieve_for_task, store_episode
from app.memory.schemas import EpisodeRecord, EvaluationRecord, safe_task_type
from app.models.llm import get_llm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _snapshot_llm() -> tuple[float, float]:
    llm = get_llm()
    return llm.total_cost_usd, llm.total_latency_s


def _delta(state: AgentState, before: tuple[float, float]) -> dict[str, float]:
    llm = get_llm()
    dc = llm.total_cost_usd - before[0]
    dl = llm.total_latency_s - before[1]
    return {
        "total_cost_usd": state.get("total_cost_usd", 0.0) + dc,
        "total_latency_s": state.get("total_latency_s", 0.0) + dl,
    }


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

def node_understand(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    rep = plan_task(state["task"])
    updates: dict[str, Any] = {
        "task_type": safe_task_type(rep["task_type"]),
        "task_representation": rep,
        "trace": [{"node": "understand", "task_type": rep["task_type"], "t": time.time()}],
    }
    updates.update(_delta(state, before))
    return updates


def node_retrieve(state: AgentState) -> dict[str, Any]:
    # heldout_task_ids is a TOP-LEVEL field — node_understand can't erase it.
    heldout = set(state.get("heldout_task_ids") or [])
    # exclude current run's own episodes ONLY when explicitly isolated
    exclude_run = state.get("run_id") if state.get("isolate_run") else None

    bundle = retrieve_for_task(
        state["task"],
        state.get("task_type") or None,
        run_id=state.get("run_id", "prod"),
        exclude_run_id=exclude_run,
        exclude_task_ids=heldout or None,
    )
    return {
        "retrieved_experiences": [e.model_dump() for e in bundle.experiences],
        "retrieved_knowledge": [k.model_dump() for k in bundle.knowledge],
        "trace": [
            {
                "node": "retrieve",
                "experiences": len(bundle.experiences),
                "knowledge": len(bundle.knowledge),
                "strategies": len(bundle.strategies),
                "excluded_run": exclude_run,
                "excluded_heldout_n": len(heldout),
                "t": time.time(),
            }
        ],
    }


def node_plan(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    ctx = _retrieval_context(state)
    strat = decide_strategy(state.get("task_type", "ambiguous"), ctx)
    updates: dict[str, Any] = {
        "strategy": strat["strategy"],
        "plan": {"strategy": strat["strategy"], "confidence": strat["confidence"]},
        "trace": [{"node": "plan", "strategy": strat["strategy"], "t": time.time()}],
    }
    updates.update(_delta(state, before))
    return updates


def node_decide(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    ctx = _retrieval_context(state)
    history = state.get("correction_history") or []
    prior = history[-1] if history else None
    prior_failure = prior["reason"] if prior else None
    exclude = {prior["failed_tool"]} if prior and prior.get("failed_tool") else set()

    d = decide_tool(
        state["task"],
        state.get("task_type", "ambiguous"),
        ctx,
        prior_failure,
        exclude_tools=exclude,
    )
    updates: dict[str, Any] = {
        "next_tool": d["next_tool"],
        "decision_confidence": d["confidence"],
        "decision_reason": d["reason"],
        "trace": [{"node": "decide", "tool": d["next_tool"], "confidence": d["confidence"], "t": time.time()}],
    }
    updates.update(_delta(state, before))
    return updates


def node_execute(state: AgentState) -> dict[str, Any]:
    tool = state.get("next_tool") or "none"
    expr_hint = str(state.get("task_representation", {}).get("expression") or "")
    result = execute(tool, state["task"], expression_hint=expr_hint)
    return {
        "tool_results": [result],
        "steps": [{"tool": tool, "success": result["success"]}],
        "trace": [{"node": "execute", "tool": tool, "success": result["success"], "t": time.time()}],
    }


def node_evaluate(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    last = state["tool_results"][-1]

    if last["tool"] == "none":
        # No tool required; defer grounding to post_evaluate.
        verdict = {
            "success": True,
            "score": 0.5,
            "sufficiency": 0.5,
            "tool_groundedness": 0.5,
            "verified": 0.5,
            "reason": "no tool required",
        }
    else:
        verdict = evaluate_tool_output(
            task=state["task"],
            task_type=state.get("task_type", "ambiguous"),
            tool_name=last["tool"],
            tool_output=last["output"],
            sources=last.get("sources", []),
        )

    updates: dict[str, Any] = {
        "evaluation": {**state.get("evaluation", {}), **verdict},
        "trace": [{"node": "evaluate", **verdict, "t": time.time()}],
    }
    updates.update(_delta(state, before))
    return updates


def node_correct(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    last = state["tool_results"][-1]
    decision = choose_correction(
        task=state["original_task"],
        failed_tool=last["tool"],
        failure_reason=state["evaluation"].get("reason", ""),
        attempt=state.get("attempt", 1),
    )
    action = decision["action"]

    updates: dict[str, Any] = {
        "attempt": state.get("attempt", 1) + 1,
        "correction_history": [
            {
                "attempt": state.get("attempt", 1),
                "action": action,
                "reason": state["evaluation"].get("reason", ""),
                "failed_tool": last["tool"],
            }
        ],
        "trace": [{"node": "correct", "action": action, "t": time.time()}],
    }

    if action == "rephrase_query":
        new_task = rephrase(state["task"], state["evaluation"].get("reason", ""))
        updates["task"] = new_task
        updates["rephrased_task"] = new_task
        updates["next_tool"] = last["tool"]
    elif action == "switch_tool":
        d = decide_tool(
            state["task"],
            state.get("task_type", "ambiguous"),
            _retrieval_context(state),
            prior_failure=state["evaluation"].get("reason", ""),
            exclude_tools={last["tool"]},
        )
        updates["next_tool"] = d["next_tool"]
        updates["decision_confidence"] = d["confidence"]

    updates.update(_delta(state, before))
    return updates


def node_finalize(state: AgentState) -> dict[str, Any]:
    before = _snapshot_llm()
    llm = get_llm()
    ev = state.get("evaluation", {})
    has_tool_evidence = any(
        r.get("output", "").strip() for r in state.get("tool_results", [])
    )

    if not has_tool_evidence or ev.get("sufficiency", 0.0) < 0.4:
        context = _retrieval_context(state, max_chars=3000)
        answer = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "I could not gather sufficient evidence to answer reliably. "
                        "Answer with explicit caveats. Do NOT invent facts. "
                        "If you don't know, say so plainly. "
                        "Output ONLY the final answer. Do not narrate your reasoning. "
                        "No preamble, no thinking-out-loud, no repetition. Start directly with the answer."
                    ),
                },
                {"role": "user", "content": f"Task: {state['original_task']}\nContext:\n{context}"},
            ]
        )
    else:
        tool_output = "\n\n".join(
            f"[{r['tool']}] {r['output']}" for r in state.get("tool_results", []) if r.get("output")
        )
        context = _retrieval_context(state, max_chars=6000)
        answer = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the final-answer writer for an AI agent. "
                        "Answer using ONLY the provided tool output and knowledge. "
                        "Do not invent facts. If evidence is insufficient, say so plainly. "
                        "Cite sources where available. "
                        "Output ONLY the final answer. Do not narrate your reasoning. "
                        "No preamble, no thinking-out-loud, no repetition. Start directly with the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {state['original_task']}\n\n"
                        f"Tool output:\n{tool_output}\n\nContext:\n{context}"
                    ),
                },
            ]
        )

    updates: dict[str, Any] = {
        "final_answer": answer,
        "trace": [{"node": "finalize", "t": time.time()}],
    }
    updates.update(_delta(state, before))
    return updates


def node_post_evaluate(state: AgentState) -> dict[str, Any]:
    """Grade the final answer against the sources, merge, and persist."""
    before = _snapshot_llm()
    last = state["tool_results"][-1] if state.get("tool_results") else {}
    verdict = evaluate_final_answer(
        task=state["original_task"],
        answer=state.get("final_answer", ""),
        tool_output=last.get("output", ""),
        sources=last.get("sources", []),
        tool_name=last.get("tool", ""),
    )
    merged = {**state.get("evaluation", {}), **verdict}

    tool_success = bool(state.get("evaluation", {}).get("success", False))
    final_success = tool_success and verdict["answer_groundedness"] >= 0.5

    merged["success"] = final_success
    merged["reason"] = (
        f"{state.get('evaluation', {}).get('reason', '')} | "
        f"answer_grounded={verdict['answer_groundedness']:.2f} "
        f"answer_relevant={verdict['answer_relevant']:.2f}"
    )

    total_cost = state.get("total_cost_usd", 0.0) + (get_llm().total_cost_usd - before[0])
    total_lat = state.get("total_latency_s", 0.0) + (get_llm().total_latency_s - before[1])

    ep = EpisodeRecord(
        id=uuid.uuid4().hex,
        task=state["original_task"],
        rephrased_task=state.get("rephrased_task", ""),
        task_id=state.get("task_id", "") or "",
        run_id=state.get("run_id", "prod"),
        task_type=safe_task_type(state.get("task_type", "ambiguous")),  # type: ignore[arg-type]
        strategy=state.get("strategy", "default"),
        tools=[r["tool"] for r in state.get("tool_results", [])],
        steps=state.get("steps", []),
        evaluation=EvaluationRecord(
            success=final_success,
            score=float(state.get("evaluation", {}).get("score", 0.0)),
            sufficiency=float(state.get("evaluation", {}).get("sufficiency", 0.0)),
            tool_groundedness=float(state.get("evaluation", {}).get("tool_groundedness", 0.0)),
            answer_groundedness=float(verdict["answer_groundedness"]),
            answer_relevant=float(verdict["answer_relevant"]),
            verified=float(state.get("evaluation", {}).get("verified", 0.0)),
            reason=merged["reason"],
        ),
        confidence=float(state.get("decision_confidence", 0.0)),
        attempts=int(state.get("attempt", 1)),
        cost_usd=total_cost,
        latency_s=total_lat,
    )

    try:
        store_episode(ep)
        stored = ep.id
    except Exception as e:
        log.exception("Failed to persist episode: %s", e)
        stored = ""

    return {
        "evaluation": merged,
        "total_cost_usd": total_cost,
        "total_latency_s": total_lat,
        "trace": [{"node": "post_evaluate", **verdict, "stored_episode": stored, "t": time.time()}],
    }


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def route_after_evaluate(state: AgentState) -> str:
    ev = state.get("evaluation", {})
    if ev.get("success"):
        return "finalize"
    attempt = state.get("attempt", 1)
    if attempt >= get_settings().max_retries + 1:
        return "finalize"
    return "correct"


def route_after_correct(state: AgentState) -> str:
    history = state.get("correction_history") or []
    if history and history[-1].get("action") == "give_up":
        return "finalize"
    return "execute"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _retrieval_context(state: AgentState, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for e in state.get("retrieved_experiences", [])[:3]:
        lines.append(
            f"- past: {e.get('task_type')} :: {e.get('strategy')} "
            f"success={e.get('evaluation', {}).get('success')}"
        )
    for k in state.get("retrieved_knowledge", [])[:3]:
        lines.append(f"- doc [{k.get('source', '')}]: {k.get('text', '')[:300]}")
    return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("understand", node_understand)
    g.add_node("retrieve", node_retrieve)
    g.add_node("plan", node_plan)
    g.add_node("decide", node_decide)
    g.add_node("execute", node_execute)
    g.add_node("evaluate", node_evaluate)
    g.add_node("correct", node_correct)
    g.add_node("finalize", node_finalize)
    g.add_node("post_evaluate", node_post_evaluate)

    g.set_entry_point("understand")
    g.add_edge("understand", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "decide")
    g.add_edge("decide", "execute")
    g.add_edge("execute", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"finalize": "finalize", "correct": "correct"},
    )
    g.add_conditional_edges(
        "correct",
        route_after_correct,
        {"execute": "execute", "finalize": "finalize"},
    )
    g.add_edge("finalize", "post_evaluate")
    g.add_edge("post_evaluate", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(
    task: str,
    *,
    run_id: str | None = None,
    task_id: str = "",
    heldout_task_ids: list[str] | None = None,
    isolate_run: bool = False,
) -> AgentState:
    graph = get_graph()
    state = new_state(
        task,
        run_id=run_id,
        task_id=task_id,
        heldout_task_ids=heldout_task_ids,
        isolate_run=isolate_run,
    )
    return graph.invoke(state)  # type: ignore[return-value]

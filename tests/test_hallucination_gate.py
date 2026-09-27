"""Verify the pipeline refuses fabrications and grounds calculator answers."""
import pytest

from app.agents.runtime_evaluator import _calculator_grounding, evaluate_final_answer
from app.graph.workflow import run


def test_calculator_grounding_deterministic():
    assert _calculator_grounding("15 * 240 / 100 = 36", "15% of 240 is 36.") == 1.0
    assert _calculator_grounding("15 * 240 / 100 = 36", "15% of 240 is 42.") == 0.0
    assert _calculator_grounding("", "any answer") == 0.0


def test_calculator_grounding_via_evaluate_final_answer():
    v = evaluate_final_answer(
        task="What is 15% of 240?",
        answer="15% of 240 is 36.",
        tool_output="15 * 240 / 100 = 36",
        sources=[],
        tool_name="calculator",
    )
    assert v["answer_groundedness"] == 1.0
    assert v["answer_relevant"] == 1.0


@pytest.mark.slow
def test_refuses_when_no_evidence():
    state = run(
        "What is our refund policy for the enterprise platinum tier?",
        run_id="test-halluc",
        isolate_run=True,
    )
    answer = (state.get("final_answer") or "").lower()
    ev = state.get("evaluation", {})
    hedged = any(
        phrase in answer
        for phrase in ["could not gather", "insufficient", "not enough", "unable to", "no evidence"]
    )
    assert (ev.get("success") is False) or hedged, (
        f"Expected refusal/hedge, got: {answer[:200]!r}"
    )


@pytest.mark.slow
def test_calculator_success_path():
    state = run("What is 15% of 240?", run_id="test-calc", isolate_run=True)
    ev = state.get("evaluation", {})
    assert ev.get("answer_groundedness", 0.0) == 1.0
    assert ev.get("success") is True, f"calc task failed: {ev}"
    assert "36" in state.get("final_answer", "")
"""Runtime evaluator: Jev-only composite verdict. Two stages:
   - evaluate_tool_output(): grades the raw tool output
   - evaluate_final_answer(): grades the LLM's answer against the sources
Includes a deterministic grounding check for calculator answers.
"""
from __future__ import annotations

import re
from typing import Any

from app.models.jev import get_jev


def evaluate_tool_output(
    task: str,
    task_type: str,
    tool_name: str,
    tool_output: str,
    sources: list[str],
) -> dict[str, Any]:
    jev = get_jev()

    sufficiency = jev.noul(
        "Given the tool output and sources, is the evidence sufficient to answer the task?",
        f"Task: {task}\nTool: {tool_name}\nOutput:\n{tool_output}\nSources: {sources}",
    )["probability"]

    grounded = jev.noul(
        "Is the tool output grounded in the provided sources (not hallucinated)?",
        f"Output:\n{tool_output}\nSources: {sources}",
        skeptic=True,
    )["probability"]

    rubric = {0: "irrelevant/empty", 1: "weak", 2: "useful", 3: "highly relevant"}
    score_resp = jev.score(
        "Score how relevant and useful this tool output is for the task.",
        rubric,
        f"Task: {task}\nTask type: {task_type}\nOutput:\n{tool_output}",
    )
    norm_score = max(0.0, min(1.0, float(score_resp["score"]) / 3.0))

    success = sufficiency >= 0.6 and grounded >= 0.5 and norm_score >= 0.5

    return {
        "success": success,
        "score": norm_score,
        "sufficiency": sufficiency,
        "tool_groundedness": grounded,
        "verified": grounded,
        "reason": f"tool_stage sufficiency={sufficiency:.2f} grounded={grounded:.2f} score={norm_score:.2f}",
    }


# ---------------------------------------------------------------------------
# Deterministic grounding for calculator
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _calculator_grounding(tool_output: str, answer: str) -> float:
    """Return 1.0 if the answer contains the calculator's numeric result, else 0.0."""
    if not tool_output.strip() or not answer.strip():
        return 0.0
    rhs = tool_output.split("=")[-1]
    rhs_nums = _NUM_RE.findall(rhs)
    if not rhs_nums:
        return 0.0
    try:
        target = float(rhs_nums[-1])
    except ValueError:
        return 0.0

    for tok in _NUM_RE.findall(answer):
        try:
            v = float(tok)
            if abs(v - target) < 1e-6 or abs(v - target) / max(abs(target), 1e-9) < 1e-4:
                return 1.0
        except ValueError:
            continue
    return 0.0


def evaluate_final_answer(
    task: str,
    answer: str,
    tool_output: str,
    sources: list[str],
    tool_name: str = "",
) -> dict[str, Any]:
    if not answer.strip():
        return {"answer_groundedness": 0.0, "answer_relevant": 0.0, "reason": "empty answer"}

    # Deterministic path for calculator
    if tool_name == "calculator" and tool_output.strip():
        cg = _calculator_grounding(tool_output, answer)
        if cg >= 1.0:
            return {"answer_groundedness": 1.0, "answer_relevant": 1.0}

    jev = get_jev()

    grounded = jev.noul(
        "Is every factual claim in the ANSWER directly supported by the TOOL OUTPUT "
        "or the SOURCES? If any claim is unsupported or invented, answer NO.",
        f"Task: {task}\n\nAnswer:\n{answer}\n\n"
        f"Tool output:\n{tool_output}\n\nSources:\n{sources}",
        skeptic=True,
    )["probability"]

    relevant = jev.noul(
        "Does the ANSWER directly address the TASK (not a tangent, not a refusal)?",
        f"Task: {task}\nAnswer: {answer}",
    )["probability"]

    return {
        "answer_groundedness": grounded,
        "answer_relevant": relevant,
    }
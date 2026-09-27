"""Offline scorer: consumes traces from benchmark runs and produces metrics."""
from __future__ import annotations

from app.evaluation.metrics import SummaryMetrics


def summarize(records: list[dict]) -> SummaryMetrics:
    if not records:
        return SummaryMetrics(0, 0, 0, 0, 0, 0, 0)

    n = len(records)
    success = sum(int(r.get("success")) for r in records) / n
    tool_ok = sum(int(r.get("tool_correct", False)) for r in records) / n
    recovered = [r for r in records if r.get("attempts", 1) > 1]
    recovery_rate = (
        sum(int(r.get("success", False)) for r in recovered) / len(recovered)
        if recovered else 0.0
    )
    grounded = sum(float(r.get("groundedness", 0.0)) for r in records) / n
    calls = sum(int(r.get("tool_calls", 0)) for r in records) / n
    latency = sum(float(r.get("latency_s", 0.0)) for r in records) / n
    cost = sum(float(r.get("cost_usd", 0.0)) for r in records) / n

    return SummaryMetrics(
        task_success_rate=success,
        tool_accuracy=tool_ok,
        recovery_rate=recovery_rate,
        groundedness=grounded,
        avg_tool_calls=calls,
        avg_latency_s=latency,
        avg_cost_usd=cost,
    )
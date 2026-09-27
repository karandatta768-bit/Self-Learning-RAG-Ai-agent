"""Metric functions."""
from __future__ import annotations

from dataclasses import dataclass


def accuracy(preds: list, golds: list) -> float:
    if not preds:
        return 0.0
    return sum(int(p == g) for p, g in zip(preds, golds)) / len(preds)


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def expected_calibration_error(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    if not probs:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(n_bins - 1, int(p * n_bins))].append((p, y))
    total = len(probs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(conf - acc)
    return ece


@dataclass
class SummaryMetrics:
    task_success_rate: float
    tool_accuracy: float
    recovery_rate: float
    groundedness: float
    avg_tool_calls: float
    avg_latency_s: float
    avg_cost_usd: float
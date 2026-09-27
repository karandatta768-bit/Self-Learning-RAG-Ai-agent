"""Compare cold vs warm JSON output files and report learning gain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cold")
    ap.add_argument("warm")
    args = ap.parse_args()

    cold = _load(args.cold)["summary"]
    warm = _load(args.warm)["summary"]

    keys = [
        ("task_success_rate", "Task Success"),
        ("tool_accuracy", "Tool Accuracy"),
        ("recovery_rate", "Recovery Rate"),
        ("groundedness", "Groundedness"),
        ("avg_tool_calls", "Avg Tool Calls"),
        ("avg_latency_s", "Avg Latency (s)"),
        ("avg_cost_usd", "Avg Cost (USD)"),
    ]
    print(f"{'Metric':<20}{'Cold':>12}{'Warm':>12}{'Delta':>12}")
    print("-" * 56)
    for k, label in keys:
        c = float(cold.get(k, 0.0))
        w = float(warm.get(k, 0.0))
        print(f"{label:<20}{c:>12.4f}{w:>12.4f}{w - c:>+12.4f}")

    print(f"\nLearning gain (success rate) = {warm['task_success_rate'] - cold['task_success_rate']:+.4f}")


if __name__ == "__main__":
    main()
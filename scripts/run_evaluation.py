"""Run benchmark in cold and warm modes with strict leakage isolation."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from app.config import get_settings
from app.evaluation.benchmark import load_benchmark
from app.evaluation.offline_evaluator import summarize
from app.graph.workflow import run


def _wipe_memory() -> None:
    s = get_settings()
    s.experiences_path.unlink(missing_ok=True)
    s.strategy_path.unlink(missing_ok=True)
    print("[clear] wiped episodic + strategy memory")


def run_once(mode: str, *, limit: int | None = None, heldout_ratio: float = 0.7) -> dict:
    tasks = load_benchmark()
    if not tasks:
        print("[warn] no benchmark tasks found at data/evaluation/benchmark.jsonl")
        return {}
    if limit:
        tasks = tasks[:limit]

    n_heldout = int(len(tasks) * heldout_ratio)
    heldout = tasks[:n_heldout]
    warmup = tasks[n_heldout:]
    heldout_ids = [t.get("id") for t in heldout if t.get("id")]

    run_id = f"{mode}-{uuid.uuid4().hex[:8]}"

    rows: list[dict] = []
    for phase, batch in (("warmup", warmup), ("heldout", heldout)):
        for i, t in enumerate(batch, 1):
            t0 = time.time()
            state = run(
                t["task"],
                run_id=run_id,
                task_id=t.get("id", ""),
                heldout_task_ids=heldout_ids,
                isolate_run=True,
            )
            dt = time.time() - t0
            tools = [r["tool"] for r in state.get("tool_results", [])]
            ev = state.get("evaluation", {})
            rows.append(
                {
                    "phase": phase,
                    "id": t.get("id", f"{phase}-{i}"),
                    "task": t["task"],
                    "expected_tool": t.get("expected_tool", ""),
                    "tool_used": tools[0] if tools else "none",
                    "tool_correct": (tools[0] if tools else "none") == t.get("expected_tool", ""),
                    "success": bool(ev.get("success", False)),
                    "groundedness": float(ev.get("answer_groundedness", 0.0)),
                    "attempts": int(state.get("attempt", 1)),
                    "tool_calls": len(tools),
                    "latency_s": dt,
                    "cost_usd": float(state.get("total_cost_usd", 0.0)),
                }
            )
            print(
                f"[{mode}::{phase}] {i}/{len(batch)} "
                f"success={rows[-1]['success']} tool={rows[-1]['tool_used']} "
                f"grounded={rows[-1]['groundedness']:.2f}"
            )

    s = summarize(rows)
    print(f"\n=== {mode.upper()} SUMMARY (run_id={run_id}) ===")
    print(json.dumps(s.__dict__, indent=2))

    Path("mlruns").mkdir(exist_ok=True)
    out = Path(f"mlruns/{mode}_{int(time.time())}.json")
    out.write_text(
        json.dumps({"run_id": run_id, "rows": rows, "summary": s.__dict__}, indent=2)
    )
    print(f"[saved] {out}")
    return {"run_id": run_id, **s.__dict__}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "warm"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clear", action="store_true", help="wipe memory before running")
    ap.add_argument("--heldout-ratio", type=float, default=0.7)
    args = ap.parse_args()

    if args.clear:
        _wipe_memory()

    run_once(args.mode, limit=args.limit, heldout_ratio=args.heldout_ratio)


if __name__ == "__main__":
    main()
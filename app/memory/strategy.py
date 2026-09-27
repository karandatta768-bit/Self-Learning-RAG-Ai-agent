"""Strategy memory: aggregate performance per (run_id, task_type, strategy)."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.memory.schemas import EpisodeRecord, StrategyRecord


class StrategyMemory:
    def __init__(self) -> None:
        s = get_settings()
        self.path: Path = s.strategy_path
        self._by_key: dict[str, StrategyRecord] = {}
        self._load()

    @staticmethod
    def _key(run_id: str, task_type: str, strategy: str) -> str:
        return f"{run_id}::{task_type}::{strategy}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for rec in data.get("strategies", []):
                r = StrategyRecord.model_validate(rec)
                self._by_key[self._key(r.run_id, r.task_type, r.strategy)] = r
        except Exception:
            self._by_key = {}

    def _persist(self) -> None:
        payload = {"strategies": [r.model_dump() for r in self._by_key.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update(self, ep: EpisodeRecord) -> StrategyRecord:
        key = self._key(ep.run_id, ep.task_type, ep.strategy)
        rec = self._by_key.get(key) or StrategyRecord(
            run_id=ep.run_id, task_type=ep.task_type, strategy=ep.strategy, tools=list(ep.tools)
        )
        rec.merge(ep)
        self._by_key[key] = rec
        self._persist()
        return rec

    def best_for(self, task_type: str, *, run_id: str = "prod") -> StrategyRecord | None:
        candidates = [
            r for r in self._by_key.values()
            if r.task_type == task_type and r.run_id == run_id
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda r: (r.success_rate, r.avg_score, -r.avg_cost),
            reverse=True,
        )[0]

    def all_for(self, task_type: str, *, run_id: str = "prod") -> list[StrategyRecord]:
        return [
            r for r in self._by_key.values()
            if r.task_type == task_type and r.run_id == run_id
        ]


_instance: StrategyMemory | None = None


def get_strategy() -> StrategyMemory:
    global _instance
    if _instance is None:
        _instance = StrategyMemory()
    return _instance


def reset_strategy() -> None:
    global _instance
    _instance = None
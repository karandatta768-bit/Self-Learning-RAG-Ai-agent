"""JSONL benchmark loader."""
from __future__ import annotations

import json

from app.config import get_settings


def load_benchmark() -> list[dict[str, str]]:
    path = Path(get_settings().data_dir) / "evaluation" / "benchmark.jsonl"
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("task"), str):
                raise ValueError(f"Invalid benchmark row at {path}:{line_number}: expected an object with a task")
            rows.append({key: str(value) for key, value in row.items()})
    return rows

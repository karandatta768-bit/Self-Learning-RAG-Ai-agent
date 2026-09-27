"""Canonical record schemas for all memory types."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


VALID_TASK_TYPES = {"research", "rag", "calculation", "ambiguous", "out_of_scope", "multi_step"}
TaskType = Literal["research", "rag", "calculation", "ambiguous", "out_of_scope", "multi_step"]


def safe_task_type(raw: str | None) -> str:
    if raw in VALID_TASK_TYPES:
        return raw  # type: ignore[return-value]
    return "ambiguous"


class EvaluationRecord(BaseModel):
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevant: float = Field(default=0.0, ge=0.0, le=1.0)
    verified: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class EpisodeRecord(BaseModel):
    id: str
    task: str
    rephrased_task: str = ""
    task_id: str = ""
    run_id: str = ""
    task_type: TaskType = "ambiguous"
    strategy: str = "default"
    tools: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    evaluation: EvaluationRecord
    confidence: float = 0.0
    attempts: int = 1
    cost_usd: float = 0.0
    latency_s: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    extra: dict[str, Any] = Field(default_factory=dict)


class StrategyRecord(BaseModel):
    run_id: str = "prod"
    task_type: TaskType = "ambiguous"
    strategy: str = "default"
    tools: list[str] = Field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    avg_score: float = 0.0
    avg_cost: float = 0.0
    avg_latency: float = 0.0
    last_updated: float = Field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count else 0.0

    def merge(self, ep: EpisodeRecord) -> "StrategyRecord":
        n = self.total_count + 1
        self.success_count += int(ep.evaluation.success)
        self.avg_score = (self.avg_score * self.total_count + ep.evaluation.score) / n
        self.avg_cost = (self.avg_cost * self.total_count + ep.cost_usd) / n
        self.avg_latency = (self.avg_latency * self.total_count + ep.latency_s) / n
        self.total_count = n
        self.last_updated = time.time()
        return self


class KnowledgeChunk(BaseModel):
    id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
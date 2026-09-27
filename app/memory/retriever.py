"""Unified memory façade with leakage guards."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.memory.episodic import get_episodic
from app.memory.schemas import EpisodeRecord, KnowledgeChunk, StrategyRecord
from app.memory.semantic import get_semantic
from app.memory.strategy import get_strategy


@dataclass
class RetrievalBundle:
    experiences: list[EpisodeRecord] = field(default_factory=list)
    knowledge: list[KnowledgeChunk] = field(default_factory=list)
    strategies: list[StrategyRecord] = field(default_factory=list)

    def as_context(self, max_chars: int = 4000) -> str:
        parts: list[str] = []
        if self.experiences:
            parts.append("### Past similar experiences (excludes the current task)")
            for e in self.experiences:
                parts.append(
                    f"- task_type={e.task_type} strategy={e.strategy} "
                    f"tools={e.tools} success={e.evaluation.success} "
                    f"score={e.evaluation.score:.2f} :: {e.task[:140]}"
                )
        if self.strategies:
            parts.append("### Known strategy performance")
            for s in self.strategies:
                parts.append(
                    f"- {s.task_type}::{s.strategy} "
                    f"rate={s.success_rate:.2f} n={s.total_count} score={s.avg_score:.2f}"
                )
        if self.knowledge:
            parts.append("### Retrieved knowledge")
            for k in self.knowledge:
                src = f" [{k.source}]" if k.source else ""
                parts.append(f"- {k.text[:300]}{src}")
        return "\n".join(parts)[:max_chars]


def retrieve_for_task(
    task: str,
    task_type: str | None = None,
    *,
    run_id: str = "prod",
    exclude_run_id: str | None = None,
    exclude_task_ids: set[str] | None = None,
) -> RetrievalBundle:
    s = get_settings()
    bundle = RetrievalBundle()

    bundle.experiences = get_episodic().search(
        task,
        k=s.experience_top_k,
        exclude_task=task,
        exclude_run_id=exclude_run_id,
        exclude_task_ids=exclude_task_ids,
        min_sim=s.memory_min_sim,
    )
    bundle.knowledge = get_semantic().search(task, k=s.knowledge_top_k, min_sim=s.memory_min_sim)
    if task_type:
        bundle.strategies = get_strategy().all_for(task_type, run_id=run_id)
    return bundle


def store_episode(ep: EpisodeRecord) -> None:
    get_episodic().write(ep)
    get_strategy().update(ep)
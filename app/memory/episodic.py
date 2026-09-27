"""Episodic memory: JSONL + FAISS. Self-task and cross-run exclusion."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from app.config import get_settings
from app.memory.schemas import EpisodeRecord
from app.memory.semantic import get_embedder


class EpisodicMemory:
    def __init__(self, path: Path | None = None) -> None:
        s = get_settings()
        self.path = path or s.experiences_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._ids: list[str] = []
        self._records: dict[str, EpisodeRecord] = {}
        self._load()

    def _load(self) -> None:
        import faiss

        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = EpisodeRecord.model_validate_json(line)
                except Exception:
                    continue
                self._records[rec.id] = rec
                self._ids.append(rec.id)
        if not self._ids:
            return
        texts = [self._records[i].task for i in self._ids]
        emb = get_embedder().encode(texts, normalize_embeddings=True).astype("float32")
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)

    def write(self, ep: EpisodeRecord) -> str:
        import faiss

        if not ep.id:
            ep.id = uuid.uuid4().hex
        self._records[ep.id] = ep
        self._ids.append(ep.id)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(ep.model_dump_json() + "\n")

        emb = get_embedder().encode([ep.task], normalize_embeddings=True).astype("float32")
        if self._index is None:
            self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        return ep.id

    def get(self, ep_id: str) -> EpisodeRecord | None:
        return self._records.get(ep_id)

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        exclude_task: str | None = None,
        exclude_run_id: str | None = None,
        exclude_task_ids: set[str] | None = None,
        min_sim: float | None = None,
    ) -> list[EpisodeRecord]:
        if self._index is None or not self._ids:
            return []
        s = get_settings()
        sim_floor = min_sim if min_sim is not None else s.memory_min_sim

        emb = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
        fetch = min(len(self._ids), max(k * 6, k + 10))
        scores, idx = self._index.search(emb, fetch)

        q_norm = (query or "").strip().lower()
        seen: set[str] = set()
        out: list[EpisodeRecord] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            rec = self._records[self._ids[int(i)]]
            if float(score) < sim_floor:
                continue
            if exclude_task and rec.task.strip().lower() == q_norm:
                continue
            if exclude_run_id and rec.run_id == exclude_run_id:
                continue
            if exclude_task_ids and rec.task_id in exclude_task_ids:
                continue
            key = rec.task.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
            if len(out) >= k:
                break
        return out

    def all(self) -> Iterable[EpisodeRecord]:
        return self._records.values()


_instance: EpisodicMemory | None = None


def get_episodic() -> EpisodicMemory:
    global _instance
    if _instance is None:
        _instance = EpisodicMemory()
    return _instance


def reset_episodic() -> None:
    global _instance
    _instance = None
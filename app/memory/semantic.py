"""Semantic memory: knowledge chunks + shared embedder."""
from __future__ import annotations

import uuid
from pathlib import Path

from app.config import get_settings
from app.memory.schemas import KnowledgeChunk


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(get_settings().embedding_model)
    return _embedder


class SemanticMemory:
    def __init__(self) -> None:
        self.dir: Path = get_settings().vectorstore_path
        self.index_path = self.dir / "knowledge.faiss"
        self.meta_path = self.dir / "knowledge.jsonl"
        self._index = None
        self._chunks: list[KnowledgeChunk] = []
        self._load()

    def _load(self) -> None:
        import faiss

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._chunks = []
            with self.meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._chunks.append(KnowledgeChunk.model_validate_json(line))

    def _persist(self) -> None:
        import faiss

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(c.model_dump_json() + "\n")

    def add(self, chunks: list[KnowledgeChunk]) -> int:
        import faiss

        if not chunks:
            return 0
        for c in chunks:
            if not c.id:
                c.id = uuid.uuid4().hex
        emb = get_embedder().encode([c.text for c in chunks], normalize_embeddings=True).astype("float32")
        if self._index is None:
            self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        self._chunks.extend(chunks)
        self._persist()
        return len(chunks)

    def search(self, query: str, k: int = 5, *, min_sim: float | None = None) -> list[KnowledgeChunk]:
        if self._index is None or not self._chunks:
            return []
        s = get_settings()
        sim_floor = min_sim if min_sim is not None else s.memory_min_sim
        emb = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
        k = min(k, len(self._chunks))
        scores, idx = self._index.search(emb, k)
        out: list[KnowledgeChunk] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or float(score) < sim_floor:
                continue
            out.append(self._chunks[int(i)])
        return out


_instance: SemanticMemory | None = None


def get_semantic() -> SemanticMemory:
    global _instance
    if _instance is None:
        _instance = SemanticMemory()
    return _instance
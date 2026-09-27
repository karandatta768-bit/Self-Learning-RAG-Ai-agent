"""Ingest .txt/.md documents from data/knowledge into semantic memory."""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from app.config import get_settings
from app.memory.schemas import KnowledgeChunk
from app.memory.semantic import get_semantic


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_dir(root: Path) -> int:
    semantic = get_semantic()
    total = 0
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [
            KnowledgeChunk(id=uuid.uuid4().hex, text=c, source=str(path))
            for c in chunk_text(text)
        ]
        total += semantic.add(chunks)
        print(f"[ingest] {path}: {len(chunks)} chunks")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(get_settings().knowledge_path))
    args = ap.parse_args()
    n = ingest_dir(Path(args.dir))
    print(f"[ingest] total chunks added: {n}")


if __name__ == "__main__":
    main()
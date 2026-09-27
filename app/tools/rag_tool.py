"""RAG tool over semantic memory."""
from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.memory.semantic import get_semantic
from app.tools.base import Tool, ToolResult


class RAGTool(Tool):
    name = "rag"
    description = (
        "Retrieve from the local knowledge base. Use for questions about internal "
        "documents, policies, and domain-specific facts indexed in the vector store."
    )
    input_schema = {"query": "string"}

    def run(self, query: str, **_: Any) -> ToolResult:
        t0 = time.time()
        s = get_settings()
        chunks = get_semantic().search(query, k=s.knowledge_top_k)
        if not chunks:
            return ToolResult(
                output="", success=False,
                error="No matching knowledge chunks found.",
                confidence=0.2, latency_s=time.time() - t0,
            )
        body = "\n\n".join(f"[{c.source or c.id}] {c.text}" for c in chunks)
        return ToolResult(
            output=body,
            sources=[c.source or c.id for c in chunks],
            confidence=0.85, latency_s=time.time() - t0,
            raw={"chunks": [c.model_dump() for c in chunks]},
        )
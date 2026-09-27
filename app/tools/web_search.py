"""Web search using DuckDuckGo HTML endpoint."""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote_plus

import httpx

from app.config import get_settings
from app.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the public web for current information. "
        "Use for recent events, external facts, and sources beyond the local knowledge base."
    )
    input_schema = {"query": "string"}

    def run(self, query: str, **_: Any) -> ToolResult:
        t0 = time.time()
        s = get_settings()
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            r = httpx.get(
                url, timeout=s.web_search_timeout,
                headers={"User-Agent": "Mozilla/5.0 (self-learning-agent)"},
            )
            r.raise_for_status()
            text = self._strip(r.text)
            return ToolResult(
                output=text[:4000], sources=[url],
                confidence=0.7, latency_s=time.time() - t0,
            )
        except Exception as e:
            return ToolResult(
                output="", success=False, error=f"Web search error: {e}",
                confidence=0.0, latency_s=time.time() - t0,
            )

    @staticmethod
    def _strip(html: str) -> str:
        import re
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
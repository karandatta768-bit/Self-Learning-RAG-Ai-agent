"""Ollama chat wrapper with cumulative latency accounting."""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.config import get_settings


_NO_REASONING = "Output ONLY the final answer. Do not narrate reasoning."


def _inject_no_reasoning(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    result = [message.copy() for message in messages]
    for message in result:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content', '').rstrip()}\n\n{_NO_REASONING}"
            return result
    return [{"role": "system", "content": _NO_REASONING}, *result]


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"(?im)^\s*(?:\.\.\.\s*)?done thinking\.?\s*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _dedupe(text: str) -> str:
    text = text.strip()
    repeated = re.fullmatch(r"(?s)(.+?)\s+\1", text)
    if repeated:
        return repeated.group(1).strip()
    sentence = r"(?P<sentence>[^.!?\n]+[.!?])"
    return re.sub(
        rf"{sentence}(?P<separator>\s+)(?P=sentence)(?:(?P=separator)(?P=sentence))*",
        lambda match: match.group("sentence") + match.group("separator"),
        text,
        flags=re.IGNORECASE,
    )


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self._base_url = s.ollama_base_url.rstrip("/")
        self._model = s.ollama_model
        self._temperature = s.llm_temperature

        self.last_cost_usd: float = 0.0
        self.last_latency_s: float = 0.0
        self.total_cost_usd: float = 0.0
        self.total_latency_s: float = 0.0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        t0 = time.time()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _inject_no_reasoning(messages),
            "stream": False,
            "think": False,
            "options": {"temperature": temperature if temperature is not None else self._temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=180.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self.last_latency_s = time.time() - t0
            raise RuntimeError(
                f"Could not reach Ollama at {self._base_url}. Start Ollama and pull {self._model}."
            ) from exc

        data = response.json()
        self.last_latency_s = time.time() - t0
        self.last_cost_usd = 0.0
        self.total_latency_s += self.last_latency_s
        message = data.get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("Ollama returned an invalid chat response")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned an invalid chat response")
        return _dedupe(_strip_thinking(content))

    def chat_json(self, messages: list[dict[str, str]], **kw: Any) -> dict:
        raw = self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            stripped = raw.strip().strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            return json.loads(stripped)


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm

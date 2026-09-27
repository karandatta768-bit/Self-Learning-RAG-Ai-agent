"""Jev decision layer (Choice / Score / Noul). Fail-closed schema validation."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.models.llm import get_llm

log = logging.getLogger(__name__)


class JevSchemaError(RuntimeError):
    pass


@dataclass
class JevCall:
    primitive: str
    question: str
    payload: dict[str, Any]
    response: dict[str, Any]
    latency_s: float
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class JevClient:
    def __init__(self) -> None:
        self.calls: list[JevCall] = []
        self._last_latency: float = 0.0
        self._last_cost: float = 0.0

    def choose(self, question: str, options: Sequence[str], context: str = "") -> dict[str, Any]:
        resp = self._choose(question, list(options), context)
        self._validate_choice(resp, list(options))
        self._log("choice", question, {"options": list(options), "context": context}, resp)
        return resp

    def score(self, question: str, rubric: dict[int, str], item: str, context: str = "") -> dict[str, Any]:
        resp = self._score(question, rubric, item, context)
        self._validate_score(resp, rubric)
        self._log("score", question, {"rubric": rubric, "item": item, "context": context}, resp)
        return resp

    def noul(self, question: str, context: str = "", *, skeptic: bool = False) -> dict[str, Any]:
        resp = self._noul(question, context, skeptic=skeptic)
        self._validate_noul(resp)
        self._log("noul", question, {"context": context, "skeptic": skeptic}, resp)
        return resp

    def _choose(self, q, options, context): raise NotImplementedError
    def _score(self, q, rubric, item, context): raise NotImplementedError
    def _noul(self, q, context, *, skeptic: bool = False): raise NotImplementedError

    @staticmethod
    def _validate_choice(resp, options):
        if "choice" not in resp or resp["choice"] not in options:
            raise JevSchemaError(f"Invalid choice: {resp!r} vs {options}")

    @staticmethod
    def _validate_score(resp, rubric):
        try:
            s = float(resp["score"])
        except (KeyError, TypeError, ValueError) as e:
            raise JevSchemaError(f"Invalid score: {resp!r}") from e
        lo, hi = min(rubric), max(rubric)
        if not (lo <= s <= hi):
            raise JevSchemaError(f"Score {s} out of [{lo},{hi}]")

    @staticmethod
    def _validate_noul(resp):
        try:
            p = float(resp["probability"])
        except (KeyError, TypeError, ValueError) as e:
            raise JevSchemaError(f"Invalid noul: {resp!r}") from e
        if not (0.0 <= p <= 1.0):
            raise JevSchemaError(f"Probability out of range: {p}")

    def _log(self, primitive, question, payload, resp):
        self.calls.append(
            JevCall(primitive, question, payload, resp, self._last_latency, self._last_cost)
        )


_SCHEMA_CHOICE = '{"choice": "<one of the options, exact string>", "confidence": <0..1>}'
_SCHEMA_SCORE = '{"score": <number in rubric range>, "confidence": <0..1>}'
_SCHEMA_NOUL = '{"probability": <0..1>, "confidence": <0..1>}'


class StubJevClient(JevClient):
    def __init__(self) -> None:
        super().__init__()
        self._llm = get_llm()
        log.warning(
            "[JEV] Local stub decisions use the same Ollama model as generation. "
            "Benchmark scores may be optimistic."
        )

    def _run(self, system: str, user: str) -> dict[str, Any]:
        t0 = time.time()
        data = self._llm.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        self._last_latency = time.time() - t0
        self._last_cost = self._llm.last_cost_usd
        return data

    def _choose(self, question, options, context):
        system = (
            "You are Jev, a decision model. Return strict JSON only. Pick exactly one option. "
            f"Schema: {_SCHEMA_CHOICE}"
        )
        user = f"Question: {question}\nOptions: {json.dumps(options)}\nContext:\n{context}\nReturn JSON only."
        return self._run(system, user)

    def _score(self, question, rubric, item, context):
        system = f"You are Jev, a scoring model. Schema: {_SCHEMA_SCORE}"
        user = (
            f"Question: {question}\nRubric: {json.dumps(rubric)}\n"
            f"Item:\n{item}\nContext:\n{context}\nReturn JSON only."
        )
        return self._run(system, user)

    def _noul(self, question, context, *, skeptic: bool = False):
        if skeptic:
            system = (
                "You are a SKEPTIC decision model. Default to NO unless the sources "
                "explicitly and directly support the claim. Do not infer. Do not use your "
                "own knowledge. Do not reward plausible-sounding text. "
                f"Schema: {_SCHEMA_NOUL}"
            )
        else:
            system = (
                "You are Jev, returning a probability in [0,1] for a yes/no question. "
                f"Schema: {_SCHEMA_NOUL}"
            )
        user = f"Question: {question}\nContext:\n{context}\nReturn JSON only."
        return self._run(system, user)


_client: JevClient | None = None


def get_jev() -> JevClient:
    global _client
    if _client is None:
        _client = StubJevClient()
    return _client

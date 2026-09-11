"""LLM-driven detection of logical dependencies between markets.

Two prediction markets are dependent when their outcomes cannot resolve
independently ("Will X win the nomination" vs "Will X win the election").
Enumerating those relations by hand across thousands of pairs is impractical,
so a reasoning model reads both descriptions and emits the valid outcome
combinations as a constraint matrix the arbitrage solver can consume.

The call is a plain OpenAI-compatible chat completion, so `TRADING_MCP_LLM_URL`
can point at xAI, any other compatible provider, or a local Switchyard
deployment that routes between them.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .config import Settings

_PROMPT = """You analyze prediction markets for logical dependencies.

Market A: {a_question}
A description: {a_description}
A outcomes: {a_outcomes}

Market B: {b_question}
B description: {b_description}
B outcomes: {b_outcomes}

Decide whether the resolutions of A and B are logically dependent — that is,
whether some combination of their outcomes is impossible or forced.

Respond with JSON only, in exactly this shape:
{{
  "dependent": true or false,
  "relation": "short description of the logical relation, or null",
  "valid_outcomes": [["A outcome", "B outcome"], ...],
  "confidence": 0.0 to 1.0
}}

"valid_outcomes" must list every jointly possible pair and omit impossible
ones. If the markets are independent, set "dependent" to false and list all
pairs. Do not include any text outside the JSON object."""


def _describe(market: dict[str, Any]) -> dict[str, str]:
    """Normalize the market fields the prompt needs, tolerating missing keys."""
    return {
        "question": str(market.get("question") or market.get("description") or "unknown"),
        "description": str(market.get("description") or ""),
        "outcomes": json.dumps(market.get("outcomes") or ["Yes", "No"]),
    }


async def detect_dependency(
    settings: Settings,
    http: httpx.AsyncClient,
    market_a: dict[str, Any],
    market_b: dict[str, Any],
) -> dict[str, Any]:
    """Ask the configured model whether two markets are logically dependent.

    Raises `RuntimeError` if no API key is configured, and propagates HTTP
    errors from the provider. A response that is not valid JSON is returned
    under `parse_error` with the raw text, so the caller can retry or inspect
    it rather than silently receiving an empty result.
    """
    if not settings.llm_api_key:
        raise RuntimeError("set XAI_API_KEY (or TRADING_MCP_LLM_API_KEY) to detect dependencies")

    a = _describe(market_a)
    b = _describe(market_b)
    prompt = _PROMPT.format(
        a_question=a["question"],
        a_description=a["description"],
        a_outcomes=a["outcomes"],
        b_question=b["question"],
        b_description=b["description"],
        b_outcomes=b["outcomes"],
    )

    response = await http.post(
        f"{settings.llm_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        json={
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        return dict(json.loads(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"parse_error": "model did not return valid JSON", "raw": content}


def constraints_from_valid_outcomes(
    valid_outcomes: list[list[str]],
    outcomes_a: list[str],
    outcomes_b: list[str],
) -> dict[str, Any]:
    """Turn a list of jointly possible outcome pairs into solver constraints.

    The outcome space is flattened to `outcomes_a + outcomes_b`, so index `i`
    selects an A outcome and index `len(outcomes_a) + j` a B outcome. Each
    market contributes an "exactly one outcome" equality, and every impossible
    pair contributes `z_i + z_j <= 1` to forbid selecting both.
    """
    width = len(outcomes_a) + len(outcomes_b)
    allowed = {(pair[0], pair[1]) for pair in valid_outcomes if len(pair) == 2}
    constraints: list[dict[str, Any]] = []

    for offset, outcomes in ((0, outcomes_a), (len(outcomes_a), outcomes_b)):
        coeffs = [0.0] * width
        for index in range(len(outcomes)):
            coeffs[offset + index] = 1.0
        constraints.append({"coeffs": coeffs, "b": 1.0, "sense": "=="})

    for i, name_a in enumerate(outcomes_a):
        for j, name_b in enumerate(outcomes_b):
            if (name_a, name_b) in allowed:
                continue
            coeffs = [0.0] * width
            coeffs[i] = 1.0
            coeffs[len(outcomes_a) + j] = 1.0
            constraints.append({"coeffs": coeffs, "b": 1.0, "sense": "<="})

    return {"width": width, "labels": list(outcomes_a) + list(outcomes_b), "constraints": constraints}

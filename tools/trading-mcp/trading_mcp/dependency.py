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
    """Build the payoff-cover matrix that the arbitrage solver minimizes against.

    Each jointly possible outcome pair is a *state of the world*, and each state
    becomes one row asserting that the basket holds at least one token paying in
    that state: `A z >= 1`. A basket satisfying every row pays at least $1 no
    matter how the markets resolve, so any basket costing less than $1 is an
    arbitrage.

    The token space is `outcomes_a` followed by `outcomes_b`, labelled `A:` and
    `B:` so identically named outcomes stay distinguishable. Dropping an
    impossible state drops its row, which is exactly how a dependency creates an
    arbitrage that neither market shows on its own.
    """
    if not valid_outcomes:
        raise ValueError("valid_outcomes is empty: no state of the world is possible")

    index_a = {name: i for i, name in enumerate(outcomes_a)}
    index_b = {name: len(outcomes_a) + i for i, name in enumerate(outcomes_b)}
    width = len(outcomes_a) + len(outcomes_b)

    rows: list[dict[str, Any]] = []
    states: list[list[str]] = []
    for pair in valid_outcomes:
        if len(pair) != 2:
            raise ValueError(f"expected [a_outcome, b_outcome] pairs, got {pair!r}")
        name_a, name_b = pair[0], pair[1]
        if name_a not in index_a:
            raise ValueError(f"{name_a!r} is not an outcome of market A ({outcomes_a})")
        if name_b not in index_b:
            raise ValueError(f"{name_b!r} is not an outcome of market B ({outcomes_b})")

        coeffs = [0.0] * width
        coeffs[index_a[name_a]] = 1.0
        coeffs[index_b[name_b]] = 1.0
        rows.append({"coeffs": coeffs, "b": 1.0, "sense": ">="})
        states.append([name_a, name_b])

    labels = [f"A:{name}" for name in outcomes_a] + [f"B:{name}" for name in outcomes_b]
    return {
        "width": width,
        "labels": labels,
        "states": states,
        "constraints": rows,
        "note": (
            "One row per jointly possible state. A basket satisfying all rows pays "
            "at least $1 in every state, so a cost below $1 is an arbitrage."
        ),
    }

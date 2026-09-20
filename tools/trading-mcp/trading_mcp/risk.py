"""Resolution risk: the gap between a logical dependency and a tradable one.

A cover basket is only risk-free if the states the solver dropped are genuinely
impossible *under the markets' own resolution rules*. Semantic implication is
not enough, and this is where a dependency arbitrage actually loses money.

The failure mode to keep in mind: "Will X be the nominee on July 1?" and "Will
X win the November election?" look strictly implied — you cannot win without
the nomination — so the model drops the (No, Yes) state. But if X is nominated
on July 15, market A resolves No while B still resolves Yes. The dropped state
happens, the basket pays nothing, and a position entered for a 15c edge loses
the full 85c of principal.

Nothing here can prove a dependency sound. What it can do is refuse to call one
tradable unless the model justified every exclusion, was confident, and the two
markets resolve on compatible terms — and price the capital lockup, since an
edge held to resolution is a carry trade, not free money.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# A basket is only risk-free if every dropped state is truly unreachable, so the
# bar for the model's own confidence is high. Below this, the edge is a wager on
# the model's reading of two prospectuses.
DEFAULT_MIN_CONFIDENCE = 0.85

# A gap between resolution dates is only dangerous when the earlier market
# settles on a *deadline* rather than on the event itself. "Will X win the
# nomination?" resolves whenever the convention happens and keeps implying the
# election outcome four months later; "Will X be nominee by June 1?" stops
# implying anything the moment June 2 arrives. Below this many days apart, even
# a deadline cannot realistically truncate the implication.
DEFAULT_DATE_BOUND_GAP_DAYS = 14.0

# Phrasing that ties a resolution to a calendar deadline rather than to the
# event. A bare year ("the 2028 election") is not a deadline, so a month name or
# digit must follow the preposition.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_DATE_BOUND_PATTERNS = (
    rf"\bby\s+(the\s+end\s+of\s+)?({_MONTHS}|\d)",
    rf"\bbefore\s+({_MONTHS}|\d)",
    rf"\bas\s+of\s+({_MONTHS}|\d)",
    r"\bon\s+or\s+before\b",
    r"\bno\s+later\s+than\b",
)


def looks_date_bounded(market: dict[str, Any]) -> bool:
    """Detect a resolution tied to a calendar deadline rather than to an event.

    A backstop for the model's own judgement, not a replacement: a deadline in
    the question text is the signature of the truncation trap, so it is worth
    catching even when the model reports no risk.
    """
    text = f"{market.get('question') or ''} {market.get('description') or ''}".lower()
    return any(re.search(pattern, text) for pattern in _DATE_BOUND_PATTERNS)


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp, returning None for anything unusable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def annualized_return(edge_per_share: float, cost_per_share: float, days: float) -> float | None:
    """Return the annualized simple return of holding a basket to resolution.

    The headline number on an arbitrage is its per-share edge, but the capital
    is locked until the later market resolves. 15c on an 85c basket is 17.6%
    over the holding period — excellent over three weeks, mediocre over two
    years — and only this figure makes those comparable.
    """
    if cost_per_share <= 0 or days <= 0:
        return None
    return (edge_per_share / cost_per_share) * (365.0 / days)


def assess_dependency_risk(
    verdict: dict[str, Any],
    market_a: dict[str, Any],
    market_b: dict[str, Any],
    *,
    now: datetime | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    date_bound_gap_days: float = DEFAULT_DATE_BOUND_GAP_DAYS,
) -> dict[str, Any]:
    """Decide whether a dependency verdict is safe to build a basket on.

    Returns `tradable` alongside the reasons against it. `blocking_risks` are
    disqualifying; `warnings` are things to price in rather than refuse.

    A long gap between resolution dates is a *cost*, reported as capital
    lockup, not a disqualification — dependent markets are usually months
    apart, and refusing them outright would reject the whole opportunity set.
    What disqualifies is *truncation*: the earlier market settling on a
    deadline that can arrive before the fact determining the later one, which
    makes an excluded state reachable.
    """
    moment = now or datetime.now(UTC)
    blocking: list[str] = []
    warnings: list[str] = []

    if not verdict.get("dependent"):
        blocking.append("model found no dependency between these markets")

    confidence = verdict.get("confidence")
    if not isinstance(confidence, int | float):
        blocking.append("verdict carries no confidence score")
    elif confidence < min_confidence:
        blocking.append(
            f"confidence {confidence:.2f} is below the {min_confidence:.2f} bar for "
            "treating dropped states as impossible"
        )

    # Every state the solver drops removes a cover row, which is exactly what
    # creates the edge. An unjustified exclusion is an unpriced risk.
    excluded = verdict.get("excluded_states") or []
    if verdict.get("dependent") and not excluded:
        blocking.append("verdict excludes no states, so it cannot produce an arbitrage")
    for state in excluded:
        if not isinstance(state, dict) or not str(state.get("justification") or "").strip():
            blocking.append(f"excluded state {state!r} has no resolution-rule justification")

    end_a = parse_timestamp(market_a.get("end_date") or market_a.get("endDate"))
    end_b = parse_timestamp(market_b.get("end_date") or market_b.get("endDate"))

    resolution_gap_days: float | None = None
    lockup_days: float | None = None
    if end_a and end_b:
        resolution_gap_days = abs((end_a - end_b).total_seconds()) / 86400.0
        lockup_days = (max(end_a, end_b) - moment).total_seconds() / 86400.0
    else:
        warnings.append("at least one market has no parseable end date; lockup is unknown")

    if lockup_days is not None and lockup_days <= 0:
        blocking.append("both markets have already passed their end date")

    truncation = verdict.get("truncation_risk")
    if isinstance(truncation, dict) and truncation.get("possible"):
        blocking.append(
            "model reports the earlier market can resolve before the later one's "
            f"outcome is determined: {truncation.get('explanation') or 'no explanation given'}"
        )
    else:
        # The trap has a lexical signature. Catch it even when the model missed
        # it, but only when the dates are far enough apart for it to bite.
        earlier = market_a if (end_a and end_b and end_a <= end_b) else market_b
        gap_matters = resolution_gap_days is None or resolution_gap_days > date_bound_gap_days
        if gap_matters and looks_date_bounded(earlier):
            blocking.append(
                "the earlier market resolves on a calendar deadline rather than on the "
                "event itself, so it can settle before the later market's outcome is "
                "determined and make an excluded state reachable"
            )
        elif truncation is None:
            warnings.append(
                "verdict does not address whether the earlier market's resolution date "
                "can truncate the implication"
            )

    if lockup_days is not None and resolution_gap_days is not None and resolution_gap_days > 0:
        warnings.append(
            f"resolution dates are {resolution_gap_days:.0f} days apart; capital is locked "
            f"for {lockup_days:.0f} days until the later market settles"
        )

    for label, market in (("A", market_a), ("B", market_b)):
        if not str(market.get("description") or "").strip():
            warnings.append(f"market {label} has no description; the verdict read only its title")

    return {
        "tradable": not blocking,
        "confidence": confidence,
        "blocking_risks": blocking,
        "warnings": warnings,
        "resolution_gap_days": resolution_gap_days,
        "capital_lockup_days": lockup_days,
        "excluded_states": excluded,
    }


def score_opportunity(
    scan_result: dict[str, Any],
    risk: dict[str, Any],
    *,
    bankroll: float | None = None,
) -> dict[str, Any]:
    """Combine an execution-validated scan with its resolution risk.

    An opportunity is only actionable when the book supports it *and* the
    dependency survives scrutiny. The return is annualized over the capital
    lockup so a fat edge held for two years does not outrank a thin one held
    for a month.
    """
    execution = scan_result.get("layer3_execution") or {}
    edge = execution.get("realized_edge_per_share")
    cost = execution.get("realized_cost_per_share")
    lockup = risk.get("capital_lockup_days")

    annualized = (
        annualized_return(edge, cost, lockup)
        if edge is not None and cost is not None and lockup is not None
        else None
    )

    reasons: list[str] = []
    if not scan_result.get("tradable"):
        reasons.append(scan_result.get("verdict") or "execution validation failed")
    reasons.extend(risk.get("blocking_risks", []))

    size = execution.get("size")
    capital = cost * size if cost is not None and size is not None else None
    if bankroll is not None and capital is not None and capital > bankroll:
        reasons.append(f"basket needs ${capital:.2f} but bankroll is ${bankroll:.2f}")

    return {
        "actionable": not reasons,
        "blocked_by": reasons,
        "edge_per_share": edge,
        "cost_per_share": cost,
        "capital_required": capital,
        "capital_lockup_days": lockup,
        "annualized_return": annualized,
        "warnings": risk.get("warnings", []),
    }

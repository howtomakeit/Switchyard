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

from datetime import UTC, datetime
from typing import Any

# A basket is only risk-free if every dropped state is truly unreachable, so the
# bar for the model's own confidence is high. Below this, the edge is a wager on
# the model's reading of two prospectuses.
DEFAULT_MIN_CONFIDENCE = 0.85

# Two markets whose resolution dates are far apart can decouple in the gap: the
# earlier one resolves on facts that the later one can still overturn.
DEFAULT_MAX_RESOLUTION_GAP_DAYS = 14.0


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
    max_resolution_gap_days: float = DEFAULT_MAX_RESOLUTION_GAP_DAYS,
) -> dict[str, Any]:
    """Decide whether a dependency verdict is safe to build a basket on.

    Returns `tradable` alongside the reasons against it. `blocking_risks` are
    disqualifying; `warnings` are things to price in rather than refuse. A
    verdict the model marked independent is not tradable here — there is no
    dependency to exploit.
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
        if resolution_gap_days > max_resolution_gap_days:
            # The earlier market settles on facts the later one can still change.
            blocking.append(
                f"resolution dates are {resolution_gap_days:.0f} days apart: the earlier "
                "market can settle before the later one's outcome is determined, which "
                "can make a dropped state reachable"
            )
        lockup_days = (max(end_a, end_b) - moment).total_seconds() / 86400.0
    else:
        warnings.append("at least one market has no parseable end date; lockup is unknown")

    if lockup_days is not None and lockup_days <= 0:
        blocking.append("both markets have already passed their end date")

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

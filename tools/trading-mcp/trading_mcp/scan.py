"""The three-layer arbitrage pipeline, run against live order books.

Layer 1 screens a candidate cluster with the LP relaxation. Layer 2 solves the
binary program exactly to get the basket. Layer 3 — the layer that kills most
paper arbitrages — re-prices that basket against real book depth at the size
actually being traded, because an edge measured at the touch is measured
against a quantity nobody can fill.

Each layer's result is reported even when a later one rejects the trade, so the
caller can see *where* an opportunity died rather than only that it did.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .arbitrage import TOLERANCE, Constraint, find_arbitrage, screen_arbitrage
from .orderbook import Level, simulate_fill

BookFetcher = Callable[[str], Awaitable[tuple[list[Level], list[Level]]]]

# A binary contract can never rationally cost more than its $1 payout, so an
# unbuyable leg is priced at the ceiling: the solver routes around it, and any
# basket forced to include it cannot clear the arbitrage threshold.
UNAVAILABLE_PRICE = 1.0


async def scan(
    fetch_book: BookFetcher,
    token_ids: list[str],
    constraints: list[Constraint],
    size: float,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Run all three layers over one dependency cluster.

    `token_ids` is positionally aligned with the constraint coefficients.
    `size` is the number of shares per leg to validate against the book.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if labels is not None and len(labels) != len(token_ids):
        raise ValueError(f"got {len(labels)} labels for {len(token_ids)} tokens")

    names = labels or list(token_ids)
    books = await asyncio.gather(*(fetch_book(token_id) for token_id in token_ids))

    # Touch prices: the best ask is what a taker pays for one share.
    touch: list[float] = []
    unavailable: list[str] = []
    for name, (_bids, asks) in zip(names, books, strict=True):
        best_ask = min((price for price, _ in asks), default=None)
        if best_ask is None:
            unavailable.append(name)
            touch.append(UNAVAILABLE_PRICE)
        else:
            touch.append(best_ask)

    layer1 = screen_arbitrage(touch, constraints)
    result: dict[str, Any] = {
        "tokens": names,
        "touch_prices": touch,
        "unavailable_legs": unavailable,
        "layer1_screen": layer1,
        "layer2_exact": None,
        "layer3_execution": None,
        "tradable": False,
    }
    if not layer1["arbitrage_possible"]:
        result["verdict"] = "no arbitrage: the LP bound rules one out at touch prices"
        return result

    layer2 = find_arbitrage(touch, constraints)
    result["layer2_exact"] = layer2
    if not layer2.get("is_arbitrage"):
        result["verdict"] = "no arbitrage: the exact solve found no profitable basket"
        return result

    layer3 = _validate_execution(
        [(names[i], books[i]) for i in layer2["selected"]], size, layer2["guaranteed_payoff"]
    )
    result["layer3_execution"] = layer3
    result["tradable"] = layer3["survives"]
    result["verdict"] = (
        f"tradable: {layer3['realized_edge_per_share']:+.4f} per share after walking the book"
        if layer3["survives"]
        else f"rejected at execution: {layer3['rejection_reason']}"
    )
    return result


def _validate_execution(
    legs: list[tuple[str, tuple[list[Level], list[Level]]]],
    size: float,
    guaranteed_payoff: float | None,
) -> dict[str, Any]:
    """Layer 3: re-price the chosen basket against real depth at `size`.

    Every leg must fill completely. A partially filled basket is not a hedged
    position — it is naked exposure on whichever legs did fill — so a short leg
    rejects the trade outright rather than scaling the basket down.
    """
    payoff = 1.0 if guaranteed_payoff is None else guaranteed_payoff
    details: list[dict[str, Any]] = []
    realized_cost = 0.0
    short_legs: list[str] = []
    depth_limit = float("inf")

    for name, (_bids, asks) in legs:
        fill = simulate_fill(asks, size, taking_asks=True)
        details.append({"token": name, **fill.as_dict()})
        depth_limit = min(depth_limit, sum(available for _price, available in asks))
        if not fill.fully_filled or fill.average_price is None:
            short_legs.append(name)
            continue
        realized_cost += fill.average_price

    realized_edge = payoff - realized_cost
    survives = not short_legs and realized_edge > TOLERANCE

    rejection_reason = None
    if short_legs:
        rejection_reason = f"insufficient depth on {', '.join(short_legs)} at {size} shares"
    elif not survives:
        rejection_reason = (
            f"edge of {realized_edge:+.4f} per share does not survive slippage at {size} shares"
        )

    return {
        "size": size,
        "legs": details,
        "realized_cost_per_share": realized_cost,
        "guaranteed_payoff": payoff,
        "realized_edge_per_share": realized_edge,
        "realized_profit": realized_edge * size if survives else 0.0,
        "max_fillable_size": None if depth_limit == float("inf") else depth_limit,
        "survives": survives,
        "rejection_reason": rejection_reason,
    }

"""Order-book fill simulation.

Pure functions with no I/O and no third-party dependencies: they take price
levels already fetched from a venue and answer "what would this order actually
cost". This is the execution-validation layer — an arbitrage that is profitable
at the touch is frequently unprofitable once it walks three levels deep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

Level = tuple[float, float]
"""A single book level as `(price, size)`, with size denominated in shares."""


@dataclass(frozen=True)
class FillSimulation:
    """The outcome of walking `requested_size` shares through one side of a book."""

    requested_size: float
    filled_size: float
    fully_filled: bool
    average_price: float | None
    notional: float
    levels_consumed: int
    best_price: float | None
    slippage_bps: float | None
    worst_price: float | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view for MCP tool responses."""
        return asdict(self)


def sort_levels(levels: list[Level], *, ascending: bool) -> list[Level]:
    """Sort book levels by price, dropping non-positive sizes.

    Venues disagree on the ordering they return (and some change it between
    endpoints), so callers sort explicitly rather than trusting wire order.
    """
    usable = [(price, size) for price, size in levels if size > 0]
    return sorted(usable, key=lambda level: level[0], reverse=not ascending)


def simulate_fill(levels: list[Level], size: float, *, taking_asks: bool) -> FillSimulation:
    """Walk `size` shares through `levels` and report the realized execution.

    `taking_asks` selects direction: a buy consumes asks cheapest-first, a sell
    consumes bids highest-first. Partial fills are reported rather than raised —
    "we can only get 60% of this at an acceptable price" is a normal answer that
    the sizing layer needs to see.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    book = sort_levels(levels, ascending=taking_asks)
    if not book:
        return FillSimulation(
            requested_size=size,
            filled_size=0.0,
            fully_filled=False,
            average_price=None,
            notional=0.0,
            levels_consumed=0,
            best_price=None,
            slippage_bps=None,
            worst_price=None,
        )

    best_price = book[0][0]
    remaining = size
    notional = 0.0
    consumed = 0
    worst_price = best_price

    for price, available in book:
        if remaining <= 0:
            break
        take = min(remaining, available)
        notional += take * price
        remaining -= take
        worst_price = price
        consumed += 1

    filled = size - remaining
    average_price = notional / filled if filled > 0 else None

    # Slippage is measured against the touch, signed so that a worse execution
    # is always positive regardless of trade direction.
    slippage_bps: float | None = None
    if average_price is not None and best_price > 0:
        drift = (average_price - best_price) if taking_asks else (best_price - average_price)
        slippage_bps = drift / best_price * 10_000

    return FillSimulation(
        requested_size=size,
        filled_size=filled,
        fully_filled=remaining <= 1e-9,
        average_price=average_price,
        notional=notional,
        levels_consumed=consumed,
        best_price=best_price,
        slippage_bps=slippage_bps,
        worst_price=worst_price if filled > 0 else None,
    )

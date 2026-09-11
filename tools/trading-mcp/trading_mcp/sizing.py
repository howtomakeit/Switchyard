"""Position sizing via a Kelly criterion modified for execution risk.

Textbook Kelly assumes the bet is placed at the quoted price with certainty.
Neither holds on a prediction-market CLOB: the fill price is the book-walked
average, and resting size can disappear before the order lands. Both are folded
in here, and the result is then scaled by a fractional-Kelly multiplier because
full Kelly is far too aggressive against an estimated edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PositionSize:
    """A sizing recommendation and the intermediate quantities behind it."""

    edge: float
    net_odds: float
    full_kelly_fraction: float
    recommended_fraction: float
    stake: float
    shares: float
    capped_by_limit: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view for MCP tool responses."""
        return asdict(self)


def kelly_position(
    true_probability: float,
    fill_price: float,
    bankroll: float,
    *,
    fill_probability: float = 1.0,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> PositionSize:
    """Size a long position in a binary outcome token paying 1.0 on resolution.

    `fill_price` should be the book-walked average from a depth simulation, not
    the touch. `fill_probability` discounts the edge by the chance the order
    never executes. A non-positive edge yields a zero-sized recommendation
    rather than a short.
    """
    if not 0.0 < fill_price < 1.0:
        raise ValueError("fill_price must be strictly between 0 and 1")
    if not 0.0 <= true_probability <= 1.0:
        raise ValueError("true_probability must be within [0, 1]")
    if not 0.0 <= fill_probability <= 1.0:
        raise ValueError("fill_probability must be within [0, 1]")
    if bankroll < 0:
        raise ValueError("bankroll must not be negative")

    # Net odds for a contract bought at `fill_price` and redeemed at 1.0.
    net_odds = (1.0 - fill_price) / fill_price
    edge = true_probability - fill_price
    full_kelly = (true_probability * net_odds - (1.0 - true_probability)) / net_odds

    # Execution risk scales the whole allocation: an edge you only capture some
    # of the time is proportionally smaller in expectation.
    adjusted = full_kelly * fill_probability * kelly_multiplier
    recommended = max(0.0, adjusted)
    capped = recommended > max_fraction
    if capped:
        recommended = max_fraction

    stake = bankroll * recommended
    return PositionSize(
        edge=edge,
        net_odds=net_odds,
        full_kelly_fraction=full_kelly,
        recommended_fraction=recommended,
        stake=stake,
        shares=stake / fill_price,
        capped_by_limit=capped,
    )

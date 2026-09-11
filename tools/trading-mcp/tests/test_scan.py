"""The three-layer pipeline, driven by synthetic order books."""

import pytest
from trading_mcp.arbitrage import Constraint
from trading_mcp.dependency import constraints_from_valid_outcomes
from trading_mcp.scan import scan

TOKENS = ["a_yes", "a_no", "b_yes", "b_no"]
LABELS = ["A:Yes", "A:No", "B:Yes", "B:No"]

# Election implies nomination, so (No, Yes) is impossible.
CONSTRAINTS = [
    Constraint.from_dict(raw)
    for raw in constraints_from_valid_outcomes(
        [["Yes", "Yes"], ["Yes", "No"], ["No", "No"]], ["Yes", "No"], ["Yes", "No"]
    )["constraints"]
]


def books(asks_by_token):
    """Build a fetcher returning fixed asks (and no bids) per token."""

    async def fetch(token_id):
        return [], asks_by_token[token_id]

    return fetch


DEEP_ARB = {
    "a_yes": [(0.30, 1000.0)],
    "a_no": [(0.70, 1000.0)],
    "b_yes": [(0.45, 1000.0)],
    "b_no": [(0.55, 1000.0)],
}


async def test_deep_book_arbitrage_is_tradable():
    result = await scan(books(DEEP_ARB), TOKENS, CONSTRAINTS, 100.0, labels=LABELS)

    assert result["tradable"]
    assert result["layer1_screen"]["arbitrage_possible"]
    assert result["layer2_exact"]["selected"] == [0, 3]
    assert result["layer3_execution"]["realized_edge_per_share"] == pytest.approx(0.15)
    assert result["layer3_execution"]["realized_profit"] == pytest.approx(15.0)


async def test_layer1_short_circuits_before_the_exact_solve():
    coherent = {**DEEP_ARB, "b_yes": [(0.20, 1000.0)], "b_no": [(0.80, 1000.0)]}

    result = await scan(books(coherent), TOKENS, CONSTRAINTS, 100.0, labels=LABELS)

    assert not result["tradable"]
    assert result["layer2_exact"] is None
    assert "LP bound" in result["verdict"]


async def test_slippage_can_destroy_an_edge_that_exists_at_the_touch():
    # The touch shows the 0.15 edge, but only 10 shares are there; the rest of
    # the size fills far worse.
    thin = {
        "a_yes": [(0.30, 10.0), (0.55, 1000.0)],
        "a_no": [(0.70, 1000.0)],
        "b_yes": [(0.45, 1000.0)],
        "b_no": [(0.55, 10.0), (0.70, 1000.0)],
    }

    result = await scan(books(thin), TOKENS, CONSTRAINTS, 100.0, labels=LABELS)

    assert result["layer2_exact"]["is_arbitrage"]
    assert not result["tradable"]
    assert "does not survive slippage" in result["layer3_execution"]["rejection_reason"]


async def test_a_leg_that_cannot_fill_rejects_the_whole_basket():
    shallow = {**DEEP_ARB, "b_no": [(0.55, 5.0)]}

    result = await scan(books(shallow), TOKENS, CONSTRAINTS, 100.0, labels=LABELS)

    assert not result["tradable"]
    assert "insufficient depth on B:No" in result["layer3_execution"]["rejection_reason"]
    assert result["layer3_execution"]["realized_profit"] == 0.0


async def test_an_empty_book_marks_the_leg_unavailable_rather_than_crashing():
    missing = {**DEEP_ARB, "b_no": []}

    result = await scan(books(missing), TOKENS, CONSTRAINTS, 100.0, labels=LABELS)

    assert result["unavailable_legs"] == ["B:No"]
    assert result["touch_prices"][3] == 1.0
    assert not result["tradable"]


async def test_smaller_size_can_still_be_tradable_where_larger_is_not():
    thin = {
        "a_yes": [(0.30, 20.0), (0.55, 1000.0)],
        "a_no": [(0.70, 1000.0)],
        "b_yes": [(0.45, 1000.0)],
        "b_no": [(0.55, 20.0), (0.70, 1000.0)],
    }

    assert (await scan(books(thin), TOKENS, CONSTRAINTS, 20.0, labels=LABELS))["tradable"]
    assert not (await scan(books(thin), TOKENS, CONSTRAINTS, 200.0, labels=LABELS))["tradable"]


async def test_labels_must_match_the_token_count():
    with pytest.raises(ValueError, match="labels"):
        await scan(books(DEEP_ARB), TOKENS, CONSTRAINTS, 10.0, labels=["only-one"])


async def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="size must be positive"):
        await scan(books(DEEP_ARB), TOKENS, CONSTRAINTS, 0.0)

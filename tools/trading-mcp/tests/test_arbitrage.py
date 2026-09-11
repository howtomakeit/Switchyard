"""Solver behavior on the payoff-cover programs the dependency detector produces."""

import pytest
from trading_mcp.arbitrage import Constraint, find_arbitrage, screen_arbitrage
from trading_mcp.dependency import constraints_from_valid_outcomes

# "X wins the nomination" (A) and "X wins the election" (B). Winning the
# election without the nomination is impossible, so that state is dropped.
NOMINATION = {
    "valid_outcomes": [["Yes", "Yes"], ["Yes", "No"], ["No", "No"]],
    "outcomes_a": ["Yes", "No"],
    "outcomes_b": ["Yes", "No"],
}
INDEPENDENT = {
    "valid_outcomes": [["Yes", "Yes"], ["Yes", "No"], ["No", "Yes"], ["No", "No"]],
    "outcomes_a": ["Yes", "No"],
    "outcomes_b": ["Yes", "No"],
}


def build(spec):
    return [Constraint.from_dict(raw) for raw in constraints_from_valid_outcomes(**spec)["constraints"]]


def test_dependency_creates_an_arbitrage_that_neither_market_shows_alone():
    # Both markets are internally coherent (each side sums to exactly 1.00), but
    # the election is priced above the nomination, which cannot happen.
    prices = [0.30, 0.70, 0.45, 0.55]

    result = find_arbitrage(prices, build(NOMINATION))

    assert result["is_arbitrage"]
    # Buy A:Yes and B:No -- one of them pays in every surviving state.
    assert result["selected"] == [0, 3]
    assert result["cost"] == pytest.approx(0.85)
    assert result["guaranteed_payoff"] == pytest.approx(1.0)
    assert result["edge"] == pytest.approx(0.15)


def test_the_same_prices_are_not_an_arbitrage_without_the_dependency():
    result = find_arbitrage([0.30, 0.70, 0.45, 0.55], build(INDEPENDENT))

    assert not result["is_arbitrage"]
    assert result["cost"] == pytest.approx(1.0)
    assert result["edge"] == pytest.approx(0.0)


def test_coherent_prices_yield_no_arbitrage_even_with_the_dependency():
    # Nomination priced above the election, as the logic requires.
    result = find_arbitrage([0.30, 0.70, 0.20, 0.80], build(NOMINATION))

    assert not result["is_arbitrage"]
    assert result["edge"] == pytest.approx(0.0)


def test_within_market_underpricing_is_found_without_any_dependency():
    # Both sides of market B for 0.90 guarantees a dollar.
    result = find_arbitrage([0.60, 0.60, 0.40, 0.50], build(INDEPENDENT))

    assert result["is_arbitrage"]
    assert result["selected"] == [2, 3]
    assert result["edge"] == pytest.approx(0.10)


def test_screen_bound_never_exceeds_the_exact_cost():
    prices = [0.30, 0.70, 0.45, 0.55]
    constraints = build(NOMINATION)

    assert screen_arbitrage(prices, constraints)["lower_bound"] <= find_arbitrage(
        prices, constraints
    )["cost"] + 1e-9


def test_screen_rules_out_arbitrage_cheaply():
    screen = screen_arbitrage([0.30, 0.70, 0.20, 0.80], build(NOMINATION))

    assert not screen["arbitrage_possible"]
    assert screen["lower_bound"] == pytest.approx(1.0)


def test_screen_flags_a_real_opportunity():
    screen = screen_arbitrage([0.30, 0.70, 0.45, 0.55], build(NOMINATION))

    assert screen["arbitrage_possible"]
    assert screen["lower_bound"] == pytest.approx(0.85)


def test_mismatched_coefficient_width_is_rejected():
    with pytest.raises(ValueError, match="2 coefficients"):
        find_arbitrage([0.5, 0.5, 0.5], [Constraint(coeffs=[1.0, 1.0], b=1.0)])


def test_unknown_sense_is_rejected():
    with pytest.raises(ValueError, match="sense must be"):
        Constraint.from_dict({"coeffs": [1.0], "b": 1.0, "sense": "!="})


def test_infeasible_constraints_report_rather_than_raise():
    contradictory = [
        Constraint(coeffs=[1.0, 1.0], b=2.0, sense="=="),
        Constraint(coeffs=[1.0, 1.0], b=0.0, sense="=="),
    ]

    result = find_arbitrage([0.5, 0.5], contradictory)

    assert result["status"] == "infeasible"
    assert result["solution"] is None


def test_payoff_is_undefined_without_cover_rows():
    result = find_arbitrage([0.5, 0.5], [Constraint(coeffs=[1.0, 1.0], b=1.0, sense="==")])

    assert result["guaranteed_payoff"] is None
    assert result["edge"] is None
    assert not result["is_arbitrage"]

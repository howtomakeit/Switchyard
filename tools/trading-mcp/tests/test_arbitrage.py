"""Solver behavior on the dependency structures the detector produces."""

import pytest
from trading_mcp.arbitrage import Constraint, find_arbitrage
from trading_mcp.dependency import constraints_from_valid_outcomes


def test_picks_the_cheapest_outcome_under_exactly_one():
    prices = [0.70, 0.25, 0.40]
    exactly_one = Constraint(coeffs=[1.0, 1.0, 1.0], b=1.0, sense="==")

    result = find_arbitrage(prices, [exactly_one])

    assert result["status"] == "optimal"
    assert result["selected"] == [1]
    assert result["cost"] == pytest.approx(0.25)


def test_forbidden_pair_forces_the_more_expensive_combination():
    # Two markets, two outcomes each; A=Yes with B=No is impossible.
    built = constraints_from_valid_outcomes(
        [["Yes", "Yes"], ["No", "Yes"], ["No", "No"]],
        outcomes_a=["Yes", "No"],
        outcomes_b=["Yes", "No"],
    )
    constraints = [Constraint.from_dict(raw) for raw in built["constraints"]]
    # Cheapest unconstrained pick would be A=Yes (0.10) with B=No (0.05).
    prices = [0.10, 0.90, 0.95, 0.05]

    result = find_arbitrage(prices, constraints)

    assert result["status"] == "optimal"
    assert result["selected"] == [1, 3]
    assert result["cost"] == pytest.approx(0.95)


def test_infeasible_constraints_report_rather_than_raise():
    contradictory = [
        Constraint(coeffs=[1.0, 1.0], b=2.0, sense="=="),
        Constraint(coeffs=[1.0, 1.0], b=0.0, sense="=="),
    ]

    result = find_arbitrage([0.5, 0.5], contradictory)

    assert result["status"] == "infeasible"
    assert result["solution"] is None


def test_mismatched_coefficient_width_is_rejected():
    with pytest.raises(ValueError, match="2 coefficients"):
        find_arbitrage([0.5, 0.5, 0.5], [Constraint(coeffs=[1.0, 1.0], b=1.0)])


def test_unknown_sense_is_rejected():
    with pytest.raises(ValueError, match="sense must be"):
        Constraint.from_dict({"coeffs": [1.0], "b": 1.0, "sense": "!="})


def test_constraint_builder_emits_one_equality_per_market():
    built = constraints_from_valid_outcomes(
        [["Yes", "Yes"]], outcomes_a=["Yes", "No"], outcomes_b=["Yes", "No"]
    )

    assert built["width"] == 4
    equalities = [c for c in built["constraints"] if c["sense"] == "=="]
    assert len(equalities) == 2
    # Three of the four pairs are impossible, so three forbidding rows.
    assert len([c for c in built["constraints"] if c["sense"] == "<="]) == 3

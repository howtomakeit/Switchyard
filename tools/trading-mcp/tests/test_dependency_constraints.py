"""Translation of dependency verdicts into payoff-cover matrices."""

import pytest
from trading_mcp.dependency import constraints_from_valid_outcomes


def test_each_feasible_state_becomes_one_cover_row():
    built = constraints_from_valid_outcomes(
        [["Yes", "Yes"], ["Yes", "No"], ["No", "No"]], ["Yes", "No"], ["Yes", "No"]
    )

    assert built["width"] == 4
    assert built["labels"] == ["A:Yes", "A:No", "B:Yes", "B:No"]
    assert len(built["constraints"]) == 3
    assert all(row["sense"] == ">=" and row["b"] == 1.0 for row in built["constraints"])


def test_a_row_marks_exactly_the_tokens_that_pay_in_that_state():
    built = constraints_from_valid_outcomes([["Yes", "No"]], ["Yes", "No"], ["Yes", "No"])

    # State (A=Yes, B=No) pays on A:Yes (index 0) and B:No (index 3).
    assert built["constraints"][0]["coeffs"] == [1.0, 0.0, 0.0, 1.0]
    assert built["states"] == [["Yes", "No"]]


def test_identically_named_outcomes_stay_distinguishable():
    built = constraints_from_valid_outcomes(
        [["Yes", "Yes"]], ["Yes", "No"], ["Yes", "No"]
    )

    assert built["labels"].count("A:Yes") == 1
    assert built["labels"].count("B:Yes") == 1


def test_multi_outcome_markets_are_supported():
    built = constraints_from_valid_outcomes(
        [["Alice", "Yes"], ["Bob", "No"], ["Carol", "No"]],
        ["Alice", "Bob", "Carol"],
        ["Yes", "No"],
    )

    assert built["width"] == 5
    assert built["constraints"][0]["coeffs"] == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_no_feasible_state_is_rejected():
    with pytest.raises(ValueError, match="no state of the world"):
        constraints_from_valid_outcomes([], ["Yes", "No"], ["Yes", "No"])


def test_a_hallucinated_outcome_name_is_rejected():
    with pytest.raises(ValueError, match="not an outcome of market A"):
        constraints_from_valid_outcomes([["Maybe", "Yes"]], ["Yes", "No"], ["Yes", "No"])


def test_a_malformed_pair_is_rejected():
    with pytest.raises(ValueError, match="pairs"):
        constraints_from_valid_outcomes([["Yes"]], ["Yes", "No"], ["Yes", "No"])

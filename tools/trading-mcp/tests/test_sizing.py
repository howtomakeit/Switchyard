"""Kelly sizing under execution risk."""

import pytest
from trading_mcp.sizing import kelly_position


def test_no_edge_yields_no_position():
    result = kelly_position(0.50, 0.50, bankroll=1000.0)

    assert result.edge == 0.0
    assert result.recommended_fraction == 0.0
    assert result.stake == 0.0


def test_negative_edge_is_floored_at_zero_rather_than_shorting():
    result = kelly_position(0.30, 0.60, bankroll=1000.0)

    assert result.full_kelly_fraction < 0
    assert result.recommended_fraction == 0.0


def test_full_kelly_matches_the_closed_form():
    # p=0.6 at a price of 0.5 gives net odds 1.0 and full Kelly of 0.2.
    result = kelly_position(0.60, 0.50, bankroll=1000.0, kelly_multiplier=1.0, max_fraction=1.0)

    assert result.net_odds == pytest.approx(1.0)
    assert result.full_kelly_fraction == pytest.approx(0.20)
    assert result.recommended_fraction == pytest.approx(0.20)


def test_execution_probability_scales_the_allocation():
    certain = kelly_position(0.60, 0.50, 1000.0, kelly_multiplier=1.0, max_fraction=1.0)
    risky = kelly_position(
        0.60, 0.50, 1000.0, fill_probability=0.5, kelly_multiplier=1.0, max_fraction=1.0
    )

    assert risky.recommended_fraction == pytest.approx(certain.recommended_fraction / 2)


def test_max_fraction_caps_the_stake_and_is_flagged():
    result = kelly_position(0.95, 0.10, bankroll=1000.0, kelly_multiplier=1.0, max_fraction=0.05)

    assert result.capped_by_limit
    assert result.recommended_fraction == 0.05
    assert result.stake == pytest.approx(50.0)
    assert result.shares == pytest.approx(500.0)


@pytest.mark.parametrize(
    ("probability", "price", "fill_probability"),
    [(0.5, 0.0, 1.0), (0.5, 1.0, 1.0), (1.5, 0.5, 1.0), (0.5, 0.5, 1.5)],
)
def test_out_of_range_inputs_are_rejected(probability, price, fill_probability):
    with pytest.raises(ValueError):
        kelly_position(probability, price, 1000.0, fill_probability=fill_probability)

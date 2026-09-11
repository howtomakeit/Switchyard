"""KL projection onto dependency constraint sets."""

import pytest
from trading_mcp.projection import bregman_projection


def test_already_feasible_prices_are_left_alone():
    # 0.6 + 0.5 >= 1.0 already holds, so nothing should move.
    result = bregman_projection([0.6, 0.5], [[1.0, 1.0]], [1.0])

    assert result["converged"]
    assert result["projected"] == pytest.approx([0.6, 0.5], abs=1e-4)
    assert result["divergence"] == pytest.approx(0.0, abs=1e-6)


def test_incoherent_prices_are_pushed_onto_the_constraint_boundary():
    # 0.3 + 0.3 = 0.6 violates the requirement that the pair sums to at least 1.
    result = bregman_projection([0.3, 0.3], [[1.0, 1.0]], [1.0])

    assert result["converged"]
    assert sum(result["projected"]) == pytest.approx(1.0, abs=1e-4)
    assert result["divergence"] > 0


def test_symmetric_inputs_receive_symmetric_corrections():
    result = bregman_projection([0.2, 0.2], [[1.0, 1.0]], [1.0])

    first, second = result["projected"]
    assert first == pytest.approx(second, abs=1e-6)


def test_projection_stays_inside_the_unit_interval():
    result = bregman_projection([0.99, 0.99], [[1.0, 1.0]], [1.9])

    assert all(0.0 < value < 1.0 for value in result["projected"])


def test_unconstrained_projection_is_the_identity():
    result = bregman_projection([0.4, 0.7], [], [])

    assert result["projected"] == pytest.approx([0.4, 0.7], abs=1e-4)
    assert result["residual"] == []


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="rows"):
        bregman_projection([0.5, 0.5], [[1.0, 1.0]], [1.0, 1.0])


def test_empty_theta_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        bregman_projection([], [], [])

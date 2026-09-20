"""Resolution-risk gating: the checks that stop a bad dependency from trading."""

from datetime import UTC, datetime, timedelta

import pytest
from trading_mcp.risk import annualized_return, assess_dependency_risk, score_opportunity

NOW = datetime(2028, 1, 1, tzinfo=UTC)


def market(end_offset_days, description="a described market"):
    return {
        "question": "a market",
        "description": description,
        "end_date": (NOW + timedelta(days=end_offset_days)).isoformat(),
    }


def verdict(**overrides):
    base = {
        "dependent": True,
        "confidence": 0.95,
        "excluded_states": [
            {"state": ["No", "Yes"], "justification": "B's rules require A to have resolved Yes"}
        ],
    }
    return {**base, **overrides}


def test_a_well_justified_dependency_is_tradable():
    result = assess_dependency_risk(verdict(), market(30), market(35), now=NOW)

    assert result["tradable"]
    assert result["blocking_risks"] == []
    assert result["capital_lockup_days"] == pytest.approx(35, abs=0.1)


def test_far_apart_resolution_dates_block_the_trade():
    # The nomination/election trap: A settles months before B's outcome exists,
    # so the excluded state can actually occur.
    result = assess_dependency_risk(verdict(), market(30), market(200), now=NOW)

    assert not result["tradable"]
    assert any("days apart" in risk for risk in result["blocking_risks"])
    assert result["resolution_gap_days"] == pytest.approx(170, abs=0.1)


def test_low_confidence_blocks_the_trade():
    result = assess_dependency_risk(verdict(confidence=0.6), market(30), market(35), now=NOW)

    assert not result["tradable"]
    assert any("confidence" in risk for risk in result["blocking_risks"])


def test_missing_confidence_blocks_the_trade():
    bad = verdict()
    del bad["confidence"]

    result = assess_dependency_risk(bad, market(30), market(35), now=NOW)

    assert any("no confidence score" in risk for risk in result["blocking_risks"])


def test_an_unjustified_exclusion_blocks_the_trade():
    result = assess_dependency_risk(
        verdict(excluded_states=[{"state": ["No", "Yes"], "justification": "  "}]),
        market(30),
        market(35),
        now=NOW,
    )

    assert not result["tradable"]
    assert any("justification" in risk for risk in result["blocking_risks"])


def test_a_dependency_excluding_nothing_cannot_be_an_arbitrage():
    result = assess_dependency_risk(verdict(excluded_states=[]), market(30), market(35), now=NOW)

    assert any("excludes no states" in risk for risk in result["blocking_risks"])


def test_an_independent_verdict_is_not_tradable():
    result = assess_dependency_risk(
        verdict(dependent=False), market(30), market(35), now=NOW
    )

    assert not result["tradable"]
    assert any("no dependency" in risk for risk in result["blocking_risks"])


def test_expired_markets_are_blocked():
    result = assess_dependency_risk(verdict(), market(-10), market(-5), now=NOW)

    assert any("already passed" in risk for risk in result["blocking_risks"])


def test_missing_end_dates_warn_rather_than_block():
    result = assess_dependency_risk(
        verdict(), {"question": "a", "description": "d"}, market(30), now=NOW
    )

    assert result["tradable"]
    assert any("end date" in warning for warning in result["warnings"])
    assert result["capital_lockup_days"] is None


def test_a_missing_description_warns():
    result = assess_dependency_risk(verdict(), market(30, ""), market(35), now=NOW)

    assert result["tradable"]
    assert any("no description" in warning for warning in result["warnings"])


def test_annualized_return_scales_with_holding_period():
    assert annualized_return(0.15, 0.85, 365) == pytest.approx(0.1765, abs=1e-3)
    assert annualized_return(0.15, 0.85, 30) == pytest.approx(2.147, abs=1e-2)
    assert annualized_return(0.15, 0.85, 0) is None
    assert annualized_return(0.15, 0.0, 30) is None


SCAN = {
    "tradable": True,
    "verdict": "tradable",
    "layer3_execution": {
        "realized_edge_per_share": 0.15,
        "realized_cost_per_share": 0.85,
        "size": 100.0,
    },
}


def test_scoring_combines_execution_and_resolution_risk():
    risk = assess_dependency_risk(verdict(), market(30), market(35), now=NOW)

    score = score_opportunity(SCAN, risk)

    assert score["actionable"]
    assert score["capital_required"] == pytest.approx(85.0)
    # 17.6% over 35 days annualizes to roughly 184%.
    assert score["annualized_return"] == pytest.approx(1.84, abs=0.05)


def test_resolution_risk_blocks_an_otherwise_perfect_execution():
    risk = assess_dependency_risk(verdict(), market(30), market(200), now=NOW)

    score = score_opportunity(SCAN, risk)

    assert not score["actionable"]
    assert any("days apart" in reason for reason in score["blocked_by"])


def test_failed_execution_blocks_a_sound_dependency():
    risk = assess_dependency_risk(verdict(), market(30), market(35), now=NOW)
    failed = {**SCAN, "tradable": False, "verdict": "rejected at execution: thin book"}

    score = score_opportunity(failed, risk)

    assert not score["actionable"]
    assert "thin book" in score["blocked_by"][0]


def test_a_basket_larger_than_the_bankroll_is_blocked():
    risk = assess_dependency_risk(verdict(), market(30), market(35), now=NOW)

    score = score_opportunity(SCAN, risk, bankroll=50.0)

    assert not score["actionable"]
    assert any("bankroll" in reason for reason in score["blocked_by"])

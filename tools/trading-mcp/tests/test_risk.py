"""Resolution-risk gating: the checks that stop a bad dependency from trading."""

from datetime import UTC, datetime, timedelta

import pytest
from trading_mcp.risk import (
    annualized_return,
    assess_dependency_risk,
    looks_date_bounded,
    score_opportunity,
)

NOW = datetime(2028, 1, 1, tzinfo=UTC)


def market(end_offset_days, description="a described market", question="a market"):
    return {
        "question": question,
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
        "truncation_risk": {"possible": False, "explanation": "A resolves on the event itself"},
    }
    return {**base, **overrides}


def test_a_well_justified_dependency_is_tradable():
    result = assess_dependency_risk(verdict(), market(30), market(35), now=NOW)

    assert result["tradable"]
    assert result["blocking_risks"] == []
    assert result["capital_lockup_days"] == pytest.approx(35, abs=0.1)


def test_the_canonical_trade_survives_a_months_long_gap():
    # "Will X win the nomination?" (July) and "Will X win the election?"
    # (November) are four months apart, but the nomination market resolves on
    # the event, so the implication holds the whole way. Rejecting this would
    # reject the entire opportunity set.
    nomination = market(180, question="Will X win the Republican nomination?")
    election = market(300, question="Will X win the 2028 presidential election?")

    result = assess_dependency_risk(verdict(), nomination, election, now=NOW)

    assert result["tradable"]
    assert result["resolution_gap_days"] == pytest.approx(120, abs=0.1)
    # The gap is priced as a cost, not treated as a disqualification.
    assert any("capital is locked" in warning for warning in result["warnings"])


def test_a_deadline_bounded_earlier_market_is_blocked():
    # The trap: A settles on a calendar deadline that can arrive before B's
    # outcome exists, making the excluded state reachable.
    deadline = market(180, question="Will X be the nominee by June 1?")
    election = market(300, question="Will X win the 2028 presidential election?")

    result = assess_dependency_risk(verdict(), deadline, election, now=NOW)

    assert not result["tradable"]
    assert any("calendar deadline" in risk for risk in result["blocking_risks"])


def test_a_model_reported_truncation_blocks_regardless_of_phrasing():
    risky = verdict(
        truncation_risk={"possible": True, "explanation": "A closes at the convention date"}
    )

    result = assess_dependency_risk(risky, market(30), market(200), now=NOW)

    assert not result["tradable"]
    assert any("can resolve before" in risk for risk in result["blocking_risks"])


def test_a_deadline_is_harmless_when_the_markets_resolve_together():
    # A deadline cannot truncate anything if both markets settle the same week.
    deadline = market(30, question="Will X be the nominee by June 1?")

    result = assess_dependency_risk(verdict(), deadline, market(33), now=NOW)

    assert result["tradable"]


def test_an_unaddressed_truncation_question_warns():
    silent = verdict()
    del silent["truncation_risk"]

    result = assess_dependency_risk(silent, market(30), market(33), now=NOW)

    assert result["tradable"]
    assert any("truncate" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("text", "bounded"),
    [
        ("Will X win the Republican nomination?", False),
        ("Will X win the 2028 presidential election?", False),
        ("Will X be the nominee by June 1?", True),
        ("Will the bill pass before March 2027?", True),
        ("Will X hold office as of January 20?", True),
        ("Will the merger close on or before closing?", True),
        ("Will X resign no later than the deadline?", True),
        ("Will the vote happen before the convention?", False),
    ],
)
def test_deadline_phrasing_detection(text, bounded):
    assert looks_date_bounded({"question": text, "description": ""}) is bounded


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
    risk = assess_dependency_risk(
        verdict(truncation_risk={"possible": True, "explanation": "deadline precedes outcome"}),
        market(30),
        market(200),
        now=NOW,
    )

    score = score_opportunity(SCAN, risk)

    assert not score["actionable"]
    assert any("can resolve before" in reason for reason in score["blocked_by"])


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

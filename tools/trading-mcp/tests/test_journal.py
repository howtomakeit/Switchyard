"""The journal as a measurement instrument, not just a log."""

import json

import pytest
from trading_mcp.journal import Journal, categorize_blocker, opportunity_id

TOKENS = ["tok-a", "tok-b"]

SCAN = {"tradable": True, "layer3_execution": {"size": 100.0, "max_fillable_size": 400.0}}
GOOD_SCORE = {
    "actionable": True,
    "blocked_by": [],
    "edge_per_share": 0.15,
    "cost_per_share": 0.85,
    "capital_required": 85.0,
    "capital_lockup_days": 35.0,
    "annualized_return": 1.84,
}


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.jsonl")


def test_opportunity_id_ignores_token_order():
    assert opportunity_id(["a", "b"]) == opportunity_id(["b", "a"])


def test_different_baskets_get_different_ids():
    assert opportunity_id(["a", "b"]) != opportunity_id(["a", "c"])


def test_an_observation_round_trips_as_jsonl(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE, labels=["A:Yes", "B:No"])

    lines = journal.path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["kind"] == "observation"
    assert entry["actionable"] is True
    assert entry["edge_per_share"] == 0.15
    assert entry["size"] == 100.0
    assert entry["at"].startswith("20")


def test_recording_requires_tokens(journal):
    with pytest.raises(ValueError, match="must not be empty"):
        journal.record_observation([], SCAN, GOOD_SCORE)


def test_a_truncated_final_line_costs_one_record_not_the_history(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE)
    with journal.path.open("a") as stream:
        stream.write('{"kind": "observation", "opportu')

    assert len(journal.entries()) == 1


def test_stats_on_an_empty_journal_are_safe(journal):
    stats = journal.stats()

    assert stats["observations"] == 0
    assert stats["actionable_rate"] is None
    assert stats["edge_per_share"] is None


def test_rejects_are_recorded_and_bucketed(journal):
    journal.record_observation(
        TOKENS,
        SCAN,
        {
            "actionable": False,
            "blocked_by": [
                "insufficient depth on B:No at 100 shares",
                "confidence 0.60 is below the 0.85 bar",
            ],
        },
    )

    stats = journal.stats()
    assert stats["actionable"] == 0
    assert stats["actionable_rate"] == 0.0
    assert stats["blocked_by"] == {"thin_book": 1, "low_confidence": 1}


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("model reports the earlier market can resolve before the later", "truncation"),
        ("the earlier market resolves on a calendar deadline", "truncation"),
        ("confidence 0.60 is below the 0.85 bar", "low_confidence"),
        ("excluded state {...} has no resolution-rule justification", "unjustified_exclusion"),
        ("verdict excludes no states, so it cannot produce an arbitrage", "unjustified_exclusion"),
        ("model found no dependency between these markets", "no_dependency"),
        ("insufficient depth on B:No at 100 shares", "thin_book"),
        ("edge of -0.01 does not survive slippage at 100 shares", "thin_book"),
        ("no arbitrage: the LP bound rules one out", "no_edge"),
        ("basket needs $85.00 but bankroll is $50.00", "bankroll"),
        ("something entirely unforeseen", "other"),
    ],
)
def test_blocker_categorization(reason, category):
    assert categorize_blocker(reason) == category


def test_actionable_rate_mixes_hits_and_misses(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE)
    journal.record_observation(["tok-c"], SCAN, {"actionable": False, "blocked_by": ["no arbitrage"]})

    stats = journal.stats()
    assert stats["observations"] == 2
    assert stats["actionable_rate"] == 0.5
    assert stats["distinct_opportunities"] == 2


def test_edge_and_capital_distributions_cover_only_actionable_rows(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE)
    journal.record_observation(TOKENS, SCAN, {**GOOD_SCORE, "edge_per_share": 0.05})
    journal.record_observation(
        ["tok-z"], SCAN, {"actionable": False, "blocked_by": ["no arbitrage"], "edge_per_share": 9.0}
    )

    edge = journal.stats()["edge_per_share"]
    assert edge["count"] == 2
    assert edge["max"] == 0.15
    assert edge["median"] == pytest.approx(0.10)


def test_a_surviving_recheck_measures_edge_lifetime(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE)
    journal.record_recheck(TOKENS, still_tradable=True, edge_per_share=0.12)

    stats = journal.stats()
    assert stats["rechecks"] == {"total": 1, "still_tradable": 1}
    assert stats["edge_lifetime_minutes"]["count"] == 1
    assert stats["edge_lifetime_minutes"]["min"] >= 0


def test_a_dead_recheck_does_not_count_toward_lifetime(journal):
    journal.record_observation(TOKENS, SCAN, GOOD_SCORE)
    journal.record_recheck(TOKENS, still_tradable=False)

    stats = journal.stats()
    assert stats["rechecks"]["still_tradable"] == 0
    assert stats["edge_lifetime_minutes"] is None


def test_open_opportunities_exclude_dead_and_settled(journal):
    journal.record_observation(["live-a", "live-b"], SCAN, GOOD_SCORE)
    journal.record_observation(["dead-a", "dead-b"], SCAN, GOOD_SCORE)
    journal.record_observation(["done-a", "done-b"], SCAN, GOOD_SCORE)
    journal.record_recheck(["dead-a", "dead-b"], still_tradable=False)
    journal.record_settlement(["done-a", "done-b"], realized_payoff=1.0)

    open_rows = journal.open_opportunities()

    assert [row["token_ids"] for row in open_rows] == [["live-a", "live-b"]]


def test_non_actionable_observations_are_not_open(journal):
    journal.record_observation(TOKENS, SCAN, {"actionable": False, "blocked_by": ["no arbitrage"]})

    assert journal.open_opportunities() == []


def test_a_settlement_that_pays_the_promise_is_marked_honoured(journal):
    entry = journal.record_settlement(TOKENS, realized_payoff=1.0)

    assert entry["paid_as_promised"]
    assert journal.stats()["settlements"]["paid_as_promised"] == 1


def test_a_short_settlement_reveals_a_wrong_dependency(journal):
    journal.record_settlement(TOKENS, realized_payoff=0.0, note="excluded state occurred")

    settlements = journal.stats()["settlements"]
    assert settlements["total"] == 1
    assert settlements["paid_as_promised"] == 0
    assert settlements["realized_payoff"]["median"] == 0.0


def test_the_journal_survives_reopening(tmp_path):
    path = tmp_path / "j.jsonl"
    Journal(path).record_observation(TOKENS, SCAN, GOOD_SCORE)

    assert Journal(path).stats()["observations"] == 1

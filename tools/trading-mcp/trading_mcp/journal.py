"""Append-only journal of arbitrage opportunities, for measuring the edge.

The strategy's expected return cannot be derived from theory: it depends on how
often real mispricings appear, how much depth they carry, what fraction
survives the risk gate, and — the question that decides whether an LLM-paced
system can trade at all — how long an edge persists before someone else takes
it. This records every opportunity seen so those numbers can be measured
before any capital is at risk.

Three entry kinds build the picture:

* `observation` — an opportunity as first seen, with all three scan layers and
  its score, actionable or not. Recording the rejects is the point: the
  histogram of *why* things fail says whether the strategy is starved of
  mispricings, of depth, or of trustworthy dependencies.
* `recheck` — the same basket re-observed later. The gap between the first
  observation and the last one that is still tradable is the edge's lifetime.
* `settlement` — what the basket actually paid at resolution, which is the only
  honest test of whether the "guaranteed" payoff was guaranteed.

Storage is JSON Lines: append-only, crash-safe, greppable, and loadable into
any analysis tool without a schema migration.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Free-text blocking reasons are bucketed so the failure histogram stays
# readable as the wording evolves. Order matters: first match wins.
_BLOCKER_CATEGORIES = (
    ("truncation", ("can resolve before", "calendar deadline")),
    ("low_confidence", ("confidence", "no confidence score")),
    ("unjustified_exclusion", ("justification", "excludes no states")),
    ("no_dependency", ("no dependency",)),
    ("expired", ("already passed",)),
    ("thin_book", ("insufficient depth", "does not survive slippage")),
    ("no_edge", ("no arbitrage",)),
    ("bankroll", ("bankroll",)),
)


def categorize_blocker(reason: str) -> str:
    """Bucket a free-text blocking reason for the failure histogram."""
    lowered = reason.lower()
    for category, needles in _BLOCKER_CATEGORIES:
        if any(needle in lowered for needle in needles):
            return category
    return "other"


def opportunity_id(token_ids: list[str]) -> str:
    """Return a stable id for the basket these tokens form.

    Keyed on the token set rather than on market ids or wording, so the same
    basket seen an hour later is recognized as the same opportunity.
    """
    digest = hashlib.sha256("\x00".join(sorted(token_ids)).encode())
    return digest.hexdigest()[:16]


def _summary(values: list[float]) -> dict[str, float] | None:
    """Return a compact distribution summary, or None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


class Journal:
    """A JSON Lines record of observed opportunities and what became of them."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _append(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Append one entry, stamping it with the current time."""
        entry["at"] = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        """Read every entry, skipping any line that is not valid JSON.

        A truncated final line (a crash mid-append) costs one record rather
        than the whole history.
        """
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows

    def record_observation(
        self,
        token_ids: list[str],
        scan_result: dict[str, Any],
        score: dict[str, Any],
        *,
        labels: list[str] | None = None,
        risk: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record an opportunity as first seen, whether or not it is actionable."""
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        execution = scan_result.get("layer3_execution") or {}
        return self._append(
            {
                "kind": "observation",
                "opportunity": opportunity_id(token_ids),
                "token_ids": token_ids,
                "labels": labels,
                "actionable": bool(score.get("actionable")),
                "blocked_by": score.get("blocked_by", []),
                "blocker_categories": sorted(
                    {categorize_blocker(r) for r in score.get("blocked_by", [])}
                ),
                "edge_per_share": score.get("edge_per_share"),
                "cost_per_share": score.get("cost_per_share"),
                "capital_required": score.get("capital_required"),
                "capital_lockup_days": score.get("capital_lockup_days"),
                "annualized_return": score.get("annualized_return"),
                "size": execution.get("size"),
                "max_fillable_size": execution.get("max_fillable_size"),
                "confidence": (risk or {}).get("confidence"),
                "note": note,
            }
        )

    def record_recheck(
        self, token_ids: list[str], *, still_tradable: bool, edge_per_share: float | None = None
    ) -> dict[str, Any]:
        """Record a later look at a basket, to measure how long its edge lasted."""
        return self._append(
            {
                "kind": "recheck",
                "opportunity": opportunity_id(token_ids),
                "still_tradable": still_tradable,
                "edge_per_share": edge_per_share,
            }
        )

    def record_settlement(
        self, token_ids: list[str], *, realized_payoff: float, note: str | None = None
    ) -> dict[str, Any]:
        """Record what a basket actually paid at resolution.

        `realized_payoff` is per share. A cover basket that was genuinely
        risk-free pays at least 1.0; anything less means an excluded state
        occurred and the dependency was wrong.
        """
        return self._append(
            {
                "kind": "settlement",
                "opportunity": opportunity_id(token_ids),
                "realized_payoff": realized_payoff,
                "paid_as_promised": realized_payoff >= 1.0 - 1e-9,
                "note": note,
            }
        )

    def open_opportunities(self) -> list[dict[str, Any]]:
        """List observed baskets that are neither settled nor known dead.

        These are what a scheduled recheck should look at again.
        """
        observations: dict[str, dict[str, Any]] = {}
        settled: set[str] = set()
        dead: set[str] = set()
        for entry in self.entries():
            key = entry.get("opportunity")
            if not isinstance(key, str):
                continue
            kind = entry.get("kind")
            if kind == "observation" and entry.get("actionable"):
                observations.setdefault(key, entry)
            elif kind == "settlement":
                settled.add(key)
            elif kind == "recheck" and not entry.get("still_tradable"):
                dead.add(key)
        return [v for k, v in observations.items() if k not in settled and k not in dead]

    def stats(self) -> dict[str, Any]:
        """Summarize what the journal says about the strategy so far.

        The headline numbers are the actionable rate (are there opportunities
        at all), the blocker histogram (what is killing them), edge lifetime
        (can a slow system capture them), and settlement accuracy (was the
        guarantee real).
        """
        rows = self.entries()
        observations = [r for r in rows if r.get("kind") == "observation"]
        actionable = [r for r in observations if r.get("actionable")]

        blockers: Counter[str] = Counter()
        for row in observations:
            for category in row.get("blocker_categories") or []:
                blockers[category] += 1

        first_seen: dict[str, str] = {}
        for row in observations:
            key = row.get("opportunity")
            if isinstance(key, str):
                first_seen.setdefault(key, row.get("at", ""))

        lifetimes: list[float] = []
        rechecks = [r for r in rows if r.get("kind") == "recheck"]
        for row in rechecks:
            key, start = row.get("opportunity"), first_seen.get(row.get("opportunity", ""))
            if not key or not start or not row.get("still_tradable"):
                continue
            try:
                delta = datetime.fromisoformat(row["at"]) - datetime.fromisoformat(start)
            except (KeyError, ValueError):
                continue
            lifetimes.append(delta.total_seconds() / 60.0)

        settlements = [r for r in rows if r.get("kind") == "settlement"]
        honoured = [r for r in settlements if r.get("paid_as_promised")]

        return {
            "entries": len(rows),
            "observations": len(observations),
            "distinct_opportunities": len(first_seen),
            "actionable": len(actionable),
            "actionable_rate": len(actionable) / len(observations) if observations else None,
            "blocked_by": dict(blockers.most_common()),
            "edge_per_share": _summary(
                [r["edge_per_share"] for r in actionable if r.get("edge_per_share") is not None]
            ),
            "capital_required": _summary(
                [r["capital_required"] for r in actionable if r.get("capital_required") is not None]
            ),
            "annualized_return": _summary(
                [
                    r["annualized_return"]
                    for r in actionable
                    if r.get("annualized_return") is not None
                ]
            ),
            "edge_lifetime_minutes": _summary(lifetimes),
            "rechecks": {
                "total": len(rechecks),
                "still_tradable": sum(1 for r in rechecks if r.get("still_tradable")),
            },
            "settlements": {
                "total": len(settlements),
                "paid_as_promised": len(honoured),
                "realized_payoff": _summary(
                    [
                        r["realized_payoff"]
                        for r in settlements
                        if r.get("realized_payoff") is not None
                    ]
                ),
            },
        }

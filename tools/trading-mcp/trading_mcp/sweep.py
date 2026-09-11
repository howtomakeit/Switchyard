"""Candidate generation and cached dependency sweeps across many markets.

Asking an LLM about every market pair is quadratic: a thousand markets is half
a million calls, almost all of which return "independent". Dependent markets
essentially always share a rare term — a candidate name, a ticker, an event —
so candidates are generated from an inverted index over distinctive terms and
only those pairs reach the model.

Terms appearing in a large share of markets ("will", "before", "2026") carry no
signal and would reconstruct the full quadratic pairing on their own, so they
are dropped by document frequency rather than by a fixed stopword list.

Pairs are ranked by the *rarest* term they share, not by how many they share.
Summing over shared terms ranks boilerplate overlap above substance: two
identically templated weather markets share four mid-frequency words and would
outrank two markets that share only a candidate's name, which is the pair that
is actually dependent.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from .cache import DependencyCache

DetectFn = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and any are as at be before by during for from in into is it its of on or
    than that the their there these this to up will with within would yes no""".split()
)


def terms(text: str, *, min_length: int = 3) -> set[str]:
    """Extract candidate index terms from market text."""
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) >= min_length and word not in _STOPWORDS
    }


def candidate_pairs(
    markets: list[dict[str, Any]],
    *,
    max_document_frequency: float = 0.25,
    min_shared_terms: int = 1,
    max_pairs: int | None = None,
) -> list[tuple[int, int]]:
    """Return index pairs worth sending to the dependency detector.

    Pairs are generated only from terms distinctive enough to be informative: a
    term present in more than `max_document_frequency` of markets is skipped.
    Surviving pairs are ordered by the inverse document frequency of the rarest
    term they share, so the most specific overlaps come first and `max_pairs`
    truncates the least informative rather than an arbitrary slice.
    """
    if not markets:
        return []

    texts = [str(market.get("question") or market.get("description") or "") for market in markets]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        for term in terms(text):
            postings[term].append(index)

    # A term must be allowed to appear in at least two markets, or no pair could
    # ever be generated from a small market set.
    cutoff = max(2, int(max_document_frequency * len(markets)))
    counts: dict[tuple[int, int], int] = defaultdict(int)
    rarest: dict[tuple[int, int], float] = {}

    for indices in postings.values():
        if len(indices) > cutoff or len(indices) < 2:
            continue
        # Rarer terms carry more signal; log(N/df) is the standard weighting.
        weight = math.log(len(markets) / len(indices))
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                pair = (left, right)
                counts[pair] += 1
                rarest[pair] = max(rarest.get(pair, 0.0), weight)

    ranked = sorted(
        (pair for pair, count in counts.items() if count >= min_shared_terms),
        key=lambda pair: (-rarest[pair], pair),
    )
    return ranked[:max_pairs] if max_pairs is not None else ranked


async def sweep_dependencies(
    markets: list[dict[str, Any]],
    detect: DetectFn,
    cache: DependencyCache,
    *,
    max_pairs: int | None = None,
    min_shared_terms: int = 1,
) -> dict[str, Any]:
    """Screen markets for dependencies, consulting the cache before the model.

    Returns only the dependent pairs, each with the verdict and the market
    indices it relates, alongside counts showing how much work the candidate
    filter and the cache avoided.
    """
    candidates = candidate_pairs(
        markets, min_shared_terms=min_shared_terms, max_pairs=max_pairs
    )
    total_possible = len(markets) * (len(markets) - 1) // 2

    dependent: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    called = 0

    for left, right in candidates:
        market_a, market_b = markets[left], markets[right]
        question_a = str(market_a.get("question") or "")
        question_b = str(market_b.get("question") or "")

        verdict = cache.get(question_a, question_b)
        if verdict is None:
            try:
                verdict = await detect(market_a, market_b)
            except Exception as exc:  # noqa: BLE001 - one bad pair must not end the sweep
                errors.append({"pair": [left, right], "error": str(exc)})
                continue
            called += 1
            cache.put(question_a, question_b, verdict)

        if verdict.get("dependent"):
            dependent.append({"pair": [left, right], "verdict": verdict})

    return {
        "markets": len(markets),
        "pairs_possible": total_possible,
        "pairs_screened": len(candidates),
        "model_calls": called,
        "cache": cache.stats(),
        "dependent": dependent,
        "errors": errors,
    }

"""Candidate generation and cached sweeps."""

from trading_mcp.cache import DependencyCache
from trading_mcp.sweep import candidate_pairs, sweep_dependencies, terms

MARKETS = [
    {"question": "Will Zephyr Callahan win the Democratic nomination?"},
    {"question": "Will Zephyr Callahan win the presidential election?"},
    {"question": "Will it rain in Reykjavik on Tuesday?"},
]


def test_terms_drop_stopwords_and_short_words():
    assert terms("Will the price of oil be above 90?") == {"price", "oil", "above"}


def test_related_markets_are_paired_and_unrelated_ones_are_not():
    pairs = candidate_pairs(MARKETS, min_shared_terms=2)

    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs


def test_one_rare_shared_term_outranks_many_boilerplate_ones():
    # The weather markets share four templated words; the two candidate markets
    # share only a name. The name is the one that signals a dependency.
    markets = [{"question": f"Will snowfall exceed 30cm in City{i} this winter?"} for i in range(8)]
    markets += [
        {"question": "Will Zephyr Callahan win the nomination?"},
        {"question": "Will Zephyr Callahan win the election?"},
    ]

    assert candidate_pairs(markets)[0] == (8, 9)


def test_pairs_are_ranked_by_shared_term_count():
    markets = [
        {"question": "Zephyr Callahan wins Iowa caucus"},
        {"question": "Zephyr Callahan wins Iowa primary"},
        {"question": "Zephyr Callahan resigns"},
    ]

    assert candidate_pairs(markets, min_shared_terms=1)[0] == (0, 1)


def test_max_pairs_truncates_the_weakest_candidates():
    assert len(candidate_pairs(MARKETS, min_shared_terms=1, max_pairs=1)) == 1


def test_ubiquitous_terms_do_not_generate_pairs():
    # "bitcoin" appears in every market, so it carries no signal and must not
    # reconstruct the full quadratic pairing on its own.
    markets = [{"question": f"Will bitcoin reach {n} thousand?"} for n in range(20)]

    assert candidate_pairs(markets, min_shared_terms=1) == []


def test_no_markets_yields_no_pairs():
    assert candidate_pairs([]) == []


async def test_sweep_returns_only_dependent_pairs(tmp_path):
    async def detect(a, b):
        return {"dependent": "nomination" in a["question"].lower()}

    result = await sweep_dependencies(
        MARKETS, detect, DependencyCache(tmp_path / "c.json"), min_shared_terms=2
    )

    assert [entry["pair"] for entry in result["dependent"]] == [[0, 1]]
    assert result["pairs_possible"] == 3
    assert result["pairs_screened"] == 1


async def test_cached_verdicts_are_not_re_requested(tmp_path):
    calls = []

    async def detect(a, b):
        calls.append((a["question"], b["question"]))
        return {"dependent": True}

    cache = DependencyCache(tmp_path / "c.json")
    first = await sweep_dependencies(MARKETS, detect, cache, min_shared_terms=2)
    second = await sweep_dependencies(MARKETS, detect, cache, min_shared_terms=2)

    assert first["model_calls"] == 1
    assert second["model_calls"] == 0
    assert len(calls) == 1
    assert second["dependent"] == first["dependent"]


async def test_one_failing_pair_does_not_abort_the_sweep(tmp_path):
    async def detect(a, b):
        raise RuntimeError("provider timeout")

    result = await sweep_dependencies(
        MARKETS, detect, DependencyCache(tmp_path / "c.json"), min_shared_terms=2
    )

    assert result["dependent"] == []
    assert result["errors"][0]["error"] == "provider timeout"

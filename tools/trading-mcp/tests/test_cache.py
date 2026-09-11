"""Persistence and keying of the dependency verdict cache."""

from trading_mcp.cache import DependencyCache, pair_key


def test_key_is_order_independent():
    assert pair_key("market A", "market B") == pair_key("market B", "market A")


def test_key_ignores_surrounding_whitespace():
    assert pair_key("  market A  ", "market B") == pair_key("market A", "market B")


def test_distinct_pairs_get_distinct_keys():
    assert pair_key("market A", "market B") != pair_key("market A", "market C")


def test_roundtrip_through_disk(tmp_path):
    path = tmp_path / "cache.json"
    DependencyCache(path).put("A", "B", {"dependent": True})

    reopened = DependencyCache(path)

    assert reopened.get("A", "B") == {"dependent": True}
    assert reopened.get("B", "A") == {"dependent": True}
    assert len(reopened) == 1


def test_miss_returns_none_and_is_counted(tmp_path):
    cache = DependencyCache(tmp_path / "cache.json")

    assert cache.get("A", "B") is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0


def test_hit_rate_reflects_lookups(tmp_path):
    cache = DependencyCache(tmp_path / "cache.json")
    cache.put("A", "B", {"dependent": False})
    cache.get("A", "B")
    cache.get("A", "C")

    assert cache.stats()["hit_rate"] == 0.5


def test_a_corrupt_cache_file_starts_empty_instead_of_raising(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")

    cache = DependencyCache(path)

    assert len(cache) == 0
    cache.put("A", "B", {"dependent": True})
    assert DependencyCache(path).get("A", "B") == {"dependent": True}


def test_missing_parent_directory_is_created(tmp_path):
    cache = DependencyCache(tmp_path / "nested" / "dir" / "cache.json")
    cache.put("A", "B", {"dependent": True})

    assert (tmp_path / "nested" / "dir" / "cache.json").exists()

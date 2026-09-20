"""Venue client behavior against a mocked CLOB."""

import httpx
import pytest
import respx
from trading_mcp.config import Settings
from trading_mcp.polymarket import PolymarketClient

BOOK = {
    "bids": [{"price": "0.44", "size": "100"}, {"price": "0.40", "size": "250"}],
    "asks": [{"price": "0.52", "size": "60"}, {"price": "0.48", "size": "40"}],
}


def make_settings(**overrides) -> Settings:
    base = {
        "live": False,
        "max_order_usd": 50.0,
        "clob_url": "https://clob.test",
        "data_url": "https://data.test",
        "private_key": None,
        "funder_address": None,
        "signature_type": 0,
        "llm_url": "https://llm.test/v1",
        "llm_model": "grok-4",
        "llm_api_key": None,
        "auth_token": None,
        "request_timeout": 5.0,
        "cache_path": "/tmp/unused-cache.json",
        "journal_path": "/tmp/unused-journal.jsonl",
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def client():
    return PolymarketClient(make_settings())


@respx.mock
async def test_quote_derives_mid_and_spread_from_the_book(client):
    respx.get("https://clob.test/book").mock(return_value=httpx.Response(200, json=BOOK))

    quote = await client.quote("tok-1")

    assert quote["best_bid"] == 0.44
    assert quote["best_ask"] == 0.48
    assert quote["mid"] == pytest.approx(0.46)
    assert quote["spread"] == pytest.approx(0.04)
    await client.aclose()


@respx.mock
async def test_quote_handles_a_one_sided_book(client):
    respx.get("https://clob.test/book").mock(
        return_value=httpx.Response(200, json={"bids": BOOK["bids"], "asks": []})
    )

    quote = await client.quote("tok-1")

    assert quote["best_ask"] is None
    assert quote["mid"] is None
    assert quote["spread"] is None
    await client.aclose()


@respx.mock
async def test_malformed_levels_are_skipped_not_fatal(client):
    respx.get("https://clob.test/book").mock(
        return_value=httpx.Response(
            200,
            json={"bids": [{"price": "oops", "size": "1"}, {"price": "0.4", "size": "1"}], "asks": []},
        )
    )

    quote = await client.quote("tok-1")

    assert quote["bid_levels"] == 1
    assert quote["best_bid"] == 0.4
    await client.aclose()


@respx.mock
async def test_depth_buy_walks_the_asks_cheapest_first(client):
    respx.get("https://clob.test/book").mock(return_value=httpx.Response(200, json=BOOK))

    fill = await client.depth("tok-1", "buy", 50.0)

    assert fill.fully_filled
    assert fill.best_price == 0.48
    # 40 @ 0.48 + 10 @ 0.52 = 24.40 over 50 shares.
    assert fill.notional == pytest.approx(24.40)
    await client.aclose()


@respx.mock
async def test_depth_reports_a_partial_fill_beyond_available_size(client):
    respx.get("https://clob.test/book").mock(return_value=httpx.Response(200, json=BOOK))

    fill = await client.depth("tok-1", "buy", 500.0)

    assert not fill.fully_filled
    assert fill.filled_size == 100.0
    await client.aclose()


async def test_depth_rejects_an_unknown_side(client):
    with pytest.raises(ValueError, match="side must be"):
        await client.depth("tok-1", "hodl", 10.0)
    await client.aclose()


@respx.mock
async def test_http_errors_propagate(client):
    respx.get("https://clob.test/book").mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        await client.quote("tok-1")
    await client.aclose()


def test_live_placement_refuses_while_in_paper_mode(client):
    with pytest.raises(RuntimeError, match="live trading is disabled"):
        client.place_live_order("tok-1", "buy", 0.5, 10.0)


def test_live_mode_without_a_key_is_rejected():
    client = PolymarketClient(make_settings(live=True))

    with pytest.raises(RuntimeError, match="POLYMARKET_PRIVATE_KEY"):
        client.place_live_order("tok-1", "buy", 0.5, 10.0)


@respx.mock
async def test_positions_are_read_without_credentials(client):
    respx.get("https://data.test/positions").mock(
        return_value=httpx.Response(
            200, json=[{"asset": "tok-1", "size": "12.5", "avgPrice": "0.44"}]
        )
    )

    positions = await client.positions("0xabc")

    assert positions == [{"asset": "tok-1", "size": "12.5", "avgPrice": "0.44"}]
    await client.aclose()


@respx.mock
async def test_positions_tolerates_a_wrapped_payload(client):
    respx.get("https://data.test/positions").mock(
        return_value=httpx.Response(200, json={"data": [{"asset": "tok-1"}]})
    )

    assert await client.positions("0xabc") == [{"asset": "tok-1"}]
    await client.aclose()


def test_cancelling_requires_live_mode(client):
    with pytest.raises(RuntimeError, match="live trading is disabled"):
        client.cancel_orders(["order-1"])
    with pytest.raises(RuntimeError, match="live trading is disabled"):
        client.cancel_all_orders()


def test_listing_open_orders_requires_live_mode(client):
    with pytest.raises(RuntimeError, match="live trading is disabled"):
        client.open_orders()


def test_cancelling_nothing_is_rejected_before_touching_the_venue():
    live = PolymarketClient(make_settings(live=True, private_key="0xkey"))

    with pytest.raises(ValueError, match="must not be empty"):
        live.cancel_orders([])

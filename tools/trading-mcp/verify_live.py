#!/usr/bin/env python3
"""Check every assumption this server makes about the Polymarket API.

The server's request and response shapes were written from documentation and
covered with mocks. Mocks prove the parsing logic, not that the wire format
matches. Run this wherever outbound access to polymarket.com is allowed:

    python verify_live.py

It is strictly read-only — it places no orders and needs no credentials — and
exits non-zero if any assumption fails, so it can gate a deploy.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx
from trading_mcp.config import Settings
from trading_mcp.orderbook import simulate_fill
from trading_mcp.polymarket import PolymarketClient

GAMMA_URL = "https://gamma-api.polymarket.com"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    """Record one assumption check and print it as it happens."""
    results.append((status, name, detail))
    marker = {PASS: "  ok ", FAIL: "FAIL ", WARN: "warn "}[status]
    print(f"{marker} {name}" + (f"\n        {detail}" if detail else ""))


async def find_liquid_token(http: httpx.AsyncClient) -> tuple[str, dict[str, Any]] | None:
    """Pick an active market with a real book to test against."""
    response = await http.get(
        f"{GAMMA_URL}/markets",
        params={"active": "true", "closed": "false", "order": "volume24hr",
                "ascending": "false", "limit": 25},
    )
    response.raise_for_status()
    payload = response.json()
    markets = payload if isinstance(payload, list) else payload.get("data", [])
    record(PASS if markets else FAIL, "Gamma /markets returns markets",
           f"{len(markets)} markets returned")
    if not markets:
        return None

    sample = markets[0]
    expected = {"question", "description", "outcomes", "clobTokenIds", "endDate"}
    missing = expected - set(sample)
    record(PASS if not missing else WARN, "Gamma market carries the fields we read",
           f"missing: {sorted(missing)}" if missing else "")

    for market in markets:
        raw_ids = market.get("clobTokenIds")
        try:
            ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        except (json.JSONDecodeError, TypeError):
            continue
        if ids:
            record(PASS, "clobTokenIds parses into token ids",
                   f"{type(raw_ids).__name__} -> {len(ids)} ids")
            return str(ids[0]), market
    record(FAIL, "clobTokenIds parses into token ids", "no market yielded usable ids")
    return None


async def check_book(client: PolymarketClient, http: httpx.AsyncClient, token_id: str) -> None:
    """Verify the CLOB book shape, ordering, and our derived quote."""
    response = await http.get(f"{client._settings.clob_url}/book", params={"token_id": token_id})
    response.raise_for_status()
    raw = response.json()

    record(PASS if isinstance(raw, dict) else FAIL, "CLOB /book returns an object",
           f"keys: {sorted(raw)[:8]}" if isinstance(raw, dict) else repr(raw)[:120])
    for side in ("bids", "asks"):
        entries = raw.get(side)
        ok = isinstance(entries, list)
        record(PASS if ok else FAIL, f"/book has a '{side}' list",
               "" if ok else f"got {type(entries).__name__}")
        if ok and entries:
            first = entries[0]
            has_fields = isinstance(first, dict) and {"price", "size"} <= set(first)
            record(PASS if has_fields else FAIL, f"{side} levels carry price and size",
                   f"sample: {first}")
            if has_fields:
                try:
                    float(first["price"]), float(first["size"])
                    record(PASS, f"{side} price/size parse as numbers", f"sample: {first}")
                except (TypeError, ValueError):
                    record(FAIL, f"{side} price/size parse as numbers", f"sample: {first}")

    # We sort defensively rather than trusting wire order; report what arrives
    # so a future change in ordering is visible rather than silent.
    for side in ("bids", "asks"):
        prices = [float(e["price"]) for e in raw.get(side, []) if "price" in e]
        if len(prices) > 1:
            order = "ascending" if prices == sorted(prices) else (
                "descending" if prices == sorted(prices, reverse=True) else "unsorted")
            record(PASS, f"{side} wire ordering observed", f"{order} ({len(prices)} levels)")

    quote = await client.quote(token_id)
    record(PASS, "quote() derives a top of book", json.dumps(quote))

    bid, ask = quote["best_bid"], quote["best_ask"]
    if bid is not None and ask is not None:
        sane = 0.0 < bid <= ask < 1.0
        record(PASS if sane else FAIL, "best bid <= best ask, both inside (0,1)",
               f"bid={bid} ask={ask}")


async def check_price_endpoints(
    client: PolymarketClient, http: httpx.AsyncClient, token_id: str, quote: dict[str, Any]
) -> None:
    """Cross-check our derived mid against the venue's own endpoints."""
    base = client._settings.clob_url
    try:
        midpoint = (await http.get(f"{base}/midpoint", params={"token_id": token_id})).json()
        theirs = float(midpoint.get("mid"))
    except (httpx.HTTPError, TypeError, ValueError, AttributeError) as exc:
        record(WARN, "CLOB /midpoint agrees with our derived mid", f"unavailable: {exc}")
        return
    ours = quote.get("mid")
    if ours is None:
        record(WARN, "CLOB /midpoint agrees with our derived mid", "one-sided book")
        return
    close = abs(ours - theirs) < 0.02
    record(PASS if close else FAIL, "CLOB /midpoint agrees with our derived mid",
           f"ours={ours:.4f} theirs={theirs:.4f}")


async def check_depth(client: PolymarketClient, token_id: str) -> None:
    """Verify depth simulation runs against a real book and stays self-consistent."""
    bids, asks = await client.fetch_book(token_id)
    if not asks:
        record(WARN, "depth simulation against a live book", "no asks on this token")
        return
    total = sum(size for _price, size in asks)
    fill = simulate_fill(asks, min(total, 10.0), taking_asks=True)
    best = min(price for price, _ in asks)
    consistent = fill.average_price is not None and fill.average_price >= best - 1e-9
    record(PASS if consistent else FAIL, "depth simulation against a live book",
           f"vwap={fill.average_price} best_ask={best} filled={fill.filled_size}")


async def main() -> int:
    settings = Settings.from_env()
    client = PolymarketClient(settings)
    http = httpx.AsyncClient(timeout=settings.request_timeout)
    print(f"Verifying against {settings.clob_url} and {GAMMA_URL}\n")

    try:
        found = await find_liquid_token(http)
        if found is None:
            return 1
        token_id, market = found
        print(f"\nUsing: {str(market.get('question'))[:70]}\n  token {token_id}\n")

        await check_book(client, http, token_id)
        await check_price_endpoints(client, http, token_id, await client.quote(token_id))
        await check_depth(client, token_id)
    except httpx.HTTPError as exc:
        record(FAIL, "network reachable", str(exc))
    finally:
        await client.aclose()
        await http.aclose()

    failures = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n{len(results) - len(failures) - len(warns)} passed, "
          f"{len(warns)} warnings, {len(failures)} failed")
    if failures:
        print("\nDo not trade until these are fixed:")
        for _status, name, detail in failures:
            print(f"  - {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

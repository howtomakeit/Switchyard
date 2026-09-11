"""Polymarket CLOB access: quotes, books, and order placement.

Read paths go straight to the CLOB REST API over httpx. The write path is
deliberately split: paper orders are simulated against the live book in this
process, and live orders are delegated to `py-clob-client`, which owns the
EIP-712 signing that real placement requires.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .orderbook import FillSimulation, Level, simulate_fill

VALID_SIDES = ("buy", "sell")


def _levels(raw: Any) -> list[Level]:
    """Coerce a CLOB book side into `(price, size)` floats.

    The API returns decimal strings; anything unparseable is dropped rather
    than crashing a quote request over one malformed level.
    """
    levels: list[Level] = []
    for entry in raw or []:
        try:
            levels.append((float(entry["price"]), float(entry["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return levels


class PolymarketClient:
    """Thin async client over the Polymarket CLOB and Gamma APIs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(timeout=settings.request_timeout)

    async def aclose(self) -> None:
        """Release the underlying connection pool."""
        await self._http.aclose()

    async def fetch_book(self, token_id: str) -> tuple[list[Level], list[Level]]:
        """Return `(bids, asks)` for an outcome token."""
        response = await self._http.get(
            f"{self._settings.clob_url}/book", params={"token_id": token_id}
        )
        response.raise_for_status()
        payload = response.json()
        return _levels(payload.get("bids")), _levels(payload.get("asks"))

    async def quote(self, token_id: str) -> dict[str, Any]:
        """Return the top of book plus derived mid and spread.

        A single `/book` call answers all of it, so this avoids the extra
        `/price` and `/midpoint` round trips a quote would otherwise cost.
        """
        bids, asks = await self.fetch_book(token_id)
        best_bid = max((price for price, _ in bids), default=None)
        best_ask = min((price for price, _ in asks), default=None)

        mid: float | None = None
        spread: float | None = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

        return {
            "token_id": token_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread": spread,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
        }

    async def depth(self, token_id: str, side: str, size: float) -> FillSimulation:
        """Simulate taking `size` shares on `side` against the live book."""
        if side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
        bids, asks = await self.fetch_book(token_id)
        taking_asks = side == "buy"
        return simulate_fill(asks if taking_asks else bids, size, taking_asks=taking_asks)

    def place_live_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> dict[str, Any]:
        """Sign and post a real limit order through `py-clob-client`.

        Imported lazily so the whole server does not require the signing stack
        (and its web3 dependency tree) just to serve quotes in paper mode.
        """
        key = self._settings.require_live_credentials()

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError as exc:  # pragma: no cover - exercised only in live mode
            raise RuntimeError(
                "live trading requires py-clob-client; install it with "
                "`pip install py-clob-client`"
            ) from exc

        client = ClobClient(
            self._settings.clob_url,
            key=key,
            chain_id=137,
            signature_type=self._settings.signature_type,
            funder=self._settings.funder_address,
        )
        client.set_api_creds(client.create_or_derive_api_creds())

        order = client.create_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY if side == "buy" else SELL,
            )
        )
        return dict(client.post_order(order))

#!/usr/bin/env python3
"""MCP server exposing Polymarket trading and arbitrage tools.

Run over stdio for a local MCP client, or over streamable HTTP when the server
needs to be reachable by a hosted model (Grok, Claude) through a tunnel:

    python server.py --transport streamable-http --port 3001
    ngrok http 3001

Order placement is simulated unless `TRADING_MCP_MODE=live` is set, and every
order is checked against `TRADING_MCP_MAX_ORDER_USD` first.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hmac
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from trading_mcp.arbitrage import Constraint, find_arbitrage, screen_arbitrage
from trading_mcp.cache import DependencyCache
from trading_mcp.config import Settings
from trading_mcp.dependency import constraints_from_valid_outcomes, detect_dependency
from trading_mcp.polymarket import VALID_SIDES, PolymarketClient
from trading_mcp.projection import bregman_projection
from trading_mcp.scan import scan
from trading_mcp.sizing import kelly_position
from trading_mcp.sweep import sweep_dependencies

logger = logging.getLogger("trading_mcp")

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def anticipated(fn: F) -> F:
    """Surface expected failures to the model instead of hiding them as crashes.

    The MCP SDK reports an arbitrary exception as a bare "Error executing tool
    <name>" with the message withheld, which leaves the caller unable to tell a
    rejected notional from an unreachable venue. Domain modules raise ordinary
    `ValueError`/`RuntimeError`, so translate those — and upstream HTTP errors —
    into `ToolError`, whose message does reach the model.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            raise ToolError(
                f"upstream returned HTTP {exc.response.status_code} for {exc.request.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"upstream request failed: {exc}") from exc
        except (ValueError, RuntimeError, KeyError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


SETTINGS = Settings.from_env()
CLIENT = PolymarketClient(SETTINGS)
LLM_HTTP = httpx.AsyncClient(timeout=SETTINGS.request_timeout)
CACHE = DependencyCache(SETTINGS.cache_path)


@contextlib.asynccontextmanager
async def lifespan(_server: MCPServer[None]) -> AsyncIterator[None]:
    """Close both connection pools when the server shuts down."""
    try:
        yield None
    finally:
        await CLIENT.aclose()
        await LLM_HTTP.aclose()


mcp: MCPServer[None] = MCPServer(
    name="polymarket-trading",
    instructions=(
        "Tools for quoting, depth-checking, and trading Polymarket outcome tokens, "
        "plus a solver for cross-market arbitrage over logically dependent markets. "
        "Call check_orderbook_depth before place_order: quoted prices at the touch "
        "rarely survive a full-size order. place_order is simulated unless the "
        "operator started the server in live mode."
    ),
    lifespan=lifespan,
)


@mcp.tool()
@anticipated
async def get_market_price(token_id: str) -> dict[str, Any]:
    """Get the current best bid, best ask, mid, and spread for an outcome token.

    Args:
        token_id: Polymarket CLOB token id for a single outcome (not a market id).
    """
    return await CLIENT.quote(token_id)


@mcp.tool()
@anticipated
async def check_orderbook_depth(token_id: str, side: str, size: float) -> dict[str, Any]:
    """Simulate an order against the live book and report the realized fill.

    Returns the volume-weighted average price, how many levels the order would
    consume, slippage against the touch, and whether the full size is available.

    Args:
        token_id: Polymarket CLOB token id for a single outcome.
        side: "buy" to walk the asks, "sell" to walk the bids.
        size: Number of shares to simulate.
    """
    simulation = await CLIENT.depth(token_id, side, size)
    return {"token_id": token_id, "side": side, **simulation.as_dict()}


@mcp.tool()
@anticipated
async def place_order(token_id: str, side: str, price: float, size: float) -> dict[str, Any]:
    """Place a limit order, simulating it unless the server runs in live mode.

    In paper mode the order is filled against the live book and the result is
    reported without any capital moving. In live mode the order is signed and
    posted to Polymarket. Either way the notional is rejected if it exceeds the
    server's configured per-order cap.

    Args:
        token_id: Polymarket CLOB token id for a single outcome.
        side: "buy" or "sell".
        price: Limit price in (0, 1).
        size: Number of shares.
    """
    if side not in VALID_SIDES:
        raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
    if not 0.0 < price < 1.0:
        raise ValueError("price must be strictly between 0 and 1")
    if size <= 0:
        raise ValueError("size must be positive")

    notional = price * size
    if notional > SETTINGS.max_order_usd:
        raise ValueError(
            f"order notional ${notional:.2f} exceeds the ${SETTINGS.max_order_usd:.2f} "
            "per-order cap (TRADING_MCP_MAX_ORDER_USD)"
        )

    if not SETTINGS.live:
        simulation = await CLIENT.depth(token_id, side, size)
        return {
            "mode": "paper",
            "token_id": token_id,
            "side": side,
            "limit_price": price,
            "requested_notional": notional,
            "simulated_fill": simulation.as_dict(),
            "note": "no capital moved; start with TRADING_MCP_MODE=live to trade for real",
        }

    logger.warning("placing LIVE order: %s %s %s @ %s", side, size, token_id, price)
    receipt = CLIENT.place_live_order(token_id, side, price, size)
    return {"mode": "live", "token_id": token_id, "side": side, "receipt": receipt}


@mcp.tool()
@anticipated
async def detect_market_dependency(
    market_a: dict[str, Any], market_b: dict[str, Any]
) -> dict[str, Any]:
    """Ask the configured reasoning model whether two markets are logically dependent.

    Returns whether the resolutions constrain each other and, if so, the list of
    jointly possible outcome pairs. Feed those to build_dependency_constraints.

    Args:
        market_a: Market dict with at least "question"; "description" and "outcomes" help.
        market_b: The market to compare against, in the same shape.
    """
    return await detect_dependency(SETTINGS, LLM_HTTP, market_a, market_b)


@mcp.tool()
@anticipated
async def build_dependency_constraints(
    valid_outcomes: list[list[str]], outcomes_a: list[str], outcomes_b: list[str]
) -> dict[str, Any]:
    """Convert jointly possible outcome pairs into constraints for find_arbitrage_basket.

    Flattens the two markets into one outcome vector (`outcomes_a` then
    `outcomes_b`) and emits an "exactly one" equality per market plus a
    forbidding inequality for every impossible pair.

    Args:
        valid_outcomes: Pairs of jointly possible outcomes, as [[a_outcome, b_outcome], ...].
        outcomes_a: Ordered outcome names for market A.
        outcomes_b: Ordered outcome names for market B.
    """
    return constraints_from_valid_outcomes(valid_outcomes, outcomes_a, outcomes_b)


@mcp.tool()
@anticipated
async def find_arbitrage_basket(
    prices: list[float], constraints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Find the cheapest basket of outcome tokens satisfying the dependency constraints.

    Solves a binary program with OR-Tools/SCIP. Compare the returned cost
    against the payoff guaranteed by the constraint set to decide whether the
    basket is an arbitrage.

    Args:
        prices: Price per outcome token, aligned with the constraint coefficients.
        constraints: Each {"coeffs": [...], "b": float, "sense": ">=" | "<=" | "=="}.
    """
    parsed = [Constraint.from_dict(raw) for raw in constraints]
    return find_arbitrage(prices, parsed)


@mcp.tool()
@anticipated
async def project_prices(
    theta: list[float], a_matrix: list[list[float]], b_vector: list[float]
) -> dict[str, Any]:
    """Project incoherent market prices onto the feasible set under KL divergence.

    Use this when observed prices violate a dependency constraint: the result is
    the nearest coherent probability vector, and the gap between it and `theta`
    localizes the mispricing.

    Args:
        theta: Observed prices, each strictly in (0, 1).
        a_matrix: Constraint matrix A for A*mu >= b.
        b_vector: Right-hand side b.
    """
    return bregman_projection(theta, a_matrix, b_vector)


@mcp.tool()
@anticipated
async def screen_for_arbitrage(
    prices: list[float], constraints: list[dict[str, Any]]
) -> dict[str, Any]:
    """Layer 1: cheaply rule out arbitrage in a cluster using the LP relaxation.

    Far faster than the exact solve and safe to run across every candidate
    cluster. `arbitrage_possible=false` proves none exists at these prices;
    `true` means find_arbitrage_basket is worth running.

    Args:
        prices: Price per outcome token, aligned with the constraint coefficients.
        constraints: Cover rows from build_dependency_constraints.
    """
    return screen_arbitrage(prices, [Constraint.from_dict(raw) for raw in constraints])


@mcp.tool()
@anticipated
async def scan_for_arbitrage(
    token_ids: list[str],
    constraints: list[dict[str, Any]],
    size: float,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline over one dependency cluster against live books.

    Fetches every leg's book, screens with the LP relaxation, solves exactly,
    then re-prices the winning basket against real depth at `size`. Reports
    each layer's result so you can see where an opportunity died, and only sets
    "tradable" when the edge survives the book.

    Args:
        token_ids: CLOB token ids, aligned with the constraint coefficients.
        constraints: Cover rows from build_dependency_constraints.
        size: Shares per leg to validate against the order book.
        labels: Optional human-readable names for each token, for the report.
    """
    return await scan(
        CLIENT.fetch_book,
        token_ids,
        [Constraint.from_dict(raw) for raw in constraints],
        size,
        labels=labels,
    )


@mcp.tool()
@anticipated
async def sweep_for_dependencies(
    markets: list[dict[str, Any]], max_pairs: int = 200, min_shared_terms: int = 2
) -> dict[str, Any]:
    """Screen many markets for dependent pairs, using cached verdicts where possible.

    Generates candidate pairs from distinctive shared terms rather than testing
    every combination, so the model is called on a small fraction of the pairs.
    Verdicts persist to disk between runs.

    Args:
        markets: Market dicts with at least "question"; "description" and "outcomes" help.
        max_pairs: Cap on candidate pairs to screen, strongest first.
        min_shared_terms: Distinctive terms two markets must share to be a candidate.
    """

    async def detect(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return await detect_dependency(SETTINGS, LLM_HTTP, a, b)

    return await sweep_dependencies(
        markets, detect, CACHE, max_pairs=max_pairs, min_shared_terms=min_shared_terms
    )


@mcp.tool()
@anticipated
async def size_position(
    true_probability: float,
    fill_price: float,
    bankroll: float,
    fill_probability: float = 1.0,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> dict[str, Any]:
    """Size a position with a Kelly criterion adjusted for execution risk.

    Pass the book-walked average from check_orderbook_depth as `fill_price`,
    not the touch price.

    Args:
        true_probability: Your estimate of the outcome's probability.
        fill_price: Expected average execution price, strictly in (0, 1).
        bankroll: Capital available for sizing, in USD.
        fill_probability: Probability the order actually executes.
        kelly_multiplier: Fraction of full Kelly to use.
        max_fraction: Hard cap on the fraction of bankroll to commit.
    """
    return kelly_position(
        true_probability,
        fill_price,
        bankroll,
        fill_probability=fill_probability,
        kelly_multiplier=kelly_multiplier,
        max_fraction=max_fraction,
    ).as_dict()


def bearer_auth_middleware(app: Any, token: str) -> Any:
    """Wrap an ASGI app so every HTTP request must carry the shared bearer token.

    A tunnel URL is public the moment it is created, and these tools can move
    money, so HTTP transport refuses to start without a token.
    """

    async def guarded(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"authorization", b"").decode()
        expected = f"Bearer {token}"
        if not hmac.compare_digest(supplied, expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
            return

        await app(scope, receive, send)

    return guarded


def main() -> None:
    """Parse arguments and run the server on the requested transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
        help="stdio for a local client, streamable-http to expose through a tunnel",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address for HTTP transports")
    parser.add_argument("--port", type=int, default=3001, help="bind port for HTTP transports")
    parser.add_argument("--path", default="/mcp", help="HTTP path to mount the MCP endpoint on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info(
        "starting in %s mode (per-order cap $%.2f)",
        "LIVE" if SETTINGS.live else "paper",
        SETTINGS.max_order_usd,
    )

    if args.transport == "stdio":
        mcp.run("stdio")
        return

    if not SETTINGS.auth_token:
        raise SystemExit(
            "TRADING_MCP_AUTH_TOKEN must be set for HTTP transports: the tunnel URL is public"
        )

    if args.transport == "sse":
        app = mcp.sse_app()
    else:
        app = mcp.streamable_http_app(streamable_http_path=args.path)

    uvicorn.run(bearer_auth_middleware(app, SETTINGS.auth_token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

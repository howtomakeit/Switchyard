# Trading MCP server

An MCP server exposing Polymarket quoting, depth-checking, and order placement,
plus a solver for cross-market arbitrage over logically dependent markets.

It is **not** part of the `nemo-switchyard` package. It lives here as a
standalone tool with its own dependencies, imports nothing from `switchyard`,
and adds nothing to the published wheel. It can be moved to its own repository
by copying this directory.

## Tools

| Tool | Purpose |
|---|---|
| `get_market_price` | Best bid, best ask, mid, and spread for an outcome token |
| `check_orderbook_depth` | Walk an order through the live book: VWAP, slippage, partial fills |
| `place_order` | Place a limit order — simulated unless the server runs in live mode |
| `detect_market_dependency` | Ask a reasoning model whether two markets are logically dependent |
| `build_dependency_constraints` | Turn valid outcome pairs into solver constraints |
| `find_arbitrage_basket` | Cheapest constraint-satisfying basket, via OR-Tools/SCIP |
| `project_prices` | KL (Bregman) projection of incoherent prices onto the feasible set |
| `size_position` | Kelly sizing adjusted for fill price and execution probability |

The intended flow is: find candidate markets → `detect_market_dependency` →
`build_dependency_constraints` → `find_arbitrage_basket` → `check_orderbook_depth`
on each leg → `size_position` with the book-walked price → `place_order`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Local MCP client over stdio:

```bash
python server.py
```

Exposed to a hosted model (Grok, Claude) through a tunnel:

```bash
export TRADING_MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
python server.py --transport streamable-http --port 3001
ngrok http 3001
```

Add the resulting `https://<subdomain>.ngrok-free.app/mcp` as a custom MCP
connector, with `Authorization: Bearer <your token>` as a header.

HTTP transports **refuse to start without `TRADING_MCP_AUTH_TOKEN`**. A tunnel
URL is public from the moment it exists, and these tools can move money.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TRADING_MCP_MODE` | `paper` | `live` enables real order placement |
| `TRADING_MCP_MAX_ORDER_USD` | `50` | Per-order notional cap, enforced in both modes |
| `TRADING_MCP_AUTH_TOKEN` | — | Shared bearer token; required for HTTP transports |
| `TRADING_MCP_LLM_URL` | `https://api.x.ai/v1` | OpenAI-compatible endpoint for dependency detection |
| `TRADING_MCP_LLM_MODEL` | `grok-4` | Model used for dependency detection |
| `XAI_API_KEY` | — | API key for that endpoint |
| `TRADING_MCP_TIMEOUT` | `20` | HTTP timeout in seconds |
| `POLYMARKET_CLOB_URL` | `https://clob.polymarket.com` | CLOB base URL |
| `POLYMARKET_PRIVATE_KEY` | — | Signing key; required in live mode |
| `POLYMARKET_FUNDER_ADDRESS` | — | Proxy wallet address, if used |
| `POLYMARKET_SIGNATURE_TYPE` | `0` | `py-clob-client` signature type |

Because dependency detection speaks the OpenAI chat-completions format,
`TRADING_MCP_LLM_URL` can point at a Switchyard deployment instead of directly
at a provider — that is the only relationship between this tool and the rest of
the repository.

### Paper vs. live

Paper mode is the default. `place_order` fills the order against the *live*
book and reports what would have happened, without moving capital. Live mode
requires all of:

1. `TRADING_MCP_MODE=live`
2. `POLYMARKET_PRIVATE_KEY` set
3. `pip install py-clob-client` (not in `requirements.txt` — it pulls in the
   web3 signing stack)

The notional cap applies in both modes and is the last line of defence when a
model sizes an order wrongly. Keep it low.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

The venue client is tested against a mocked CLOB (`respx`); everything else is
pure computation and tested directly.

## Known gaps

- **The Polymarket request/response shapes are not verified against the live
  API.** They were written from the documented CLOB schema and covered with
  mocks. Run `get_market_price` against a real token id before trusting it.
- `place_order` places orders but does not track, amend, or cancel them. There
  is no position or P&L state anywhere in this server.
- The optimizer is a single exact binary program. The Frank-Wolfe / Gurobi
  layer is not implemented: for dependency clusters of this size, SCIP solves
  the problem exactly and the extra layer would not pay for itself.
- Dependency detection analyzes one market pair per call. Sweeping thousands of
  pairs needs a candidate-generation step (and caching) that lives outside this
  server.
- No WebSocket feed. Every quote is a fresh REST call, which is too slow for
  latency-sensitive execution.

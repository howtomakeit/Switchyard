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
| `sweep_for_dependencies` | Screen many markets for dependent pairs, with caching |
| `detect_market_dependency` | Ask a reasoning model whether two markets are logically dependent |
| `build_dependency_constraints` | Turn valid outcome pairs into the payoff-cover matrix |
| `screen_for_arbitrage` | Layer 1: rule out arbitrage cheaply with the LP relaxation |
| `find_arbitrage_basket` | Layer 2: cheapest covering basket, exactly, via OR-Tools/SCIP |
| `scan_for_arbitrage` | All three layers against live books, in one call |
| `project_prices` | KL (Bregman) projection of incoherent prices onto the feasible set |
| `size_position` | Kelly sizing adjusted for fill price and execution probability |

The intended flow is `sweep_for_dependencies` → `build_dependency_constraints`
→ `scan_for_arbitrage` → `place_order` on each leg. `scan_for_arbitrage` runs
the whole pipeline itself; the individual layer tools exist for when you want
to drive it step by step.

## How the arbitrage is found

Dependency detection returns the outcome combinations that are jointly
possible. Each one is a *state of the world*, and each becomes one row of a
cover matrix asserting that the basket holds at least one token paying in that
state:

```
A z >= 1,  minimize  price · z,  z binary
```

A basket satisfying every row pays at least $1 however the markets resolve, so
any basket costing less than $1 is an arbitrage. Dropping an impossible state
drops its row — which is exactly how a dependency creates an opportunity that
neither market shows alone.

Worked example. "X wins the nomination" and "X wins the election", where
winning the election without the nomination is impossible:

| | A:Yes | A:No | B:Yes | B:No |
|---|---|---|---|---|
| price | 0.30 | 0.70 | 0.45 | 0.55 |

Both markets are internally coherent — each side sums to exactly 1.00 — so
neither looks mispriced on its own. But the election is priced *above* the
nomination, which cannot happen. Buying `A:Yes` + `B:No` costs 0.85 and pays
$1 in all three surviving states: **15¢ per share, risk free**. Delete the
dependency and the same basket no longer covers `(No, Yes)`, the cheapest cover
costs exactly 1.00, and the edge vanishes. Both cases are pinned in
`tests/test_arbitrage.py`.

### The three layers

1. **Screen** (`screen_arbitrage`, GLOP). The LP relaxation's cost is a lower
   bound on the integer optimum, so a bound at or above the guaranteed payoff
   *proves* no arbitrage exists and the exact solve is skipped.
2. **Solve** (`find_arbitrage`, SCIP). The exact binary program, yielding the
   basket to trade.
3. **Execute** (`scan`). Re-prices that basket against real book depth at the
   size actually being traded. This is the layer that kills most paper
   arbitrages: an edge measured at the touch is measured against a quantity
   nobody can fill. Every leg must fill completely — a partially filled basket
   is not a hedged position, it is naked exposure on whichever legs did fill —
   so a short leg rejects the trade rather than scaling it down.

### Sweeping thousands of pairs

Asking a model about every market pair is quadratic: a thousand markets is half
a million calls. `sweep_for_dependencies` builds an inverted index over
distinctive terms, drops terms too common to carry signal, and ranks pairs by
the *rarest* term they share — summing over shared terms ranks templated
boilerplate above substance. Verdicts persist to disk, keyed order-independently
by market text.

On a 440-market synthetic set (96,580 possible pairs) with 360 genuinely
dependent pairs, candidate generation takes 20 ms and ranks all 360 true pairs
above every false one: **100% recall at 360 model calls, a 268× reduction**.

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
| `TRADING_MCP_CACHE` | `.dependency-cache.json` | Where dependency verdicts persist |
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
  is no position or P&L state anywhere in this server. **This matters most for
  multi-leg baskets**: if leg 2 fails after leg 1 fills, you hold unhedged
  exposure and must unwind by hand. `scan_for_arbitrage` checks every leg's
  depth before you trade, which reduces the risk but does not remove it.
- No Frank-Wolfe / Gurobi layer. The LP relaxation already gives a sound bound
  for Layer 1 and SCIP solves these clusters exactly in milliseconds, so the
  extra layer would add a licensed dependency for no accuracy.
- **No WebSocket feed, deliberately.** A real-time feed belongs in a separate
  always-on process that maintains book state, not in an MCP request path whose
  latency is dominated by model turnaround. Adding one here would buy
  milliseconds an LLM-driven caller cannot use, at the cost of a background
  task and another unverified wire protocol.
- Dependency verdicts never expire. If a market's text is edited materially,
  clear the cache file.
- Candidate generation is lexical. Two dependent markets sharing no distinctive
  vocabulary ("Will the incumbent lose?" / "Will Callahan win?") are never
  paired.

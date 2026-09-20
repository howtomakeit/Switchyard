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
| `assess_resolution_risk` | Refuse a dependency whose exclusions could actually occur |
| `build_dependency_constraints` | Turn valid outcome pairs into the payoff-cover matrix |
| `screen_for_arbitrage` | Layer 1: rule out arbitrage cheaply with the LP relaxation |
| `find_arbitrage_basket` | Layer 2: cheapest covering basket, exactly, via OR-Tools/SCIP |
| `scan_for_arbitrage` | All three layers against live books, in one call |
| `score_arbitrage_opportunity` | Combine execution + resolution risk; annualize over the lockup |
| `project_prices` | KL (Bregman) projection of incoherent prices onto the feasible set |
| `size_position` | Kelly sizing adjusted for fill price and execution probability |
| `journal_observation` | Record an opportunity — actionable or not — for measurement |
| `journal_recheck` | Re-observe a basket to measure how long its edge survives |
| `journal_settlement` | Record what a basket actually paid at resolution |
| `journal_open` / `journal_report` | What to recheck, and what the data says so far |
| `get_positions` | Current wallet positions (read-only, no credentials) |
| `list_open_orders` | Resting orders (live mode) |
| `cancel_orders` / `cancel_all_orders` | Unwind path for a half-filled basket (live mode) |

The intended flow is `sweep_for_dependencies` → `assess_resolution_risk` →
`build_dependency_constraints` → `scan_for_arbitrage` →
`score_arbitrage_opportunity` → `place_order` on each leg.
`scan_for_arbitrage` runs the three solver layers itself; the individual layer
tools exist for when you want to drive it step by step.

**Do not skip `assess_resolution_risk`.** It is the difference between an
arbitrage and an unhedged bet — see below.

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

### Resolution risk — where this loses money

A cover basket is risk-free only if the dropped states are unreachable **under
the markets' own resolution rules**. Semantic implication is not enough, and
this is the failure mode that costs real money:

> "Will X be the nominee **on July 1**?" and "Will X win the **November**
> election?" look strictly implied — you cannot win without the nomination — so
> a model drops the `(No, Yes)` state. But if X is nominated on July 15, market
> A resolves No while B still resolves Yes. The dropped state happens, the
> basket pays nothing, and a position entered for a 15¢ edge loses the full 85¢
> of principal.

The critical distinction is **timing gap vs. truncation**, and conflating them
breaks the strategy in one direction or the other:

- A **gap** between resolution dates is a *cost*. "Will X win the nomination?"
  settles at the convention and keeps implying the November outcome four months
  later. The gap locks your capital; it does not endanger the basket. Dependent
  markets are nearly always months apart, so refusing them would reject the
  entire opportunity set.
- **Truncation** is the *risk*. "Will X be the nominee **by June 1**?" stops
  implying anything on June 2. That is what makes an excluded state reachable.

`assess_resolution_risk` blocks on truncation, never on gap alone. A verdict is
refused when the model reports truncation is possible, when the earlier market's
wording carries a calendar deadline (`by June 1`, `on or before`, `as of March`)
and the dates are far enough apart for it to bite, when any exclusion lacks a
resolution-rule justification, when confidence is under 0.85, or when a market
has expired. Long lockups come back as priced warnings.

The deadline detector is a backstop for the model, not a replacement — it fires
even when the model reports no risk, because a deadline in the question text is
the trap's signature. A bare year ("the 2028 election") is not a deadline.

None of this can *prove* a dependency sound. It refuses the ones that are
plainly unsound and makes the rest legible.

### Edges are carry trades, not free money

`score_arbitrage_opportunity` annualizes the edge over the capital lockup,
because the two numbers tell opposite stories:

| Edge | Cost | Lockup | Period return | Annualized |
|---|---|---|---|---|
| 15¢ | 85¢ | 35 days | 17.6% | **184%** |
| 15¢ | 85¢ | 659 days | 17.6% | **9.8%** |

Same headline edge. One is excellent; the other underperforms a savings
account once you price the resolution risk. Capital is locked until the *later*
market resolves.

### Measure before you trade

The strategy's expected return cannot be derived from theory. It depends on how
often real mispricings appear, how much depth they carry, what fraction
survives the risk gate, and — the question that decides whether a model-paced
system can trade at all — how long an edge persists before someone else takes
it. The journal exists to measure those before any capital is at risk.

Record **every** opportunity, including the rejects: the histogram of why
things fail is the most useful early output.

```
scan_for_arbitrage → score_arbitrage_opportunity → journal_observation
                                                   (actionable or not)
journal_open  → journal_recheck  (on a schedule; this measures edge lifetime)
at resolution → journal_settlement
anytime       → journal_report
```

`journal_report` answers four questions:

| Output | Question it answers |
|---|---|
| `actionable_rate` | Are there opportunities at all? |
| `blocked_by` | What is killing them — no edge, thin books, bad dependencies? |
| `edge_lifetime_minutes` | Can a system this slow actually capture them? |
| `settlements.paid_as_promised` | Was the "guaranteed" payoff ever guaranteed? |

That last row is the one that matters most. A cover basket that was genuinely
risk-free pays at least 1.0 per share. Anything less means an excluded state
occurred and the dependency was wrong — the loss is the whole position, so a
settlement rate below 100% invalidates the strategy no matter how good the
edges looked.

Storage is JSON Lines: append-only, crash-safe, and loadable into pandas
without a schema migration. A truncated final line costs one record, not the
history.

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

## Verify before you trade

The Polymarket wire format is the one thing mocks cannot prove. Run this
wherever outbound access to `polymarket.com` is allowed:

```bash
python verify_live.py
```

It is read-only, needs no credentials, and checks every assumption the server
makes: that Gamma returns the fields we read, that `clobTokenIds` parses, that
book levels carry string `price`/`size`, what order the venue actually returns
levels in, that our derived mid matches the venue's own `/midpoint`, and that
depth simulation against a real book stays self-consistent. It exits non-zero
on any failure, so it can gate a deploy.

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
| `TRADING_MCP_JOURNAL` | `.opportunity-journal.jsonl` | Where observed opportunities are logged |
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
- **Multi-leg execution is not atomic.** `scan_for_arbitrage` checks every
  leg's depth first and `cancel_orders` gives you an unwind path, but between
  placing leg 1 and leg 2 the book can move. Nothing here makes a basket fill
  all-or-nothing; that risk is real and unhedged. Place the least liquid leg
  first.
- There is no P&L or position state beyond what `get_positions` reads back from
  the venue.
- The order-lifecycle calls (`cancel_orders`, `list_open_orders`) are thin
  passthroughs to `py-clob-client` and are **not** covered by `verify_live.py`,
  which is read-only. Test them with one tiny resting order before relying on
  them to unwind anything.
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

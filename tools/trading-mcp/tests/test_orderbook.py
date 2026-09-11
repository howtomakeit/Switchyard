"""Fill-simulation behavior: partial fills, slippage, and book ordering."""

from trading_mcp.orderbook import simulate_fill, sort_levels


def test_sorts_asks_cheapest_first_regardless_of_wire_order():
    levels = [(0.60, 10.0), (0.40, 5.0), (0.50, 8.0)]
    assert sort_levels(levels, ascending=True) == [(0.40, 5.0), (0.50, 8.0), (0.60, 10.0)]


def test_drops_zero_size_levels():
    assert sort_levels([(0.4, 0.0), (0.5, 3.0)], ascending=True) == [(0.5, 3.0)]


def test_buy_walks_asks_and_reports_vwap():
    result = simulate_fill([(0.50, 10.0), (0.60, 10.0)], 15.0, taking_asks=True)

    assert result.fully_filled
    assert result.filled_size == 15.0
    assert result.levels_consumed == 2
    # 10 @ 0.50 + 5 @ 0.60 = 8.00 over 15 shares.
    assert result.notional == 8.0
    assert result.average_price == 8.0 / 15.0
    assert result.best_price == 0.50
    assert result.worst_price == 0.60


def test_sell_walks_bids_highest_first():
    result = simulate_fill([(0.30, 10.0), (0.45, 4.0)], 4.0, taking_asks=False)

    assert result.average_price == 0.45
    assert result.levels_consumed == 1
    assert result.slippage_bps == 0.0


def test_partial_fill_is_reported_not_raised():
    result = simulate_fill([(0.50, 3.0)], 10.0, taking_asks=True)

    assert not result.fully_filled
    assert result.filled_size == 3.0
    assert result.average_price == 0.50


def test_empty_book_yields_no_fill():
    result = simulate_fill([], 5.0, taking_asks=True)

    assert result.filled_size == 0.0
    assert result.average_price is None
    assert result.best_price is None


def test_slippage_is_positive_for_worse_execution_on_both_sides():
    buy = simulate_fill([(0.50, 5.0), (0.55, 5.0)], 10.0, taking_asks=True)
    sell = simulate_fill([(0.50, 5.0), (0.45, 5.0)], 10.0, taking_asks=False)

    assert buy.slippage_bps is not None and buy.slippage_bps > 0
    assert sell.slippage_bps is not None and sell.slippage_bps > 0

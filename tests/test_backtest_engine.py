from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from polymarket_bot.backtest_engine import (
    POLYMARKET_WIN_FEE,
    BacktestEngine,
    _timestamp_sort_key,
    _total_liquidity,
)


class LiquidityTests(unittest.TestCase):
    def test_total_liquidity_uses_ask_side_only(self):
        book = {
            "bids": [{"price": 0.40, "size": 1000}],
            "asks": [{"price": 0.50, "size": 100}, {"price": 0.55, "size": 100}],
        }
        # Only asks: 0.50 * 100 + 0.55 * 100 = 105.0; bids are ignored.
        self.assertAlmostEqual(_total_liquidity(book), 105.0)

    def test_total_liquidity_floor(self):
        self.assertEqual(_total_liquidity({"bids": [], "asks": []}), 1.0)


class TimestampOrderingTests(unittest.TestCase):
    def test_numeric_timestamps_sorted_numerically(self):
        rows = [{"timestamp": "10"}, {"timestamp": "2"}, {"timestamp": "1"}]
        ordered = sorted(rows, key=_timestamp_sort_key)
        self.assertEqual([r["timestamp"] for r in ordered], ["1", "2", "10"])

    def test_grouped_rows_orders_numeric_timestamps(self):
        records = [
            {"market_id": "m", "timestamp": 10, "close": 0.5},
            {"market_id": "m", "timestamp": 2, "close": 0.5},
            {"market_id": "m", "timestamp": 1, "close": 0.5},
        ]
        engine = BacktestEngine(records, min_history=2)
        grouped = engine._grouped_rows()
        self.assertEqual([r["timestamp"] for r in grouped[0]], [1, 2, 10])


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine([], min_history=2)

    def test_settle_trade_zero_position(self):
        self.assertEqual(self.engine._settle_trade(0.0, 0.5, True), (0.0, 0.0))

    def test_settle_trade_loss_returns_negative_stake(self):
        pnl, fee = self.engine._settle_trade(10.0, 0.5, False)
        self.assertEqual(pnl, -10.0)
        self.assertEqual(fee, 0.0)

    def test_settle_trade_win_applies_fee(self):
        position, price = 10.0, 0.5
        pnl, fee = self.engine._settle_trade(position, price, True)
        shares = position / price
        expected_fee = shares * POLYMARKET_WIN_FEE
        self.assertAlmostEqual(fee, expected_fee)
        self.assertAlmostEqual(pnl, shares - expected_fee - position)


class RunSmokeTests(unittest.TestCase):
    def _records(self):
        book = {"asks": [{"price": 0.5, "size": 500}], "bids": [{"price": 0.49, "size": 500}]}
        rows = []
        for i in range(6):
            rows.append(
                {
                    "market_id": "m1",
                    "timestamp": i,
                    "close": 0.40 + i * 0.05,
                    "order_book": book,
                    "category": "politics",
                    "outcome": "YES",
                    "actual_outcome": True,
                }
            )
        return rows

    def test_run_produces_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            engine = BacktestEngine(self._records(), min_history=2, output_path=out)
            engine.run()
            self.assertTrue(out.exists())
            for key in ("decisions", "trades", "sharpe_ratio", "max_drawdown", "win_rate"):
                self.assertIn(key, engine.summary)
            self.assertGreaterEqual(engine.summary["decisions"], 1)


if __name__ == "__main__":
    unittest.main()

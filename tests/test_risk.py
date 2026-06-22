from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot import risk
from polymarket_bot.risk import (
    drawdown_factor,
    kelly_fraction,
    size_positions,
    volatility_factor,
)


def _opp(**overrides):
    base = {
        "current_price": 0.40,
        "fair_value_estimate": 0.60,
        "volatility": 0.0,
        "category": "politics",
    }
    base.update(overrides)
    return base


class KellyTests(unittest.TestCase):
    def test_positive_with_real_edge(self):
        self.assertGreater(kelly_fraction(0.25, 0.45), 0.0)

    def test_zero_without_edge(self):
        self.assertEqual(kelly_fraction(0.50, 0.50), 0.0)

    def test_capped(self):
        self.assertLessEqual(kelly_fraction(0.01, 0.99), 1.0)


class VolatilityFactorTests(unittest.TestCase):
    def test_no_vol_is_full_size(self):
        self.assertEqual(volatility_factor(0.0), 1.0)

    def test_higher_vol_shrinks(self):
        self.assertLess(volatility_factor(0.02), volatility_factor(0.005))

    def test_floor(self):
        self.assertGreaterEqual(volatility_factor(10.0), 0.25)

    def test_disabled(self):
        with patch.object(risk, "VOL_SIZING", False):
            self.assertEqual(volatility_factor(0.05), 1.0)


class DrawdownFactorTests(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(drawdown_factor([{"pnl_usdc": 5}, {"pnl_usdc": -2}]), 1.0)

    def test_throttle(self):
        with patch.object(risk, "MAX_DRAWDOWN_USDC", 30):
            self.assertEqual(drawdown_factor([{"pnl_usdc": -40}]), 0.5)

    def test_halt(self):
        with patch.object(risk, "MAX_DRAWDOWN_USDC", 30):
            self.assertEqual(drawdown_factor([{"pnl_usdc": -70}]), 0.0)

    def test_disabled_when_threshold_zero(self):
        with patch.object(risk, "MAX_DRAWDOWN_USDC", 0):
            self.assertEqual(drawdown_factor([{"pnl_usdc": -1000}]), 1.0)


class SizePositionsTests(unittest.TestCase):
    def test_respects_per_bet_cap(self):
        with patch.object(risk, "MAX_BET_USDC", 5):
            opps = [_opp(current_price=0.1, fair_value_estimate=0.9)]
            sized, _ = size_positions(opps, resolved=[])
            self.assertLessEqual(sized[0]["stake_usdc"], 5)

    def test_respects_portfolio_cap(self):
        with patch.object(risk, "MAX_PORTFOLIO_EXPOSURE_USDC", 5), patch.object(risk, "MAX_BET_USDC", 10):
            opps = [_opp(current_price=0.1, fair_value_estimate=0.9, category=f"c{i}") for i in range(6)]
            _, summary = size_positions(opps, resolved=[])
            self.assertLessEqual(summary["total_exposure_usdc"], 5.0 + 1e-6)

    def test_respects_category_cap(self):
        with patch.object(risk, "MAX_CATEGORY_EXPOSURE_USDC", 3), patch.object(risk, "MAX_BET_USDC", 10):
            opps = [_opp(current_price=0.1, fair_value_estimate=0.9) for _ in range(5)]
            sized, summary = size_positions(opps, resolved=[])
            self.assertLessEqual(summary["by_category_usdc"].get("politics", 0), 3.0 + 1e-6)

    def test_respects_concurrency_cap(self):
        with patch.object(risk, "MAX_CONCURRENT_POSITIONS", 2):
            opps = [_opp(current_price=0.1, fair_value_estimate=0.9, category=f"c{i}") for i in range(5)]
            sized, _ = size_positions(opps, resolved=[])
            self.assertLessEqual(sum(1 for o in sized if o["stake_usdc"] > 0), 2)

    def test_annotates_each_opportunity(self):
        sized, _ = size_positions([_opp()], resolved=[])
        self.assertIn("stake_usdc", sized[0])
        self.assertIn("vol_factor", sized[0])
        self.assertIn("drawdown_factor", sized[0])


if __name__ == "__main__":
    unittest.main()

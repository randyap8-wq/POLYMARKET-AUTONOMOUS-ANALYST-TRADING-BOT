from __future__ import annotations

import unittest

from polymarket_bot.backtest_full import _proxy_ai_score, _summary


class FullBacktestHelpersTests(unittest.TestCase):
    def test_proxy_ai_score_picks_highest_current_price(self):
        market = {
            "condition_id": "c1",
            "outcomes": ["A", "B", "C"],
            "prices": [0.2, 0.6, 0.2],
            "category": "politics",
        }

        score = _proxy_ai_score(market)

        self.assertEqual(score["recommended_outcome"], "B")
        self.assertEqual(score["recommended_outcome_index"], 1)
        self.assertGreater(score["edge"], 0.02)

    def test_summary_calculates_trade_metrics(self):
        summary = _summary(
            [
                {"stake_usdc": 10.0, "pnl_usdc": 5.0, "win": True},
                {"stake_usdc": 10.0, "pnl_usdc": -10.0, "win": False},
                {"stake_usdc": 0.0, "pnl_usdc": 0.0, "win": None},
            ]
        )

        self.assertEqual(summary["decisions"], 3)
        self.assertEqual(summary["trades"], 2)
        self.assertEqual(summary["win_rate"], 0.5)
        self.assertEqual(summary["total_pnl_usdc"], -5.0)


if __name__ == "__main__":
    unittest.main()

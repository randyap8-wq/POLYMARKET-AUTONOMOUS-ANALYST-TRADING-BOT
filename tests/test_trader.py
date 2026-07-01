from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot.trader import place_bet


class PlaceBetTests(unittest.TestCase):
    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.30)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_place_bet_returns_dry_run_result(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.29,
            "edge": 0.20,
        }

        result = place_bet(market, score)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["token_id"], "token-123")

    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.30)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_place_bet_uses_precomputed_stake(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.29,
            "edge": 0.20,
            "stake_usdc": 7.5,
        }

        result = place_bet(market, score)

        self.assertIsNotNone(result)
        # The risk-layer stake must be used verbatim, not recomputed from edge.
        self.assertEqual(result["usdc_spent"], 7.5)

    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.30)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_place_bet_skips_when_stake_zeroed_by_risk_layer(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.29,
            "edge": 0.20,
            "stake_usdc": 0.0,  # deliberate veto (e.g. drawdown breaker / caps)
        }

        # A present-but-zero stake is a veto: do not fall back to edge sizing.
        self.assertIsNone(place_bet(market, score))

    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.25)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_lower_ask_is_not_treated_as_slippage(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {"recommended_outcome": "Yes", "current_price": 0.29, "edge": 0.20}

        result = place_bet(market, score)

        self.assertIsNotNone(result)
        self.assertEqual(result["price"], 0.25)

    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.32)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_higher_ask_aborts_when_price_moves_up(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {"recommended_outcome": "Yes", "current_price": 0.29, "edge": 0.20}

        self.assertIsNone(place_bet(market, score))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot import scorer
from polymarket_bot.scorer import (
    _coerce_score,
    get_token_usage,
    reset_token_usage,
    score_market,
)


MARKET = {
    "condition_id": "cond-1",
    "question": "Will it rain tomorrow?",
    "outcomes": ["Yes", "No"],
    "prices": [0.4, 0.6],
    "category": "weather",
    "end_date": "2026-06-30T00:00:00Z",
    "volume": 50000,
    "url": "https://example.com",
}


class ScoreMarketTests(unittest.TestCase):
    def setUp(self):
        reset_token_usage()
        self.addCleanup(reset_token_usage)

    @patch.object(scorer, "GEMINI_API_KEY", "")
    def test_returns_none_without_api_key(self):
        self.assertIsNone(score_market(MARKET))

    @patch.object(scorer, "GEMINI_API_KEY", "fake-key")
    @patch.object(scorer, "_call_gemini")
    def test_coerces_recommended_outcome_from_index(self, mock_call):
        mock_call.return_value = {
            "recommended_outcome": None,
            "recommended_outcome_index": 0,
            "fair_value_estimate": 0.6,
            "edge": 0.2,
            "confidence": "medium",
            "news_supports_bet": True,
            "reasoning": "rain expected",
        }
        score = score_market(MARKET)
        self.assertEqual(score["recommended_outcome"], "Yes")
        self.assertEqual(score["recommended_outcome_index"], 0)
        # current_price falls back to the snapshot price for the index.
        self.assertEqual(score["current_price"], 0.4)
        self.assertEqual(score["condition_id"], "cond-1")
        self.assertEqual(score["category"], "weather")

    @patch.object(scorer, "GEMINI_API_KEY", "fake-key")
    @patch.object(scorer, "_call_gemini")
    def test_invalid_outcome_is_nulled(self, mock_call):
        mock_call.return_value = {
            "recommended_outcome": "Maybe",
            "recommended_outcome_index": 9,
            "fair_value_estimate": 0.6,
            "edge": 0.2,
            "confidence": "low",
            "news_supports_bet": False,
            "reasoning": "unclear",
        }
        score = score_market(MARKET)
        self.assertIsNone(score["recommended_outcome"])
        self.assertIsNone(score["recommended_outcome_index"])

    def test_coerce_score_handles_short_prices(self):
        market = {**MARKET, "prices": [0.4]}
        score = _coerce_score(
            market,
            {
                "recommended_outcome": None,
                "recommended_outcome_index": 1,
                "current_price": 0,
            },
        )

        self.assertEqual(score["current_price"], 0.0)

    @patch.object(scorer, "GEMINI_API_KEY", "fake-key")
    @patch.object(scorer, "GEMINI_PRO_RECHECK", True)
    @patch.object(scorer, "_call_gemini")
    def test_pro_recheck_downgrades_on_disagreement(self, mock_call):
        primary = {
            "recommended_outcome": "Yes",
            "recommended_outcome_index": 0,
            "current_price": 0.4,
            "fair_value_estimate": 0.6,
            "edge": 0.2,
            "confidence": "high",
            "news_supports_bet": True,
            "reasoning": "strong",
        }
        recheck = {**primary, "recommended_outcome": "No", "recommended_outcome_index": 1}
        mock_call.side_effect = [primary, recheck]

        score = score_market(MARKET)
        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(score["confidence"], "medium")
        self.assertTrue(score["pro_recheck_disagreed"])

    @patch.object(scorer, "GEMINI_API_KEY", "fake-key")
    @patch.object(scorer, "_call_gemini", return_value=None)
    def test_returns_none_when_call_fails(self, _mock_call):
        self.assertIsNone(score_market(MARKET))


class TokenUsageTests(unittest.TestCase):
    def setUp(self):
        reset_token_usage()
        self.addCleanup(reset_token_usage)

    def test_record_usage_accepts_dict(self):
        scorer._record_usage(scorer.GEMINI_MODEL, {"prompt_tokens": 100, "completion_tokens": 50})
        usage = get_token_usage()
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)
        # Free tier pricing -> zero cost.
        self.assertEqual(usage["cost_usd"], 0.0)

    def test_record_usage_accepts_metadata_object(self):
        class Meta:
            prompt_token_count = 200
            candidates_token_count = 100

        scorer._record_usage(scorer.GEMINI_MODEL, Meta())
        usage = get_token_usage()
        self.assertEqual(usage["prompt_tokens"], 200)
        self.assertEqual(usage["completion_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 300)

    def test_record_usage_applies_paid_pricing(self):
        pricing = {scorer.GEMINI_MODEL: {"input": 1.0, "output": 2.0}}
        with patch.object(scorer, "GEMINI_PRICING", pricing):
            scorer._record_usage(scorer.GEMINI_MODEL, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        self.assertAlmostEqual(get_token_usage()["cost_usd"], 3.0, places=6)

    def test_reset_zeroes_usage(self):
        scorer._record_usage(scorer.GEMINI_MODEL, {"prompt_tokens": 10, "completion_tokens": 5})
        reset_token_usage()
        usage = get_token_usage()
        self.assertEqual(usage["calls"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_record_usage_ignores_missing(self):
        scorer._record_usage(scorer.GEMINI_MODEL, None)
        self.assertEqual(get_token_usage()["calls"], 0)


if __name__ == "__main__":
    unittest.main()

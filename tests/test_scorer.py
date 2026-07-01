from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from polymarket_bot import scorer
from polymarket_bot.scorer import (
    _build_prompt,
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
        self.assertTrue(score["edge_threshold_met"])

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
        self.assertEqual(score["edge"], 0.0)

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

    def test_coerce_score_infers_index_from_valid_label(self):
        score = _coerce_score(
            MARKET,
            {
                "recommended_outcome": "No",
                "recommended_outcome_index": None,
                "fair_value_estimate": 0.7,
                "edge": 0.1,
                "confidence": "medium",
                "news_supports_bet": True,
            },
        )

        self.assertEqual(score["recommended_outcome_index"], 1)
        self.assertEqual(score["current_price"], 0.6)
        self.assertEqual(score["edge"], 0.1)

    def test_build_prompt_contains_structured_reasoning_contract(self):
        prompt = _build_prompt({**MARKET, "category": "politics"})

        self.assertIn("ANALYSIS PATH", prompt)
        self.assertIn("evidence hierarchy", prompt)
        self.assertIn("Compress estimates toward 50%", prompt)
        self.assertIn("more than 0.02", prompt)
        self.assertIn('"base_rate"', prompt)
        self.assertIn('"bayesian_updates"', prompt)
        self.assertIn("Weight polling data heavily", prompt)

    def test_coerce_score_preserves_structured_reasoning_fields(self):
        score = _coerce_score(
            MARKET,
            {
                "recommended_outcome": "Yes",
                "recommended_outcome_index": 0,
                "current_price": 0.4,
                "fair_value_estimate": "0.55",
                "edge": 0.99,  # recomputed from fair value and current price
                "confidence": "HIGH",
                "base_rate": "0.45",
                "evidence_summary": {"official_data": "NWS forecast", "market_signals": ["book moved"]},
                "bayesian_updates": [
                    {
                        "direction": "toward",
                        "magnitude": "medium",
                        "evidence": "forecast strengthened",
                        "probability_after": "0.55",
                    }
                ],
                "reasoning": "rain expected",
                "key_risks": "forecast changes",
                "information_quality": "medium",
                "edge_threshold_met": True,
                "news_supports_bet": True,
                "counter_evidence_considered": True,
                "news_headlines": ["Forecast update"],
            },
        )

        self.assertEqual(score["edge"], 0.15)
        self.assertEqual(score["confidence"], 85.0)
        self.assertEqual(score["confidence_level"], "high")
        self.assertEqual(score["base_rate"], 0.45)
        self.assertEqual(score["evidence_summary"]["official_data"], ["NWS forecast"])
        self.assertEqual(score["bayesian_updates"][0]["probability_after"], 0.55)
        self.assertEqual(score["key_risks"], ["forecast changes"])
        self.assertEqual(score["information_quality"], "medium")
        self.assertTrue(score["edge_threshold_met"])
        self.assertEqual(score["news_headlines"], ["Forecast update"])

    def test_coerce_score_vetoes_sub_threshold_edge(self):
        score = _coerce_score(
            MARKET,
            {
                "recommended_outcome": "Yes",
                "recommended_outcome_index": 0,
                "current_price": 0.4,
                "fair_value_estimate": 0.41,
                "edge": 0.01,
                "confidence": "medium",
                "news_supports_bet": True,
            },
        )

        self.assertIsNone(score["recommended_outcome"])
        self.assertIsNone(score["recommended_outcome_index"])
        self.assertEqual(score["edge"], 0.0)
        self.assertFalse(score["edge_threshold_met"])
        self.assertFalse(score["news_supports_bet"])

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
        self.assertEqual(score["confidence"], 65.0)
        self.assertEqual(score["confidence_level"], "medium")
        self.assertTrue(score["pro_recheck_disagreed"])

    @patch.object(scorer, "GEMINI_API_KEY", "fake-key")
    @patch.object(scorer, "_call_gemini", return_value=None)
    def test_returns_none_when_call_fails(self, _mock_call):
        self.assertIsNone(score_market(MARKET))

    @patch.object(scorer, "GEMINI_USE_SEARCH", False)
    @patch.object(scorer, "_rate_limit_sleep")
    @patch("polymarket_bot.scorer.time.sleep")
    @patch.object(scorer, "_get_client")
    def test_rate_limit_applied_even_when_parse_fails(self, mock_client_fn, _mock_time_sleep, mock_rl):
        client = Mock()
        bad = Mock()
        bad.text = "not json"
        bad.usage_metadata = None
        good = Mock()
        good.text = '{"edge": 0.1, "recommended_outcome": "Yes"}'
        good.usage_metadata = None
        client.models.generate_content.side_effect = [bad, good]
        mock_client_fn.return_value = client

        result = scorer._call_gemini("prompt")

        self.assertIsNotNone(result)
        self.assertEqual(mock_rl.call_count, 2)


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

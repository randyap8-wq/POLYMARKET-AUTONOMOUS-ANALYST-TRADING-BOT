from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot import pipeline


MARKET = {
    "condition_id": "c1",
    "question": "Will X happen?",
    "outcomes": ["Yes", "No"],
    "prices": [0.50, 0.50],
    "token_ids": ["tokA", "tokB"],
    "category": "politics",
    "end_date": "2026-06-30T00:00:00Z",
    "volume": 50000,
    "url": "https://example.com",
}

LIQUID_BOOK = {
    "bids": [{"price": 0.49, "size": 1000}],
    "asks": [{"price": 0.51, "size": 1000}],
}
THIN_BOOK = {
    "bids": [{"price": 0.49, "size": 5}],
    "asks": [{"price": 0.51, "size": 5}],
}


def _rising():
    return [{"t": i, "p": 0.30 + i * 0.01} for i in range(30)]


def _ai_pick():
    return {
        "recommended_outcome": "Yes",
        "recommended_outcome_index": 0,
        "current_price": 0.50,
        "fair_value_estimate": 0.62,
        "edge": 0.12,
        "confidence": "medium",
        "news_supports_bet": True,
        "category": "politics",
        "reasoning": "news",
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(pipeline, "fetch_market_stats", return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch.object(pipeline, "QUANT_ENABLED", False)
    @patch.object(pipeline, "score_market")
    def test_quant_disabled_uses_ai_only(self, mock_score):
        mock_score.return_value = _ai_pick()
        result = pipeline.analyze_market(MARKET)
        self.assertEqual(result["recommended_outcome"], "Yes")
        mock_score.assert_called_once()

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_price_history", return_value=[])
    @patch.object(pipeline, "fetch_order_book", return_value=THIN_BOOK)
    @patch.object(pipeline, "score_market")
    def test_prefilter_skips_untradeable_without_calling_ai(self, mock_score, *_):
        result = pipeline.analyze_market(MARKET)
        self.assertIsNone(result)
        mock_score.assert_not_called()

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_price_history", return_value=_rising())
    @patch.object(pipeline, "fetch_order_book", return_value={"bids": [], "asks": []})
    @patch.object(pipeline, "score_market")
    def test_prefilter_fails_soft_when_book_missing_but_history_exists(self, mock_score, *_):
        mock_score.return_value = _ai_pick()
        result = pipeline.analyze_market(MARKET)
        self.assertIsNotNone(result)
        mock_score.assert_called_once()

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_price_history")
    @patch.object(pipeline, "fetch_order_book", return_value=LIQUID_BOOK)
    @patch.object(pipeline, "score_market")
    def test_full_path_fuses_quant_and_ai(self, mock_score, mock_book, mock_hist):
        mock_score.return_value = _ai_pick()
        mock_hist.return_value = _rising()
        result = pipeline.analyze_market(MARKET)
        self.assertIsNotNone(result)
        self.assertIn("quant", result)
        self.assertIn(result["agreement"], {"agree", "neutral", "disagree"})
        mock_score.assert_called_once()

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_price_history", return_value=[])
    @patch.object(pipeline, "fetch_order_book", return_value=LIQUID_BOOK)
    @patch.object(pipeline, "score_market", return_value=None)
    def test_ai_failure_returns_none(self, *_):
        self.assertIsNone(pipeline.analyze_market(MARKET))

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_price_history", return_value=_rising())
    @patch.object(pipeline, "fetch_order_book", return_value=LIQUID_BOOK)
    @patch.object(pipeline, "score_market")
    def test_no_recommendation_is_ai_only(self, mock_score, *_):
        ai = _ai_pick()
        ai["recommended_outcome"] = None
        ai["recommended_outcome_index"] = None
        mock_score.return_value = ai
        result = pipeline.analyze_market(MARKET)
        self.assertEqual(result["agreement"], "ai_only")

    @patch.object(pipeline, "QUANT_ENABLED", True)
    @patch.object(pipeline, "QUANT_PREFILTER", True)
    @patch.object(pipeline, "fetch_order_book", return_value=LIQUID_BOOK)
    @patch.object(pipeline, "fetch_price_history", return_value=_rising())
    @patch.object(pipeline, "score_market")
    def test_multi_outcome_can_switch_to_better_quant_edge(self, mock_score, *_):
        market = {
            **MARKET,
            "outcomes": ["A", "B", "C"],
            "prices": [0.60, 0.20, 0.20],
            "token_ids": ["tokA", "tokB", "tokC"],
        }
        mock_score.return_value = {
            **_ai_pick(),
            "recommended_outcome": "A",
            "recommended_outcome_index": 0,
            "current_price": 0.60,
            "fair_value_estimate": 0.63,
            "probability": 0.63,
            "edge": 0.03,
        }

        result = pipeline.analyze_market(market)

        self.assertEqual(result["recommended_outcome"], "B")
        self.assertTrue(result["multi_outcome_override"])


if __name__ == "__main__":
    unittest.main()

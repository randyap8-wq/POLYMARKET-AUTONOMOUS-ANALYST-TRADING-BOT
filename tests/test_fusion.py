from __future__ import annotations

import unittest

from polymarket_bot.fusion import fuse_signals


def _ai(**overrides):
    base = {
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
    base.update(overrides)
    return base


def _quant(score, tradeable=True, available=True, **overrides):
    base = {
        "available": available,
        "tradeable": tradeable,
        "quant_score": score,
        "quant_edge": round(score * 0.10, 4),
        "quant_fair_value": 0.51,
        "volatility": 0.005,
        "reason": "ok" if tradeable else "liquidity $10 < $200",
    }
    base.update(overrides)
    return base


class FuseSignalsTests(unittest.TestCase):
    def test_agreement_blends_edge(self):
        result = fuse_signals(_ai(), _quant(0.30))
        self.assertEqual(result["agreement"], "agree")
        self.assertGreater(result["edge"], 0)
        self.assertLess(result["edge"], 0.12)  # blended below the AI-only edge
        self.assertEqual(result["recommended_outcome"], "Yes")

    def test_strong_agreement_boosts_confidence(self):
        result = fuse_signals(_ai(confidence="medium"), _quant(0.5))
        self.assertEqual(result["confidence"], 75.0)
        self.assertEqual(result["confidence_level"], "high")

    def test_mild_disagreement_shrinks_edge_and_downgrades(self):
        result = fuse_signals(_ai(confidence="medium"), _quant(-0.2))
        self.assertEqual(result["agreement"], "disagree")
        self.assertEqual(result["confidence"], 45.0)
        self.assertEqual(result["confidence_level"], "low")
        self.assertEqual(result["position_size_multiplier"], 0.3)
        self.assertLess(result["edge"], 0.12)
        self.assertEqual(result["recommended_outcome"], "Yes")  # not vetoed

    def test_strong_disagreement_vetoes(self):
        result = fuse_signals(_ai(), _quant(-0.6))
        self.assertEqual(result["agreement"], "strong_disagree_veto")
        self.assertEqual(result["edge"], 0.0)
        self.assertEqual(result["position_size_multiplier"], 0.0)
        self.assertEqual(result["direction"], "HOLD")
        self.assertIsNone(result["recommended_outcome"])
        self.assertIn("strong quant disagreement", result["veto_reason"])

    def test_untradeable_book_vetoes(self):
        result = fuse_signals(_ai(), _quant(0.3, tradeable=False))
        self.assertEqual(result["agreement"], "untradeable")
        self.assertEqual(result["edge"], 0.0)
        self.assertIsNone(result["recommended_outcome"])

    def test_no_quant_data_falls_back_to_ai(self):
        result = fuse_signals(_ai(), _quant(0.0, available=False))
        self.assertEqual(result["agreement"], "ai_only")
        self.assertEqual(result["edge"], 0.12)  # unchanged

    def test_none_quant_falls_back_to_ai(self):
        result = fuse_signals(_ai(), None)
        self.assertEqual(result["agreement"], "ai_only")
        self.assertEqual(result["edge"], 0.12)

    def test_no_recommendation_is_ai_only(self):
        result = fuse_signals(_ai(recommended_outcome=None, recommended_outcome_index=None), _quant(0.4))
        self.assertEqual(result["agreement"], "ai_only")

    def test_quant_context_always_attached(self):
        result = fuse_signals(_ai(), _quant(0.3))
        self.assertIn("quant", result)
        self.assertIn("ai_edge", result)
        self.assertIn("quant_edge", result)
        self.assertIn("position_size_multiplier", result)


if __name__ == "__main__":
    unittest.main()

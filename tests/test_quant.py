from __future__ import annotations

import unittest

from polymarket_bot import quant
from polymarket_bot.quant import _feature_payload, book_features, compute_quant_signal, price_features


def _rising(n=30, start=0.30, step=0.01):
    return [{"t": i, "p": start + i * step} for i in range(n)]


def _falling(n=30, start=0.70, step=0.01):
    return [{"t": i, "p": start - i * step} for i in range(n)]


LIQUID_BOOK = {
    "bids": [{"price": 0.49, "size": 1000}, {"price": 0.48, "size": 800}],
    "asks": [{"price": 0.51, "size": 1000}, {"price": 0.52, "size": 800}],
}


class PriceFeatureTests(unittest.TestCase):
    def test_empty_history(self):
        self.assertFalse(price_features([])["available"])

    def test_rising_series_has_positive_momentum_and_trend(self):
        pf = price_features(_rising())
        self.assertTrue(pf["available"])
        self.assertGreater(pf["momentum"], 0)
        self.assertEqual(pf["trend"], 1)
        self.assertEqual(pf["n_samples"], 30)

    def test_falling_series_has_negative_momentum(self):
        pf = price_features(_falling())
        self.assertLess(pf["momentum"], 0)
        self.assertEqual(pf["trend"], -1)

    def test_single_point_is_safe(self):
        pf = price_features([{"t": 1, "p": 0.5}])
        self.assertTrue(pf["available"])
        self.assertEqual(pf["momentum"], 0.0)


class BookFeatureTests(unittest.TestCase):
    def test_empty_book_unavailable(self):
        self.assertFalse(book_features({"bids": [], "asks": []})["available"])

    def test_best_levels_and_spread(self):
        bf = book_features(LIQUID_BOOK)
        self.assertEqual(bf["best_bid"], 0.49)
        self.assertEqual(bf["best_ask"], 0.51)
        self.assertAlmostEqual(bf["spread"], 0.02, places=4)
        self.assertAlmostEqual(bf["mid"], 0.50, places=4)

    def test_imbalance_sign(self):
        heavy_bids = {
            "bids": [{"price": 0.49, "size": 5000}],
            "asks": [{"price": 0.51, "size": 100}],
        }
        self.assertGreater(book_features(heavy_bids)["imbalance"], 0)


class QuantSignalTests(unittest.TestCase):
    def test_liquid_tight_book_is_tradeable(self):
        sig = compute_quant_signal(0.50, _rising(), LIQUID_BOOK)
        self.assertTrue(sig["tradeable"])
        self.assertGreater(sig["quant_score"], 0)

    def test_wide_spread_not_tradeable(self):
        wide = {"bids": [{"price": 0.40, "size": 1000}], "asks": [{"price": 0.60, "size": 1000}]}
        sig = compute_quant_signal(0.50, _rising(), wide)
        self.assertFalse(sig["tradeable"])
        self.assertIn("spread", sig["reason"])

    def test_thin_liquidity_not_tradeable(self):
        thin = {"bids": [{"price": 0.49, "size": 5}], "asks": [{"price": 0.51, "size": 5}]}
        sig = compute_quant_signal(0.50, _rising(), thin)
        self.assertFalse(sig["tradeable"])
        self.assertIn("liquidity", sig["reason"])

    def test_extreme_price_not_tradeable(self):
        book = {"bids": [{"price": 0.01, "size": 100000}], "asks": [{"price": 0.02, "size": 100000}]}
        sig = compute_quant_signal(0.015, _rising(start=0.01, step=0.0001), book)
        self.assertFalse(sig["tradeable"])

    def test_falling_market_negative_score(self):
        sig = compute_quant_signal(0.50, _falling(), LIQUID_BOOK)
        self.assertLess(sig["quant_score"], 0)

    def test_quant_edge_bounded(self):
        sig = compute_quant_signal(0.50, _rising(), LIQUID_BOOK)
        self.assertLessEqual(abs(sig["quant_edge"]), quant.QUANT_MAX_EDGE)

    def test_no_data_degrades_gracefully(self):
        sig = compute_quant_signal(0.50, [], {"bids": [], "asks": []})
        self.assertFalse(sig["tradeable"])
        self.assertFalse(sig["available"])

    def test_empty_feature_payload_zeroes_all_model_features(self):
        features = _feature_payload([], {"bids": [], "asks": []})
        for key in (
            "last_price",
            "momentum",
            "velocity",
            "volatility",
            "zscore",
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_position",
            "vwap",
            "order_imbalance",
            "spread",
            "liquidity_usdc",
        ):
            self.assertEqual(features[key], 0.0)

    def test_low_24h_volume_blocks_tradeability_when_available(self):
        sig = compute_quant_signal(0.50, _rising(), LIQUID_BOOK, {"volume_24h": 1000.0})
        self.assertFalse(sig["tradeable"])
        self.assertIn("24h volume", sig["reason"])


if __name__ == "__main__":
    unittest.main()

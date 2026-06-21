from __future__ import annotations

import unittest

from polymarket_bot.categories import categorize


class CategorizeTests(unittest.TestCase):
    def test_detects_crypto(self):
        self.assertEqual(categorize("Will Bitcoin close above $100k in 2026?"), "crypto")

    def test_detects_politics(self):
        self.assertEqual(categorize("Who will win the presidential election?"), "politics")

    def test_detects_economics(self):
        self.assertEqual(categorize("Will the Fed cut the interest rate in July?"), "economics")

    def test_detects_sports(self):
        self.assertEqual(categorize("Will the Lakers win the NBA championship?"), "sports")

    def test_unknown_is_other(self):
        self.assertEqual(categorize("Will the price of bananas change?"), "other")

    def test_word_boundary_avoids_false_positive(self):
        # "whether" must not match the "eth" crypto keyword.
        self.assertEqual(categorize("Question about whether something happens"), "other")

    def test_empty_question_is_other(self):
        self.assertEqual(categorize(""), "other")
        self.assertEqual(categorize(None), "other")


if __name__ == "__main__":
    unittest.main()

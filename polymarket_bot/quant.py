"""Quant engine: structured signals from numeric market data.

This is the "quant" half of the bot. Unlike the AI scorer (which reads news and
reasons in prose), these functions derive signals purely from numbers:

* **Price-history features** — moving averages, momentum, velocity, realised
  volatility and a mean-reversion z-score from the CLOB price series.
* **Order-book microstructure** — best bid/ask, spread, depth-weighted
  microprice, order-book imbalance and tradeable liquidity.
* **A directional signal** — a single conviction score in ``[-1, 1]`` for the
  token's price rising, plus a tradeability gate that filters illiquid or
  wide-spread markets before any capital (or AI budget) is committed.

Everything is pure Python (``statistics`` only) so it stays dependency-light and
unit-testable without network access.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

try:
    from .config import (
        MAX_PRICE,
        MAX_SPREAD,
        MIN_BOOK_LIQUIDITY_USDC,
        MIN_PRICE,
        QUANT_MA_LONG,
        QUANT_MA_SHORT,
        QUANT_MOMENTUM_LOOKBACK,
    )
except ImportError:  # pragma: no cover
    from config import (
        MAX_PRICE,
        MAX_SPREAD,
        MIN_BOOK_LIQUIDITY_USDC,
        MIN_PRICE,
        QUANT_MA_LONG,
        QUANT_MA_SHORT,
        QUANT_MOMENTUM_LOOKBACK,
    )

# A maxed-out quant conviction (|score| == 1) maps to this much implied edge.
# Keeps the quant contribution bounded versus the AI's fundamental edge.
QUANT_MAX_EDGE = 0.10

# Scaling constants for squashing raw indicators into [-1, 1] via tanh.
_MOMENTUM_SCALE = 0.05  # a 5-cent move over the lookback ~ strong momentum
_ZSCORE_SCALE = 2.0

_TOP_LEVELS = 5


def _prices(history: list[dict]) -> list[float]:
    return [float(pt["p"]) for pt in history if "p" in pt]


def _diffs(values: list[float]) -> list[float]:
    return [b - a for a, b in zip(values, values[1:])]


def price_features(history: list[dict]) -> dict[str, Any]:
    """Compute price-history features from a ``[{"t","p"}, ...]`` series."""
    prices = _prices(history)
    n = len(prices)
    if n == 0:
        return {"available": False, "n_samples": 0}
    if n == 1:
        return {
            "available": True,
            "n_samples": 1,
            "last_price": prices[0],
            "ma_short": prices[0],
            "ma_long": prices[0],
            "momentum": 0.0,
            "velocity": 0.0,
            "volatility": 0.0,
            "zscore": 0.0,
            "trend": 0,
        }

    last_price = prices[-1]
    ma_short = mean(prices[-QUANT_MA_SHORT:])
    ma_long = mean(prices[-QUANT_MA_LONG:])

    lookback = min(QUANT_MOMENTUM_LOOKBACK, n - 1)
    momentum = last_price - prices[-1 - lookback]

    vel_window = min(QUANT_MA_SHORT, n - 1)
    velocity = (last_price - prices[-1 - vel_window]) / vel_window if vel_window else 0.0

    diffs = _diffs(prices[-(lookback + 1):])
    volatility = pstdev(diffs) if len(diffs) > 1 else 0.0

    window = prices[-QUANT_MA_LONG:]
    spread_std = pstdev(window) if len(window) > 1 else 0.0
    zscore = (last_price - mean(window)) / spread_std if spread_std > 1e-9 else 0.0

    trend = 1 if ma_short > ma_long else (-1 if ma_short < ma_long else 0)

    return {
        "available": True,
        "n_samples": n,
        "last_price": round(last_price, 4),
        "ma_short": round(ma_short, 4),
        "ma_long": round(ma_long, 4),
        "momentum": round(momentum, 4),
        "velocity": round(velocity, 5),
        "volatility": round(volatility, 5),
        "zscore": round(zscore, 3),
        "trend": trend,
    }


def book_features(book: dict, top_n: int = _TOP_LEVELS) -> dict[str, Any]:
    """Compute order-book microstructure features from ``{bids, asks}``."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return {"available": False}

    best_bid = max(bids, key=lambda lvl: lvl["price"])
    best_ask = min(asks, key=lambda lvl: lvl["price"])
    bb, ba = best_bid["price"], best_ask["price"]
    mid = (bb + ba) / 2.0
    spread = max(ba - bb, 0.0)

    # Depth-weighted microprice using the best levels: heavier opposite-side size
    # pulls the fair price toward that side (classic microstructure estimator).
    bsz, asz = best_bid["size"], best_ask["size"]
    denom = bsz + asz
    microprice = (bb * asz + ba * bsz) / denom if denom > 0 else mid

    top_bids = sorted(bids, key=lambda lvl: lvl["price"], reverse=True)[:top_n]
    top_asks = sorted(asks, key=lambda lvl: lvl["price"])[:top_n]
    bid_vol = sum(lvl["size"] for lvl in top_bids)
    ask_vol = sum(lvl["size"] for lvl in top_asks)
    total_vol = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

    bid_liquidity = sum(lvl["price"] * lvl["size"] for lvl in top_bids)
    ask_liquidity = sum(lvl["price"] * lvl["size"] for lvl in top_asks)

    return {
        "available": True,
        "best_bid": round(bb, 4),
        "best_ask": round(ba, 4),
        "mid": round(mid, 4),
        "spread": round(spread, 4),
        "microprice": round(microprice, 4),
        "imbalance": round(imbalance, 4),
        "bid_liquidity_usdc": round(bid_liquidity, 2),
        "ask_liquidity_usdc": round(ask_liquidity, 2),
        "liquidity_usdc": round(min(bid_liquidity, ask_liquidity), 2),
    }


def _tanh(x: float) -> float:
    return math.tanh(x)


def compute_quant_signal(
    snapshot_price: float,
    history: list[dict],
    book: dict,
) -> dict[str, Any]:
    """Fuse price-history + order-book features into one directional signal.

    Returns a dict describing the token's microstructure, a ``quant_score`` in
    ``[-1, 1]`` (conviction that the token price rises), a microstructure-anchored
    ``quant_fair_value``, and a ``tradeable`` gate based on liquidity, spread and
    extreme-price filters.
    """
    pf = price_features(history)
    bf = book_features(book)

    price = float(snapshot_price or 0.0)
    if bf.get("available"):
        price = bf["mid"] or price
    elif pf.get("available"):
        price = pf.get("last_price", price)

    momentum = pf.get("momentum", 0.0) if pf.get("available") else 0.0
    zscore = pf.get("zscore", 0.0) if pf.get("available") else 0.0
    imbalance = bf.get("imbalance", 0.0) if bf.get("available") else 0.0

    # Continuation (momentum + book imbalance) net of a mild mean-reversion pull.
    momentum_term = _tanh(momentum / _MOMENTUM_SCALE)
    reversion_term = _tanh(zscore / _ZSCORE_SCALE)
    quant_score = 0.40 * momentum_term + 0.35 * imbalance - 0.25 * reversion_term
    quant_score = max(-1.0, min(1.0, quant_score))

    micro_fv = bf.get("microprice") if bf.get("available") else None
    quant_fair_value = micro_fv if micro_fv is not None else price

    spread = bf.get("spread", 1.0) if bf.get("available") else 1.0
    liquidity = bf.get("liquidity_usdc", 0.0) if bf.get("available") else 0.0

    reasons: list[str] = []
    if not bf.get("available"):
        reasons.append("no order book")
    if bf.get("available") and spread > MAX_SPREAD:
        reasons.append(f"spread {spread:.3f} > {MAX_SPREAD}")
    if bf.get("available") and liquidity < MIN_BOOK_LIQUIDITY_USDC:
        reasons.append(f"liquidity ${liquidity:.0f} < ${MIN_BOOK_LIQUIDITY_USDC:.0f}")
    if not (MIN_PRICE <= price <= MAX_PRICE):
        reasons.append(f"price {price:.3f} outside [{MIN_PRICE}, {MAX_PRICE}]")

    tradeable = bf.get("available", False) and not reasons

    return {
        "available": bool(pf.get("available") or bf.get("available")),
        "tradeable": bool(tradeable),
        "price": round(price, 4),
        "quant_score": round(quant_score, 4),
        "quant_fair_value": round(float(quant_fair_value), 4),
        "quant_edge": round(quant_score * QUANT_MAX_EDGE, 4),
        "momentum": momentum,
        "velocity": pf.get("velocity", 0.0),
        "volatility": pf.get("volatility", 0.0),
        "zscore": zscore,
        "trend": pf.get("trend", 0),
        "ma_short": pf.get("ma_short"),
        "ma_long": pf.get("ma_long"),
        "imbalance": imbalance,
        "spread": bf.get("spread"),
        "microprice": bf.get("microprice"),
        "liquidity_usdc": liquidity,
        "n_samples": pf.get("n_samples", 0),
        "reason": "; ".join(reasons) if reasons else "ok",
    }

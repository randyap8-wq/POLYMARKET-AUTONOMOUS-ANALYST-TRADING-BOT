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
import pickle
from pathlib import Path
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
_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "quant_model.pkl"
_MODEL_WARNING_PRINTED = False
_MODEL_FEATURES = [
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
]


def _prices(history: list[dict]) -> list[float]:
    return [float(pt["p"]) for pt in history if "p" in pt]


def _coerce_prices(values: Any) -> list[float]:
    """Return a clean price list from floats or common history dict shapes."""
    prices: list[float] = []
    for item in values or []:
        try:
            if isinstance(item, dict):
                raw = item.get("p", item.get("price", item.get("close")))
            else:
                raw = item
            if raw is not None:
                prices.append(float(raw))
        except (TypeError, ValueError):
            continue
    return prices


def _diffs(values: list[float]) -> list[float]:
    return [b - a for a, b in zip(values, values[1:])]


def _ema(values: list[float], period: int) -> list[float]:
    """Compute an exponential moving average series."""
    if not values:
        return []
    period = max(int(period), 1)
    alpha = 2.0 / (period + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def calculate_rsi(prices: list[float] | list[dict], period: int = 14) -> float:
    """Return the latest Relative Strength Index in ``[0, 100]``.

    The function accepts either raw price floats or history dictionaries with
    ``p``, ``price`` or ``close`` keys. Short/flat series return a neutral 50.
    """
    series = _coerce_prices(prices)
    period = max(int(period), 1)
    if len(series) <= period:
        return 50.0

    changes = _diffs(series)
    seed = changes[:period]
    avg_gain = mean(max(change, 0.0) for change in seed)
    avg_loss = mean(abs(min(change, 0.0)) for change in seed)

    for change in changes[period:]:
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


def calculate_macd(
    prices: list[float] | list[dict],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float]:
    """Return latest MACD line, signal line and histogram."""
    series = _coerce_prices(prices)
    if not series:
        return {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}

    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_series = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_series = _ema(macd_series, signal)
    macd_line = macd_series[-1] if macd_series else 0.0
    signal_line = signal_series[-1] if signal_series else 0.0
    return {
        "macd_line": round(macd_line, 6),
        "signal_line": round(signal_line, 6),
        "histogram": round(macd_line - signal_line, 6),
    }


def calculate_bollinger_bands(
    prices: list[float] | list[dict],
    period: int = 20,
    num_std: float = 2,
) -> dict[str, float]:
    """Return latest Bollinger upper/middle/lower bands."""
    series = _coerce_prices(prices)
    if not series:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}

    window = series[-max(int(period), 1):]
    middle = mean(window)
    std = pstdev(window) if len(window) > 1 else 0.0
    width = float(num_std) * std
    return {
        "upper": round(middle + width, 6),
        "middle": round(middle, 6),
        "lower": round(middle - width, 6),
    }


def _sorted_levels(levels: Any, *, bids: bool, depth: int | None = None) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for level in levels or []:
        try:
            out.append({"price": float(level["price"]), "size": float(level["size"])})
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda lvl: lvl["price"], reverse=bids)
    return out[:depth] if depth is not None else out


def calculate_vwap(order_book: dict) -> float:
    """Return volume-weighted average price across displayed bid/ask depth."""
    levels = _sorted_levels(order_book.get("bids"), bids=True) + _sorted_levels(order_book.get("asks"), bids=False)
    total_size = sum(level["size"] for level in levels)
    if total_size <= 0:
        return 0.0
    return round(sum(level["price"] * level["size"] for level in levels) / total_size, 6)


def calculate_order_imbalance(order_book: dict, depth: int = 5) -> float:
    """Return top-of-book imbalance ``(bid_vol - ask_vol) / total_vol``."""
    top_bids = _sorted_levels(order_book.get("bids"), bids=True, depth=depth)
    top_asks = _sorted_levels(order_book.get("asks"), bids=False, depth=depth)
    bid_vol = sum(level["size"] for level in top_bids)
    ask_vol = sum(level["size"] for level in top_asks)
    total_vol = bid_vol + ask_vol
    if total_vol <= 0:
        return 0.0
    return round((bid_vol - ask_vol) / total_vol, 6)


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

    top_bids = _sorted_levels(bids, bids=True, depth=top_n)
    top_asks = _sorted_levels(asks, bids=False, depth=top_n)
    bid_vol = sum(lvl["size"] for lvl in top_bids)
    ask_vol = sum(lvl["size"] for lvl in top_asks)
    total_vol = bid_vol + ask_vol
    imbalance = calculate_order_imbalance(book, depth=top_n) if total_vol > 0 else 0.0

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
        "vwap": calculate_vwap(book),
        "bid_liquidity_usdc": round(bid_liquidity, 2),
        "ask_liquidity_usdc": round(ask_liquidity, 2),
        "liquidity_usdc": round(min(bid_liquidity, ask_liquidity), 2),
    }


def _tanh(x: float) -> float:
    return math.tanh(x)


def _bollinger_position(last_price: float, bands: dict[str, float]) -> float:
    width = float(bands.get("upper", 0.0)) - float(bands.get("lower", 0.0))
    if width <= 1e-12:
        return 0.5
    return max(0.0, min(1.0, (last_price - float(bands.get("lower", 0.0))) / width))


def _feature_payload(prices: list[float], order_book: dict) -> dict[str, float]:
    """Build the enhanced numerical feature set used by ML and fallback logic."""
    history = [{"p": p} for p in prices]
    pf = price_features(history)
    bf = book_features(order_book)
    last_price = prices[-1] if prices else float(bf.get("mid") or 0.0)
    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    bands = calculate_bollinger_bands(prices)
    vwap = calculate_vwap(order_book)
    imbalance = calculate_order_imbalance(order_book)

    return {
        "last_price": round(last_price, 6),
        "momentum": float(pf.get("momentum", 0.0) or 0.0),
        "velocity": float(pf.get("velocity", 0.0) or 0.0),
        "volatility": float(pf.get("volatility", 0.0) or 0.0),
        "zscore": float(pf.get("zscore", 0.0) or 0.0),
        "rsi": rsi,
        "macd_line": float(macd["macd_line"]),
        "macd_signal": float(macd["signal_line"]),
        "macd_histogram": float(macd["histogram"]),
        "bollinger_upper": float(bands["upper"]),
        "bollinger_middle": float(bands["middle"]),
        "bollinger_lower": float(bands["lower"]),
        "bb_position": round(_bollinger_position(last_price, bands), 6),
        "vwap": vwap,
        "order_imbalance": imbalance,
        "spread": float(bf.get("spread", 0.0) or 0.0),
        "liquidity_usdc": float(bf.get("liquidity_usdc", 0.0) or 0.0),
    }


def _load_quant_model(path: Path = _MODEL_PATH) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # pragma: no cover - corrupt model/dependency issues
        print(f"Quant ML model failed to load, using rule-based fallback: {exc}")
        return None


def _ml_probability(model: Any, features: dict[str, float]) -> float | None:
    """Run a duck-typed sklearn/xgboost estimator if one is available."""
    names = list(getattr(model, "feature_names_in_", _MODEL_FEATURES))
    vector = [[float(features.get(name, 0.0) or 0.0) for name in names]]
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)
            return max(0.0, min(1.0, float(proba[0][1])))
        if hasattr(model, "predict"):
            pred = model.predict(vector)
            value = float(pred[0])
            return max(0.0, min(1.0, value if value <= 1.0 else value / 100.0))
    except Exception as exc:  # pragma: no cover - model-specific runtime issues
        print(f"Quant ML inference failed, using rule-based fallback: {exc}")
    return None


def _rule_based_probability(features: dict[str, float]) -> float:
    """Fallback scorer using bounded weighted Z-score style features."""
    rsi_term = (float(features.get("rsi", 50.0)) - 50.0) / 50.0
    macd_term = _tanh(float(features.get("macd_histogram", 0.0)) / 0.03)
    bollinger_term = (float(features.get("bb_position", 0.5)) - 0.5) * 2.0
    momentum_term = _tanh(float(features.get("momentum", 0.0)) / _MOMENTUM_SCALE)
    zscore_term = -_tanh(float(features.get("zscore", 0.0)) / _ZSCORE_SCALE)
    imbalance_term = float(features.get("order_imbalance", 0.0))

    score = (
        0.20 * rsi_term
        + 0.20 * macd_term
        + 0.15 * bollinger_term
        + 0.20 * momentum_term
        + 0.15 * zscore_term
        + 0.10 * imbalance_term
    )
    return round(max(0.01, min(0.99, 0.5 + 0.25 * score)), 6)


def get_enhanced_quant_signal(market_data: dict[str, Any]) -> dict[str, Any]:
    """Return ML-backed probability, confidence and engineered features.

    ``market_data`` may contain ``prices``/``history``/``price_history`` and an
    ``order_book``/``book``. If ``models/quant_model.pkl`` is absent or unusable,
    the function prints the requested warning and falls back to a deterministic
    feature-weighted scorer.
    """
    raw_prices = (
        market_data.get("prices")
        or market_data.get("history")
        or market_data.get("price_history")
        or market_data.get("ohlcv")
        or []
    )
    prices = _coerce_prices(raw_prices)
    if not prices and market_data.get("current_price") is not None:
        prices = [float(market_data["current_price"])]

    order_book = market_data.get("order_book") or market_data.get("book") or {"bids": [], "asks": []}
    features = _feature_payload(prices, order_book)

    model = _load_quant_model()
    probability = _ml_probability(model, features) if model is not None else None
    if probability is None:
        global _MODEL_WARNING_PRINTED
        if not _MODEL_WARNING_PRINTED:
            print("ML model missing, using rule-based fallback")
            _MODEL_WARNING_PRINTED = True
        probability = _rule_based_probability(features)

    completeness = sum(
        1
        for key in ("rsi", "macd_histogram", "bb_position", "vwap", "order_imbalance")
        if features.get(key) not in (None, 0.0)
    ) / 5.0
    confidence = min(1.0, abs(probability - 0.5) * 2.0 * (0.5 + 0.5 * completeness))
    return {
        "probability": round(float(probability), 6),
        "confidence": round(confidence, 6),
        "features": features,
    }


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
    raw_prices = _prices(history)
    rsi = calculate_rsi(raw_prices)
    macd = calculate_macd(raw_prices)
    bands = calculate_bollinger_bands(raw_prices)

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
        "price_history": raw_prices[-120:],
        "zscore": zscore,
        "rsi": rsi,
        "macd": macd,
        "bollinger_bands": bands,
        "vwap": bf.get("vwap"),
        "order_imbalance": imbalance,
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

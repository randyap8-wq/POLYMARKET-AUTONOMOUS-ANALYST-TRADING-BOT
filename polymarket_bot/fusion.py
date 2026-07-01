"""Fusion layer: combine the AI fair-value signal with the quant signal.

The AI scorer contributes a *fundamental* view (news-driven fair value). The
quant engine contributes a *microstructure/momentum* view plus a tradeability
gate. ``fuse_signals`` blends the two into a single actionable edge and adjusts
confidence based on whether the two signals agree:

* **Agree** (both lean the same direction) -> blended edge, optional confidence
  boost when the quant confirmation is strong and the book is tradeable.
* **Disagree** -> shrink the edge by ``DISAGREEMENT_PENALTY`` and downgrade
  confidence; a *strong* quant disagreement vetoes the pick entirely.
* **Untradeable** book (wide spread / thin liquidity) -> veto.
* **No quant data** (e.g. CLOB unreachable) -> fall back to the AI signal alone.

This is the "use both AI and a quant" core: neither signal trades on its own
without at least passing the other's sanity check.
"""

from __future__ import annotations

import logging
import os
from statistics import pstdev
from typing import Any

try:
    from .config import CONFIDENCE_RANK, DISAGREEMENT_PENALTY
except ImportError:  # pragma: no cover
    from config import CONFIDENCE_RANK, DISAGREEMENT_PENALTY

LOGGER = logging.getLogger("fusion")

# Quant conviction thresholds for calling agreement / disagreement.
_AGREE_EPS = 0.05
_STRONG_AGREE = 0.40
DISAGREEMENT_DISCOUNT = max(0.0, min(1.0, float(os.getenv("DISAGREEMENT_DISCOUNT", "0.3"))))
VOLATILITY_REGIME_THRESHOLD = float(os.getenv("VOLATILITY_REGIME_THRESHOLD", "0"))
_VOL_WINDOW = 20


def _coerce_confidence_score(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in CONFIDENCE_RANK:
            return {"low": 35.0, "medium": 65.0, "high": 85.0}[text]
        value = text.rstrip("%")
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    if 0.0 <= raw <= 1.0:
        raw *= 100.0
    return max(0.0, min(100.0, raw))


def _confidence_level(value: Any) -> str:
    score = _coerce_confidence_score(value)
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _adjust_confidence(value: Any, delta: float) -> float:
    return round(max(0.0, min(100.0, _coerce_confidence_score(value) + delta)), 2)


def _neutral_quant() -> dict[str, Any]:
    return {"available": False, "tradeable": False, "quant_score": 0.0, "quant_edge": 0.0}


def _extract_prices(payload: Any) -> list[float]:
    """Accept common market-data shapes and return a clean price series."""
    if not payload:
        return []
    if isinstance(payload, dict):
        raw = (
            payload.get("prices")
            or payload.get("price_history")
            or payload.get("history")
            or payload.get("ohlcv")
            or []
        )
    else:
        raw = payload

    prices: list[float] = []
    for item in raw or []:
        try:
            if isinstance(item, dict):
                value = item.get("p", item.get("price", item.get("close")))
            else:
                value = item
            if value is not None:
                prices.append(float(value))
        except (TypeError, ValueError):
            continue
    return prices


def calculate_volatility_regime(prices: list[float] | list[dict]) -> str:
    """Classify current volatility as ``HIGH`` or ``LOW`` from rolling stdevs."""
    series = _extract_prices(prices)
    if len(series) <= _VOL_WINDOW:
        return "LOW"

    diffs = [b - a for a, b in zip(series, series[1:])]
    rolling = [
        pstdev(diffs[i - _VOL_WINDOW:i])
        for i in range(_VOL_WINDOW, len(diffs) + 1)
        if len(diffs[i - _VOL_WINDOW:i]) > 1
    ]
    if not rolling:
        return "LOW"

    current = rolling[-1]
    if VOLATILITY_REGIME_THRESHOLD > 0:
        threshold = VOLATILITY_REGIME_THRESHOLD
    else:
        sorted_vols = sorted(rolling)
        threshold = sorted_vols[int(0.75 * (len(sorted_vols) - 1))]
    return "HIGH" if current > threshold else "LOW"


def _weights_for_regime(regime: str) -> tuple[float, float]:
    if regime == "HIGH":
        return 0.7, 0.3
    return 0.3, 0.7


def _clamp_probability(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _direction_from_edge(edge: float, eps: float = _AGREE_EPS) -> int:
    if edge > eps:
        return 1
    if edge < -eps:
        return -1
    return 0


def fuse_signals(
    ai_signal: dict[str, Any],
    quant_signal: dict[str, Any] | None,
    market_data: dict[str, Any] | None = None,
    *,
    ai_weight: float | None = None,
    quant_weight: float | None = None,
    penalty: float = DISAGREEMENT_PENALTY,
) -> dict[str, Any]:
    """Return a new score dict blending AI and quant signals adaptively."""
    result = dict(ai_signal)
    quant = quant_signal or _neutral_quant()

    # Surface the quant context on every scored market for transparency.
    result["quant"] = quant
    result["quant_score"] = quant.get("quant_score", 0.0)
    result["quant_fair_value"] = quant.get("quant_fair_value")
    result["quant_tradeable"] = quant.get("tradeable", False)
    result["volatility"] = quant.get("volatility", 0.0)
    result["liquidity_usdc"] = quant.get("liquidity_usdc")

    ai_edge = float(ai_signal.get("edge") or 0.0)
    ai_fair_value = _clamp_probability(ai_signal.get("fair_value_estimate", ai_signal.get("probability")))
    recommended = ai_signal.get("recommended_outcome")
    current_price = _clamp_probability(ai_signal.get("current_price"))
    result["confidence"] = _coerce_confidence_score(result.get("confidence", result.get("confidence_level")))
    result["confidence_level"] = _confidence_level(result["confidence"])

    result["ai_edge"] = round(ai_edge, 4)
    result["quant_edge"] = round(float(quant.get("quant_edge") or 0.0), 4)
    result["position_size_multiplier"] = 1.0
    result["direction"] = "BUY" if recommended else "HOLD"
    result["probability"] = ai_fair_value

    # Nothing to confirm: the AI declined to pick. Keep the quant context only.
    if not recommended:
        result["agreement"] = "ai_only"
        result["position_size_multiplier"] = 0.0
        result["direction"] = "HOLD"
        return result

    quant_available = bool(quant.get("available"))
    quant_score = float(quant.get("quant_score") or 0.0)
    quant_edge = float(quant.get("quant_edge") or 0.0)
    quant_probability = _clamp_probability(quant.get("probability"))
    if quant_probability <= 0 and current_price > 0:
        quant_probability = _clamp_probability(current_price + quant_edge)

    # Graceful degradation: no quant data -> trade on the AI signal alone.
    if not quant_available:
        result["agreement"] = "ai_only"
        result["fused"] = True
        result["probability"] = ai_fair_value
        return result

    # Tradeability veto: an untradeable book means we cannot realistically fill.
    if not quant.get("tradeable", False):
        result.update(
            {
                "agreement": "untradeable",
                "fused": True,
                "edge": 0.0,
                "probability": ai_fair_value,
                "position_size_multiplier": 0.0,
                "direction": "HOLD",
                "news_supports_bet": False,
                "recommended_outcome": None,
                "recommended_outcome_index": None,
                "outcome_index": None,
                "veto_reason": f"quant gate: {quant.get('reason', 'untradeable')}",
            }
        )
        return result

    prices = _extract_prices(market_data) or _extract_prices(quant)
    regime = calculate_volatility_regime(prices) if prices else ("HIGH" if float(quant.get("volatility") or 0.0) > 0.02 else "LOW")
    # Only apply regime-switching defaults for weights the caller did not supply.
    regime_ai_weight, regime_quant_weight = _weights_for_regime(regime)
    if ai_weight is None:
        ai_weight = regime_ai_weight
    if quant_weight is None:
        quant_weight = regime_quant_weight
    weight_sum = (ai_weight + quant_weight) or 1.0
    fused_probability = (ai_weight * ai_fair_value + quant_weight * quant_probability) / weight_sum
    fused_edge = fused_probability - current_price if current_price > 0 else (ai_weight * ai_edge + quant_weight * quant_edge) / weight_sum

    # The AI recommends a BUY, i.e. it expects this token to rise. The quant
    # agrees when its directional conviction is also positive.
    ai_direction = _direction_from_edge(ai_edge)
    quant_direction = _direction_from_edge(quant_score if abs(quant_score) >= _AGREE_EPS else quant_edge)
    if quant_direction > 0:
        agreement = "agree"
    elif quant_direction < 0:
        agreement = "disagree"
    else:
        agreement = "neutral"
    if ai_direction and quant_direction and ai_direction != quant_direction:
        agreement = "disagree"

    confidence = result["confidence"]
    position_size_multiplier = 1.0

    if agreement == "agree":
        if quant_score >= _STRONG_AGREE and quant.get("tradeable"):
            confidence = _adjust_confidence(confidence, +10.0)
    elif agreement == "disagree":
        position_size_multiplier = DISAGREEMENT_DISCOUNT
        confidence = _adjust_confidence(confidence, -20.0)
        LOGGER.info(
            "AI/quant disagreement: ai_reason=%s quant_reason=%s quant_score=%.3f",
            ai_signal.get("reasoning", ""),
            quant.get("reason", ""),
            quant_score,
        )

    result.update(
        {
            "agreement": agreement,
            "fused": True,
            "edge": round(fused_edge, 4),
            "confidence": confidence,
            "confidence_level": _confidence_level(confidence),
            "volatility_regime": regime,
            "ai_weight": ai_weight,
            "quant_weight": quant_weight,
            "probability": round(fused_probability, 4),
            "position_size_multiplier": round(position_size_multiplier, 4),
            "direction": "BUY" if fused_edge > 0 else "HOLD",
            "blended_fair_value": round(fused_probability, 4),
        }
    )
    return result

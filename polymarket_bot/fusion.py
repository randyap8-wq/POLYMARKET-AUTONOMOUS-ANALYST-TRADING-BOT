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

from typing import Any

try:
    from .config import AI_WEIGHT, CONFIDENCE_RANK, DISAGREEMENT_PENALTY, QUANT_WEIGHT
except ImportError:  # pragma: no cover
    from config import AI_WEIGHT, CONFIDENCE_RANK, DISAGREEMENT_PENALTY, QUANT_WEIGHT

_RANK_TO_LEVEL = {rank: level for level, rank in CONFIDENCE_RANK.items()}

# Quant conviction thresholds for calling agreement / disagreement.
_AGREE_EPS = 0.05
_STRONG_AGREE = 0.40
_STRONG_DISAGREE = -0.40


def _bump(level: str, steps: int) -> str:
    rank = CONFIDENCE_RANK.get(level, 0) + steps
    rank = max(min(rank, max(_RANK_TO_LEVEL)), min(_RANK_TO_LEVEL))
    return _RANK_TO_LEVEL[rank]


def _neutral_quant() -> dict[str, Any]:
    return {"available": False, "tradeable": False, "quant_score": 0.0, "quant_edge": 0.0}


def fuse_signals(
    ai_score: dict[str, Any],
    quant_signal: dict[str, Any] | None,
    *,
    ai_weight: float = AI_WEIGHT,
    quant_weight: float = QUANT_WEIGHT,
    penalty: float = DISAGREEMENT_PENALTY,
) -> dict[str, Any]:
    """Return a new score dict blending ``ai_score`` with ``quant_signal``."""
    result = dict(ai_score)
    quant = quant_signal or _neutral_quant()

    # Surface the quant context on every scored market for transparency.
    result["quant"] = quant
    result["quant_score"] = quant.get("quant_score", 0.0)
    result["quant_fair_value"] = quant.get("quant_fair_value")
    result["quant_tradeable"] = quant.get("tradeable", False)
    result["volatility"] = quant.get("volatility", 0.0)
    result["liquidity_usdc"] = quant.get("liquidity_usdc")

    ai_edge = float(ai_score.get("edge") or 0.0)
    ai_fair_value = float(ai_score.get("fair_value_estimate") or 0.0)
    recommended = ai_score.get("recommended_outcome")

    result["ai_edge"] = round(ai_edge, 4)
    result["quant_edge"] = round(float(quant.get("quant_edge") or 0.0), 4)

    # Nothing to confirm: the AI declined to pick. Keep the quant context only.
    if not recommended:
        result["agreement"] = "ai_only"
        return result

    quant_available = bool(quant.get("available"))
    quant_score = float(quant.get("quant_score") or 0.0)
    quant_edge = float(quant.get("quant_edge") or 0.0)

    # Graceful degradation: no quant data -> trade on the AI signal alone.
    if not quant_available:
        result["agreement"] = "ai_only"
        result["fused"] = True
        return result

    # Tradeability veto: an untradeable book means we cannot realistically fill.
    if not quant.get("tradeable", False):
        result.update(
            {
                "agreement": "untradeable",
                "fused": True,
                "edge": 0.0,
                "news_supports_bet": False,
                "recommended_outcome": None,
                "recommended_outcome_index": None,
                "outcome_index": None,
                "veto_reason": f"quant gate: {quant.get('reason', 'untradeable')}",
            }
        )
        return result

    weight_sum = (ai_weight + quant_weight) or 1.0
    fused_edge = (ai_weight * ai_edge + quant_weight * quant_edge) / weight_sum

    # The AI recommends a BUY, i.e. it expects this token to rise. The quant
    # agrees when its directional conviction is also positive.
    if quant_score >= _AGREE_EPS:
        agreement = "agree"
    elif quant_score <= -_AGREE_EPS:
        agreement = "disagree"
    else:
        agreement = "neutral"

    confidence = str(ai_score.get("confidence") or "low").lower()

    if agreement == "agree":
        if quant_score >= _STRONG_AGREE and quant.get("tradeable"):
            confidence = _bump(confidence, +1)
    elif agreement == "disagree":
        fused_edge *= max(0.0, 1.0 - penalty)
        confidence = _bump(confidence, -1)
        if quant_score <= _STRONG_DISAGREE:
            result.update(
                {
                    "agreement": "disagree",
                    "fused": True,
                    "edge": 0.0,
                    "confidence": confidence,
                    "news_supports_bet": False,
                    "recommended_outcome": None,
                    "recommended_outcome_index": None,
                    "outcome_index": None,
                    "veto_reason": f"quant strongly disagrees (score {quant_score:.2f})",
                }
            )
            return result

    result.update(
        {
            "agreement": agreement,
            "fused": True,
            "edge": round(fused_edge, 4),
            "confidence": confidence,
            "blended_fair_value": round(
                (ai_weight * ai_fair_value + quant_weight * float(quant.get("quant_fair_value") or ai_fair_value))
                / weight_sum,
                4,
            ),
        }
    )
    return result

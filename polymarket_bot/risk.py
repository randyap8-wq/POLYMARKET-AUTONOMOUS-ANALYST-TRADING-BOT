"""Systematic risk controls and position sizing.

Turns a ranked list of opportunities into capital allocations using rules a
quant desk would recognise:

* **Volatility-scaled half-Kelly** — Kelly fraction from the edge, halved (or
  ``KELLY_FRACTION``), then shrunk further when the order book is choppy.
* **Portfolio exposure cap** — total deployed capital across one scan.
* **Category (correlation) cap** — markets in the same category are correlated;
  limit concentration per category.
* **Concurrency cap** — at most ``MAX_CONCURRENT_POSITIONS`` live bets.
* **Drawdown circuit breaker** — throttle (or halt) sizing when recent realised
  P&L breaches ``MAX_DRAWDOWN_USDC``.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from .config import (
        KELLY_FRACTION,
        MAX_BET_USDC,
        MAX_CATEGORY_EXPOSURE_USDC,
        MAX_CONCURRENT_POSITIONS,
        MAX_DRAWDOWN_USDC,
        MAX_PORTFOLIO_EXPOSURE_USDC,
        VOL_SIZING,
    )
except ImportError:  # pragma: no cover
    from config import (
        KELLY_FRACTION,
        MAX_BET_USDC,
        MAX_CATEGORY_EXPOSURE_USDC,
        MAX_CONCURRENT_POSITIONS,
        MAX_DRAWDOWN_USDC,
        MAX_PORTFOLIO_EXPOSURE_USDC,
        VOL_SIZING,
    )

LOGGER = logging.getLogger("risk")

_VOL_REF = 0.01  # ~1 cent per-sample move treated as "normal" volatility
_MIN_VOL_FACTOR = 0.25
_MIN_STAKE_USDC = 1.0


def kelly_fraction(price: float, fair_value: float, cap: float = KELLY_FRACTION) -> float:
    """Fractional-Kelly stake for a $1-payout binary share.

    ``price`` is the entry cost, ``fair_value`` the win probability. Returns a
    capped, non-negative fraction (``full_kelly * cap``).
    """
    p = max(min(price, 0.999), 0.001)
    q = max(min(fair_value, 1.0), 0.0)
    b = (1.0 - p) / p  # net odds on a win
    if b <= 0:
        return 0.0
    full_kelly = q - (1.0 - q) / b
    return round(max(0.0, min(full_kelly * cap, 1.0)), 4)


def volatility_factor(volatility: float) -> float:
    """Map realised volatility to a sizing shrink factor in ``[0.25, 1.0]``."""
    if not VOL_SIZING or volatility <= 0:
        return 1.0
    factor = 1.0 / (1.0 + volatility / _VOL_REF)
    return round(max(_MIN_VOL_FACTOR, min(1.0, factor)), 4)


def drawdown_factor(resolved: list[dict] | None, window: int = 20) -> float:
    """Circuit breaker: throttle sizing when recent realised P&L is poor.

    Returns 1.0 normally, 0.5 once the recent loss breaches ``MAX_DRAWDOWN_USDC``
    and 0.0 (halt) once it breaches twice that. Disabled when the threshold is 0.
    """
    if MAX_DRAWDOWN_USDC <= 0 or not resolved:
        return 1.0
    recent = resolved[-window:]
    recent_pnl = sum(float(b.get("pnl_usdc") or 0.0) for b in recent)
    if recent_pnl <= -2 * MAX_DRAWDOWN_USDC:
        LOGGER.warning("drawdown breaker HALT: recent P&L $%.2f", recent_pnl)
        return 0.0
    if recent_pnl <= -MAX_DRAWDOWN_USDC:
        LOGGER.warning("drawdown breaker THROTTLE: recent P&L $%.2f", recent_pnl)
        return 0.5
    return 1.0


def size_positions(
    opportunities: list[dict[str, Any]],
    resolved: list[dict] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate opportunities (in place) with ``stake_usdc`` honouring all caps.

    Expects ``opportunities`` already filtered and ranked by edge (best first).
    Returns the same list plus a portfolio summary.
    """
    if resolved is None:
        try:
            from .paper_trader import load_resolved
        except ImportError:  # pragma: no cover
            try:
                from paper_trader import load_resolved
            except Exception:  # pragma: no cover
                load_resolved = lambda: []  # noqa: E731
        try:
            resolved = load_resolved()
        except Exception:  # pragma: no cover
            resolved = []

    dd_factor = drawdown_factor(resolved)

    # Kelly is a fraction of bankroll; treat the portfolio exposure budget as the
    # bankroll and cap each ticket at MAX_BET_USDC.
    bankroll = MAX_PORTFOLIO_EXPOSURE_USDC if MAX_PORTFOLIO_EXPOSURE_USDC > 0 else MAX_BET_USDC

    portfolio_used = 0.0
    category_used: dict[str, float] = {}
    n_sized = 0

    for opp in opportunities:
        price = float(opp.get("current_price") or 0.0)
        fair_value = float(opp.get("fair_value_estimate") or 0.0)
        volatility = float(opp.get("volatility") or 0.0)
        category = opp.get("category", "other")
        position_multiplier = max(0.0, min(1.0, float(opp.get("position_size_multiplier", 1.0) or 0.0)))

        base_fraction = kelly_fraction(price, fair_value)
        vol_factor = volatility_factor(volatility)
        stake = min(bankroll * base_fraction * vol_factor * dd_factor * position_multiplier, MAX_BET_USDC)

        # Concurrency cap.
        if n_sized >= MAX_CONCURRENT_POSITIONS:
            stake = 0.0
            opp["sizing_note"] = "beyond max concurrent positions"

        # Portfolio + category exposure caps.
        if stake > 0:
            portfolio_room = max(0.0, MAX_PORTFOLIO_EXPOSURE_USDC - portfolio_used)
            cat_room = max(0.0, MAX_CATEGORY_EXPOSURE_USDC - category_used.get(category, 0.0))
            capped = min(stake, portfolio_room, cat_room)
            if capped < stake:
                opp["sizing_note"] = "capped by portfolio/category limit"
            stake = capped

        stake = round(stake, 2)
        if 0 < stake < _MIN_STAKE_USDC:
            stake = 0.0
            opp.setdefault("sizing_note", "below minimum stake")

        if stake > 0:
            portfolio_used += stake
            category_used[category] = category_used.get(category, 0.0) + stake
            n_sized += 1

        opp["stake_usdc"] = stake
        opp["stake_fraction"] = round(base_fraction, 4)
        opp["vol_factor"] = vol_factor
        opp["position_size_multiplier"] = position_multiplier
        opp["drawdown_factor"] = dd_factor

    summary = {
        "total_exposure_usdc": round(portfolio_used, 2),
        "positions_sized": n_sized,
        "by_category_usdc": {k: round(v, 2) for k, v in category_used.items()},
        "drawdown_factor": dd_factor,
        "max_portfolio_exposure_usdc": MAX_PORTFOLIO_EXPOSURE_USDC,
        "max_category_exposure_usdc": MAX_CATEGORY_EXPOSURE_USDC,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    }
    return opportunities, summary

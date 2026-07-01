"""Point-in-time backtest for the quant price-history signal.

Gist critique: the AI backfill has look-ahead bias because news is fetched *now*,
not at the historical decision time. The quant signal does not share that flaw —
CLOB price history is timestamped, so we can replay it honestly:

For each resolved market we take an entry point partway through its price history,
compute the quant signal using **only** the data up to that point (no future
leakage), then score the prediction against the known resolution. The AI is
intentionally excluded because news cannot be replayed point-in-time — this
isolates and measures the *quant* edge with no look-ahead bias.

Order-book microstructure is also excluded (no historical book snapshots), so the
backtest evaluates the price-history features (momentum / mean-reversion / trend).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from statistics import mean

import requests

try:
    from .config import BASE_DIR, GAMMA_BASE
    from .market_data import fetch_price_history
    from .quant import compute_quant_signal
    from .utils import retry_with_backoff
except ImportError:  # pragma: no cover
    from config import BASE_DIR, GAMMA_BASE
    from market_data import fetch_price_history
    from quant import compute_quant_signal
    from utils import retry_with_backoff

LOGGER = logging.getLogger("backtest")

BACKTEST_PATH = BASE_DIR / "data" / "backtest.json"

_MIN_SAMPLES = 8
_EMPTY_BOOK = {"bids": [], "asks": []}


@retry_with_backoff(max_retries=5, base_delay=1)
def _fetch_closed_gamma_markets(limit: int) -> list[dict]:
    response = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"closed": "true", "limit": limit, "order": "volume", "ascending": "false"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def _clamp_price(price: float) -> float:
    return max(0.01, min(0.99, price))


def evaluate_market(
    histories: list[list[dict]],
    winner_index: int,
    *,
    entry_fraction: float = 0.6,
    threshold: float = 0.15,
) -> list[dict]:
    """Replay the quant signal point-in-time for one market.

    ``histories[i]`` is the price series for outcome ``i``. Returns one record per
    triggered signal (``quant_score`` past ``threshold``), each with the simulated
    per-$1 P&L versus the known ``winner_index``.
    """
    signals: list[dict] = []
    for index, history in enumerate(histories):
        if len(history) < _MIN_SAMPLES:
            continue

        entry_idx = int(len(history) * entry_fraction)
        entry_idx = max(_MIN_SAMPLES - 1, min(entry_idx, len(history) - 2))
        if entry_idx < _MIN_SAMPLES - 1:
            continue

        entry_price = float(history[entry_idx]["p"])
        past = history[: entry_idx + 1]  # strictly point-in-time, no look-ahead
        sig = compute_quant_signal(entry_price, past, _EMPTY_BOOK)
        qs = sig["quant_score"]

        if qs >= threshold:
            # Predict this token rises -> buy it; wins if it is the resolved outcome.
            price = _clamp_price(entry_price)
            win = index == winner_index
            pnl = (1.0 / price - 1.0) if win else -1.0
            signals.append(
                {"index": index, "direction": "up", "quant_score": qs, "entry_price": round(price, 4), "win": win, "pnl": round(pnl, 4)}
            )
        elif qs <= -threshold:
            # Predict this token falls -> buy the complement at (1 - price).
            price = _clamp_price(1.0 - entry_price)
            win = index != winner_index
            pnl = (1.0 / price - 1.0) if win else -1.0
            signals.append(
                {"index": index, "direction": "down", "quant_score": qs, "entry_price": round(price, 4), "win": win, "pnl": round(pnl, 4)}
            )
    return signals


def _fetch_resolved_markets(days_back: int, limit: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        raw = _fetch_closed_gamma_markets(limit)
    except Exception as exc:
        LOGGER.error("failed to fetch closed markets: %s", exc)
        return []

    out: list[dict] = []
    for m in raw:
        try:
            resolution_time = m.get("resolutionTime") or m.get("resolution_time")
            if not resolution_time:
                continue
            res_dt = datetime.fromisoformat(str(resolution_time).replace("Z", "+00:00"))
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            if res_dt < cutoff:
                continue

            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            token_ids = [str(t) for t in json.loads(m.get("clobTokenIds") or "[]")]
            if len(outcomes) != 2 or not token_ids or max(prices, default=0) < 0.99:
                continue

            out.append(
                {
                    "question": m.get("question", ""),
                    "outcomes": outcomes,
                    "token_ids": token_ids,
                    "winner_index": prices.index(max(prices)),
                }
            )
        except Exception as exc:
            LOGGER.debug("skipping market in backtest fetch: %s", exc)
    return out


def run_backtest(
    days_back: int = 30,
    limit: int = 60,
    entry_fraction: float = 0.6,
    threshold: float = 0.15,
) -> dict:
    """Run the point-in-time quant backtest over recently resolved markets."""
    markets = _fetch_resolved_markets(days_back, limit)
    LOGGER.info("backtest: %s resolved binary markets", len(markets))

    all_signals: list[dict] = []
    for market in markets:
        histories = [fetch_price_history(tid) for tid in market["token_ids"]]
        signals = evaluate_market(histories, market["winner_index"], entry_fraction=entry_fraction, threshold=threshold)
        for sig in signals:
            sig["question"] = market["question"][:80]
        all_signals.extend(signals)

    n = len(all_signals)
    wins = sum(1 for s in all_signals if s["win"])
    total_pnl = sum(s["pnl"] for s in all_signals)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "point-in-time quant replay (no look-ahead, price-history only)",
        "markets_tested": len(markets),
        "signals": n,
        "hit_rate": round(wins / n, 4) if n else 0.0,
        "total_pnl_per_unit": round(total_pnl, 4),
        "avg_pnl_per_signal": round(total_pnl / n, 4) if n else 0.0,
        "entry_fraction": entry_fraction,
        "threshold": threshold,
        "up_signals": sum(1 for s in all_signals if s["direction"] == "up"),
        "down_signals": sum(1 for s in all_signals if s["direction"] == "down"),
        "avg_quant_score": round(mean(abs(s["quant_score"]) for s in all_signals), 4) if n else 0.0,
    }

    BACKTEST_PATH.parent.mkdir(exist_ok=True)
    BACKTEST_PATH.write_text(json.dumps({"summary": summary, "signals": all_signals[:200]}, indent=2), encoding="utf-8")

    print("\n" + "=" * 52)
    print(" QUANT BACKTEST — point-in-time (no look-ahead)")
    print("=" * 52)
    print(f" Markets tested : {summary['markets_tested']}")
    print(f" Signals fired  : {summary['signals']}  (up {summary['up_signals']} / down {summary['down_signals']})")
    print(f" Hit rate       : {summary['hit_rate']*100:.1f}%")
    print(f" Total P&L/unit : {summary['total_pnl_per_unit']:+.3f}")
    print(f" Avg P&L/signal : {summary['avg_pnl_per_signal']:+.4f}")
    print(f" Saved to {BACKTEST_PATH.name}")
    print("=" * 52 + "\n")
    return summary

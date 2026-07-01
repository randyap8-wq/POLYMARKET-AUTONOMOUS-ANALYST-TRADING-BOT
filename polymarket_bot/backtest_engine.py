"""Full-pipeline historical backtesting engine.

The existing ``backtest.py`` intentionally isolates the quant signal. This
module simulates the broader production path for historical OHLCV rows plus
order-book snapshots: enhanced quant features, cached/proxy AI scoring, adaptive
fusion, Kelly sizing, fees, slippage and portfolio-level performance metrics.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    from .fusion import fuse_signals
    from .quant import compute_quant_signal, get_enhanced_quant_signal
    from .risk import kelly_fraction
except ImportError:  # pragma: no cover
    from fusion import fuse_signals
    from quant import compute_quant_signal, get_enhanced_quant_signal
    from risk import kelly_fraction

RESULTS_PATH = Path(__file__).resolve().parents[1] / "backtest_results.csv"
POLYMARKET_WIN_FEE = 0.02
SLIPPAGE_DEPTH_FACTOR = 0.1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_probability(value: Any) -> float:
    return max(0.01, min(0.99, _safe_float(value, 0.5)))


def _price_from_row(row: dict[str, Any]) -> float:
    for key in ("close", "price", "current_price", "mid"):
        if row.get(key) is not None:
            return _clamp_probability(row[key])
    return 0.5


def _order_book(row: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    book = row.get("order_book") or row.get("book")
    if isinstance(book, dict):
        return {"bids": book.get("bids") or [], "asks": book.get("asks") or []}
    return {"bids": row.get("bids") or [], "asks": row.get("asks") or []}


def _total_liquidity(order_book: dict[str, list[dict[str, float]]]) -> float:
    # BUY-only simulator: only ask-side depth can fill the trade, so bid-side
    # liquidity is excluded to avoid under-estimating slippage/costs.
    total = 0.0
    for level in order_book.get("asks") or []:
        total += _safe_float(level.get("price")) * _safe_float(level.get("size"))
    return max(total, 1.0)


def _to_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return list(data.to_dict("records"))
    return [dict(row) for row in data]


def _timestamp_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    """Order rows chronologically, sorting numeric timestamps numerically."""
    raw = row.get("timestamp")
    if raw is None:
        raw = row.get("t")
    if raw is None:
        return (2, 0.0, "")
    try:
        return (0, float(raw), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(raw))


class BacktestEngine:
    """Replay historical market rows through the production signal pipeline."""

    def __init__(
        self,
        historical_market_data: Iterable[dict[str, Any]] | Any,
        *,
        initial_bankroll: float = 1_000.0,
        max_position_usdc: float = 25.0,
        min_history: int = 20,
        ai_responses: dict[str, dict[str, Any]] | None = None,
        output_path: str | Path = RESULTS_PATH,
    ) -> None:
        self.records = _to_records(historical_market_data)
        self.initial_bankroll = float(initial_bankroll)
        self.max_position_usdc = float(max_position_usdc)
        self.min_history = max(2, int(min_history))
        self.ai_responses = ai_responses or {}
        self.output_path = Path(output_path)
        self.results: Any = None
        self.summary: dict[str, Any] = {}

    def run(self) -> Any:
        """Run the simulation, save results, print a summary and return results."""
        bankroll = self.initial_bankroll
        peak_equity = bankroll
        rows: list[dict[str, Any]] = []

        for market_rows in self._grouped_rows():
            if len(market_rows) <= self.min_history:
                continue

            final_row = market_rows[-1]
            for idx in range(self.min_history, len(market_rows) - 1):
                history_rows = market_rows[: idx + 1]
                row = market_rows[idx]
                price_history = [_price_from_row(item) for item in history_rows]
                current_price = price_history[-1]
                book = _order_book(row)

                quant_signal = self._quant_signal(current_price, price_history, book)
                ai_signal = self._ai_signal(row, price_history, quant_signal)
                fused = fuse_signals(ai_signal, quant_signal, {"prices": price_history, "order_book": book})

                kelly = kelly_fraction(current_price, fused.get("probability", current_price))
                size_multiplier = _safe_float(fused.get("position_size_multiplier"), 1.0)
                position_size = min(bankroll * kelly * size_multiplier, self.max_position_usdc)
                if fused.get("direction") != "BUY" or not fused.get("recommended_outcome"):
                    position_size = 0.0

                liquidity = _total_liquidity(book)
                slippage = (position_size / liquidity) * SLIPPAGE_DEPTH_FACTOR if position_size > 0 else 0.0
                execution_price = _clamp_probability(current_price + slippage)
                win = self._resolved_win(row, final_row)
                pnl, fee = self._settle_trade(position_size, execution_price, win)
                bankroll += pnl
                peak_equity = max(peak_equity, bankroll)
                drawdown = (bankroll - peak_equity) / peak_equity if peak_equity else 0.0

                rows.append(
                    {
                        "timestamp": row.get("timestamp") or row.get("t") or idx,
                        "market_id": row.get("market_id") or row.get("condition_id") or "market",
                        "question": row.get("question", ""),
                        "category": str(row.get("category", "other")).lower(),
                        "current_price": round(current_price, 4),
                        "execution_price": round(execution_price, 4),
                        "slippage": round(slippage, 6),
                        "total_liquidity": round(liquidity, 2),
                        "ai_probability": round(_safe_float(ai_signal.get("probability")), 4),
                        "quant_probability": round(_safe_float(quant_signal.get("probability")), 4),
                        "fused_probability": round(_safe_float(fused.get("probability")), 4),
                        "confidence": round(_safe_float(fused.get("confidence")), 2),
                        "agreement": fused.get("agreement"),
                        "volatility_regime": fused.get("volatility_regime"),
                        "kelly_fraction": round(kelly, 4),
                        "position_size_multiplier": round(size_multiplier, 4),
                        "position_size_usdc": round(position_size, 2),
                        "win": bool(win) if position_size > 0 else None,
                        "fee_usdc": round(fee, 4),
                        "pnl_usdc": round(pnl, 4),
                        "return_on_stake": round(pnl / position_size, 4) if position_size > 0 else 0.0,
                        "bankroll": round(bankroll, 4),
                        "drawdown": round(drawdown, 4),
                    }
                )

        self.results = self._dataframe(rows)
        self.summary = self._calculate_summary(rows)
        self._save(rows)
        self.print_summary()
        return self.results

    def _grouped_rows(self) -> list[list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.records:
            key = str(row.get("market_id") or row.get("condition_id") or "market")
            groups[key].append(row)
        return [
            sorted(rows, key=_timestamp_sort_key)
            for rows in groups.values()
        ]

    def _quant_signal(
        self,
        current_price: float,
        price_history: list[float],
        order_book: dict[str, list[dict[str, float]]],
    ) -> dict[str, Any]:
        history = [{"t": i, "p": price} for i, price in enumerate(price_history)]
        legacy = compute_quant_signal(current_price, history, order_book)
        enhanced = get_enhanced_quant_signal(
            {"prices": price_history, "order_book": order_book, "current_price": current_price}
        )
        probability = _clamp_probability(enhanced["probability"])
        legacy.update(
            {
                "probability": probability,
                "confidence": enhanced["confidence"],
                "features": enhanced["features"],
                "quant_edge": round(probability - current_price, 4),
                "quant_score": round((probability - 0.5) * 2.0, 4),
            }
        )
        return legacy

    def _ai_signal(
        self,
        row: dict[str, Any],
        price_history: list[float],
        quant_signal: dict[str, Any],
    ) -> dict[str, Any]:
        cache_key = str(row.get("market_id") or row.get("condition_id") or row.get("timestamp") or "")
        cached = self.ai_responses.get(cache_key)
        if cached:
            probability = _clamp_probability(cached.get("probability", cached.get("fair_value_estimate")))
            confidence = _safe_float(cached.get("confidence"), 60.0)
            reasoning = str(cached.get("reasoning", "cached AI response"))
        else:
            momentum = price_history[-1] - price_history[max(0, len(price_history) - 6)]
            probability = _clamp_probability(0.55 * _safe_float(quant_signal.get("probability"), 0.5) + 0.45 * (price_history[-1] + momentum))
            confidence = min(90.0, 50.0 + abs(probability - price_history[-1]) * 300.0)
            reasoning = "rule-based historical AI proxy"

        current_price = price_history[-1]
        edge = probability - current_price
        should_buy = edge > 0.02
        return {
            "recommended_outcome": row.get("outcome", "YES") if should_buy else None,
            "recommended_outcome_index": 0 if should_buy else None,
            "current_price": current_price,
            "fair_value_estimate": probability,
            "probability": probability,
            "edge": round(edge if should_buy else 0.0, 4),
            "confidence": round(confidence, 2),
            "confidence_level": "high" if confidence >= 75 else "medium" if confidence >= 50 else "low",
            "direction": "BUY" if should_buy else "SKIP",
            "news_supports_bet": bool(should_buy),
            "reasoning": reasoning,
            "category": row.get("category", "other"),
        }

    def _resolved_win(self, row: dict[str, Any], final_row: dict[str, Any]) -> bool:
        for source in (row, final_row):
            if source.get("actual_outcome") is not None:
                return bool(source["actual_outcome"])
            if source.get("win") is not None:
                return bool(source["win"])
            if source.get("resolved_price") is not None:
                return _safe_float(source["resolved_price"]) >= 0.5
        return _price_from_row(final_row) >= 0.5

    def _settle_trade(self, position_size: float, execution_price: float, win: bool) -> tuple[float, float]:
        if position_size <= 0:
            return 0.0, 0.0
        if not win:
            return -position_size, 0.0
        shares = position_size / execution_price
        gross_payout = shares
        fee = gross_payout * POLYMARKET_WIN_FEE
        return gross_payout - fee - position_size, fee

    def _dataframe(self, rows: list[dict[str, Any]]) -> Any:
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.results, "to_csv"):
            self.results.to_csv(self.output_path, index=False)
            return
        if not rows:
            self.output_path.write_text("", encoding="utf-8")
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _calculate_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        trades = [row for row in rows if row["position_size_usdc"] > 0]
        returns = [row["return_on_stake"] for row in trades]
        total_pnl = sum(row["pnl_usdc"] for row in trades)
        total_return = total_pnl / self.initial_bankroll if self.initial_bankroll else 0.0
        sharpe = 0.0
        if len(returns) > 1 and pstdev(returns) > 1e-12:
            sharpe = mean(returns) / pstdev(returns) * math.sqrt(len(returns))

        by_category: dict[str, dict[str, Any]] = {}
        for category in sorted({row["category"] for row in trades}):
            bucket = [row for row in trades if row["category"] == category]
            by_category[category] = {
                "trades": len(bucket),
                "win_rate": round(sum(1 for row in bucket if row["win"]) / len(bucket), 4) if bucket else 0.0,
                "pnl_usdc": round(sum(row["pnl_usdc"] for row in bucket), 4),
            }

        return {
            "decisions": len(rows),
            "trades": len(trades),
            "total_return": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(min((row["drawdown"] for row in rows), default=0.0), 4),
            "win_rate": round(sum(1 for row in trades if row["win"]) / len(trades), 4) if trades else 0.0,
            "total_pnl_usdc": round(total_pnl, 4),
            "by_category": by_category,
            "results_path": str(self.output_path),
        }

    def print_summary(self) -> None:
        """Print a compact summary table for CLI usage."""
        summary = self.summary
        print("\n" + "=" * 64)
        print(" FULL PIPELINE BACKTEST")
        print("=" * 64)
        print(f" Decisions     : {summary.get('decisions', 0)}")
        print(f" Trades        : {summary.get('trades', 0)}")
        print(f" Total Return  : {summary.get('total_return', 0.0) * 100:+.2f}%")
        print(f" Sharpe Ratio  : {summary.get('sharpe_ratio', 0.0):+.2f}")
        print(f" Max Drawdown  : {summary.get('max_drawdown', 0.0) * 100:.2f}%")
        print(f" Win Rate      : {summary.get('win_rate', 0.0) * 100:.1f}%")
        print("-" * 64)
        print(" Category       Trades   Win Rate   P&L")
        for category, stats in (summary.get("by_category") or {}).items():
            print(
                f" {category:<14} {stats['trades']:>6}   "
                f"{stats['win_rate'] * 100:>7.1f}%   ${stats['pnl_usdc']:>+.2f}"
            )
        print(f" Saved to {summary.get('results_path', self.output_path)}")
        print("=" * 64 + "\n")

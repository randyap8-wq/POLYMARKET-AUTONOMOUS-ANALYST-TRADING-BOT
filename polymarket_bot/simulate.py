"""Risk-parameter simulation over cached full-backtest decisions."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

try:
    from .backtest_full import FULL_BACKTEST_PATH
    from .config import BASE_DIR
    from .risk import kelly_fraction
except ImportError:  # pragma: no cover
    from backtest_full import FULL_BACKTEST_PATH
    from config import BASE_DIR
    from risk import kelly_fraction

SIMULATION_PATH = BASE_DIR / "data" / "risk_simulation.json"


def _load_trades(path: str | Path = FULL_BACKTEST_PATH) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("trades", payload) if isinstance(payload, dict) else payload
    return [dict(row) for row in rows if isinstance(row, dict)]


def _grid(parameter_grid: dict[str, Iterable[Any]] | None) -> list[dict[str, float]]:
    grid = parameter_grid or {
        "kelly_fraction": [0.25, 0.5, 0.75],
        "max_portfolio_exposure_usdc": [25.0, 50.0, 100.0],
        "max_bet_usdc": [5.0, 10.0, 20.0],
    }
    keys = list(grid)
    combos: list[dict[str, float]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combos.append({key: float(value) for key, value in zip(keys, values)})
    return combos


def _simulate_one(trades: list[dict[str, Any]], params: dict[str, float], initial_bankroll: float) -> dict[str, Any]:
    bankroll = initial_bankroll
    peak = bankroll
    returns: list[float] = []
    total_pnl = 0.0
    wins = 0
    n_trades = 0

    for row in trades:
        if row.get("win") is None or not row.get("recommended_outcome"):
            continue
        price = max(0.01, min(0.99, float(row.get("current_price") or 0.5)))
        fair_value = max(0.0, min(1.0, float(row.get("probability", row.get("fair_value_estimate", price)) or price)))
        fraction = kelly_fraction(price, fair_value, cap=params["kelly_fraction"])
        stake = min(
            bankroll * fraction,
            params["max_bet_usdc"],
            params["max_portfolio_exposure_usdc"],
        )
        if stake <= 0:
            continue

        win = bool(row.get("win"))
        pnl = stake * (1.0 / price - 1.0) if win else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        total_pnl += pnl
        wins += 1 if win else 0
        n_trades += 1
        returns.append(pnl / stake)

    sharpe = 0.0
    if len(returns) > 1 and pstdev(returns) > 1e-12:
        sharpe = mean(returns) / pstdev(returns) * math.sqrt(len(returns))
    return {
        **params,
        "trades": n_trades,
        "win_rate": round(wins / n_trades, 4) if n_trades else 0.0,
        "total_return": round((bankroll - initial_bankroll) / initial_bankroll, 4) if initial_bankroll else 0.0,
        "total_pnl_usdc": round(total_pnl, 4),
        "sharpe_ratio": round(sharpe, 4),
    }


def run_simulation(
    parameter_grid: dict[str, Iterable[Any]] | None = None,
    *,
    backtest_path: str | Path = FULL_BACKTEST_PATH,
    initial_bankroll: float = 1_000.0,
) -> dict[str, Any]:
    """Run a risk-parameter sweep and save a ranked results table."""
    trades = _load_trades(backtest_path)
    results = [_simulate_one(trades, params, initial_bankroll) for params in _grid(parameter_grid)]
    results.sort(key=lambda row: (row["sharpe_ratio"], row["total_return"]), reverse=True)
    best = results[0] if results else {}

    payload = {
        "source": str(backtest_path),
        "initial_bankroll": initial_bankroll,
        "best": best,
        "results": results,
    }
    SIMULATION_PATH.parent.mkdir(exist_ok=True)
    SIMULATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(" RISK PARAMETER SIMULATION")
    print("=" * 78)
    print(" Sharpe   Return    P&L      Trades   Kelly   MaxExp   MaxBet")
    for row in results[:10]:
        print(
            f" {row['sharpe_ratio']:>+6.2f}  {row['total_return'] * 100:>+6.1f}%  "
            f"${row['total_pnl_usdc']:>+7.2f}  {row['trades']:>6}   "
            f"{row['kelly_fraction']:<5.2f}   ${row['max_portfolio_exposure_usdc']:<6.0f}  "
            f"${row['max_bet_usdc']:<5.0f}"
        )
    if best:
        print("-" * 78)
        print(
            " Suggested: "
            f"KELLY_FRACTION={best['kelly_fraction']}, "
            f"MAX_PORTFOLIO_EXPOSURE_USDC={best['max_portfolio_exposure_usdc']}, "
            f"MAX_BET_USDC={best['max_bet_usdc']}"
        )
    print(f" Saved to {SIMULATION_PATH}")
    print("=" * 78 + "\n")
    return payload

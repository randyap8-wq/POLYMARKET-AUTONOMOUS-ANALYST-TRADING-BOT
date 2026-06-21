from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from statistics import mean
from pathlib import Path

try:
    from .paper_trader import load_resolved
    from .config import BASE_DIR
except ImportError:  # pragma: no cover
    from paper_trader import load_resolved
    from config import BASE_DIR

LOGGER = logging.getLogger("validator")

PERFORMANCE_PATH = BASE_DIR / "data" / "performance.json"


def _win_rate(bets: list[dict]) -> float:
    if not bets:
        return 0.0
    return sum(1 for b in bets if b.get("correct")) / len(bets)


def _avg_pnl(bets: list[dict]) -> float:
    if not bets:
        return 0.0
    return mean(b.get("pnl_usdc", 0.0) for b in bets)


def generate_performance_report() -> dict:
    resolved = load_resolved()
    total = len(resolved)

    if total == 0:
        print("\n[validator] No resolved bets yet. Keep the bot running in --paper mode.\n")
        return {"total_bets": 0}

    correct = sum(1 for b in resolved if b.get("correct"))
    win_rate = _win_rate(resolved)
    total_pnl = sum(b.get("pnl_usdc", 0.0) for b in resolved)
    avg_pnl = total_pnl / total
    avg_edge = mean(b.get("edge", 0.0) for b in resolved)

    # By confidence
    conf_groups = {"high": [], "medium": [], "low": []}
    for b in resolved:
        c = b.get("confidence", "low")
        if c in conf_groups:
            conf_groups[c].append(b)

    conf_stats = {
        level: {
            "count": len(bets),
            "win_rate": round(_win_rate(bets), 4),
            "avg_pnl": round(_avg_pnl(bets), 4),
        }
        for level, bets in conf_groups.items()
    }

    # By edge bucket
    edge_groups = {
        "0.05-0.10": [b for b in resolved if 0.05 <= b.get("edge", 0) < 0.10],
        "0.10-0.15": [b for b in resolved if 0.10 <= b.get("edge", 0) < 0.15],
        "0.15+":     [b for b in resolved if b.get("edge", 0) >= 0.15],
    }
    edge_stats = {
        bucket: {
            "count": len(bets),
            "win_rate": round(_win_rate(bets), 4),
            "avg_pnl": round(_avg_pnl(bets), 4),
        }
        for bucket, bets in edge_groups.items()
    }

    # By category
    cat_groups: dict[str, list[dict]] = {}
    for b in resolved:
        cat_groups.setdefault(b.get("category", "other"), []).append(b)
    category_stats = {
        category: {
            "count": len(bets),
            "win_rate": round(_win_rate(bets), 4),
            "avg_pnl": round(_avg_pnl(bets), 4),
        }
        for category, bets in sorted(cat_groups.items())
    }

    # Generate recommendation
    recommendations = []
    if total < 30:
        recommendations.append(f"⏳ Only {total}/30 resolved bets — keep running before drawing conclusions.")
    else:
        if win_rate >= 0.60:
            recommendations.append("✅ SIGNAL VALIDATED — win rate >60% on 30+ bets. Consider going live with MAX_BET_USDC=2.")
        elif win_rate >= 0.53:
            recommendations.append("⚠️  MARGINAL SIGNAL — win rate 53-60%. Go live only with minimum stake ($2).")
        else:
            recommendations.append("❌ SIGNAL WEAK — win rate <53% on 30+ bets. Do NOT go live. Tune prompts first.")

        if conf_stats["high"]["count"] > 5 and conf_stats["high"]["win_rate"] >= 0.65:
            recommendations.append("✅ High-confidence picks are strong. Consider raising MIN_CONFIDENCE=high in .env")

        weak_buckets = [b for b, s in edge_stats.items() if s["count"] > 3 and s["win_rate"] < 0.50]
        if weak_buckets:
            recommendations.append(f"⚠️  Edge buckets with win rate <50%: {', '.join(weak_buckets)}. Raise MIN_EDGE.")

        best_bucket = max(edge_stats.items(), key=lambda x: x[1]["win_rate"] if x[1]["count"] > 2 else 0)
        if best_bucket[1]["win_rate"] >= 0.65 and best_bucket[1]["count"] > 3:
            recommendations.append(f"✅ Best edge bucket: {best_bucket[0]} (win rate {best_bucket[1]['win_rate']*100:.0f}%). Focus here.")

        weak_categories = [
            c for c, s in category_stats.items() if s["count"] >= 4 and s["win_rate"] < 0.50
        ]
        if weak_categories:
            recommendations.append(
                f"⚠️  Weak categories (<50% win on 4+ bets): {', '.join(weak_categories)}. "
                f"Consider DISABLED_CATEGORIES={','.join(weak_categories)} in .env."
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_bets": total,
        "correct": correct,
        "win_rate": round(win_rate, 4),
        "total_pnl_usdc": round(total_pnl, 4),
        "avg_pnl_per_bet": round(avg_pnl, 4),
        "avg_edge_detected": round(avg_edge, 4),
        "by_confidence": conf_stats,
        "by_edge_bucket": edge_stats,
        "by_category": category_stats,
        "recommendations": recommendations,
    }

    PERFORMANCE_PATH.parent.mkdir(exist_ok=True)
    PERFORMANCE_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty print
    w = 56

    def row(label, value):
        return f"║  {label:<22} {str(value):<{w - 27}}║"

    print("\n" + "╔" + "═" * w + "╗")
    print(f"║{'  POLYMARKET BOT — SIGNAL VALIDATION':^{w}}║")
    print("╠" + "═" * w + "╣")
    print(row("Total resolved bets", f"{total}"))
    print(row("Win rate", f"{win_rate*100:.1f}%  ({correct}/{total})"))
    print(row("Total hypothetical P&L", f"${total_pnl:+.2f}"))
    print(row("Avg P&L per bet", f"${avg_pnl:+.2f}"))
    print(row("Avg edge detected", f"{avg_edge:.3f}"))
    print("╠" + "═" * w + "╣")
    print(f"║{'  BY CONFIDENCE':^{w}}║")
    for level in ["high", "medium", "low"]:
        s = conf_stats[level]
        if s["count"] > 0:
            print(row(f"  {level}", f"{s['win_rate']*100:.1f}% win  ({s['count']} bets)  avg ${s['avg_pnl']:+.2f}"))
    print("╠" + "═" * w + "╣")
    print(f"║{'  BY EDGE BUCKET':^{w}}║")
    for bucket, s in edge_stats.items():
        if s["count"] > 0:
            print(row(f"  {bucket}", f"{s['win_rate']*100:.1f}% win  ({s['count']} bets)  avg ${s['avg_pnl']:+.2f}"))
    print("╠" + "═" * w + "╣")
    print(f"║{'  BY CATEGORY':^{w}}║")
    for category, s in category_stats.items():
        if s["count"] > 0:
            print(row(f"  {category}", f"{s['win_rate']*100:.1f}% win  ({s['count']} bets)  avg ${s['avg_pnl']:+.2f}"))
    print("╠" + "═" * w + "╣")
    print(f"║{'  RECOMMENDATIONS':^{w}}║")
    for rec in recommendations:
        # Word-wrap at w-4
        while len(rec) > w - 4:
            print(f"║  {rec[:w-4]:<{w-4}}║")
            rec = "    " + rec[w-4:]
        print(f"║  {rec:<{w-4}}║")
    print("╚" + "═" * w + "╝\n")

    return report

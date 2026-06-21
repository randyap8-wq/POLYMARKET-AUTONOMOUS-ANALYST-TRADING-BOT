from __future__ import annotations

import json
from datetime import datetime, timezone

try:
    from .config import (
        CONFIDENCE_RANK,
        MIN_CONFIDENCE,
        MIN_EDGE,
        REPORT_PATH,
        MIN_PRICE,
        MAX_PRICE,
    )
except ImportError:  # pragma: no cover
    from config import (
        CONFIDENCE_RANK,
        MIN_CONFIDENCE,
        MIN_EDGE,
        REPORT_PATH,
        MIN_PRICE,
        MAX_PRICE,
    )


def _kelly_fraction(price: float, fair_value: float) -> float:
    """Half-Kelly stake fraction for a $1-payout binary share.

    ``price`` is the entry cost per share and ``fair_value`` is the model's
    estimated win probability. Returns a capped, conservative fraction used as
    an informational sizing hint (actual stakes are governed elsewhere).
    """
    p = max(min(price, 0.999), 0.001)
    q = max(min(fair_value, 1.0), 0.0)
    b = (1.0 - p) / p  # net odds received on a win
    if b <= 0:
        return 0.0
    full_kelly = q - (1.0 - q) / b
    return round(max(0.0, min(full_kelly * 0.5, 1.0)), 4)


def build_report(all_markets, scored_results, token_usage: dict | None = None) -> dict:
    opportunities = [
        result
        for result in scored_results
        if result["edge"] >= MIN_EDGE
        and CONFIDENCE_RANK[result["confidence"]] >= CONFIDENCE_RANK[MIN_CONFIDENCE]
        and result["news_supports_bet"]
        and result["recommended_outcome"] is not None
        and MIN_PRICE <= float(result.get("current_price") or 0.0) <= MAX_PRICE
    ]
    opportunities.sort(key=lambda item: item["edge"], reverse=True)

    report_items = []
    for index, item in enumerate(opportunities, start=1):
        report_items.append(
            {
                "rank": index,
                "question": item["question"],
                "url": item["url"],
                "category": item.get("category", "other"),
                "recommended_outcome": item["recommended_outcome"],
                "current_price": item["current_price"],
                "fair_value_estimate": item["fair_value_estimate"],
                "edge": item["edge"],
                "confidence": item["confidence"],
                "kelly_fraction": _kelly_fraction(
                    float(item.get("current_price") or 0.0),
                    float(item.get("fair_value_estimate") or 0.0),
                ),
                "counter_evidence_considered": bool(item.get("counter_evidence_considered", False)),
                "reasoning": item["reasoning"],
                "top_news": [
                    {
                        "title": news_item.get("title", ""),
                        "url": news_item.get("url", ""),
                        "stance": news_item.get("stance", "supporting"),
                        "published_date": news_item.get("published_date", news_item.get("date", "")),
                    }
                    for news_item in item.get("top_news", [])[:3]
                ],
                "volume": item["volume"],
                "end_date": item["end_date"],
                "condition_id": item["condition_id"],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "markets_scanned": len(all_markets),
        "markets_scored": len(scored_results),
        "opportunities_found": len(report_items),
        "token_usage": token_usage or {},
        "opportunities": report_items,
    }


def save_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_summary(report: dict) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 40)
    print(f" POLYMARKET BOT — {timestamp}")
    print("=" * 40)
    print(f" Scanned : {report['markets_scanned']} markets")
    print(f" Scored  : {report['markets_scored']} markets")
    print(f" Picks   : {report['opportunities_found']} opportunities")
    print()
    print(" RANK  EDGE    CONF    OUTCOME   QUESTION")
    print(" ----  ------  ------  --------  ---------------------------------")
    for item in report["opportunities"]:
        print(
            f"  #{item['rank']:<1}   +{item['edge']:.2f}   {item['confidence']:<6}  "
            f"{item['recommended_outcome']:<8}  {item['question'][:33]}"
        )
    print()
    usage = report.get("token_usage") or {}
    if usage.get("total_tokens"):
        print(
            f" Tokens  : {usage.get('total_tokens', 0):,} "
            f"(~${usage.get('cost_usd', 0.0):.4f}, {usage.get('calls', 0)} calls)"
        )
    print(f" Full report saved to {REPORT_PATH.name}")
    print("=" * 40)

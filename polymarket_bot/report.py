from __future__ import annotations

import json
from datetime import datetime, timezone

try:
    from .config import CONFIDENCE_RANK, MIN_CONFIDENCE, MIN_EDGE, REPORT_PATH
except ImportError:  # pragma: no cover
    from config import CONFIDENCE_RANK, MIN_CONFIDENCE, MIN_EDGE, REPORT_PATH


def build_report(all_markets, scored_results) -> dict:
    opportunities = [
        result
        for result in scored_results
        if result["edge"] >= MIN_EDGE
        and CONFIDENCE_RANK[result["confidence"]] >= CONFIDENCE_RANK[MIN_CONFIDENCE]
        and result["news_supports_bet"]
        and result["recommended_outcome"] is not None
    ]
    opportunities.sort(key=lambda item: item["edge"], reverse=True)

    report_items = []
    for index, item in enumerate(opportunities, start=1):
        report_items.append(
            {
                "rank": index,
                "question": item["question"],
                "url": item["url"],
                "recommended_outcome": item["recommended_outcome"],
                "current_price": item["current_price"],
                "fair_value_estimate": item["fair_value_estimate"],
                "edge": item["edge"],
                "confidence": item["confidence"],
                "reasoning": item["reasoning"],
                "top_news": [
                    {
                        "title": news_item.get("title", ""),
                        "url": news_item.get("url", ""),
                        "date": news_item.get("published_date", news_item.get("date", "")),
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
    print(f" Full report saved to {REPORT_PATH.name}")
    print("=" * 40)

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Polymarket Bot")
    parser.add_argument("--analyze", action="store_true", help="Scan markets and produce report only")
    parser.add_argument("--trade", action="store_true", help="Scan markets and place bets on top picks")
    parser.add_argument("--loop", type=int, default=0, help="Repeat every N minutes (0 = run once)")
    args = parser.parse_args()

    if not args.analyze and not args.trade:
        parser.print_help()
        sys.exit(1)

    try:
        from .fetcher import fetch_markets
        from .news import fetch_news
        from .report import build_report, print_summary, save_report
        from .scorer import score_market
    except ImportError:  # pragma: no cover
        from fetcher import fetch_markets
        from news import fetch_news
        from report import build_report, print_summary, save_report
        from scorer import score_market

    def run():
        markets = fetch_markets()
        print(f"[fetcher] {len(markets)} markets after filtering")

        scored = []
        for market in markets:
            try:
                news = fetch_news(market["question"])
                score = score_market(market, news)
                if score:
                    scored.append({**market, **score, "top_news": news[:3]})
            except Exception as exc:  # pragma: no cover - integration path
                print(f"[error] {market['question'][:60]}: {exc}", file=sys.stderr)

        report = build_report(markets, scored)
        save_report(report)
        print_summary(report)

        if args.trade:
            try:
                from .trader import place_bet
            except ImportError:  # pragma: no cover
                from trader import place_bet
            for opportunity in report["opportunities"][:3]:
                result = place_bet(opportunity, opportunity)
                if result:
                    print(
                        f"[trade] {opportunity['recommended_outcome']} on "
                        f"'{opportunity['question'][:50]}' → {result['status']}"
                    )

    if args.loop > 0:
        import schedule
        import time

        schedule.every(args.loop).minutes.do(run)
        run()
        while True:
            schedule.run_pending()
            time.sleep(10)
    else:
        run()


if __name__ == "__main__":
    main()

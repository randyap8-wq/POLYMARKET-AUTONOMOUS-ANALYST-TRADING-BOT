from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Polymarket Bot")
    parser.add_argument("--analyze",   action="store_true", help="Scan markets and produce report only")
    parser.add_argument("--trade",     action="store_true", help="Scan markets and place real/dry-run bets")
    parser.add_argument("--paper",     action="store_true", help="Record hypothetical bets, no real trades")
    parser.add_argument("--validate",  action="store_true", help="Print signal validation report and exit")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard (requires fastapi + uvicorn)")
    parser.add_argument("--loop",      type=int, default=0, help="Repeat every N minutes (0 = run once)")
    parser.add_argument("--port",      type=int, default=8080, help="Dashboard port (default 8080)")
    args = parser.parse_args()

    # --validate: just print stats and exit
    if args.validate:
        try:
            from .validator import generate_performance_report
        except ImportError:
            from validator import generate_performance_report
        generate_performance_report()
        sys.exit(0)

    # --dashboard: launch web UI and exit
    if args.dashboard:
        try:
            from .dashboard import run_dashboard
        except ImportError:
            from dashboard import run_dashboard
        run_dashboard(port=args.port)
        sys.exit(0)

    if not args.analyze and not args.trade and not args.paper:
        parser.print_help()
        sys.exit(1)

    try:
        from .fetcher       import fetch_markets
        from .news          import fetch_news
        from .report        import build_report, print_summary, save_report
        from .scorer        import score_market
        from .paper_trader  import check_resolutions, record_paper_bet
    except ImportError:
        from fetcher       import fetch_markets
        from news          import fetch_news
        from report        import build_report, print_summary, save_report
        from scorer        import score_market
        from paper_trader  import check_resolutions, record_paper_bet

    def run():
        # Always check if any previously recorded paper bets have now resolved
        newly_resolved = check_resolutions()
        if newly_resolved:
            print(f"[paper] {newly_resolved} bet(s) resolved since last run")

        markets = fetch_markets()
        print(f"[fetcher] {len(markets)} markets after filtering")

        scored = []
        for market in markets:
            try:
                news  = fetch_news(market["question"])
                score = score_market(market, news)
                if score:
                    scored.append({**market, **score, "top_news": news[:3]})
            except Exception as exc:
                print(f"[error] {market['question'][:60]}: {exc}", file=sys.stderr)

        report = build_report(markets, scored)
        save_report(report)
        print_summary(report)

        if args.trade:
            try:
                from .trader import place_bet
            except ImportError:
                from trader import place_bet
            for opportunity in report["opportunities"][:3]:
                result = place_bet(opportunity, opportunity)
                if result:
                    print(
                        f"[trade] {opportunity['recommended_outcome']} on "
                        f"'{opportunity['question'][:50]}' → {result['status']}"
                    )

        elif args.paper:
            for opportunity in report["opportunities"][:3]:
                record_paper_bet(opportunity, opportunity, opportunity.get("top_news", []))

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

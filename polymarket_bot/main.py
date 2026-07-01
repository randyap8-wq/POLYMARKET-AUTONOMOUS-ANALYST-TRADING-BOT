from __future__ import annotations

import argparse
import sys

try:
    from .config import DASHBOARD_HOST
except ImportError:
    from config import DASHBOARD_HOST


def main():
    parser = argparse.ArgumentParser(description="Polymarket Bot")
    parser.add_argument("--analyze",   action="store_true", help="Scan markets and produce report only")
    parser.add_argument("--trade",     action="store_true", help="Scan markets and place real/dry-run bets")
    parser.add_argument("--paper",     action="store_true", help="Record hypothetical bets, no real trades")
    parser.add_argument("--validate",  action="store_true", help="Print signal validation report and exit")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard (requires fastapi + uvicorn)")
    parser.add_argument("--backfill",  action="store_true", help="Score recently closed markets and bootstrap resolved.jsonl")
    parser.add_argument("--backfill-days", type=int, default=14, help="How many days back to look for closed markets (default 14)")
    parser.add_argument("--backtest",  action="store_true", help="Point-in-time quant backtest on resolved markets (no look-ahead bias)")
    parser.add_argument("--backtest-full", action="store_true", help="Full pipeline backtest with cached/proxy AI, quant fusion, and risk sizing")
    parser.add_argument("--backtest-full-live-ai", action="store_true", help="Allow live Gemini calls during --backtest-full when no cached score exists")
    parser.add_argument("--backtest-days", type=int, default=30, help="How many days back to look for resolved markets (default 30)")
    parser.add_argument("--backtest-limit", type=int, default=60, help="Resolved market fetch limit for backtests (default 60)")
    parser.add_argument("--train-quant", action="store_true", help="Train the optional ML quant model from historical rows and exit")
    parser.add_argument("--train-quant-path", type=str, default="", help="Optional JSON/JSONL/CSV training data path for --train-quant")
    parser.add_argument("--simulate-risk", action="store_true", help="Tune risk settings using data/full_backtest.json and exit")
    parser.add_argument("--loop",      type=int, default=0, help="Repeat every N minutes (0 = run once)")
    parser.add_argument("--host",      type=str, default=DASHBOARD_HOST, help="Dashboard bind host. Defaults to 127.0.0.1 (localhost) or the DASHBOARD_HOST env var. Pass --host 0.0.0.0 (or set DASHBOARD_HOST=0.0.0.0) for VPS/remote access, and put it behind a firewall/reverse proxy/auth.")
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

    # --backfill: bootstrap resolved.jsonl from historical closed markets and exit
    if args.backfill:
        try:
            from .backfill import run_backfill
        except ImportError:
            from backfill import run_backfill
        count = run_backfill(days_back=args.backfill_days)
        print(f"[backfill] wrote {count} synthetic resolved bets")
        print("[backfill] NOTE: Gemini uses current news — has look-ahead bias. Use --backtest for the bias-free quant signal, and --validate after.")
        sys.exit(0)

    # --backtest: point-in-time quant backtest and exit
    if args.backtest:
        try:
            from .backtest import run_backtest
        except ImportError:
            from backtest import run_backtest
        run_backtest(days_back=args.backtest_days, limit=args.backtest_limit)
        sys.exit(0)

    # --backtest-full: production path replay with cached/proxy AI and risk sizing
    if args.backtest_full:
        try:
            from .backtest_full import run_full_backtest
        except ImportError:
            from backtest_full import run_full_backtest
        run_full_backtest(days_back=args.backtest_days, limit=args.backtest_limit, allow_live_ai=args.backtest_full_live_ai)
        sys.exit(0)

    # --train-quant: fit the optional sklearn model and exit
    if args.train_quant:
        try:
            from .quant import train_quant_model
        except ImportError:
            from quant import train_quant_model
        summary = train_quant_model(args.train_quant_path or None)
        if summary.get("trained"):
            print(f"[quant] trained model on {summary['examples']} examples -> {summary['model_path']}")
        else:
            print(f"[quant] model not trained: {summary.get('reason', 'unknown reason')}")
        sys.exit(0)

    # --simulate-risk: parameter sweep over full-backtest decisions and exit
    if args.simulate_risk:
        try:
            from .simulate import run_simulation
        except ImportError:
            from simulate import run_simulation
        run_simulation()
        sys.exit(0)

    # --dashboard: launch web UI and exit
    if args.dashboard:
        try:
            from .dashboard import run_dashboard
        except ImportError:
            from dashboard import run_dashboard
        run_dashboard(host=args.host, port=args.port)
        sys.exit(0)

    if not args.analyze and not args.trade and not args.paper:
        parser.print_help()
        sys.exit(1)

    try:
        from .fetcher       import fetch_markets
        from .pipeline      import analyze_market
        from .report        import build_report, print_summary, save_report
        from .scorer        import reset_token_usage, get_token_usage
        from .paper_trader  import check_resolutions, record_paper_bet
    except ImportError:
        from fetcher       import fetch_markets
        from pipeline      import analyze_market
        from report        import build_report, print_summary, save_report
        from scorer        import reset_token_usage, get_token_usage
        from paper_trader  import check_resolutions, record_paper_bet

    def run():
        # Always check if any previously recorded paper bets have now resolved
        newly_resolved = check_resolutions()
        if newly_resolved:
            print(f"[paper] {newly_resolved} bet(s) resolved since last run")

        reset_token_usage()
        markets = fetch_markets()
        print(f"[fetcher] {len(markets)} markets after filtering")

        scored = []
        skipped = 0
        for market in markets:
            try:
                # Quant prefilter -> Gemini (search grounding) -> AI/quant fusion.
                score = analyze_market(market)
                if score:
                    headlines = score.get("news_headlines") or []
                    top_news = [{"title": h, "url": "", "date": ""} for h in headlines[:3]]
                    scored.append({**market, **score, "top_news": top_news})
                else:
                    skipped += 1
            except Exception as exc:
                print(f"[error] {market['question'][:60]}: {exc}", file=sys.stderr)

        if skipped:
            print(f"[pipeline] {skipped} markets skipped by quant prefilter / scoring")

        report = build_report(markets, scored, token_usage=get_token_usage())
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
            for opportunity in report["opportunities"][:5]:
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

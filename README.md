# POLYMARKET-AUTONOMOUS-ANALYST-TRADING-BOT

POLYMARKET AUTONOMOUS ANALYST & TRADING BOT.

The bot scans Polymarket prediction markets, pulls fresh news for each market,
asks DeepSeek to estimate fair value, and surfaces mispriced opportunities. It can
run in analyze-only mode, place real (or dry-run) trades, record hypothetical
**paper bets** to validate the signal before risking capital, and serve a live
**web dashboard**.

## Project layout

The implementation lives under `polymarket_bot/`.

| File | Purpose |
| --- | --- |
| `config.py` | Environment, paths, and tunable filters/thresholds. |
| `fetcher.py` | Fetches and filters active markets from the Gamma API (paginated, up to 150); tags each with a `market_category` and drops disabled categories. |
| `categories.py` | Keyword-based market categorisation used for breakdowns and filtering. |
| `news.py` | Tavily news search per market with a dynamic time window plus a counter-evidence query. |
| `scorer.py` | DeepSeek fair-value scoring with a reasoner second opinion and per-run token-cost tracking. |
| `report.py` | Builds, saves, and prints the opportunities report (`report.json`). |
| `trader.py` | Places real/dry-run orders via the CLOB client. |
| `wallet.py` | Builds the authenticated CLOB client. |
| `paper_trader.py` | Records hypothetical bets and resolves them against the API. |
| `validator.py` | Computes win rate / P&L / calibration and a go-live recommendation. |
| `dashboard.py` | Self-contained FastAPI web dashboard reading local JSON/JSONL. |
| `main.py` | CLI entrypoint wiring all modes together. |

Generated data is written to `polymarket_bot/data/` (paper bets, resolved bets,
performance summary) and `polymarket_bot/logs/trades.jsonl` (live trades). These
files are git-ignored.

## Quick start

```bash
cd polymarket_bot
cp .env.example .env        # add your API keys
pip install -r requirements.txt
python main.py --analyze
```

### Environment variables

Set these in `polymarket_bot/.env`:

- `DEEPSEEK_API_KEY` — required for market scoring.
- `TAVILY_API_KEY` — required for news search.
- `POLYGON_PRIVATE_KEY` / `POLYGON_WALLET_ADDRESS` — required for live trading.
- `MAX_BET_USDC` (default `10`), `MIN_EDGE` (default `0.07`),
  `MIN_CONFIDENCE` (default `medium`), `DRY_RUN` (default `true`).
- Signal tuning (all optional): `NEWS_WINDOW_MIN_DAYS` / `NEWS_WINDOW_MAX_DAYS`
  (dynamic news window), `COUNTER_EVIDENCE` (counter-evidence query, default `true`),
  `MIN_PRICE` / `MAX_PRICE` (extreme-price guard), and `DISABLED_CATEGORIES`
  (comma-separated categories to skip).

## CLI modes

```bash
# Analyze only — scan markets and write report.json
python main.py --analyze

# Paper trading — record hypothetical bets (no real USDC), no trades placed.
# Pending paper bets are auto-checked for resolution on every run.
python main.py --paper

# Live/dry-run trading — place orders on the top opportunities
python main.py --trade

# Signal validation — print win rate, P&L, calibration, and a recommendation
python main.py --validate

# Web dashboard — browse opportunities, paper bets, resolved bets, and trades
python main.py --dashboard            # http://localhost:8080
python main.py --dashboard --port 9000

# Repeat any scanning mode every N minutes
python main.py --paper --loop 60
```

`--analyze`, `--trade`, and `--paper` all perform a market scan. `--validate` and
`--dashboard` only read previously generated data and exit / serve.

## Recommended workflow

```bash
# Install all deps (existing + dashboard)
pip install -r requirements.txt

# --- PHASE 1: Paper trading (2-3 weeks) ---
# Scan every 60 minutes, record paper bets, auto-check resolutions
python main.py --paper --loop 60

# Check signal quality anytime
python main.py --validate

# Launch the dashboard in a separate terminal
python main.py --dashboard
# Open http://localhost:8080

# --- PHASE 2: Go live (after --validate shows >60% win rate on 30+ bets) ---
# Set DRY_RUN=false and a small MAX_BET_USDC (e.g. 2) in .env first
python main.py --trade --loop 60
```

## Dashboard

The dashboard is a single FastAPI server with an inline HTML/JS front end — no
database required. It reads `report.json`, `data/paper_bets.jsonl`,
`data/resolved.jsonl`, `data/performance.json`, and `logs/trades.jsonl`, and
auto-refreshes every 60 seconds. Pages:

- **Overview** — high-level stats and the latest top opportunities.
- **Opportunities** — every market flagged as mispriced in the latest scan.
- **Paper Bets** — pending hypothetical bets.
- **Resolved** — closed markets scored against the bot's predictions.
- **Validation** — calibration by confidence and edge bucket, plus recommendations.
- **Live Trades** — real/dry-run orders from the trade log.

Requires `fastapi` and `uvicorn` (included in `requirements.txt`).

## How paper trading validates the signal

1. In `--paper` mode the bot records a fixed $10 hypothetical bet for each top
   opportunity to `data/paper_bets.jsonl`.
2. On every run it queries Polymarket for any pending bet whose market has now
   closed, computes P&L, and appends the outcome to `data/resolved.jsonl`.
3. `--validate` aggregates resolved bets into `data/performance.json`: overall
   win rate, P&L, calibration by confidence/edge bucket, and a go-live
   recommendation. Wait for 30+ resolved bets before drawing conclusions.

## Signal improvements (implemented)

These signal-quality features are built into the pipeline and tuned via the
environment variables above:

- **Dynamic news window** — the Tavily search window widens for slower-moving
  markets (those closing further out) and stays tight for fast-moving ones
  (`NEWS_WINDOW_MIN_DAYS` / `NEWS_WINDOW_MAX_DAYS`).
- **Counter-evidence query** — a second, negation-focused Tavily query per market
  feeds the scorer so it weighs reasons the favoured outcome may *not* happen,
  reducing overconfidence (`COUNTER_EVIDENCE`).
- **DeepSeek token-cost tracking** — every scan records prompt/completion tokens
  and an estimated USD cost (shown in the summary, `report.json`, and dashboard).
- **Market categories** — each market is tagged with a `category`; weak
  categories can be disabled via `DISABLED_CATEGORIES`, and `--validate` reports
  win rate per category.
- **Extreme-price guard** — opportunities at very low/high prices (noisy edge,
  poor risk/reward) are filtered out (`MIN_PRICE` / `MAX_PRICE`).
- **Kelly sizing hint** — each opportunity includes a conservative half-Kelly
  stake fraction as an informational sizing guide.

## Tests

```bash
pip install pytest
python -m pytest
```

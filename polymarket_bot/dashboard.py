from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import BASE_DIR
except ImportError:
    from config import BASE_DIR

LOGGER = logging.getLogger("dashboard")

DATA_DIR   = BASE_DIR / "data"
REPORT_PATH = BASE_DIR / "report.json"
PAPER_BETS_PATH = DATA_DIR / "paper_bets.jsonl"
RESOLVED_PATH   = DATA_DIR / "resolved.jsonl"
PERFORMANCE_PATH = DATA_DIR / "performance.json"
TRADES_LOG_PATH  = BASE_DIR / "logs" / "trades.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymarket Bot</title>
<style>
  :root {
    --bg: #0a0e17;
    --surface: #111827;
    --surface2: #1a2236;
    --border: #1f2d45;
    --accent: #00d4b8;
    --accent2: #7c3aed;
    --win: #10b981;
    --loss: #ef4444;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --font: 'Inter', system-ui, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; min-height: 100vh; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Layout */
  .shell { display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px 16px;
    display: flex; flex-direction: column; gap: 4px;
    position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }
  .sidebar-logo {
    font-size: 13px; font-weight: 700; letter-spacing: .08em;
    color: var(--accent); text-transform: uppercase;
    padding: 0 8px 20px; border-bottom: 1px solid var(--border); margin-bottom: 12px;
  }
  .nav-item {
    padding: 8px 12px; border-radius: 6px; cursor: pointer;
    color: var(--muted); font-weight: 500; font-size: 13px;
    transition: background .15s, color .15s;
    display: flex; align-items: center; gap: 8px;
  }
  .nav-item:hover { background: var(--surface2); color: var(--text); }
  .nav-item.active { background: var(--surface2); color: var(--accent); }
  .nav-icon { font-size: 15px; width: 18px; text-align: center; }

  .main { padding: 28px 32px; overflow-y: auto; }
  .page { display: none; }
  .page.active { display: block; }

  /* Header */
  .page-header { margin-bottom: 24px; }
  .page-header h1 { font-size: 22px; font-weight: 700; color: var(--text); }
  .page-header p { color: var(--muted); font-size: 13px; margin-top: 4px; }

  /* Stat cards */
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; margin-bottom: 28px; }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px;
  }
  .stat-label { color: var(--muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
  .stat-value { font-size: 26px; font-weight: 700; margin-top: 6px; font-family: var(--mono); }
  .stat-value.green { color: var(--win); }
  .stat-value.red { color: var(--loss); }
  .stat-value.accent { color: var(--accent); }
  .stat-sub { font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* Tables */
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 24px; }
  .card-header { padding: 14px 18px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; color: var(--text); display: flex; align-items: center; justify-content: space-between; }
  .card-header .badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 2px 10px; font-size: 11px; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; }
  th { padding: 10px 14px; text-align: left; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); background: var(--surface2); }
  td { padding: 11px 14px; font-size: 13px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,.02); }

  /* Pills / badges */
  .pill { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .pill-high   { background: rgba(16,185,129,.15); color: var(--win); }
  .pill-medium { background: rgba(245,158,11,.15); color: var(--warn); }
  .pill-low    { background: rgba(100,116,139,.15); color: var(--muted); }
  .pill-win    { background: rgba(16,185,129,.15); color: var(--win); }
  .pill-loss   { background: rgba(239,68,68,.15);  color: var(--loss); }
  .pill-pending{ background: rgba(124,58,237,.15); color: #a78bfa; }

  /* Edge bar */
  .edge-bar { display: flex; align-items: center; gap: 8px; }
  .edge-track { flex: 1; height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  .edge-fill  { height: 100%; background: var(--accent); border-radius: 2px; }

  /* Recommendations */
  .recs { display: flex; flex-direction: column; gap: 8px; padding: 16px 18px; }
  .rec { padding: 10px 14px; border-radius: 8px; font-size: 13px; border-left: 3px solid; }
  .rec-ok   { background: rgba(16,185,129,.08);  border-color: var(--win);  color: #a7f3d0; }
  .rec-warn { background: rgba(245,158,11,.08);  border-color: var(--warn); color: #fde68a; }
  .rec-bad  { background: rgba(239,68,68,.08);   border-color: var(--loss); color: #fca5a5; }
  .rec-info { background: rgba(100,116,139,.08); border-color: var(--muted); color: var(--muted); }

  /* Calibration grid */
  .cal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 16px 18px; }
  .cal-section h4 { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }
  .cal-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid var(--border); }
  .cal-row:last-child { border-bottom: none; }
  .cal-label { color: var(--muted); font-size: 12px; width: 80px; flex-shrink: 0; }
  .cal-bar-wrap { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .cal-bar { height: 100%; border-radius: 3px; }
  .cal-val { font-size: 12px; font-family: var(--mono); width: 40px; text-align: right; flex-shrink: 0; }

  /* Question cell truncation */
  .q-cell { max-width: 320px; }
  .q-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
  .q-sub  { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* Refresh btn */
  .refresh-btn {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text); cursor: pointer; font-size: 12px;
    transition: background .15s;
  }
  .refresh-btn:hover { background: var(--border); }

  /* Empty state */
  .empty { padding: 48px; text-align: center; color: var(--muted); font-size: 13px; }
  .empty-icon { font-size: 36px; margin-bottom: 12px; }

  /* Last updated */
  .last-updated { font-size: 11px; color: var(--muted); margin-bottom: 20px; }

  @media (max-width: 768px) {
    .shell { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    .main { padding: 16px; }
    .cal-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="shell">
  <nav class="sidebar">
    <div class="sidebar-logo">⬡ Polymarket Bot</div>
    <div class="nav-item active" data-page="overview" onclick="nav(this)"><span class="nav-icon">◈</span> Overview</div>
    <div class="nav-item" data-page="opportunities" onclick="nav(this)"><span class="nav-icon">◎</span> Opportunities</div>
    <div class="nav-item" data-page="paper" onclick="nav(this)"><span class="nav-icon">◷</span> Paper Bets</div>
    <div class="nav-item" data-page="resolved" onclick="nav(this)"><span class="nav-icon">◉</span> Resolved</div>
    <div class="nav-item" data-page="validation" onclick="nav(this)"><span class="nav-icon">◆</span> Validation</div>
    <div class="nav-item" data-page="trades" onclick="nav(this)"><span class="nav-icon">◀</span> Live Trades</div>
  </nav>

  <main class="main">

    <!-- OVERVIEW -->
    <div class="page active" id="page-overview">
      <div class="page-header">
        <h1>Overview</h1>
        <p>Live snapshot of bot activity across all modes.</p>
      </div>
      <div class="last-updated" id="overview-updated"></div>
      <div class="stat-grid" id="overview-stats"></div>
      <div class="card">
        <div class="card-header">Top Opportunities This Run</div>
        <div id="overview-opps"></div>
      </div>
    </div>

    <!-- OPPORTUNITIES -->
    <div class="page" id="page-opportunities">
      <div class="page-header">
        <h1>Current Opportunities</h1>
        <p>Markets the bot flagged as potentially mispriced in the latest scan.</p>
      </div>
      <div class="last-updated" id="opps-updated"></div>
      <div class="card">
        <div class="card-header">
          Flagged Markets
          <button class="refresh-btn" onclick="loadAll()">↻ Refresh</button>
        </div>
        <div id="opps-table"></div>
      </div>
    </div>

    <!-- PAPER BETS -->
    <div class="page" id="page-paper">
      <div class="page-header">
        <h1>Paper Bets</h1>
        <p>Hypothetical bets recorded in --paper mode. No real USDC spent.</p>
      </div>
      <div class="stat-grid" id="paper-stats"></div>
      <div class="card">
        <div class="card-header">Pending Bets <span class="badge" id="pending-count">0</span></div>
        <div id="paper-table"></div>
      </div>
    </div>

    <!-- RESOLVED -->
    <div class="page" id="page-resolved">
      <div class="page-header">
        <h1>Resolved Bets</h1>
        <p>Markets that closed and were scored against the bot's predictions.</p>
      </div>
      <div class="stat-grid" id="resolved-stats"></div>
      <div class="card">
        <div class="card-header">Resolved History <span class="badge" id="resolved-count">0</span></div>
        <div id="resolved-table"></div>
      </div>
    </div>

    <!-- VALIDATION -->
    <div class="page" id="page-validation">
      <div class="page-header">
        <h1>Signal Validation</h1>
        <p>Is the bot's edge real? Calibration by confidence and edge bucket.</p>
      </div>
      <div class="stat-grid" id="val-stats"></div>
      <div class="card">
        <div class="card-header">Recommendations</div>
        <div class="recs" id="val-recs"></div>
      </div>
      <div class="card">
        <div class="card-header">Calibration</div>
        <div class="cal-grid" id="val-calibration"></div>
      </div>
    </div>

    <!-- LIVE TRADES -->
    <div class="page" id="page-trades">
      <div class="page-header">
        <h1>Live Trades</h1>
        <p>Real orders placed via the CLOB API (dry-run or live).</p>
      </div>
      <div class="card">
        <div class="card-header">Trade Log</div>
        <div id="trades-table"></div>
      </div>
    </div>

  </main>
</div>

<script>
let DATA = {};

function nav(el) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('page-' + el.dataset.page).classList.add('active');
}

async function loadAll() {
  try {
    const r = await fetch('/api/all');
    DATA = await r.json();
    renderOverview();
    renderOpportunities();
    renderPaper();
    renderResolved();
    renderValidation();
    renderTrades();
  } catch(e) {
    console.error('Failed to load data', e);
  }
}

function fmt_pnl(v) {
  if (v == null) return '—';
  const s = v >= 0 ? `+$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`;
  return `<span style="color:${v >= 0 ? 'var(--win)' : 'var(--loss)'}">${s}</span>`;
}
function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function safeUrl(u){ if (!u) return '#'; try { const x = new URL(u, location.origin); return (x.protocol==='http:'||x.protocol==='https:') ? x.href : '#'; } catch(e){ return '#'; } }
function fmt_pct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—'; }
function fmt_price(v) { return v != null ? '$' + (+v).toFixed(2) : '—'; }
function pill_conf(c) {
  const level = String(c || 'low');
  const cls = ['high','medium','low'].includes(level) ? level : 'low';
  return `<span class="pill pill-${cls}">${esc(level)}</span>`;
}
function pill_status(s, correct) {
  if (s === 'pending') return `<span class="pill pill-pending">pending</span>`;
  return correct ? `<span class="pill pill-win">✓ correct</span>` : `<span class="pill pill-loss">✗ wrong</span>`;
}
function edge_bar(e) {
  const pct = Math.min((e||0) * 400, 100);
  return `<div class="edge-bar"><div class="edge-track"><div class="edge-fill" style="width:${pct}%"></div></div><span style="font-family:var(--mono);font-size:12px;color:var(--accent)">${(+(e||0)).toFixed(2)}</span></div>`;
}
function q_cell(q, url, sub) {
  return `<div class="q-cell"><div class="q-text"><a href="${safeUrl(url)}" target="_blank" rel="noopener noreferrer">${esc(q)}</a></div>${sub ? `<div class="q-sub">${esc(sub)}</div>` : ''}</div>`;
}
function stat_card(label, value, cls, sub) {
  return `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value ${cls||''}">${value}</div>${sub ? `<div class="stat-sub">${sub}</div>` : ''}</div>`;
}
function empty_state(msg) {
  return `<div class="empty"><div class="empty-icon">◌</div>${msg}</div>`;
}

function renderOverview() {
  const r = DATA.report || {};
  const perf = DATA.performance || {};
  const ts = r.generated_at ? new Date(r.generated_at).toLocaleString() : '—';
  document.getElementById('overview-updated').textContent = `Last scan: ${ts}`;

  const wr = perf.win_rate != null ? (perf.win_rate * 100).toFixed(1) + '%' : '—';
  const pnl = perf.total_pnl_usdc != null ? `$${perf.total_pnl_usdc > 0 ? '+' : ''}${(+perf.total_pnl_usdc).toFixed(2)}` : '—';
  const pnl_cls = (perf.total_pnl_usdc || 0) >= 0 ? 'green' : 'red';
  const wr_cls = (perf.win_rate || 0) >= 0.55 ? 'green' : (perf.win_rate || 0) >= 0.5 ? 'accent' : 'red';
  const usage = r.token_usage || {};
  const cost = usage.cost_usd != null ? `$${(+usage.cost_usd).toFixed(4)}` : '—';

  document.getElementById('overview-stats').innerHTML = [
    stat_card('Markets Scanned', r.markets_scanned || 0, 'accent'),
    stat_card('Opportunities', r.opportunities_found || 0, 'accent'),
    stat_card('Paper Bets', DATA.paper_pending_count || 0, ''),
    stat_card('Resolved Bets', DATA.resolved_count || 0, ''),
    stat_card('Win Rate', wr, wr_cls, `${perf.total_bets || 0} resolved`),
    stat_card('Paper P&L', pnl, pnl_cls, 'hypothetical'),
    stat_card('LLM Cost / Scan', cost, '', `${usage.total_tokens || 0} tokens`),
  ].join('');

  const opps = (r.opportunities || []).slice(0, 5);
  if (!opps.length) {
    document.getElementById('overview-opps').innerHTML = empty_state('No opportunities in latest scan.');
    return;
  }
  document.getElementById('overview-opps').innerHTML = `<table>
    <tr><th>Market</th><th>Outcome</th><th>Edge</th><th>Confidence</th><th>Price</th></tr>
    ${opps.map(o => `<tr>
      <td>${q_cell(o.question, o.url, o.end_date ? 'Closes ' + o.end_date.slice(0,10) : '')}</td>
      <td><b>${esc(o.recommended_outcome)}</b></td>
      <td>${edge_bar(o.edge)}</td>
      <td>${pill_conf(o.confidence)}</td>
      <td style="font-family:var(--mono)">${fmt_price(o.current_price)} → ${fmt_price(o.fair_value_estimate)}</td>
    </tr>`).join('')}
  </table>`;
}

function renderOpportunities() {
  const opps = (DATA.report || {}).opportunities || [];
  const ts = (DATA.report || {}).generated_at;
  if (ts) document.getElementById('opps-updated').textContent = `Last scan: ${new Date(ts).toLocaleString()}`;

  if (!opps.length) {
    document.getElementById('opps-table').innerHTML = empty_state('No opportunities found in latest scan. Bot may still be running.');
    return;
  }
  document.getElementById('opps-table').innerHTML = `<table>
    <tr><th>#</th><th>Market</th><th>Category</th><th>Outcome</th><th>Edge</th><th>Confidence</th><th>Current Price</th><th>Fair Value</th><th>Volume</th></tr>
    ${opps.map(o => `<tr>
      <td style="color:var(--muted);font-family:var(--mono)">#${esc(o.rank)}</td>
      <td>${q_cell(o.question, o.url, o.reasoning ? o.reasoning.slice(0,80)+'…' : '')}</td>
      <td style="color:var(--muted)">${esc(o.category || 'other')}</td>
      <td><b>${esc(o.recommended_outcome)}</b></td>
      <td>${edge_bar(o.edge)}</td>
      <td>${pill_conf(o.confidence)}</td>
      <td style="font-family:var(--mono)">${fmt_price(o.current_price)}</td>
      <td style="font-family:var(--mono);color:var(--accent)">${fmt_price(o.fair_value_estimate)}</td>
      <td style="color:var(--muted);font-family:var(--mono)">$${((o.volume||0)/1000).toFixed(0)}k</td>
    </tr>`).join('')}
  </table>`;
}

function renderPaper() {
  const bets = DATA.paper_bets || [];
  const pending = bets.filter(b => b.status === 'pending');
  document.getElementById('pending-count').textContent = pending.length;

  const total_stake = pending.reduce((s, b) => s + (b.hypothetical_usdc || 0), 0);
  const avg_edge = pending.length ? pending.reduce((s, b) => s + (b.edge || 0), 0) / pending.length : 0;

  document.getElementById('paper-stats').innerHTML = [
    stat_card('Pending Bets', pending.length, 'accent'),
    stat_card('Total Stake (Paper)', `$${total_stake.toFixed(0)}`, '', 'hypothetical USDC'),
    stat_card('Avg Edge', avg_edge.toFixed(3), 'accent'),
  ].join('');

  if (!pending.length) {
    document.getElementById('paper-table').innerHTML = empty_state('No pending paper bets. Run the bot with --paper to start recording.');
    return;
  }
  document.getElementById('paper-table').innerHTML = `<table>
    <tr><th>Recorded</th><th>Market</th><th>Outcome</th><th>Edge</th><th>Confidence</th><th>Price</th><th>Stake</th></tr>
    ${pending.map(b => `<tr>
      <td style="color:var(--muted);font-size:11px;white-space:nowrap">${esc(b.recorded_at ? b.recorded_at.slice(0,16).replace('T',' ') : '—')}</td>
      <td>${q_cell(b.question, b.url, 'Closes ' + (b.end_date||'').slice(0,10))}</td>
      <td><b>${esc(b.recommended_outcome)}</b></td>
      <td>${edge_bar(b.edge)}</td>
      <td>${pill_conf(b.confidence)}</td>
      <td style="font-family:var(--mono)">${fmt_price(b.current_price)}</td>
      <td style="font-family:var(--mono)">$${(b.hypothetical_usdc||10).toFixed(0)}</td>
    </tr>`).join('')}
  </table>`;
}

function renderResolved() {
  const res = DATA.resolved_bets || [];
  document.getElementById('resolved-count').textContent = res.length;

  const wins = res.filter(b => b.correct).length;
  const total_pnl = res.reduce((s, b) => s + (b.pnl_usdc || 0), 0);
  const wr = res.length ? wins / res.length : 0;

  document.getElementById('resolved-stats').innerHTML = [
    stat_card('Total Resolved', res.length, 'accent'),
    stat_card('Correct', wins, 'green'),
    stat_card('Win Rate', fmt_pct(wr), wr >= 0.55 ? 'green' : wr >= 0.5 ? 'accent' : 'red'),
    stat_card('Total P&L', `${total_pnl >= 0 ? '+' : ''}$${total_pnl.toFixed(2)}`, total_pnl >= 0 ? 'green' : 'red', 'hypothetical'),
  ].join('');

  if (!res.length) {
    document.getElementById('resolved-table').innerHTML = empty_state('No resolved bets yet. Markets take time to close — check back in a few days.');
    return;
  }
  const sorted = [...res].sort((a, b) => new Date(b.resolved_at||0) - new Date(a.resolved_at||0));
  document.getElementById('resolved-table').innerHTML = `<table>
    <tr><th>Resolved</th><th>Market</th><th>Predicted</th><th>Actual</th><th>Result</th><th>P&L</th><th>Edge</th></tr>
    ${sorted.map(b => `<tr>
      <td style="color:var(--muted);font-size:11px;white-space:nowrap">${esc((b.resolved_at||'').slice(0,10))}</td>
      <td>${q_cell(b.question, b.url)}</td>
      <td style="font-family:var(--mono)">${esc(b.recommended_outcome || '—')}</td>
      <td style="font-family:var(--mono)">${esc(b.resolved_outcome || '—')}</td>
      <td>${pill_status(b.status, b.correct)}</td>
      <td style="font-family:var(--mono)">${fmt_pnl(b.pnl_usdc)}</td>
      <td style="font-family:var(--mono);color:var(--accent)">${(+(b.edge||0)).toFixed(2)}</td>
    </tr>`).join('')}
  </table>`;
}

function renderValidation() {
  const perf = DATA.performance || {};
  if (!perf.total_bets) {
    document.getElementById('val-stats').innerHTML = stat_card('Status', 'No data yet', 'accent', 'Run --paper mode first');
    document.getElementById('val-recs').innerHTML = '<p style="color:var(--muted);padding:16px">Validation data will appear here once markets resolve.</p>';
    document.getElementById('val-calibration').innerHTML = '';
    return;
  }

  const wr = perf.win_rate || 0;
  document.getElementById('val-stats').innerHTML = [
    stat_card('Total Bets', perf.total_bets, 'accent'),
    stat_card('Win Rate', fmt_pct(wr), wr >= 0.58 ? 'green' : wr >= 0.52 ? 'accent' : 'red'),
    stat_card('Total P&L', `${(perf.total_pnl_usdc||0) >= 0 ? '+' : ''}$${(+(perf.total_pnl_usdc||0)).toFixed(2)}`, (perf.total_pnl_usdc||0) >= 0 ? 'green' : 'red', 'hypothetical'),
    stat_card('Avg P&L / Bet', `${(perf.avg_pnl_per_bet||0) >= 0 ? '+' : ''}$${(+(perf.avg_pnl_per_bet||0)).toFixed(2)}`, (perf.avg_pnl_per_bet||0) >= 0 ? 'green' : 'red'),
    stat_card('Avg Edge', (+(perf.avg_edge_detected||0)).toFixed(3), 'accent'),
  ].join('');

  const recs = perf.recommendations || [];
  document.getElementById('val-recs').innerHTML = recs.length
    ? recs.map(r => {
        const cls = r.startsWith('✅') ? 'rec-ok' : r.startsWith('❌') ? 'rec-bad' : r.startsWith('⚠') ? 'rec-warn' : 'rec-info';
        return `<div class="rec ${cls}">${esc(r)}</div>`;
      }).join('')
    : '<p style="color:var(--muted);padding:4px">No recommendations yet.</p>';

  const byConf = perf.by_confidence || {};
  const byEdge = perf.by_edge_bucket || {};
  const byCat = perf.by_category || {};
  const cal_bar = (wr, color) => `<div class="cal-bar-wrap"><div class="cal-bar" style="width:${Math.min(wr*100,100).toFixed(0)}%;background:${color}"></div></div>`;

  document.getElementById('val-calibration').innerHTML = `
    <div class="cal-section">
      <h4>By Confidence</h4>
      ${['high','medium','low'].map(level => {
        const s = byConf[level] || {};
        if (!s.count) return '';
        const color = s.win_rate >= 0.58 ? 'var(--win)' : s.win_rate >= 0.5 ? 'var(--warn)' : 'var(--loss)';
        return `<div class="cal-row">
          <span class="cal-label">${level}</span>
          ${cal_bar(s.win_rate||0, color)}
          <span class="cal-val" style="color:${color}">${((s.win_rate||0)*100).toFixed(0)}%</span>
        </div>`;
      }).join('')}
    </div>
    <div class="cal-section">
      <h4>By Edge Bucket</h4>
      ${Object.entries(byEdge).map(([bucket, s]) => {
        if (!s.count) return '';
        const color = s.win_rate >= 0.58 ? 'var(--win)' : s.win_rate >= 0.5 ? 'var(--warn)' : 'var(--loss)';
        return `<div class="cal-row">
          <span class="cal-label">${esc(bucket)}</span>
          ${cal_bar(s.win_rate||0, color)}
          <span class="cal-val" style="color:${color}">${((s.win_rate||0)*100).toFixed(0)}%</span>
        </div>`;
      }).join('')}
    </div>
    <div class="cal-section">
      <h4>By Category</h4>
      ${Object.entries(byCat).map(([category, s]) => {
        if (!s.count) return '';
        const color = s.win_rate >= 0.58 ? 'var(--win)' : s.win_rate >= 0.5 ? 'var(--warn)' : 'var(--loss)';
        return `<div class="cal-row">
          <span class="cal-label">${esc(category)}</span>
          ${cal_bar(s.win_rate||0, color)}
          <span class="cal-val" style="color:${color}">${((s.win_rate||0)*100).toFixed(0)}% (${s.count})</span>
        </div>`;
      }).join('')}
    </div>`;
}

function renderTrades() {
  const trades = DATA.trades || [];
  if (!trades.length) {
    document.getElementById('trades-table').innerHTML = empty_state('No trades logged yet. Trades appear when running --trade mode (live or dry-run).');
    return;
  }
  const sorted = [...trades].sort((a, b) => new Date(b.timestamp||0) - new Date(a.timestamp||0));
  document.getElementById('trades-table').innerHTML = `<table>
    <tr><th>Time</th><th>Market</th><th>Outcome</th><th>Price</th><th>USDC</th><th>Tokens</th><th>Status</th></tr>
    ${sorted.map(t => `<tr>
      <td style="color:var(--muted);font-size:11px;white-space:nowrap">${esc((t.timestamp||'').slice(0,16).replace('T',' '))}</td>
      <td>${q_cell(t.question)}</td>
      <td><b>${esc(t.outcome)}</b></td>
      <td style="font-family:var(--mono)">${fmt_price(t.price)}</td>
      <td style="font-family:var(--mono)">$${(+(t.usdc_spent||0)).toFixed(2)}</td>
      <td style="font-family:var(--mono)">${(+(t.tokens_bought||0)).toFixed(1)}</td>
      <td><span class="pill ${t.status==='dry_run'?'pill-pending':t.status==='filled'?'pill-win':'pill-loss'}">${esc(t.status || '—')}</span></td>
    </tr>`).join('')}
  </table>`;
}

loadAll();
setInterval(loadAll, 60000);  // auto-refresh every 60s
</script>
</body>
</html>
"""


def create_app():
    """Build and return the FastAPI application. Import FastAPI lazily so the
    rest of the bot works without it installed."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn") from exc

    app = FastAPI(title="Polymarket Bot Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/all")
    def api_all():
        report       = _read_json(REPORT_PATH)
        performance  = _read_json(PERFORMANCE_PATH)
        paper_bets   = _read_jsonl(PAPER_BETS_PATH)
        resolved     = _read_jsonl(RESOLVED_PATH)
        trades       = _read_jsonl(TRADES_LOG_PATH)

        pending = [b for b in paper_bets if b.get("status") == "pending"]

        return JSONResponse({
            "report": report,
            "performance": performance,
            "paper_bets": pending,
            "paper_pending_count": len(pending),
            "resolved_bets": resolved,
            "resolved_count": len(resolved),
            "trades": trades,
        })

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8080):
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install uvicorn: pip install uvicorn") from exc
    app = create_app()
    LOGGER.info("Dashboard running at http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")

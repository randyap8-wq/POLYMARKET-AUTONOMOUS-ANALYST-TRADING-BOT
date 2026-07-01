"""AI market scorer backed by Gemini with built-in Google Search grounding.

This replaces the previous DeepSeek (LLM) + Tavily (news) stack. A single
Gemini call retrieves its own news via Google Search grounding and returns a
fair-value estimate, so there is no separate ``fetch_news`` step and zero paid
API cost on the free tier.
"""

from __future__ import annotations

import json
import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from .config import (
        GEMINI_API_KEY,
        GEMINI_CACHE_PATH,
        GEMINI_CACHE_TTL_HOURS,
        GEMINI_MODEL,
        GEMINI_MODEL_PRO,
        GEMINI_PRICING,
        GEMINI_PRO_RECHECK,
        GEMINI_RPM,
        GEMINI_USE_SEARCH,
    )
    from .utils import retry_with_backoff
except ImportError:  # pragma: no cover
    from config import (
        GEMINI_API_KEY,
        GEMINI_CACHE_PATH,
        GEMINI_CACHE_TTL_HOURS,
        GEMINI_MODEL,
        GEMINI_MODEL_PRO,
        GEMINI_PRICING,
        GEMINI_PRO_RECHECK,
        GEMINI_RPM,
        GEMINI_USE_SEARCH,
    )
    from utils import retry_with_backoff

LOGGER = logging.getLogger("scorer")

_client = None
_CACHE_LOCK = threading.RLock()

ANALYTICAL_EDGE_THRESHOLD = 0.02
_EVIDENCE_KEYS = ("official_data", "reputable_reporting", "market_signals", "social_or_unverified")
CATEGORY_INSTRUCTIONS = {
    "politics": "Weight polling data heavily. Account for electoral college dynamics and incumbency advantage.",
    "crypto": "Consider on-chain metrics (active addresses, exchange flows), whale movements, and macro-economic conditions.",
    "sports": "Weight injury reports, head-to-head history, recent form, and home/away advantage.",
    "default": "Use general event forecasting principles.",
}
_CATEGORY_GUIDANCE = CATEGORY_INSTRUCTIONS

_TOKEN_USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}


def _cache_date(market: dict) -> str:
    raw = market.get("cache_date") or market.get("entry_date")
    if raw:
        return str(raw)[:10]
    return datetime.now(timezone.utc).date().isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cache_matches(entry: dict[str, Any], condition_id: str, cache_date: str) -> bool:
    return entry.get("condition_id") == condition_id and entry.get("date") == cache_date


def read_cached_score(
    market: dict,
    *,
    max_age_hours: float | None = GEMINI_CACHE_TTL_HOURS,
) -> dict[str, Any] | None:
    """Return the newest cached Gemini score for this market/date, if valid."""
    condition_id = str(market.get("condition_id") or "")
    if not condition_id or not GEMINI_CACHE_PATH.exists():
        return None
    cache_date = _cache_date(market)
    now = datetime.now(timezone.utc)
    newest: dict[str, Any] | None = None

    with _CACHE_LOCK:
        try:
            with GEMINI_CACHE_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not _cache_matches(entry, condition_id, cache_date):
                        continue
                    timestamp = _parse_timestamp(entry.get("timestamp"))
                    if max_age_hours is not None and max_age_hours > 0:
                        if timestamp is None or now - timestamp > timedelta(hours=max_age_hours):
                            continue
                    newest = entry
        except OSError as exc:
            LOGGER.debug("Gemini cache read failed for %s: %s", condition_id, exc)
            return None

    score = newest.get("score") if newest else None
    return dict(score) if isinstance(score, dict) else None


def write_cached_score(market: dict, score: dict[str, Any]) -> None:
    """Append a normalized score to the Gemini JSONL cache."""
    condition_id = str(market.get("condition_id") or "")
    if not condition_id:
        return
    entry = {
        "condition_id": condition_id,
        "date": _cache_date(market),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": score,
    }
    with _CACHE_LOCK:
        GEMINI_CACHE_PATH.parent.mkdir(exist_ok=True)
        with GEMINI_CACHE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def reset_token_usage() -> None:
    """Zero the per-run token accountant. Call once at the start of a scan."""
    _TOKEN_USAGE.update(
        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    )


def get_token_usage() -> dict[str, Any]:
    """Return a snapshot of Gemini token usage and estimated cost so far."""
    snapshot = dict(_TOKEN_USAGE)
    snapshot["cost_usd"] = round(snapshot["cost_usd"], 6)
    return snapshot


def _record_usage(model: str, usage: Any) -> None:
    """Accumulate token counts and (optional) cost from a Gemini response.

    ``usage`` may be a ``usage_metadata`` object (live SDK) or a plain dict
    (tests). On the Flash free tier the configured price is 0, so ``cost_usd``
    stays 0 — but the accounting is wired up for paid Vertex keys.
    """
    if not usage:
        return

    def _get(name: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    prompt_tokens = _get("prompt_token_count") or _get("prompt_tokens")
    completion_tokens = _get("candidates_token_count") or _get("completion_tokens")

    pricing = GEMINI_PRICING.get(model, GEMINI_PRICING.get(GEMINI_MODEL, {}))
    cost = (
        prompt_tokens / 1_000_000 * pricing.get("input", 0.0)
        + completion_tokens / 1_000_000 * pricing.get("output", 0.0)
    )

    _TOKEN_USAGE["calls"] += 1
    _TOKEN_USAGE["prompt_tokens"] += prompt_tokens
    _TOKEN_USAGE["completion_tokens"] += completion_tokens
    _TOKEN_USAGE["total_tokens"] += prompt_tokens + completion_tokens
    _TOKEN_USAGE["cost_usd"] += cost


def _days_remaining(end_date: str) -> str:
    try:
        end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        days = (end - datetime.now(timezone.utc)).days
        return f"{days} days remaining"
    except Exception:
        return "unknown time remaining"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_probability(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _coerce_confidence_score(value: Any, default: float = 0.0) -> float:
    """Normalize model confidence to a 0-100 calibration score."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"low", "medium", "high"}:
            return {"low": 35.0, "medium": 65.0, "high": 85.0}[text]
        value = text.rstrip("%")
    raw = _safe_float(value, default)
    if 0.0 <= raw <= 1.0:
        raw *= 100.0
    return round(max(0.0, min(100.0, raw)), 2)


def _confidence_level(confidence: Any) -> str:
    """Map numeric confidence to the existing low/medium/high buckets."""
    score = _coerce_confidence_score(confidence)
    if score >= 75.0:
        return "high"
    if score >= 50.0:
        return "medium"
    return "low"


def _coerce_string_list(value: Any, limit: int = 5) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _coerce_evidence_summary(value: Any) -> dict[str, list[str]]:
    summary = value if isinstance(value, dict) else {}
    return {key: _coerce_string_list(summary.get(key), limit=4) for key in _EVIDENCE_KEYS}


def _coerce_bayesian_updates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    updates = []
    for item in value[:5]:
        if isinstance(item, dict):
            probability = item.get("probability_after")
            probability_after = None if probability is None else _coerce_probability(probability)
            updates.append(
                {
                    "direction": str(item.get("direction") or "neutral").lower(),
                    "magnitude": str(item.get("magnitude") or "small").lower(),
                    "evidence": str(item.get("evidence") or "").strip(),
                    "probability_after": probability_after,
                }
            )
        else:
            updates.append(
                {
                    "direction": "neutral",
                    "magnitude": "small",
                    "evidence": str(item).strip(),
                    "probability_after": None,
                }
            )
    return updates


def _category_guidance(category: str) -> str:
    return CATEGORY_INSTRUCTIONS.get(str(category or "").lower(), CATEGORY_INSTRUCTIONS["default"])


def _format_liquidity_context(market: dict) -> str:
    lines = [f"Total volume traded: ${_safe_float(market.get('volume')):,.0f}"]
    liquidity = market.get("liquidity_usdc", market.get("liquidity"))
    if liquidity is not None:
        lines.append(f"Displayed liquidity: ${_safe_float(liquidity):,.0f}")
    spread = market.get("spread")
    if spread is not None:
        lines.append(f"Quoted spread: {_safe_float(spread) * 100:.1f}%")
    return "\n".join(lines)


def build_prompt(market: dict, category: str | None = None) -> str:
    """Build the Gemini analysis prompt with category-specific instructions."""
    outcomes = market.get("outcomes", [])
    prices = market.get("prices", [])
    outcome_lines = []
    for outcome, raw_price in zip(outcomes, prices):
        price = _safe_float(raw_price)
        outcome_lines.append(
            f"  {outcome}: current market price {price:.2f} (implied probability {price * 100:.1f}%)"
        )
    outcomes_str = "\n".join(outcome_lines) or "  (no outcome prices supplied)"
    category = category or market.get("category", "general")

    return f"""You are a disciplined prediction market analyst with access to Google Search.

Market question: {market.get("question", "")}
Category: {category}
Resolution date: {market.get("end_date", "unknown")} ({_days_remaining(market.get("end_date", ""))})
Market URL: {market.get("url", "")}

Current outcome prices:
{outcomes_str}

Liquidity / tradeability context:
{_format_liquidity_context(market)}

ANALYSIS PATH:
1. Decompose the market into the exact resolution criteria, likely drivers, and time remaining.
2. Set a base rate before looking at the latest news.
3. Search for recent supporting evidence and counter-evidence for each outcome.
4. Apply this evidence hierarchy: official data / filings / primary sources > reputable reporting and expert consensus > market/liquidity signals > social media or unsourced commentary.
5. Make Bayesian updates from the base rate. State the main updates in the JSON.
6. Compress estimates toward 50% when evidence is thin, stale, contradictory, or outside your expertise.
7. Calculate edge as fair_value_estimate minus current_price for the recommended outcome.

Category-specific guidance:
{_category_guidance(str(category))}

A market is mispriced only when the evidence-supported fair value differs from
the current price by more than {ANALYTICAL_EDGE_THRESHOLD:.2f}. Treat 2-5 cents
as a weak analytical edge, under 10 cents with skepticism, and over 15 cents as
actionable only when evidence quality is high. Weigh counter-evidence seriously:
if it materially undercuts the case, lower confidence or decline to bet.

Return ONLY a valid JSON object with this schema. No markdown, no explanation
outside the JSON:

{{
  "recommended_outcome": "<outcome label or null if no clear edge>",
  "recommended_outcome_index": <integer index or null>,
  "current_price": <float — price of the recommended outcome>,
  "probability": <float — same value as fair_value_estimate>,
  "fair_value_estimate": <float — your probability estimate>,
  "edge": <fair_value_estimate minus current_price>,
  "confidence": <integer or float from 0 to 100>,
  "direction": "<BUY|SKIP>",
  "base_rate": <float probability before latest evidence>,
  "evidence_summary": {{
    "official_data": ["<primary-source evidence>"],
    "reputable_reporting": ["<reported evidence>"],
    "market_signals": ["<price/liquidity/positioning signal>"],
    "social_or_unverified": ["<weak evidence, if any>"]
  }},
  "bayesian_updates": [
    {{
      "direction": "<toward|away|neutral>",
      "magnitude": "<small|medium|large>",
      "evidence": "<what changed the estimate>",
      "probability_after": <float>
    }}
  ],
  "reasoning": "<2-3 sentences max — what news drives this and why>",
  "key_risks": ["<risk that could break the thesis>", "<another risk>"],
  "information_quality": "<low|medium|high>",
  "edge_threshold_met": <true|false>,
  "news_supports_bet": <true|false>,
  "counter_evidence_considered": <true|false>,
  "news_headlines": ["<headline 1>", "<headline 2>", "<headline 3>"]
}}

If news is absent, contradictory, stale, low quality, or the edge is <=
{ANALYTICAL_EDGE_THRESHOLD:.2f}, set recommended_outcome to null, edge to 0,
edge_threshold_met to false, and news_supports_bet to false. Do not force a pick.
""".strip()


_build_prompt = build_prompt


def _parse_json(raw: str) -> dict[str, Any]:
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    return json.loads(clean)


def _rate_limit_sleep() -> None:
    if GEMINI_RPM and GEMINI_RPM > 0:
        time.sleep(60.0 / GEMINI_RPM)


@retry_with_backoff(max_retries=5, base_delay=1)
def _generate_content(client: Any, model: str, prompt: str, config: Any) -> Any:
    """Call Gemini with backoff around transient SDK/network failures."""
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )


def _call_gemini(prompt: str, use_pro: bool = False) -> dict[str, Any] | None:
    from google.genai import types

    client = _get_client()
    model = GEMINI_MODEL_PRO if use_pro else GEMINI_MODEL

    tools = []
    if GEMINI_USE_SEARCH:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    config = types.GenerateContentConfig(
        tools=tools or None,
        temperature=0.1,
        max_output_tokens=1200,
    )

    raw = ""
    for attempt in range(2):
        try:
            response = _generate_content(client, model, prompt, config)
            _record_usage(model, getattr(response, "usage_metadata", None))
            raw = (response.text or "").strip()
            # A successful API call consumes RPM budget, so throttle here (before
            # parsing) -- a retry after malformed JSON must still respect GEMINI_RPM
            # instead of firing again after only a short backoff.
            _rate_limit_sleep()
            return _parse_json(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Gemini JSON parse failed (attempt %s): %s | raw: %s", attempt + 1, exc, raw[:200])
        except Exception as exc:  # pragma: no cover - network/SDK errors
            LOGGER.warning("Gemini call failed (attempt %s): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(5)
    return None


def _neutral_score(market: dict, reason: str) -> dict[str, Any]:
    return {
        "recommended_outcome": None,
        "recommended_outcome_index": None,
        "current_price": 0.0,
        "fair_value_estimate": 0.0,
        "edge": 0.0,
        "probability": 0.0,
        "confidence": 0.0,
        "confidence_level": "low",
        "direction": "SKIP",
        "reasoning": reason,
        "base_rate": 0.5,
        "evidence_summary": _coerce_evidence_summary({}),
        "bayesian_updates": [],
        "key_risks": [],
        "information_quality": "low",
        "edge_threshold_met": False,
        "news_supports_bet": False,
        "counter_evidence_considered": False,
        "news_headlines": [],
        "token_id": None,
        "condition_id": market.get("condition_id", ""),
        "category": market.get("category", "other"),
        "outcome_index": None,
    }


def _coerce_score(market: dict, score: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(score)
    outcomes = market.get("outcomes", [])
    prices = market.get("prices", [])

    index = normalized.get("recommended_outcome_index")
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None

    recommended_outcome = normalized.get("recommended_outcome")
    if index is not None and 0 <= index < len(outcomes) and not recommended_outcome:
        recommended_outcome = outcomes[index]
    if recommended_outcome in outcomes and index is None:
        index = outcomes.index(recommended_outcome)
    if recommended_outcome not in outcomes:
        recommended_outcome = None
        index = None

    current_price = _coerce_probability(normalized.get("current_price"), 0.0)
    if current_price <= 0 and index is not None and 0 <= index < len(prices):
        current_price = _coerce_probability(prices[index], 0.0)

    fair_value = _coerce_probability(normalized.get("fair_value_estimate", normalized.get("probability")), 0.0)
    edge = _safe_float(normalized.get("edge"), 0.0)
    if recommended_outcome and current_price > 0 and fair_value > 0:
        edge = round(fair_value - current_price, 4)

    news_supports_bet = bool(normalized.get("news_supports_bet", False))
    if (
        not recommended_outcome
        or current_price <= 0
        or fair_value <= 0
        or edge <= ANALYTICAL_EDGE_THRESHOLD
    ):
        recommended_outcome = None
        index = None
        edge = 0.0
        news_supports_bet = False
    confidence = _coerce_confidence_score(normalized.get("confidence", normalized.get("confidence_score")), 0.0)
    confidence_level = _confidence_level(confidence)

    information_quality = str(normalized.get("information_quality") or "low").lower()
    if information_quality not in {"low", "medium", "high"}:
        information_quality = "low"

    normalized.update(
        {
            "recommended_outcome": recommended_outcome,
            "recommended_outcome_index": index,
            "current_price": current_price,
            "fair_value_estimate": fair_value,
            "probability": fair_value,
            "edge": edge,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "direction": str(normalized.get("direction") or ("BUY" if recommended_outcome else "SKIP")).upper(),
            "base_rate": _coerce_probability(normalized.get("base_rate"), 0.5),
            "evidence_summary": _coerce_evidence_summary(normalized.get("evidence_summary")),
            "bayesian_updates": _coerce_bayesian_updates(normalized.get("bayesian_updates")),
            "reasoning": str(normalized.get("reasoning") or ""),
            "key_risks": _coerce_string_list(normalized.get("key_risks"), limit=5),
            "information_quality": information_quality,
            "edge_threshold_met": bool(recommended_outcome and edge > ANALYTICAL_EDGE_THRESHOLD),
            "news_supports_bet": news_supports_bet,
            "counter_evidence_considered": bool(normalized.get("counter_evidence_considered", False)),
            "news_headlines": _coerce_string_list(normalized.get("news_headlines"), limit=5),
            "token_id": normalized.get("token_id"),
            "condition_id": market.get("condition_id", ""),
            "category": market.get("category", "other"),
            "outcome_index": index,
        }
    )
    return normalized


def score_market(market: dict, news: list[dict] | None = None) -> dict | None:
    """Score a market with Gemini + Google Search grounding.

    The ``news`` parameter is accepted for backwards compatibility but ignored —
    Gemini fetches its own news internally via search grounding.
    """
    cached = read_cached_score(market)
    if cached is not None:
        LOGGER.debug("using cached Gemini score for %s", market.get("condition_id", ""))
        return cached

    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY is not configured; skipping market scoring")
        return None

    prompt = build_prompt(market, category=market.get("category"))
    score = _call_gemini(prompt, use_pro=False)
    if score is None:
        return None

    if "edge" not in score or "recommended_outcome" not in score:
        LOGGER.warning("score missing required fields for: %s", str(market.get("question", ""))[:50])
        return None

    normalized = _coerce_score(market, score)

    # High-confidence recheck with the Pro model on significant edges only.
    if (
        GEMINI_PRO_RECHECK
        and normalized["confidence_level"] == "high"
        and normalized["edge"] >= 0.12
    ):
        LOGGER.info("running Pro recheck for high-edge pick: %s", str(market.get("question", ""))[:50])
        recheck_raw = _call_gemini(prompt, use_pro=True)
        if recheck_raw is not None:
            recheck = _coerce_score(market, recheck_raw)
            if recheck["recommended_outcome"] != normalized["recommended_outcome"]:
                LOGGER.info("Pro model disagrees — downgrading confidence to medium")
                normalized["confidence"] = 65.0
                normalized["confidence_level"] = "medium"
                normalized["pro_recheck_disagreed"] = True

    write_cached_score(market, normalized)
    return normalized

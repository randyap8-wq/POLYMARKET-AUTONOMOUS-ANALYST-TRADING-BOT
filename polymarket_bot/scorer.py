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
from datetime import datetime, timezone
from typing import Any

try:
    from .config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GEMINI_MODEL_PRO,
        GEMINI_PRICING,
        GEMINI_PRO_RECHECK,
        GEMINI_RPM,
        GEMINI_USE_SEARCH,
    )
except ImportError:  # pragma: no cover
    from config import (
        GEMINI_API_KEY,
        GEMINI_MODEL,
        GEMINI_MODEL_PRO,
        GEMINI_PRICING,
        GEMINI_PRO_RECHECK,
        GEMINI_RPM,
        GEMINI_USE_SEARCH,
    )

LOGGER = logging.getLogger("scorer")

_client = None

_TOKEN_USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}


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


def _build_prompt(market: dict) -> str:
    outcomes = market.get("outcomes", [])
    prices = market.get("prices", [])
    outcomes_str = "\n".join(
        f"  {outcome}: current market price {price:.2f} (implied probability {price * 100:.1f}%)"
        for outcome, price in zip(outcomes, prices)
    )
    category = market.get("category", "general")

    return f"""You are a prediction market analyst with access to Google Search.

Market question: {market.get("question", "")}
Category: {category}
Market closes: {market.get("end_date", "unknown")} ({_days_remaining(market.get("end_date", ""))})
Total volume traded: ${market.get("volume", 0):,.0f}
Market URL: {market.get("url", "")}

Current outcome prices:
{outcomes_str}

TASK:
1. Use Google Search to find news and information from the LAST FEW DAYS relevant to this market.
2. Search for both supporting evidence AND counter-evidence for each outcome.
3. Based on what you find, determine if any outcome is significantly mispriced.

A market is mispriced when recent news suggests the true probability differs
meaningfully from the current price. A 5-cent edge (0.05) is the minimum worth
noting. Under 10 cents, be skeptical. Over 15 cents with strong news support is
actionable. Weigh the counter-evidence seriously: if it materially undercuts the
case, lower your confidence or decline to bet.

Return ONLY a valid JSON object — no markdown, no explanation outside the JSON:

{{
  "recommended_outcome": "<outcome label or null if no clear edge>",
  "recommended_outcome_index": <integer index or null>,
  "current_price": <float — price of the recommended outcome>,
  "fair_value_estimate": <float — your probability estimate>,
  "edge": <fair_value_estimate minus current_price>,
  "confidence": "<low|medium|high>",
  "reasoning": "<2-3 sentences max — what news drives this and why>",
  "news_supports_bet": <true|false>,
  "counter_evidence_considered": <true|false>,
  "news_headlines": ["<headline 1>", "<headline 2>", "<headline 3>"]
}}

If news is absent, contradictory, or the market looks fairly priced, set
recommended_outcome to null and edge to 0. Do not force a pick.
""".strip()


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
        max_output_tokens=700,
    )

    raw = ""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
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
        "confidence": "low",
        "reasoning": reason,
        "news_supports_bet": False,
        "counter_evidence_considered": False,
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
    if recommended_outcome not in outcomes:
        recommended_outcome = None
        index = None

    current_price = normalized.get("current_price")
    if (
        (current_price is None or float(current_price or 0) <= 0)
        and index is not None
        and 0 <= index < len(prices)
    ):
        current_price = prices[index]

    normalized.update(
        {
            "recommended_outcome": recommended_outcome,
            "recommended_outcome_index": index,
            "current_price": float(current_price or 0.0),
            "fair_value_estimate": float(normalized.get("fair_value_estimate") or 0.0),
            "edge": float(normalized.get("edge") or 0.0),
            "confidence": str(normalized.get("confidence") or "low").lower(),
            "reasoning": str(normalized.get("reasoning") or ""),
            "news_supports_bet": bool(normalized.get("news_supports_bet", False)),
            "counter_evidence_considered": bool(normalized.get("counter_evidence_considered", False)),
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
    if not GEMINI_API_KEY:
        LOGGER.warning("GEMINI_API_KEY is not configured; skipping market scoring")
        return None

    prompt = _build_prompt(market)
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
        and normalized["confidence"] == "high"
        and normalized["edge"] >= 0.12
    ):
        LOGGER.info("running Pro recheck for high-edge pick: %s", str(market.get("question", ""))[:50])
        recheck_raw = _call_gemini(prompt, use_pro=True)
        if recheck_raw is not None:
            recheck = _coerce_score(market, recheck_raw)
            if recheck["recommended_outcome"] != normalized["recommended_outcome"]:
                LOGGER.info("Pro model disagrees — downgrading confidence to medium")
                normalized["confidence"] = "medium"
                normalized["pro_recheck_disagreed"] = True

    return normalized

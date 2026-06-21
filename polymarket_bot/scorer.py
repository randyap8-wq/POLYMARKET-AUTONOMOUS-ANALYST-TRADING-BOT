from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

try:
    from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_REASONER_MODEL, DEEPSEEK_URL
except ImportError:  # pragma: no cover
    from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_REASONER_MODEL, DEEPSEEK_URL

LOGGER = logging.getLogger("scorer")


def _build_user_message(market: dict, news: list[dict]) -> str:
    outcomes_str = "\n".join(
        f"  {outcome}: current price {price:.2f} (implied probability {price * 100:.1f}%)"
        for outcome, price in zip(market["outcomes"], market["prices"])
    )
    news_str = "\n\n".join(
        f"[{item['published_date']}] {item['title']}\n{item['snippet']}" for item in news
    ) or "No recent news found."

    days_remaining = ""
    try:
        from datetime import datetime, timezone

        end = datetime.fromisoformat(market.get("end_date", "").replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        delta = (end - datetime.now(timezone.utc)).days
        days_remaining = f"Days until market closes: {delta}"
    except Exception:
        pass

    return f"""
Market: {market['question']}
End date: {market['end_date']}
Volume traded: ${market['volume']:,.0f}
URL: {market['url']}

Current outcome prices:
{outcomes_str}

Recent news (last 3 days):
{news_str}

{days_remaining}

Analyze whether any outcome is mispriced given this news.
Return ONLY a valid JSON object, no markdown, no explanation outside the JSON:

{{
  "recommended_outcome": "<outcome label or null>",
  "recommended_outcome_index": <integer index or null>,
  "current_price": <float>,
  "fair_value_estimate": <float>,
  "edge": <fair_value_estimate minus current_price as float>,
  "confidence": "<low|medium|high>",
  "reasoning": "<2-3 sentence max>",
  "news_supports_bet": <true|false>,
  "token_id": null
}}

If news is absent, irrelevant, or contradictory, set recommended_outcome to null and edge to 0.
""".strip()


def _call_deepseek(user_message: str, model: str) -> dict[str, Any] | None:
    headers = {
        "Authorization": "Bearer " + DEEPSEEK_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a prediction market analyst. Output only valid JSON. No markdown fences.",
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }

    last_error: Exception | None = None
    for attempt in range(2):
        raw = ""
        try:
            response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
            if response.status_code >= 400:
                if attempt == 0:
                    LOGGER.warning("deepseek returned %s; retrying once", response.status_code)
                    time.sleep(5)
                    continue
                response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            LOGGER.error("deepseek returned non-json payload: %s", raw)
            return None
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 0:
                LOGGER.warning("deepseek request failed: %s; retrying once", exc)
                time.sleep(5)
                continue
            raise

    if last_error:
        raise last_error
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
        "token_id": None,
        "condition_id": market["condition_id"],
        "outcome_index": None,
    }


def _coerce_score(market: dict, score: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(score)
    index = normalized.get("recommended_outcome_index")
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None
    recommended_outcome = normalized.get("recommended_outcome")
    if index is not None and 0 <= index < len(market["outcomes"]) and not recommended_outcome:
        recommended_outcome = market["outcomes"][index]
    if recommended_outcome not in market["outcomes"]:
        recommended_outcome = None
        index = None

    current_price = normalized.get("current_price")
    if current_price is None and index is not None:
        current_price = market["prices"][index]

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
            "token_id": normalized.get("token_id"),
            "condition_id": market["condition_id"],
            "outcome_index": index,
        }
    )
    return normalized


def score_market(market: dict, news: list[dict]) -> dict | None:
    if not news:
        return _neutral_score(market, "No recent news found.")

    if not DEEPSEEK_API_KEY:
        LOGGER.warning("DEEPSEEK_API_KEY is not configured; skipping market scoring")
        return None

    user_message = _build_user_message(market, news)
    score = _call_deepseek(user_message, DEEPSEEK_MODEL)
    if score is None:
        return None

    normalized = _coerce_score(market, score)

    if normalized["confidence"] == "high" and normalized["edge"] > 0.10:
        recheck = _call_deepseek(user_message, DEEPSEEK_REASONER_MODEL)
        if recheck is None:
            return normalized
        second_opinion = _coerce_score(market, recheck)
        if second_opinion["recommended_outcome"] != normalized["recommended_outcome"]:
            LOGGER.info(
                "deepseek models disagreed for market '%s': %s vs %s",
                market["question"],
                normalized["recommended_outcome"],
                second_opinion["recommended_outcome"],
            )
            normalized.update(
                {
                    "recommended_outcome": None,
                    "recommended_outcome_index": None,
                    "edge": 0.0,
                    "news_supports_bet": False,
                    "outcome_index": None,
                }
            )

    return normalized

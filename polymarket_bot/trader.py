from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from py_clob_client.clob_types import OrderArgs, OrderType

try:
    from .config import CLOB_BASE, DRY_RUN, MAX_BET_USDC, POLYGON_WALLET_ADDRESS, TRADES_LOG_PATH
    from .wallet import build_clob_client
except ImportError:  # pragma: no cover
    from config import CLOB_BASE, DRY_RUN, MAX_BET_USDC, POLYGON_WALLET_ADDRESS, TRADES_LOG_PATH
    from wallet import build_clob_client

LOGGER = logging.getLogger("trader")


def _fetch_market_tokens(condition_id: str) -> list[dict[str, Any]]:
    response = requests.get(f"{CLOB_BASE}/markets/{condition_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("tokens", [])


def _match_token(tokens: list[dict[str, Any]], outcome_label: str) -> dict[str, Any] | None:
    for token in tokens:
        label = token.get("outcome") or token.get("label") or token.get("name") or ""
        if label.lower() == outcome_label.lower():
            return token
    return None


def _best_ask_price(token_id: str) -> float | None:
    response = requests.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    asks = payload.get("asks", [])
    if not asks:
        return None
    try:
        return min(float(ask["price"]) for ask in asks)
    except (KeyError, TypeError, ValueError):
        return None


def _append_trade_log(entry: dict[str, Any]) -> None:
    TRADES_LOG_PATH.parent.mkdir(exist_ok=True)
    with TRADES_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def place_bet(market: dict, score: dict) -> dict | None:
    outcome_label = score.get("recommended_outcome")
    if not outcome_label:
        LOGGER.info("skipping '%s' because no outcome was recommended", market.get("question", ""))
        return None

    tokens = _fetch_market_tokens(market["condition_id"])
    token = _match_token(tokens, outcome_label)
    if not token:
        LOGGER.warning("could not find token for outcome '%s'", outcome_label)
        return None

    token_id = str(token.get("token_id") or token.get("tokenId") or token.get("id") or "")
    if not token_id:
        LOGGER.warning("market token for '%s' did not include a token id", outcome_label)
        return None

    best_ask_price = _best_ask_price(token_id)
    if best_ask_price is None:
        LOGGER.warning("no ask liquidity available for token %s", token_id)
        return None

    current_price = float(score.get("current_price") or 0.0)
    if abs(best_ask_price - current_price) > 0.02:
        LOGGER.info(
            "aborting bet on '%s'; price moved from %.2f to %.2f",
            market["question"],
            current_price,
            best_ask_price,
        )
        return None

    edge = max(float(score.get("edge") or 0.0), 0.0)
    usdc_to_spend = round(min(MAX_BET_USDC, MAX_BET_USDC * edge * 2), 2)
    if usdc_to_spend <= 0:
        LOGGER.info("skipping '%s'; computed stake is zero", market["question"])
        return None

    token_amount = round(usdc_to_spend / best_ask_price, 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    if DRY_RUN:
        result = {
            "status": "dry_run",
            "token_id": token_id,
            "price": best_ask_price,
            "size": token_amount,
            "usdc_spent": usdc_to_spend,
        }
    else:
        client = build_clob_client(level_2=True)
        expiration = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
        order = client.create_order(
            OrderArgs(
                token_id=token_id,
                price=best_ask_price,
                size=token_amount,
                side="BUY",
                expiration=expiration,
            )
        )
        result = client.post_order(order, OrderType.FOK)

    _append_trade_log(
        {
            "timestamp": timestamp,
            "question": market["question"],
            "outcome": outcome_label,
            "usdc_spent": usdc_to_spend,
            "price": best_ask_price,
            "tokens_bought": token_amount,
            "order_id": result.get("orderID") or result.get("id") or "",
            "status": result.get("status", "filled" if not DRY_RUN else "dry_run"),
        }
    )
    return result

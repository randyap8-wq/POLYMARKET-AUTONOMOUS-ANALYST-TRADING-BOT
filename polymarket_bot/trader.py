from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from py_clob_client.clob_types import OrderArgs, OrderType

try:
    from .config import CLOB_BASE, DRY_RUN, MAX_BET_USDC, POLYGON_WALLET_ADDRESS, TRADES_LOG_PATH, USE_LIMIT_ORDERS
    from .wallet import build_clob_client
except ImportError:  # pragma: no cover
    from config import CLOB_BASE, DRY_RUN, MAX_BET_USDC, POLYGON_WALLET_ADDRESS, TRADES_LOG_PATH, USE_LIMIT_ORDERS
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


def _build_order_args(
    token_id: str,
    price: float,
    size: float,
    *,
    expiration: int | None = None,
    time_in_force: str | None = None,
) -> OrderArgs:
    kwargs: dict[str, Any] = {
        "token_id": token_id,
        "price": price,
        "size": size,
        "side": "BUY",
    }
    if expiration is not None:
        kwargs["expiration"] = expiration
    if time_in_force:
        kwargs["time_in_force"] = time_in_force
    try:
        return OrderArgs(**kwargs)
    except TypeError:
        kwargs.pop("time_in_force", None)
        return OrderArgs(**kwargs)


def _order_type(name: str, fallback: Any) -> Any:
    return getattr(OrderType, name, fallback)


def _extract_order_id(result: dict[str, Any]) -> str:
    return str(result.get("orderID") or result.get("order_id") or result.get("id") or "")


def _order_status(client: Any, order_id: str) -> dict[str, Any]:
    for method_name in ("get_order", "get_order_status"):
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            status = method(order_id)
            return status if isinstance(status, dict) else {"status": str(status)}
        except Exception as exc:  # pragma: no cover - client/network dependent
            LOGGER.debug("order status check failed via %s: %s", method_name, exc)
    return {}


def _cancel_order(client: Any, order_id: str) -> None:
    for method_name in ("cancel_order", "cancel"):
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            method(order_id)
            return
        except Exception as exc:  # pragma: no cover - client/network dependent
            LOGGER.debug("order cancel failed via %s: %s", method_name, exc)


def _is_filled(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or payload.get("state") or "").lower()
    filled_size = float(payload.get("filled_size") or payload.get("filledSize") or 0.0)
    return status in {"filled", "matched", "complete", "completed"} or filled_size > 0


def _post_limit_order_with_retry(client: Any, token_id: str, price: float, size: float) -> dict[str, Any]:
    limit_type = _order_type("LIMIT", _order_type("GTC", OrderType.FOK))
    last_result: dict[str, Any] = {}

    for attempt in range(2):
        limit_price = round(min(price + 0.01 * attempt, 0.99), 4)
        order = client.create_order(
            _build_order_args(token_id, limit_price, size, time_in_force="GTC")
        )
        result = client.post_order(order, limit_type)
        last_result = result if isinstance(result, dict) else {"status": str(result)}
        order_id = _extract_order_id(last_result)
        if order_id:
            last_result.setdefault("orderID", order_id)
        if _is_filled(last_result) or not order_id:
            LOGGER.info("limit order finished immediately at %.4f: %s", limit_price, last_result.get("status", "posted"))
            return last_result

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(10)
            status = _order_status(client, order_id)
            if status:
                last_result = {**last_result, **status}
            if _is_filled(last_result):
                LOGGER.info("limit order filled at %.4f after polling", limit_price)
                return last_result

        LOGGER.info("limit order %s not filled at %.4f; cancelling", order_id, limit_price)
        _cancel_order(client, order_id)

    last_result.setdefault("status", "unfilled")
    return last_result


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
    if best_ask_price > current_price + 0.02:
        LOGGER.info(
            "aborting bet on '%s'; price moved up from %.2f to %.2f",
            market["question"],
            current_price,
            best_ask_price,
        )
        return None

    # Honour the stake the risk layer already computed (risk.size_positions,
    # surfaced as stake_usdc in report.json) so the order we place matches what
    # the report shows. A stake key that is *present but zero* is a deliberate
    # veto — a concurrency/exposure cap or the drawdown circuit breaker said
    # "don't size this" — so we must not bet. Only fall back to edge-scaled
    # sizing when the key is absent entirely (risk layer never ran).
    raw_stake = score.get("stake_usdc")
    if raw_stake is None:
        edge = max(float(score.get("edge") or 0.0), 0.0)
        usdc_to_spend = round(min(MAX_BET_USDC * edge * 2, MAX_BET_USDC), 2)
    else:
        usdc_to_spend = round(max(float(raw_stake), 0.0), 2)
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
        if USE_LIMIT_ORDERS:
            result = _post_limit_order_with_retry(client, token_id, best_ask_price, token_amount)
        else:
            expiration = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
            order = client.create_order(
                _build_order_args(
                    token_id=token_id,
                    price=best_ask_price,
                    size=token_amount,
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

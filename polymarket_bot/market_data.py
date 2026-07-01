"""CLOB market-data layer for the quant engine.

Fetches the numeric, point-in-time data the quant signals are built from:

* ``fetch_price_history`` — timestamped price series (CLOB ``/prices-history``).
* ``fetch_order_book``    — live bids/asks (CLOB ``/book``).
* ``fetch_market_tokens`` — outcome -> token id mapping (CLOB ``/markets``).

All calls fail soft: on any network/parse error they return an empty structure
so the quant layer degrades to "no data" rather than crashing the scan.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

try:
    from .config import CLOB_BASE, QUANT_HISTORY_FIDELITY, QUANT_HISTORY_INTERVAL
    from .utils import retry_with_backoff
except ImportError:  # pragma: no cover
    from config import CLOB_BASE, QUANT_HISTORY_FIDELITY, QUANT_HISTORY_INTERVAL
    from utils import retry_with_backoff

LOGGER = logging.getLogger("market_data")

_TIMEOUT = 15


@retry_with_backoff(max_retries=5, base_delay=1)
def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET JSON with retries for transient CLOB/API failures."""
    response = requests.get(url, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_price_history(
    token_id: str,
    interval: str | None = None,
    fidelity: int | None = None,
) -> list[dict[str, float]]:
    """Return a chronological list of ``{"t": unix_ts, "p": price}`` points.

    Empty list on failure or when the token has no history.
    """
    if not token_id:
        return []
    interval = interval or QUANT_HISTORY_INTERVAL
    fidelity = fidelity or QUANT_HISTORY_FIDELITY
    try:
        payload = _get_json(
            f"{CLOB_BASE}/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": fidelity},
        )
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.debug("price history fetch failed for %s: %s", token_id, exc)
        return []

    history = payload.get("history", payload) if isinstance(payload, dict) else payload
    points: list[dict[str, float]] = []
    for item in history or []:
        try:
            points.append({"t": float(item["t"]), "p": float(item["p"])})
        except (KeyError, TypeError, ValueError):
            continue
    points.sort(key=lambda pt: pt["t"])
    return points


def fetch_order_book(token_id: str) -> dict[str, list[dict[str, float]]]:
    """Return ``{"bids": [...], "asks": [...]}`` with float price/size levels.

    Empty book on failure.
    """
    empty: dict[str, list[dict[str, float]]] = {"bids": [], "asks": []}
    if not token_id:
        return empty
    try:
        payload = _get_json(f"{CLOB_BASE}/book", params={"token_id": token_id})
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.debug("order book fetch failed for %s: %s", token_id, exc)
        return empty

    def _levels(raw: Any) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for level in raw or []:
            try:
                out.append({"price": float(level["price"]), "size": float(level["size"])})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    return {"bids": _levels(payload.get("bids")), "asks": _levels(payload.get("asks"))}


def fetch_market_tokens(condition_id: str) -> list[dict[str, Any]]:
    """Return the CLOB token records for a market (outcome + token_id)."""
    if not condition_id:
        return []
    try:
        payload = _get_json(f"{CLOB_BASE}/markets/{condition_id}")
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.debug("market tokens fetch failed for %s: %s", condition_id, exc)
        return []
    return payload.get("tokens", [])


def _first_number(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def fetch_market_stats(condition_id: str) -> dict[str, float]:
    """Return activity stats for a CLOB market.

    The CLOB/Gamma payload names vary by endpoint version, so the parser accepts
    the common spellings and returns only normalized keys. Empty dict on failure.
    """
    if not condition_id:
        return {}
    try:
        payload = _get_json(f"{CLOB_BASE}/markets/{condition_id}")
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.debug("market stats fetch failed for %s: %s", condition_id, exc)
        return {}
    if not isinstance(payload, dict):
        return {}

    volume_24h = _first_number(
        payload,
        ("volume_24h", "volume24h", "volume24hr", "volume_24hr", "volume24H", "volumeNum24Hr"),
    )
    open_interest = _first_number(
        payload,
        ("open_interest", "openInterest", "open_interest_usdc", "openInterestUsd"),
    )

    stats: dict[str, float] = {}
    if volume_24h is not None:
        stats["volume_24h"] = volume_24h
    if open_interest is not None:
        stats["open_interest"] = open_interest
    return stats

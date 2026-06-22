from __future__ import annotations

from functools import lru_cache

from eth_account import Account
from eth_account.messages import encode_defunct
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from py_order_utils.model.signatures import POLY_PROXY

try:
    from .config import CLOB_BASE, POLYGON_CHAIN_ID, POLYGON_PRIVATE_KEY, POLYGON_WALLET_ADDRESS
except ImportError:  # pragma: no cover
    from config import CLOB_BASE, POLYGON_CHAIN_ID, POLYGON_PRIVATE_KEY, POLYGON_WALLET_ADDRESS


def _require_wallet_settings() -> None:
    if not POLYGON_PRIVATE_KEY:
        raise ValueError("POLYGON_PRIVATE_KEY is required for trade mode")
    if not POLYGON_WALLET_ADDRESS:
        raise ValueError("POLYGON_WALLET_ADDRESS is required for trade mode")


def get_account():
    _require_wallet_settings()
    return Account.from_key(POLYGON_PRIVATE_KEY)


def sign_message(message: str) -> str:
    account = get_account()
    msg = encode_defunct(text=message)
    signed = account.sign_message(msg)
    return signed.signature.hex()


def build_clob_client(level_2: bool = True) -> ClobClient:
    _require_wallet_settings()
    client = ClobClient(
        CLOB_BASE,
        chain_id=POLYGON_CHAIN_ID,
        key=POLYGON_PRIVATE_KEY,
        signature_type=POLY_PROXY,
        funder=POLYGON_WALLET_ADDRESS,
    )
    if level_2:
        creds = get_clob_api_key()
        client.set_api_creds(
            ApiCreds(
                api_key=creds["apiKey"],
                api_secret=creds["secret"],
                api_passphrase=creds["passphrase"],
            )
        )
    return client


# Derive the CLOB API credentials once per process and reuse them. Deriving is a
# signed round-trip to the exchange, and in --loop mode build_clob_client() runs
# on every scan; maxsize=1 means we sign once and cache. This is intentional —
# do not remove the cache or the bot will re-derive credentials on every loop.
@lru_cache(maxsize=1)
def get_clob_api_key():
    client = build_clob_client(level_2=False)
    creds = client.derive_api_key(nonce=0)
    return {
        "apiKey": creds.api_key,
        "secret": creds.api_secret,
        "passphrase": creds.api_passphrase,
    }

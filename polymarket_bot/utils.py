"""Shared operational utilities."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(max_retries: int = 5, base_delay: float = 1) -> Callable[[F], F]:
    """Retry transient failures with exponential backoff.

    Delays are ``base_delay * 2 ** attempt`` seconds, so the default schedule is
    1, 2, 4, 8 and 16 seconds across five attempts. The final exception is
    re-raised so callers can keep their existing fail-soft behavior.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            attempts = max(1, int(max_retries))
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt >= attempts - 1:
                        raise
                    delay = float(base_delay) * (2 ** attempt)
                    logger.warning(
                        "%s failed (%s/%s): %s; retrying in %.1fs",
                        func.__name__,
                        attempt + 1,
                        attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            return None

        return cast(F, wrapper)

    return decorator

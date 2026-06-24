"""Small in-memory rate limiter for the starter backend.

This is good enough for local demos and tests. A deployed system should use a
shared store such as Redis so limits work across multiple API instances.
"""

from collections import defaultdict
from time import monotonic

from fastapi import HTTPException, status


REQUEST_LIMIT = 60
WINDOW_SECONDS = 60
_requests: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(user_id: str) -> None:
    """Allow at most REQUEST_LIMIT requests per user in a rolling window."""

    now = monotonic()
    window_start = now - WINDOW_SECONDS
    recent_requests = [timestamp for timestamp in _requests[user_id] if timestamp >= window_start]

    if len(recent_requests) >= REQUEST_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again shortly.",
        )

    recent_requests.append(now)
    _requests[user_id] = recent_requests

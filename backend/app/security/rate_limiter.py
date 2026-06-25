"""Async-safe per-user token bucket rate limiter.

The implementation is process-local for the assignment prototype. In production,
use a shared store such as Redis so limits apply across all API instances.
"""

import asyncio
from dataclasses import dataclass
from time import monotonic

from app.config import get_settings


@dataclass
class TokenBucket:
    """Token bucket state for one user."""

    tokens: float
    updated_at: float


class RateLimitExceeded(Exception):
    """Raised when a user has no tokens left."""

    def __init__(self, user_id: str, retry_after_seconds: float) -> None:
        self.user_id = user_id
        self.retry_after_seconds = max(retry_after_seconds, 0.0)
        super().__init__(f"Rate limit exceeded for user {user_id}.")


class TokenBucketRateLimiter:
    """Async-safe token bucket limiter keyed by user id."""

    def __init__(
        self,
        capacity: int,
        refill_rate_per_second: float,
        time_provider=monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")
        if refill_rate_per_second < 0:
            raise ValueError("refill_rate_per_second cannot be negative.")

        self.capacity = float(capacity)
        self.refill_rate_per_second = refill_rate_per_second
        self.time_provider = time_provider
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, user_id: str, tokens: float = 1.0) -> None:
        """Consume tokens for a user or raise RateLimitExceeded."""

        if tokens <= 0:
            raise ValueError("tokens must be greater than zero.")

        async with self._lock:
            now = self.time_provider()
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = TokenBucket(tokens=self.capacity, updated_at=now)

            elapsed_seconds = max(now - bucket.updated_at, 0.0)
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + (elapsed_seconds * self.refill_rate_per_second),
            )
            bucket.updated_at = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                self._buckets[user_id] = bucket
                return

            missing_tokens = tokens - bucket.tokens
            retry_after = (
                missing_tokens / self.refill_rate_per_second
                if self.refill_rate_per_second > 0
                else 60.0
            )
            self._buckets[user_id] = bucket
            raise RateLimitExceeded(user_id=user_id, retry_after_seconds=retry_after)

    async def get_tokens(self, user_id: str) -> float:
        """Return current token count for tests and diagnostics."""

        async with self._lock:
            bucket = self._buckets.get(user_id)
            return self.capacity if bucket is None else bucket.tokens


_rate_limiter: TokenBucketRateLimiter | None = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Return the configured global rate limiter."""

    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = TokenBucketRateLimiter(
            capacity=settings.rate_limit_capacity,
            refill_rate_per_second=settings.rate_limit_refill_rate_per_second,
        )

    return _rate_limiter


def reset_rate_limiter(
    capacity: int | None = None,
    refill_rate_per_second: float | None = None,
) -> None:
    """Reset the global limiter. Intended for tests and local development."""

    global _rate_limiter
    settings = get_settings()
    _rate_limiter = TokenBucketRateLimiter(
        capacity=capacity if capacity is not None else settings.rate_limit_capacity,
        refill_rate_per_second=(
            refill_rate_per_second
            if refill_rate_per_second is not None
            else settings.rate_limit_refill_rate_per_second
        ),
    )


async def check_rate_limit(user_id: str) -> None:
    """Consume one request token for a user."""

    await get_rate_limiter().check(user_id)

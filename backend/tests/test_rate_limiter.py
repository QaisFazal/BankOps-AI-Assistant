"""Tests for per-user token bucket rate limiting."""

import asyncio

import pytest

from app.security.rate_limiter import RateLimitExceeded, TokenBucketRateLimiter


class FakeClock:
    """Controllable monotonic clock for deterministic limiter tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_token_bucket_allows_capacity_then_blocks() -> None:
    """A user can consume capacity, then receives a rate-limit error."""

    limiter = TokenBucketRateLimiter(capacity=2, refill_rate_per_second=0)

    asyncio.run(limiter.check("user-1"))
    asyncio.run(limiter.check("user-1"))

    with pytest.raises(RateLimitExceeded):
        asyncio.run(limiter.check("user-1"))


def test_token_bucket_refills_over_time() -> None:
    """Tokens should refill according to configured refill rate."""

    clock = FakeClock()
    limiter = TokenBucketRateLimiter(
        capacity=1,
        refill_rate_per_second=0.5,
        time_provider=clock,
    )

    asyncio.run(limiter.check("user-1"))

    with pytest.raises(RateLimitExceeded):
        asyncio.run(limiter.check("user-1"))

    clock.advance(2)
    asyncio.run(limiter.check("user-1"))


def test_token_bucket_is_per_user() -> None:
    """One user's exhausted bucket should not affect another user."""

    limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_second=0)

    asyncio.run(limiter.check("user-1"))
    asyncio.run(limiter.check("user-2"))

    with pytest.raises(RateLimitExceeded):
        asyncio.run(limiter.check("user-1"))


def test_token_bucket_is_async_safe() -> None:
    """Concurrent checks should not over-consume beyond capacity."""

    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_second=0)

    async def consume() -> bool:
        try:
            await limiter.check("user-1")
        except RateLimitExceeded:
            return False
        return True

    async def consume_many() -> list[bool]:
        return await asyncio.gather(*(consume() for _ in range(10)))

    results = asyncio.run(consume_many())

    assert results.count(True) == 3
    assert results.count(False) == 7

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Rate Limiting for ZeroTrustML.

Provides token bucket and sliding window rate limiters
to protect against DoS attacks and resource exhaustion.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from threading import Lock
from collections import defaultdict


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, identifier: str, limit: int, window: float, retry_after: float):
        self.identifier = identifier
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {identifier}: {limit} requests per {window}s. "
            f"Retry after {retry_after:.1f}s"
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_seconds: float = 60.0
    max_requests_per_window: int = 100


@dataclass
class TokenBucket:
    """Token bucket state for a single identifier."""
    tokens: float
    last_update: float
    request_count: int = 0
    window_start: float = 0.0


class RateLimiter:
    """
    Combined token bucket and sliding window rate limiter.
    
    Uses token bucket for burst protection and sliding window
    for sustained rate limiting.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = Lock()
    
    def _get_or_create_bucket(self, identifier: str) -> TokenBucket:
        """Get or create a token bucket for an identifier."""
        now = time.time()
        if identifier not in self._buckets:
            self._buckets[identifier] = TokenBucket(
                tokens=self.config.burst_size,
                last_update=now,
                request_count=0,
                window_start=now
            )
        return self._buckets[identifier]
    
    def _refill_tokens(self, bucket: TokenBucket) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - bucket.last_update
        tokens_to_add = elapsed * self.config.requests_per_second
        bucket.tokens = min(self.config.burst_size, bucket.tokens + tokens_to_add)
        bucket.last_update = now
    
    def _check_window(self, bucket: TokenBucket) -> bool:
        """Check sliding window limit."""
        now = time.time()
        
        # Reset window if expired
        if now - bucket.window_start >= self.config.window_seconds:
            bucket.request_count = 0
            bucket.window_start = now
        
        return bucket.request_count < self.config.max_requests_per_window
    
    def check(self, identifier: str) -> bool:
        """
        Check if a request is allowed without consuming tokens.
        
        Args:
            identifier: Unique identifier (node_id, IP, etc.)
            
        Returns:
            True if request would be allowed
        """
        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_tokens(bucket)
            return bucket.tokens >= 1.0 and self._check_window(bucket)
    
    def acquire(self, identifier: str, tokens: int = 1) -> None:
        """
        Acquire tokens for a request.
        
        Args:
            identifier: Unique identifier
            tokens: Number of tokens to acquire
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_tokens(bucket)
            
            # Check token bucket
            if bucket.tokens < tokens:
                retry_after = (tokens - bucket.tokens) / self.config.requests_per_second
                raise RateLimitExceeded(
                    identifier,
                    int(self.config.requests_per_second),
                    1.0,
                    retry_after
                )
            
            # Check sliding window
            if not self._check_window(bucket):
                window_remaining = self.config.window_seconds - (time.time() - bucket.window_start)
                raise RateLimitExceeded(
                    identifier,
                    self.config.max_requests_per_window,
                    self.config.window_seconds,
                    window_remaining
                )
            
            # Consume tokens
            bucket.tokens -= tokens
            bucket.request_count += 1
    
    def reset(self, identifier: str) -> None:
        """Reset rate limit for an identifier."""
        with self._lock:
            if identifier in self._buckets:
                del self._buckets[identifier]
    
    def get_remaining(self, identifier: str) -> Dict[str, float]:
        """Get remaining capacity for an identifier."""
        with self._lock:
            bucket = self._get_or_create_bucket(identifier)
            self._refill_tokens(bucket)
            
            return {
                "tokens": bucket.tokens,
                "window_remaining": self.config.max_requests_per_window - bucket.request_count,
                "window_reset_in": max(0, self.config.window_seconds - (time.time() - bucket.window_start))
            }
    
    def cleanup_expired(self, max_age: float = 3600.0) -> int:
        """Remove expired buckets to free memory."""
        now = time.time()
        removed = 0
        
        with self._lock:
            expired = [
                k for k, v in self._buckets.items()
                if now - v.last_update > max_age
            ]
            for key in expired:
                del self._buckets[key]
                removed += 1
        
        return removed


# Pre-configured limiters for common use cases
gradient_rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_second=5.0,      # 5 gradient submissions/sec
    burst_size=10,                 # Allow burst of 10
    window_seconds=60.0,           # Per minute window
    max_requests_per_window=100    # Max 100 per minute
))

verification_rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_second=20.0,      # 20 verifications/sec
    burst_size=50,                 # Allow burst of 50
    window_seconds=60.0,
    max_requests_per_window=500
))

api_rate_limiter = RateLimiter(RateLimitConfig(
    requests_per_second=10.0,
    burst_size=30,
    window_seconds=60.0,
    max_requests_per_window=200
))

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for rate limiter module."""

import pytest
import time
from zerotrustml.security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitConfig,
)


class TestRateLimiter:
    """Tests for rate limiter."""
    
    def test_basic_acquire(self):
        """Basic token acquisition should work."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=10.0,
            burst_size=5,
            max_requests_per_window=100
        ))
        
        # Should succeed
        limiter.acquire("test-id")
    
    def test_burst_limit(self):
        """Burst limit should be enforced."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,  # Slow refill
            burst_size=3,
            max_requests_per_window=100
        ))
        
        # Use up burst
        for _ in range(3):
            limiter.acquire("test-id")
        
        # 4th should fail
        with pytest.raises(RateLimitExceeded):
            limiter.acquire("test-id")
    
    def test_window_limit(self):
        """Window limit should be enforced."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=100.0,  # Fast refill
            burst_size=100,
            window_seconds=1.0,
            max_requests_per_window=5
        ))
        
        # Use up window
        for _ in range(5):
            limiter.acquire("test-id")
        
        # 6th should fail
        with pytest.raises(RateLimitExceeded):
            limiter.acquire("test-id")
    
    def test_check_without_consume(self):
        """Check should not consume tokens."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,
            burst_size=2,
            max_requests_per_window=100
        ))
        
        # Check multiple times
        assert limiter.check("test-id")
        assert limiter.check("test-id")
        assert limiter.check("test-id")
        
        # Still should be able to acquire 2
        limiter.acquire("test-id")
        limiter.acquire("test-id")
    
    def test_separate_identifiers(self):
        """Different identifiers should have separate limits."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,
            burst_size=2,
            max_requests_per_window=100
        ))
        
        # Exhaust id1
        limiter.acquire("id1")
        limiter.acquire("id1")
        
        # id2 should still work
        limiter.acquire("id2")
        limiter.acquire("id2")
    
    def test_reset(self):
        """Reset should clear limits for identifier."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,
            burst_size=2,
            max_requests_per_window=100
        ))
        
        # Exhaust
        limiter.acquire("test-id")
        limiter.acquire("test-id")
        
        with pytest.raises(RateLimitExceeded):
            limiter.acquire("test-id")
        
        # Reset
        limiter.reset("test-id")
        
        # Should work again
        limiter.acquire("test-id")
    
    def test_get_remaining(self):
        """Should report remaining capacity."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=10.0,
            burst_size=5,
            max_requests_per_window=100
        ))
        
        remaining = limiter.get_remaining("test-id")
        assert remaining["tokens"] == 5
        assert remaining["window_remaining"] == 100
    
    def test_cleanup_expired(self):
        """Cleanup should remove old buckets."""
        limiter = RateLimiter()
        
        # Create some buckets
        limiter.acquire("id1")
        limiter.acquire("id2")
        
        # Artificially age the buckets
        for bucket in limiter._buckets.values():
            bucket.last_update -= 7200  # 2 hours ago
        
        # Cleanup with 1 hour max age
        removed = limiter.cleanup_expired(max_age=3600)
        assert removed == 2
    
    def test_rate_limit_exception_info(self):
        """Exception should contain useful info."""
        limiter = RateLimiter(RateLimitConfig(
            requests_per_second=1.0,
            burst_size=1,
            max_requests_per_window=100
        ))
        
        limiter.acquire("test-id")
        
        try:
            limiter.acquire("test-id")
            assert False, "Should have raised"
        except RateLimitExceeded as e:
            assert e.identifier == "test-id"
            assert e.retry_after > 0

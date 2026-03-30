# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for ZeroTrustML Credits System

Tests:
1. Event handler functionality
2. Reputation multipliers
3. Rate limiting
4. Economic validation
5. End-to-end integration with ZeroTrustML components
"""

import pytest
import pytest_asyncio
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml_credits_integration import (
    ZeroTrustMLCreditsIntegration,
    CreditEventType,
    ReputationLevel,
    CreditIssuanceConfig,
    RateLimiter
)
from holochain_credits_bridge import HolochainCreditsBridge


class TestEventHandlers:
    """Test each event handler independently"""

    @pytest_asyncio.fixture
    async def setup(self):
        """Setup integration with mock bridge"""
        bridge = HolochainCreditsBridge(enabled=False)  # Mock mode
        config = CreditIssuanceConfig(enabled=True)
        integration = ZeroTrustMLCreditsIntegration(bridge, config)
        return integration

    @pytest.mark.asyncio
    async def test_quality_gradient_credits(self, setup):
        """Test quality gradient credit issuance"""
        integration = setup

        # Issue credits for high-quality gradient
        result = await integration.on_quality_gradient(
            node_id="node_1",
            pogq_score=0.95,  # High quality
            reputation_level="NORMAL",
            verifiers=["node_2"]
        )

        assert result is not None  # Credits issued

        # Check audit trail
        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 1
        assert audit[0]["event_type"] == "quality_gradient"
        assert audit[0]["credits_issued"] == 95.0  # 0.95 * 100 * 1.0 (NORMAL multiplier)

    @pytest.mark.asyncio
    async def test_quality_gradient_with_reputation_multiplier(self, setup):
        """Test reputation multipliers work correctly"""
        integration = setup

        # ELITE reputation (1.5x multiplier)
        result_elite = await integration.on_quality_gradient(
            node_id="node_elite",
            pogq_score=1.0,
            reputation_level="ELITE",
            verifiers=["node_2"]
        )

        audit_elite = await integration.get_audit_trail("node_elite")
        assert audit_elite[0]["credits_issued"] == 150.0  # 1.0 * 100 * 1.5

        # BLACKLISTED reputation (0.0x multiplier)
        result_blacklisted = await integration.on_quality_gradient(
            node_id="node_blacklisted",
            pogq_score=1.0,
            reputation_level="BLACKLISTED",
            verifiers=["node_2"]
        )

        assert result_blacklisted is None  # No credits issued

    @pytest.mark.asyncio
    async def test_quality_gradient_below_threshold(self, setup):
        """Test minimum PoGQ threshold"""
        integration = setup

        # Below minimum threshold (0.7)
        result = await integration.on_quality_gradient(
            node_id="node_1",
            pogq_score=0.65,
            reputation_level="NORMAL",
            verifiers=["node_2"]
        )

        assert result is None  # No credits issued
        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 0  # No record

    @pytest.mark.asyncio
    async def test_byzantine_detection_credits(self, setup):
        """Test Byzantine detection reward"""
        integration = setup

        result = await integration.on_byzantine_detection(
            detector_node_id="node_1",
            detected_node_id="node_2",
            reputation_level="TRUSTED",
            evidence={"consecutive_failures": 15}
        )

        assert result is not None

        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 1
        assert audit[0]["event_type"] == "byzantine_detection"
        assert audit[0]["credits_issued"] == 60.0  # 50 * 1.2 (TRUSTED multiplier)

    @pytest.mark.asyncio
    async def test_peer_validation_credits(self, setup):
        """Test peer validation reward"""
        integration = setup

        result = await integration.on_peer_validation(
            validator_node_id="node_1",
            validated_node_id="node_2",
            reputation_level="NORMAL"
        )

        assert result is not None

        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 1
        assert audit[0]["event_type"] == "peer_validation"
        assert audit[0]["credits_issued"] == 10.0  # 10 * 1.0 (NORMAL multiplier)

    @pytest.mark.asyncio
    async def test_network_contribution_credits(self, setup):
        """Test network uptime reward"""
        integration = setup

        result = await integration.on_network_contribution(
            node_id="node_1",
            uptime_percentage=0.98,
            reputation_level="NORMAL",
            hours_online=1.0
        )

        assert result is not None

        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 1
        assert audit[0]["event_type"] == "network_contribution"
        assert audit[0]["credits_issued"] == 1.0  # 1 * 1.0 (NORMAL multiplier)

    @pytest.mark.asyncio
    async def test_network_contribution_below_threshold(self, setup):
        """Test minimum uptime threshold"""
        integration = setup

        # Below minimum (0.95)
        result = await integration.on_network_contribution(
            node_id="node_1",
            uptime_percentage=0.90,
            reputation_level="NORMAL",
            hours_online=1.0
        )

        assert result is None  # No credits issued


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter()

        # First request should pass
        allowed, reason = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            100.0,
            hourly_limit=1000
        )
        assert allowed is True

        # Add to tracking
        limiter.track_issuance("node_1", CreditEventType.QUALITY_GRADIENT, 100.0)

        # Second request should pass (under limit)
        allowed, reason = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            100.0,
            hourly_limit=1000
        )
        assert allowed is True

    def test_rate_limiter_exceeds_hourly(self):
        """Test hourly limit enforcement"""
        limiter = RateLimiter()

        # Issue many credits to exceed limit
        for i in range(100):
            limiter.track_issuance(
                "node_1",
                CreditEventType.QUALITY_GRADIENT,
                100.0
            )

        # Total: 10,000 credits issued
        # Now check if next issuance would exceed limit
        allowed, reason = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            100.0,
            hourly_limit=10000
        )
        assert allowed is False
        assert "hourly limit" in reason.lower()

    def test_rate_limiter_sliding_window(self):
        """Test sliding window behavior"""
        limiter = RateLimiter()
        import time

        # Issue credits
        limiter.track_issuance(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            5000.0
        )

        # Just under limit (5000 + 4999 = 9999 < 10000)
        allowed, _ = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            4999.0,
            hourly_limit=10000
        )
        assert allowed is True  # Just under limit

        # Over limit (5000 + 4999 + 2 = 10001 > 10000)
        allowed, reason = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            2.0,
            hourly_limit=10000
        )
        # Note: previous check_limit added 4999 to history, so total is now 9999

        allowed, reason = limiter.check_limit(
            "node_1",
            CreditEventType.QUALITY_GRADIENT,
            2.0,  # 9999 + 2 = 10001 > 10000 (exceeds limit)
            hourly_limit=10000
        )
        assert allowed is False  # Should be rejected


class TestEconomicValidation:
    """Test economic model correctness"""

    @pytest_asyncio.fixture
    async def setup(self):
        """Setup integration"""
        bridge = HolochainCreditsBridge(enabled=False)
        config = CreditIssuanceConfig(enabled=True)
        integration = ZeroTrustMLCreditsIntegration(bridge, config)
        return integration

    @pytest.mark.asyncio
    async def test_reputation_multipliers_all_levels(self, setup):
        """Test all reputation level multipliers"""
        integration = setup

        test_cases = [
            ("BLACKLISTED", 0.0),
            ("CRITICAL", 0.5),
            ("WARNING", 0.75),
            ("NORMAL", 1.0),
            ("TRUSTED", 1.2),
            ("ELITE", 1.5)
        ]

        for level, expected_multiplier in test_cases:
            result = await integration.on_quality_gradient(
                node_id=f"node_{level}",
                pogq_score=1.0,  # Perfect quality
                reputation_level=level,
                verifiers=["node_validator"]
            )

            if expected_multiplier == 0.0:
                assert result is None  # BLACKLISTED gets no credits
            else:
                audit = await integration.get_audit_trail(f"node_{level}")
                expected_credits = 100.0 * expected_multiplier
                assert audit[0]["credits_issued"] == expected_credits

    @pytest.mark.asyncio
    async def test_total_credits_cap(self, setup):
        """Test that total credits don't exceed daily caps"""
        integration = setup

        # Try to issue many quality gradient credits
        total_issued = 0
        for i in range(150):  # Try 150 high-quality gradients
            result = await integration.on_quality_gradient(
                node_id="node_1",
                pogq_score=1.0,
                reputation_level="NORMAL",
                verifiers=["node_2"]
            )
            if result:
                total_issued += 100.0

        # Should cap at hourly limit (10,000)
        assert total_issued <= 10000

    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, setup):
        """Test that all issuances are logged"""
        integration = setup

        # Issue various types of credits
        await integration.on_quality_gradient(
            "node_1", 0.9, "NORMAL", ["node_2"]
        )
        await integration.on_byzantine_detection(
            "node_1", "node_3", "TRUSTED", {}
        )
        await integration.on_peer_validation(
            "node_1", "node_2", "NORMAL"
        )
        await integration.on_network_contribution(
            "node_1", 0.97, "NORMAL", 1.0
        )

        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 4  # All 4 events recorded

        # Check each event type present
        event_types = {record["event_type"] for record in audit}
        assert event_types == {
            "quality_gradient",
            "byzantine_detection",
            "peer_validation",
            "network_contribution"
        }


class TestIntegrationStatistics:
    """Test statistics and reporting"""

    @pytest_asyncio.fixture
    async def setup(self):
        """Setup with some activity"""
        bridge = HolochainCreditsBridge(enabled=False)
        config = CreditIssuanceConfig(enabled=True)
        integration = ZeroTrustMLCreditsIntegration(bridge, config)

        # Generate activity
        await integration.on_quality_gradient(
            "node_1", 0.9, "NORMAL", ["node_2"]
        )
        await integration.on_byzantine_detection(
            "node_2", "node_3", "TRUSTED", {}
        )
        await integration.on_peer_validation(
            "node_1", "node_4", "NORMAL"
        )

        return integration

    @pytest.mark.asyncio
    async def test_integration_statistics(self, setup):
        """Test overall statistics"""
        integration = setup

        stats = integration.get_integration_stats()

        assert stats["total_nodes"] >= 2  # At least node_1 and node_2
        assert stats["total_events"] == 3
        assert stats["total_credits_issued"] > 0
        assert "credits_by_event_type" in stats
        assert stats["credits_by_event_type"]["quality_gradient"] > 0

    @pytest.mark.asyncio
    async def test_rate_limit_statistics(self, setup):
        """Test rate limit tracking stats"""
        integration = setup

        # Issue more to test rate limits
        for i in range(10):
            await integration.on_peer_validation(
                "node_heavy_user", f"node_{i}", "NORMAL"
            )

        stats = integration.get_rate_limit_stats("node_heavy_user")
        assert stats["total_issuances"] == 10
        assert CreditEventType.PEER_VALIDATION.value in stats["by_event_type"]


class TestDisabledIntegration:
    """Test behavior when integration is disabled"""

    @pytest.mark.asyncio
    async def test_disabled_integration_no_credits(self):
        """Test that no credits issued when disabled"""
        bridge = HolochainCreditsBridge(enabled=False)
        config = CreditIssuanceConfig(enabled=False)  # DISABLED
        integration = ZeroTrustMLCreditsIntegration(bridge, config)

        result = await integration.on_quality_gradient(
            "node_1", 0.9, "NORMAL", ["node_2"]
        )

        assert result is None  # No credits issued
        audit = await integration.get_audit_trail("node_1")
        assert len(audit) == 0  # No records


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
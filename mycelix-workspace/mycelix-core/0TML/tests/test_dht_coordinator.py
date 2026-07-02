#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit Tests for DHT_IdentityCoordinator

Tests the Python-side identity coordinator that bridges local identity
management with Holochain DHT infrastructure.

Week 5-6 Phase 7: End-to-End Testing
"""

import pytest
import asyncio
import time
from typing import Dict, List
from unittest.mock import Mock, AsyncMock, patch

# Import identity modules
try:
    from zerotrustml.identity import (
        DHT_IdentityCoordinator,
        IdentityCoordinatorConfig,
        AgentType,
        FactorStatus,
        FactorCategory
    )
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False


@pytest.fixture
def coordinator_config():
    """Fixture providing test configuration"""
    return IdentityCoordinatorConfig(
        dht_enabled=False,  # Disable DHT for unit tests
        auto_sync_to_dht=False,
        cache_dht_queries=True,
        cache_ttl=300,
        enable_local_fallback=True
    )


@pytest.fixture
def coordinator(coordinator_config):
    """Fixture providing DHT coordinator instance"""
    return DHT_IdentityCoordinator(config=coordinator_config)


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
class TestDHTIdentityCoordinator:
    """Unit tests for DHT_IdentityCoordinator"""

    # ========================================
    # Identity Creation Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_create_identity_basic(self, coordinator):
        """Test basic identity creation"""
        did = await coordinator.create_identity(
            participant_id="test_user",
            agent_type=AgentType.HUMAN_MEMBER
        )

        assert did.startswith("did:mycelix:")
        assert coordinator.has_identity("test_user")
        assert coordinator.get_did("test_user") == did

    @pytest.mark.asyncio
    async def test_create_identity_with_factors(self, coordinator):
        """Test identity creation with initial factors"""
        initial_factors = [
            {
                "factor_id": "crypto-key-test",
                "factor_type": "CryptoKey",
                "category": "PRIMARY",
                "status": "ACTIVE",
                "metadata": '{"algorithm": "Ed25519"}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
        ]

        did = await coordinator.create_identity(
            participant_id="test_user_factors",
            agent_type=AgentType.HUMAN_MEMBER,
            initial_factors=initial_factors
        )

        assert did is not None
        assert coordinator.has_identity("test_user_factors")

    @pytest.mark.asyncio
    async def test_create_duplicate_identity_fails(self, coordinator):
        """Test that creating duplicate identity fails"""
        await coordinator.create_identity(
            participant_id="duplicate_test",
            agent_type=AgentType.HUMAN_MEMBER
        )

        # Second attempt should fail
        with pytest.raises(ValueError, match="already exists"):
            await coordinator.create_identity(
                participant_id="duplicate_test",
                agent_type=AgentType.HUMAN_MEMBER
            )

    # ========================================
    # Identity Retrieval Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_get_identity_by_participant_id(self, coordinator):
        """Test retrieving identity by participant ID"""
        did = await coordinator.create_identity(
            participant_id="retrieve_test",
            agent_type=AgentType.HUMAN_MEMBER
        )

        identity = await coordinator.get_identity(participant_id="retrieve_test")
        assert identity is not None
        assert identity["did"] == did

    @pytest.mark.asyncio
    async def test_get_identity_by_did(self, coordinator):
        """Test retrieving identity by DID"""
        did = await coordinator.create_identity(
            participant_id="did_test",
            agent_type=AgentType.HUMAN_MEMBER
        )

        identity = await coordinator.get_identity(did=did)
        assert identity is not None
        assert identity["did"] == did

    @pytest.mark.asyncio
    async def test_get_nonexistent_identity(self, coordinator):
        """Test retrieving non-existent identity returns None"""
        identity = await coordinator.get_identity(participant_id="nonexistent")
        assert identity is None

    # ========================================
    # Caching Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_identity_caching(self, coordinator):
        """Test that identity queries are cached"""
        did = await coordinator.create_identity(
            participant_id="cache_test",
            agent_type=AgentType.HUMAN_MEMBER
        )

        # First query (uncached)
        start = time.time()
        identity1 = await coordinator.get_identity(participant_id="cache_test", use_cache=False)
        uncached_time = time.time() - start

        # Second query (cached)
        start = time.time()
        identity2 = await coordinator.get_identity(participant_id="cache_test", use_cache=True)
        cached_time = time.time() - start

        assert identity1 == identity2
        assert cached_time < uncached_time  # Cached should be faster

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, coordinator):
        """Test that cache respects TTL"""
        # Create coordinator with short TTL
        short_ttl_config = IdentityCoordinatorConfig(
            dht_enabled=False,
            cache_ttl=1  # 1 second TTL
        )
        ttl_coordinator = DHT_IdentityCoordinator(config=short_ttl_config)

        did = await ttl_coordinator.create_identity(
            participant_id="ttl_test",
            agent_type=AgentType.HUMAN_MEMBER
        )

        # Query to populate cache
        await ttl_coordinator.get_identity(participant_id="ttl_test")

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Should query again (not from cache)
        identity = await ttl_coordinator.get_identity(participant_id="ttl_test")
        assert identity is not None

    # ========================================
    # Guardian Network Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_add_guardian_basic(self, coordinator):
        """Test adding a guardian relationship"""
        # Create subject and guardian
        await coordinator.create_identity("subject", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian", AgentType.HUMAN_MEMBER)

        # Add guardian
        success = await coordinator.add_guardian(
            subject_participant_id="subject",
            guardian_participant_id="guardian",
            relationship_type="RECOVERY",
            weight=0.5
        )

        assert success

    @pytest.mark.asyncio
    async def test_add_guardian_invalid_weight(self, coordinator):
        """Test that invalid guardian weight fails"""
        await coordinator.create_identity("subject", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian", AgentType.HUMAN_MEMBER)

        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            await coordinator.add_guardian(
                subject_participant_id="subject",
                guardian_participant_id="guardian",
                relationship_type="RECOVERY",
                weight=1.5  # Invalid
            )

    @pytest.mark.asyncio
    async def test_guardian_weights_sum_validation(self, coordinator):
        """Test that guardian weights cannot exceed 1.0 total"""
        await coordinator.create_identity("subject", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian2", AgentType.HUMAN_MEMBER)

        # Add first guardian (0.8 weight)
        await coordinator.add_guardian(
            subject_participant_id="subject",
            guardian_participant_id="guardian1",
            relationship_type="RECOVERY",
            weight=0.8
        )

        # Adding second guardian with 0.3 weight should fail (total = 1.1)
        with pytest.raises(ValueError, match="Total guardian weight exceeds 1.0"):
            await coordinator.add_guardian(
                subject_participant_id="subject",
                guardian_participant_id="guardian2",
                relationship_type="RECOVERY",
                weight=0.3
            )

    @pytest.mark.asyncio
    async def test_authorize_recovery_success(self, coordinator):
        """Test successful recovery authorization"""
        # Setup: Subject with 3 guardians
        await coordinator.create_identity("alice", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("bob", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("carol", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("dave", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("alice", "bob", "RECOVERY", 0.4)
        await coordinator.add_guardian("alice", "carol", "RECOVERY", 0.3)
        await coordinator.add_guardian("alice", "dave", "RECOVERY", 0.3)

        # Test: Bob + Carol approve (0.7 total, threshold 0.6)
        result = await coordinator.authorize_recovery(
            subject_participant_id="alice",
            approving_guardian_ids=["bob", "carol"],
            required_threshold=0.6
        )

        assert result["authorized"]
        assert result["approval_weight"] == 0.7
        assert result["required_weight"] == 0.6

    @pytest.mark.asyncio
    async def test_authorize_recovery_failure(self, coordinator):
        """Test failed recovery authorization"""
        # Setup
        await coordinator.create_identity("alice", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("bob", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("carol", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("alice", "bob", "RECOVERY", 0.4)
        await coordinator.add_guardian("alice", "carol", "RECOVERY", 0.3)

        # Test: Only Carol approves (0.3 total, threshold 0.6)
        result = await coordinator.authorize_recovery(
            subject_participant_id="alice",
            approving_guardian_ids=["carol"],
            required_threshold=0.6
        )

        assert not result["authorized"]
        assert result["approval_weight"] == 0.3
        assert result["required_weight"] == 0.6
        assert "insufficient weight" in result["reason"].lower()

    # ========================================
    # FL Integration Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_verify_identity_for_fl_basic(self, coordinator):
        """Test basic FL identity verification"""
        await coordinator.create_identity("fl_user", AgentType.HUMAN_MEMBER)

        verification = await coordinator.verify_identity_for_fl(
            participant_id="fl_user",
            required_assurance="E0"  # Minimum assurance
        )

        assert "verified" in verification
        assert "assurance_level" in verification
        assert "sybil_resistance" in verification

    @pytest.mark.asyncio
    async def test_verify_identity_for_fl_with_factors(self, coordinator):
        """Test FL verification with multiple factors"""
        initial_factors = [
            {
                "factor_id": "crypto-key",
                "factor_type": "CryptoKey",
                "category": "PRIMARY",
                "status": "ACTIVE",
                "metadata": '{"algorithm": "Ed25519"}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            },
            {
                "factor_id": "gitcoin",
                "factor_type": "GitcoinPassport",
                "category": "REPUTATION",
                "status": "ACTIVE",
                "metadata": '{"score": 80.0}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
        ]

        await coordinator.create_identity(
            participant_id="fl_user_verified",
            agent_type=AgentType.HUMAN_MEMBER,
            initial_factors=initial_factors
        )

        verification = await coordinator.verify_identity_for_fl(
            participant_id="fl_user_verified",
            required_assurance="E1"
        )

        # With factors, should have higher Sybil resistance
        assert verification["sybil_resistance"] > 0.0

    @pytest.mark.asyncio
    async def test_sync_reputation_to_local(self, coordinator):
        """Test reputation synchronization (local fallback)"""
        await coordinator.create_identity("rep_test", AgentType.HUMAN_MEMBER)

        # Should not raise error even with DHT disabled
        await coordinator.sync_reputation_to_dht(
            participant_id="rep_test",
            reputation_score=0.85,
            score_type="trust",
            metadata={"rounds": 10}
        )

    # ========================================
    # Statistics Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_get_identity_statistics(self, coordinator):
        """Test retrieving identity statistics"""
        # Create several identities
        await coordinator.create_identity("user1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("user2", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("user3", AgentType.HUMAN_MEMBER)

        stats = await coordinator.get_identity_statistics()

        assert stats["total_participants"] == 3
        assert "dht_connected" in stats
        assert "cache_size" in stats

    # ========================================
    # Edge Cases and Error Handling
    # ========================================

    @pytest.mark.asyncio
    async def test_add_guardian_to_nonexistent_subject(self, coordinator):
        """Test adding guardian to non-existent subject fails"""
        await coordinator.create_identity("guardian", AgentType.HUMAN_MEMBER)

        with pytest.raises(ValueError, match="Subject identity not found"):
            await coordinator.add_guardian(
                subject_participant_id="nonexistent",
                guardian_participant_id="guardian",
                relationship_type="RECOVERY",
                weight=0.5
            )

    @pytest.mark.asyncio
    async def test_add_nonexistent_guardian(self, coordinator):
        """Test adding non-existent guardian fails"""
        await coordinator.create_identity("subject", AgentType.HUMAN_MEMBER)

        with pytest.raises(ValueError, match="Guardian identity not found"):
            await coordinator.add_guardian(
                subject_participant_id="subject",
                guardian_participant_id="nonexistent",
                relationship_type="RECOVERY",
                weight=0.5
            )

    @pytest.mark.asyncio
    async def test_self_guardian_prevention(self, coordinator):
        """Test that identity cannot be its own guardian"""
        await coordinator.create_identity("self_test", AgentType.HUMAN_MEMBER)

        with pytest.raises(ValueError, match="cannot be its own guardian"):
            await coordinator.add_guardian(
                subject_participant_id="self_test",
                guardian_participant_id="self_test",
                relationship_type="RECOVERY",
                weight=0.5
            )

    @pytest.mark.asyncio
    async def test_concurrent_identity_creation(self, coordinator):
        """Test concurrent identity creation"""
        # Create multiple identities concurrently
        tasks = [
            coordinator.create_identity(f"concurrent_{i}", AgentType.HUMAN_MEMBER)
            for i in range(10)
        ]

        dids = await asyncio.gather(*tasks)

        # All should succeed
        assert len(dids) == 10
        assert len(set(dids)) == 10  # All unique


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
class TestDHTCoordinatorIntegration:
    """Integration tests with mocked DHT"""

    @pytest.mark.asyncio
    async def test_dht_sync_on_creation(self, coordinator_config):
        """Test that identity syncs to DHT when enabled"""
        config = coordinator_config
        config.dht_enabled = True
        config.auto_sync_to_dht = True

        coordinator = DHT_IdentityCoordinator(config=config)

        # Mock DHT client
        coordinator.dht_client = AsyncMock()
        coordinator.dht_connected = True

        await coordinator.create_identity("sync_test", AgentType.HUMAN_MEMBER)

        # Verify DHT methods were called
        assert coordinator.dht_client.create_did.called

    @pytest.mark.asyncio
    async def test_fallback_when_dht_unavailable(self, coordinator_config):
        """Test graceful fallback when DHT is unavailable"""
        config = coordinator_config
        config.dht_enabled = True
        config.enable_local_fallback = True

        coordinator = DHT_IdentityCoordinator(config=config)

        # DHT connection fails
        coordinator.dht_connected = False

        # Should still work (local fallback)
        did = await coordinator.create_identity("fallback_test", AgentType.HUMAN_MEMBER)
        assert did is not None

        identity = await coordinator.get_identity(participant_id="fallback_test")
        assert identity is not None
        assert identity.get("source") == "local"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

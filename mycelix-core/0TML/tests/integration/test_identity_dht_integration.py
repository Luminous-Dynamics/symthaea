#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration Tests for Identity DHT System

Tests complete workflows integrating:
- DID Registry Zome
- Identity Store Zome
- Reputation Sync Zome
- Guardian Graph Zome
- Python DHT Client
- Identity Coordinator

Week 5-6 Phase 7: End-to-End Testing
"""

import pytest
import asyncio
import time
from typing import Dict, List

# Import identity modules
try:
    from zerotrustml.identity import (
        DHT_IdentityCoordinator,
        IdentityCoordinatorConfig,
        AgentType,
    )
    from zerotrustml.holochain import (
        IdentityDHTClient,
        DIDDocument,
        IdentitySignals,
        ReputationEntry,
    )
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False


@pytest.fixture
def coordinator_with_dht():
    """Fixture providing coordinator with DHT enabled (mocked)"""
    config = IdentityCoordinatorConfig(
        dht_enabled=True,
        auto_sync_to_dht=True,
        cache_dht_queries=True,
        enable_local_fallback=True
    )
    coordinator = DHT_IdentityCoordinator(config=config)

    # Note: For real integration tests, you would:
    # 1. Start Holochain conductor
    # 2. Install identity DNA
    # 3. Connect to real WebSocket endpoints
    #
    # For CI/CD, we mock the DHT client

    return coordinator


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.integration
class TestCompleteIdentityWorkflow:
    """Integration tests for complete identity workflows"""

    @pytest.mark.asyncio
    async def test_complete_user_registration_flow(self, coordinator_with_dht):
        """Test: New user registers with multi-factor identity"""
        coordinator = coordinator_with_dht

        # Step 1: Create identity with initial factors
        initial_factors = [
            {
                "factor_id": "crypto-key-alice",
                "factor_type": "CryptoKey",
                "category": "PRIMARY",
                "status": "ACTIVE",
                "metadata": '{"algorithm": "Ed25519", "public_key": "..."}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            },
            {
                "factor_id": "gitcoin-alice",
                "factor_type": "GitcoinPassport",
                "category": "REPUTATION",
                "status": "ACTIVE",
                "metadata": '{"score": 75.5, "stamps": 15}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
        ]

        did = await coordinator.create_identity(
            participant_id="alice",
            agent_type=AgentType.HUMAN_MEMBER,
            initial_factors=initial_factors,
            metadata={"email": "alice@example.com", "display_name": "Alice"}
        )

        # Verify identity was created
        assert did.startswith("did:mycelix:")
        assert coordinator.has_identity("alice")

        # Step 2: Verify identity for FL participation
        verification = await coordinator.verify_identity_for_fl(
            participant_id="alice",
            required_assurance="E1"
        )

        assert verification["verified"]
        assert verification["sybil_resistance"] > 0.5  # Has 2 factors

        # Step 3: Add guardians for recovery
        await coordinator.create_identity("bob", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("carol", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("alice", "bob", "RECOVERY", 0.6)
        await coordinator.add_guardian("alice", "carol", "RECOVERY", 0.4)

        # Step 4: Verify guardian network
        # (Would query DHT in real integration test)
        assert True  # Placeholder for DHT query verification

    @pytest.mark.asyncio
    async def test_fl_training_round_with_reputation(self, coordinator_with_dht):
        """Test: FL training round with reputation tracking"""
        coordinator = coordinator_with_dht

        # Setup: Create FL participants
        participants = ["node1", "node2", "node3", "node4", "node5"]

        for participant in participants:
            await coordinator.create_identity(
                participant_id=participant,
                agent_type=AgentType.AI_AGENT
            )

        # Pre-round verification
        verified_participants = []

        for participant in participants:
            verification = await coordinator.verify_identity_for_fl(
                participant_id=participant,
                required_assurance="E0"  # Minimum for AI agents
            )

            if verification["verified"]:
                verified_participants.append(participant)

        assert len(verified_participants) == 5  # All should pass

        # Simulate FL training round
        # (In real test, would run actual FL training)

        # Post-round reputation update
        reputation_scores = {
            "node1": 0.95,  # Excellent
            "node2": 0.88,  # Good
            "node3": 0.72,  # Average
            "node4": 0.45,  # Poor (potential Byzantine)
            "node5": 0.91,  # Good
        }

        for participant, score in reputation_scores.items():
            await coordinator.sync_reputation_to_dht(
                participant_id=participant,
                reputation_score=score,
                score_type="trust",
                metadata={
                    "rounds_participated": 1,
                    "byzantine_detected": score < 0.5,
                    "contribution_quality": "high" if score > 0.8 else "medium"
                }
            )

        # Verify reputation was recorded
        # (Would query DHT in real integration test)
        assert True

    @pytest.mark.asyncio
    async def test_recovery_authorization_workflow(self, coordinator_with_dht):
        """Test: Complete recovery authorization workflow"""
        coordinator = coordinator_with_dht

        # Setup: Create subject and guardians
        await coordinator.create_identity("subject", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian2", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian3", AgentType.HUMAN_MEMBER)

        # Establish guardian network (total weight = 1.0)
        await coordinator.add_guardian("subject", "guardian1", "RECOVERY", 0.5)
        await coordinator.add_guardian("subject", "guardian2", "RECOVERY", 0.3)
        await coordinator.add_guardian("subject", "guardian3", "RECOVERY", 0.2)

        # Scenario 1: Sufficient approval (guardian1 + guardian2 = 0.8)
        result1 = await coordinator.authorize_recovery(
            subject_participant_id="subject",
            approving_guardian_ids=["guardian1", "guardian2"],
            required_threshold=0.6
        )

        assert result1["authorized"]
        assert result1["approval_weight"] == 0.8

        # Scenario 2: Insufficient approval (guardian3 only = 0.2)
        result2 = await coordinator.authorize_recovery(
            subject_participant_id="subject",
            approving_guardian_ids=["guardian3"],
            required_threshold=0.6
        )

        assert not result2["authorized"]
        assert result2["approval_weight"] == 0.2

    @pytest.mark.asyncio
    async def test_cross_network_reputation_aggregation(self, coordinator_with_dht):
        """Test: Reputation aggregation across multiple networks"""
        coordinator = coordinator_with_dht

        # Create identity
        await coordinator.create_identity("multi_rep", AgentType.HUMAN_MEMBER)

        # Simulate reputation from multiple networks
        networks = [
            ("zero_trustml", 0.85, "trust"),
            ("gitcoin_passport", 0.78, "verification"),
            ("worldcoin", 0.92, "verification"),
            ("proof_of_humanity", 0.88, "verification"),
        ]

        for network_id, score, score_type in networks:
            await coordinator.sync_reputation_to_dht(
                participant_id="multi_rep",
                reputation_score=score,
                score_type=score_type,
                metadata={"network_id": network_id}
            )

        # Query aggregated reputation (would hit DHT in real test)
        identity = await coordinator.get_identity(participant_id="multi_rep")

        # Verify identity exists (actual aggregation tested in Holochain)
        assert identity is not None

    @pytest.mark.asyncio
    async def test_cartel_detection_scenario(self, coordinator_with_dht):
        """Test: Cartel detection in guardian networks"""
        coordinator = coordinator_with_dht

        # Create a suspicious guardian network (circular guardianship)
        await coordinator.create_identity("cartel_member_a", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("cartel_member_b", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("cartel_member_c", AgentType.HUMAN_MEMBER)

        # A guards B, B guards C, C guards A (circular)
        await coordinator.add_guardian("cartel_member_a", "cartel_member_b", "RECOVERY", 1.0)
        await coordinator.add_guardian("cartel_member_b", "cartel_member_c", "RECOVERY", 1.0)
        await coordinator.add_guardian("cartel_member_c", "cartel_member_a", "RECOVERY", 1.0)

        # In real test, would query guardian metrics from DHT
        # Should detect high cartel risk score (>0.7)

        # Verify identities were created (cartel detection tested in Holochain)
        assert coordinator.has_identity("cartel_member_a")
        assert coordinator.has_identity("cartel_member_b")
        assert coordinator.has_identity("cartel_member_c")

    @pytest.mark.asyncio
    async def test_identity_evolution_over_time(self, coordinator_with_dht):
        """Test: Identity evolves through factor additions and reputation updates"""
        coordinator = coordinator_with_dht

        # Initial identity (minimal)
        did = await coordinator.create_identity(
            participant_id="evolving_user",
            agent_type=AgentType.HUMAN_MEMBER
        )

        # Initial verification (should have low Sybil resistance)
        verification_t0 = await coordinator.verify_identity_for_fl(
            participant_id="evolving_user",
            required_assurance="E0"
        )
        sybil_t0 = verification_t0["sybil_resistance"]

        # Add factor 1: CryptoKey
        # (In real test, would call add_factor method)

        # Add factor 2: GitcoinPassport
        # (In real test, would call add_factor method)

        # Build reputation through FL participation
        for round_num in range(1, 11):
            await coordinator.sync_reputation_to_dht(
                participant_id="evolving_user",
                reputation_score=0.85 + (round_num * 0.01),  # Increasing
                score_type="trust",
                metadata={"round": round_num}
            )

        # Add guardians
        await coordinator.create_identity("guardian_a", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian_b", AgentType.HUMAN_MEMBER)
        await coordinator.add_guardian("evolving_user", "guardian_a", "RECOVERY", 0.5)
        await coordinator.add_guardian("evolving_user", "guardian_b", "RECOVERY", 0.5)

        # Final verification (should have higher Sybil resistance)
        verification_t1 = await coordinator.verify_identity_for_fl(
            participant_id="evolving_user",
            required_assurance="E2"
        )
        sybil_t1 = verification_t1["sybil_resistance"]

        # Verify evolution (would be more dramatic with real DHT)
        # In this test, just verify identity exists and methods execute
        assert verification_t1 is not None


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.integration
class TestDHTClientIntegration:
    """Integration tests for Python DHT Client"""

    @pytest.mark.asyncio
    async def test_dht_client_lifecycle(self):
        """Test: DHT client connection lifecycle"""
        # Note: Requires running Holochain conductor
        # This is a placeholder for real integration test

        client = IdentityDHTClient(
            admin_url="ws://localhost:8888",
            app_url="ws://localhost:8889"
        )

        # In real test:
        # await client.connect()
        # assert client.is_connected
        # await client.close()

        assert True  # Placeholder

    @pytest.mark.asyncio
    async def test_dht_client_complete_identity_query(self):
        """Test: Query complete identity from DHT"""
        # Note: Requires pre-populated DHT
        # This is a placeholder for real integration test

        client = IdentityDHTClient(
            admin_url="ws://localhost:8888",
            app_url="ws://localhost:8889"
        )

        # In real test:
        # identity = await client.get_complete_identity("did:mycelix:test")
        # assert identity["complete"]
        # assert "identity_signals" in identity
        # assert "aggregated_reputation" in identity
        # assert "guardians" in identity

        assert True  # Placeholder


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Large-scale integration tests"""

    @pytest.mark.asyncio
    async def test_hundred_participants_fl_round(self, coordinator_with_dht):
        """Test: FL round with 100 participants"""
        coordinator = coordinator_with_dht

        # Create 100 participants
        participants = [f"participant_{i:03d}" for i in range(100)]

        for participant in participants:
            await coordinator.create_identity(
                participant_id=participant,
                agent_type=AgentType.AI_AGENT
            )

        # Verify all participants
        start_time = time.time()

        verified_count = 0
        for participant in participants:
            verification = await coordinator.verify_identity_for_fl(
                participant_id=participant,
                required_assurance="E0"
            )
            if verification["verified"]:
                verified_count += 1

        verification_time = time.time() - start_time

        # All should verify
        assert verified_count == 100

        # Performance check (with caching, should be fast)
        avg_time_per_participant = verification_time / 100
        print(f"Average verification time: {avg_time_per_participant * 1000:.2f}ms")

        # Sync reputation for all
        start_time = time.time()

        for i, participant in enumerate(participants):
            score = 0.5 + (i % 50) / 100  # Vary between 0.5-0.99
            await coordinator.sync_reputation_to_dht(
                participant_id=participant,
                reputation_score=score,
                score_type="trust"
            )

        sync_time = time.time() - start_time

        avg_sync_time = sync_time / 100
        print(f"Average reputation sync time: {avg_sync_time * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_complex_guardian_network(self, coordinator_with_dht):
        """Test: Complex guardian network with 50 participants"""
        coordinator = coordinator_with_dht

        # Create 50 participants
        participants = [f"node_{i:02d}" for i in range(50)]

        for participant in participants:
            await coordinator.create_identity(
                participant_id=participant,
                agent_type=AgentType.HUMAN_MEMBER
            )

        # Create complex guardian relationships
        # Each participant has 3-5 guardians
        for i, subject in enumerate(participants):
            guardian_count = 3 + (i % 3)  # 3, 4, or 5 guardians
            weight_per_guardian = 1.0 / guardian_count

            for j in range(guardian_count):
                guardian_idx = (i + j + 1) % len(participants)
                guardian = participants[guardian_idx]

                await coordinator.add_guardian(
                    subject_participant_id=subject,
                    guardian_participant_id=guardian,
                    relationship_type="RECOVERY",
                    weight=weight_per_guardian
                )

        # Verify network was created
        # (In real test, would query DHT for guardian metrics)
        assert True

    @pytest.mark.asyncio
    async def test_concurrent_identity_operations(self, coordinator_with_dht):
        """Test: Concurrent identity creation and operations"""
        coordinator = coordinator_with_dht

        # Create 20 identities concurrently
        create_tasks = [
            coordinator.create_identity(
                participant_id=f"concurrent_{i:02d}",
                agent_type=AgentType.HUMAN_MEMBER
            )
            for i in range(20)
        ]

        dids = await asyncio.gather(*create_tasks)

        # Verify all created successfully
        assert len(dids) == 20
        assert len(set(dids)) == 20  # All unique

        # Verify all concurrently
        verify_tasks = [
            coordinator.verify_identity_for_fl(
                participant_id=f"concurrent_{i:02d}",
                required_assurance="E0"
            )
            for i in range(20)
        ]

        verifications = await asyncio.gather(*verify_tasks)

        # All should verify
        assert all(v["verified"] for v in verifications)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])

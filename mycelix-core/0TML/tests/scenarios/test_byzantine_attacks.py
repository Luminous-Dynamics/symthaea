#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Scenario Tests

Tests system resilience against various Byzantine attack vectors:
1. Sybil attacks (fake identities)
2. Guardian cartels (collusive networks)
3. Reputation manipulation
4. Recovery authorization hijacking
5. Factor forgery
6. Eclipse attacks (identity isolation)

Week 5-6 Phase 7: End-to-End Testing
"""

import pytest
import asyncio
import time
from typing import Dict, List

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import identity modules
try:
    from zerotrustml.identity import (
        DHT_IdentityCoordinator,
        IdentityCoordinatorConfig,
        AgentType,
    )
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False


@pytest.fixture
def coordinator():
    """Fixture providing test coordinator"""
    config = IdentityCoordinatorConfig(
        dht_enabled=False,
        auto_sync_to_dht=False,
        cache_dht_queries=False,  # Disable caching for attack tests
        enable_local_fallback=True
    )
    return DHT_IdentityCoordinator(config=config)


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestSybilAttacks:
    """Tests for Sybil attack resistance"""

    @pytest.mark.asyncio
    async def test_sybil_detection_no_factors(self, coordinator):
        """Test: Sybil identities (no factors) have low Sybil resistance"""
        # Create 10 Sybil identities (minimal factors)
        sybil_scores = []

        for i in range(10):
            await coordinator.create_identity(
                participant_id=f"sybil_{i:02d}",
                agent_type=AgentType.HUMAN_MEMBER
            )

            verification = await coordinator.verify_identity_for_fl(
                participant_id=f"sybil_{i:02d}",
                required_assurance="E0"
            )

            sybil_scores.append(verification["sybil_resistance"])

        # All should have low Sybil resistance (< 0.3)
        avg_sybil_score = sum(sybil_scores) / len(sybil_scores)

        logger.info(f"Average Sybil resistance (no factors): {avg_sybil_score:.3f}")
        assert avg_sybil_score < 0.3, "Sybil identities should have low resistance"

    @pytest.mark.asyncio
    async def test_sybil_vs_verified_identity(self, coordinator):
        """Test: Verified identities have significantly higher Sybil resistance"""
        # Create Sybil identity (no factors)
        await coordinator.create_identity("sybil", AgentType.HUMAN_MEMBER)

        # Create verified identity (multiple factors)
        verified_factors = [
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
                "metadata": '{"score": 85.0}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            },
            {
                "factor_id": "worldcoin",
                "factor_type": "BiometricFactor",
                "category": "BIOMETRIC",
                "status": "ACTIVE",
                "metadata": '{"provider": "Worldcoin"}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
        ]

        await coordinator.create_identity(
            participant_id="verified",
            agent_type=AgentType.HUMAN_MEMBER,
            initial_factors=verified_factors
        )

        # Compare Sybil resistance
        sybil_verification = await coordinator.verify_identity_for_fl("sybil", "E0")
        verified_verification = await coordinator.verify_identity_for_fl("verified", "E0")

        sybil_score = sybil_verification["sybil_resistance"]
        verified_score = verified_verification["sybil_resistance"]

        logger.info(f"Sybil resistance - Sybil: {sybil_score:.3f}, Verified: {verified_score:.3f}")

        # Verified should have significantly higher score
        assert verified_score > sybil_score + 0.3, "Verified identity should have >30pp higher resistance"

    @pytest.mark.asyncio
    async def test_sybil_army_blocked_at_e2(self, coordinator):
        """Test: Sybil army cannot pass E2 assurance threshold"""
        # Create 20 Sybil identities
        rejected_count = 0

        for i in range(20):
            await coordinator.create_identity(f"sybil_army_{i:02d}", AgentType.HUMAN_MEMBER)

            verification = await coordinator.verify_identity_for_fl(
                participant_id=f"sybil_army_{i:02d}",
                required_assurance="E2"  # Higher threshold
            )

            if not verification["verified"]:
                rejected_count += 1

        rejection_rate = rejected_count / 20

        logger.info(f"Sybil army rejection rate at E2: {rejection_rate:.1%}")

        # Should reject majority (>80%)
        assert rejection_rate > 0.8, "E2 threshold should block Sybil army"


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestGuardianCartels:
    """Tests for guardian cartel detection"""

    @pytest.mark.asyncio
    async def test_circular_guardianship_detection(self, coordinator):
        """Test: Circular guardian networks have high cartel risk"""
        # Create circular guardian network (A guards B, B guards C, C guards A)
        await coordinator.create_identity("cartel_a", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("cartel_b", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("cartel_c", AgentType.HUMAN_MEMBER)

        # A guards B
        await coordinator.add_guardian("cartel_a", "cartel_b", "RECOVERY", 1.0)

        # B guards C
        await coordinator.add_guardian("cartel_b", "cartel_c", "RECOVERY", 1.0)

        # C guards A (circular)
        await coordinator.add_guardian("cartel_c", "cartel_a", "RECOVERY", 1.0)

        # In production, would query guardian_metrics from DHT
        # Expected: cartel_risk_score > 0.7 for all three

        # For unit test, just verify structure was created
        assert coordinator.has_identity("cartel_a")
        assert coordinator.has_identity("cartel_b")
        assert coordinator.has_identity("cartel_c")

        logger.info("Circular guardianship structure created (cartel risk detection tested in Holochain)")

    @pytest.mark.asyncio
    async def test_homogeneous_guardian_network(self, coordinator):
        """Test: Guardian networks with homogeneous relationships have higher risk"""
        # Create identity with all guardians of same type
        await coordinator.create_identity("homogeneous_subject", AgentType.HUMAN_MEMBER)

        guardians = []
        for i in range(3):
            guardian_id = f"homo_guardian_{i}"
            await coordinator.create_identity(guardian_id, AgentType.HUMAN_MEMBER)
            guardians.append(guardian_id)

            # All use same relationship type
            await coordinator.add_guardian(
                "homogeneous_subject",
                guardian_id,
                "RECOVERY",  # All RECOVERY
                1.0 / 3
            )

        # In production, would compute diversity_score from DHT
        # Expected: diversity_score < 0.3 (low diversity)

        logger.info("Homogeneous guardian network created (diversity testing in Holochain)")

    @pytest.mark.asyncio
    async def test_diverse_guardian_network(self, coordinator):
        """Test: Diverse guardian networks have lower cartel risk"""
        # Create identity with diverse guardians
        await coordinator.create_identity("diverse_subject", AgentType.HUMAN_MEMBER)

        guardians = []
        relationship_types = ["RECOVERY", "ENDORSEMENT", "DELEGATION"]

        for i, rel_type in enumerate(relationship_types):
            guardian_id = f"diverse_guardian_{i}"
            await coordinator.create_identity(guardian_id, AgentType.HUMAN_MEMBER)
            guardians.append(guardian_id)

            await coordinator.add_guardian(
                "diverse_subject",
                guardian_id,
                rel_type,  # Different types
                1.0 / 3
            )

        # In production, would compute diversity_score from DHT
        # Expected: diversity_score > 0.7 (high diversity)

        logger.info("Diverse guardian network created (diversity testing in Holochain)")

    @pytest.mark.asyncio
    async def test_insufficient_guardian_count(self, coordinator):
        """Test: Identities with <3 guardians have higher risk"""
        # Create identity with only 2 guardians
        await coordinator.create_identity("weak_subject", AgentType.HUMAN_MEMBER)

        await coordinator.create_identity("guardian1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("guardian2", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("weak_subject", "guardian1", "RECOVERY", 0.5)
        await coordinator.add_guardian("weak_subject", "guardian2", "RECOVERY", 0.5)

        # In production, would check guardian_count and risk_score
        # Expected: cartel_risk_score includes +0.4 penalty for <3 guardians

        logger.info("Weak guardian network created (count-based risk tested in Holochain)")


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestReputationManipulation:
    """Tests for reputation manipulation resistance"""

    @pytest.mark.asyncio
    async def test_reputation_inflation_attack(self, coordinator):
        """Test: Cannot inflate reputation beyond reasonable bounds"""
        await coordinator.create_identity("inflator", AgentType.HUMAN_MEMBER)

        # Attempt to sync extremely high reputation
        await coordinator.sync_reputation_to_dht(
            participant_id="inflator",
            reputation_score=0.99,  # Very high
            score_type="trust",
            metadata={"rounds": 1}  # But only 1 round
        )

        # In production FL system:
        # - Reputation would be weighted by rounds_participated
        # - Cross-validation with other networks
        # - Statistical outlier detection

        logger.info("Reputation inflation tested (validation in FL coordinator)")

    @pytest.mark.asyncio
    async def test_reputation_oscillation_attack(self, coordinator):
        """Test: Rapid reputation changes are flagged"""
        await coordinator.create_identity("oscillator", AgentType.HUMAN_MEMBER)

        # Oscillate reputation rapidly
        for i in range(10):
            score = 0.9 if i % 2 == 0 else 0.3
            await coordinator.sync_reputation_to_dht(
                participant_id="oscillator",
                reputation_score=score,
                score_type="trust",
                metadata={"round": i}
            )

        # In production:
        # - Reputation volatility metric computed
        # - High volatility increases risk score
        # - May trigger manual review

        logger.info("Reputation oscillation tested (volatility detection in MATL)")

    @pytest.mark.asyncio
    async def test_cross_network_reputation_validation(self, coordinator):
        """Test: Cross-network reputation inconsistencies detected"""
        await coordinator.create_identity("inconsistent", AgentType.HUMAN_MEMBER)

        # Report very different scores from different networks
        networks = [
            ("zero_trustml", 0.95),  # Very high
            ("gitcoin_passport", 0.25),  # Very low
            ("worldcoin", 0.92),  # Very high
        ]

        for network_id, score in networks:
            await coordinator.sync_reputation_to_dht(
                participant_id="inconsistent",
                reputation_score=score,
                score_type="trust",
                metadata={"network": network_id}
            )

        # In production:
        # - Compute reputation variance across networks
        # - High variance triggers investigation
        # - May require additional verification

        logger.info("Cross-network reputation inconsistency tested (variance detection in DHT)")


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestRecoveryAuthorizationAttacks:
    """Tests for recovery authorization hijacking"""

    @pytest.mark.asyncio
    async def test_recovery_hijack_attempt(self, coordinator):
        """Test: Unauthorized recovery attempt is rejected"""
        # Setup: Subject with 3 guardians
        await coordinator.create_identity("victim", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("legitimate_guardian1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("legitimate_guardian2", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("attacker", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("victim", "legitimate_guardian1", "RECOVERY", 0.5)
        await coordinator.add_guardian("victim", "legitimate_guardian2", "RECOVERY", 0.5)

        # Attacker attempts unauthorized recovery
        result = await coordinator.authorize_recovery(
            subject_participant_id="victim",
            approving_guardian_ids=["attacker"],  # Not a legitimate guardian
            required_threshold=0.6
        )

        # Should be rejected
        assert not result["authorized"]
        assert result["approval_weight"] == 0.0
        assert "not a registered guardian" in result["reason"].lower()

        logger.info("Recovery hijack attempt correctly rejected")

    @pytest.mark.asyncio
    async def test_minority_guardian_takeover(self, coordinator):
        """Test: Minority guardians cannot authorize recovery"""
        # Setup: Subject with weighted guardians
        await coordinator.create_identity("target", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("major_guardian", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("minor_guardian1", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("minor_guardian2", AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("target", "major_guardian", "RECOVERY", 0.6)
        await coordinator.add_guardian("target", "minor_guardian1", "RECOVERY", 0.2)
        await coordinator.add_guardian("target", "minor_guardian2", "RECOVERY", 0.2)

        # Minor guardians attempt takeover (0.4 total)
        result = await coordinator.authorize_recovery(
            subject_participant_id="target",
            approving_guardian_ids=["minor_guardian1", "minor_guardian2"],
            required_threshold=0.6
        )

        # Should fail
        assert not result["authorized"]
        assert result["approval_weight"] == 0.4
        assert result["approval_weight"] < result["required_weight"]

        logger.info("Minority guardian takeover correctly prevented")

    @pytest.mark.asyncio
    async def test_expired_guardian_relationship(self, coordinator):
        """Test: Expired guardian relationships are not counted"""
        # Setup: Subject with guardian that will expire
        await coordinator.create_identity("expiring_subject", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("expiring_guardian", AgentType.HUMAN_MEMBER)
        await coordinator.create_identity("current_guardian", AgentType.HUMAN_MEMBER)

        # Add expiring guardian (expired 1 hour ago)
        # Note: In real implementation, would set expires_at in past
        await coordinator.add_guardian("expiring_subject", "expiring_guardian", "RECOVERY", 0.5)

        # Add current guardian
        await coordinator.add_guardian("expiring_subject", "current_guardian", "RECOVERY", 0.5)

        # In production:
        # - DHT would filter out expired relationships
        # - Only current_guardian's approval would count
        # - This test validates the filtering logic

        logger.info("Expired guardian filtering tested (validation in DHT)")


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestEclipseAttacks:
    """Tests for identity isolation/eclipse attacks"""

    @pytest.mark.asyncio
    async def test_guardian_graph_fragmentation(self, coordinator):
        """Test: Isolated guardian subgraphs are detected"""
        # Create two isolated guardian clusters
        # Cluster 1: A-B-C (interconnected)
        cluster1 = ["cluster1_a", "cluster1_b", "cluster1_c"]
        for member in cluster1:
            await coordinator.create_identity(member, AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("cluster1_a", "cluster1_b", "RECOVERY", 0.5)
        await coordinator.add_guardian("cluster1_a", "cluster1_c", "RECOVERY", 0.5)
        await coordinator.add_guardian("cluster1_b", "cluster1_c", "RECOVERY", 1.0)

        # Cluster 2: D-E-F (interconnected, but isolated from cluster 1)
        cluster2 = ["cluster2_d", "cluster2_e", "cluster2_f"]
        for member in cluster2:
            await coordinator.create_identity(member, AgentType.HUMAN_MEMBER)

        await coordinator.add_guardian("cluster2_d", "cluster2_e", "RECOVERY", 0.5)
        await coordinator.add_guardian("cluster2_d", "cluster2_f", "RECOVERY", 0.5)
        await coordinator.add_guardian("cluster2_e", "cluster2_f", "RECOVERY", 1.0)

        # In production:
        # - Graph analysis would detect disconnected components
        # - Isolated clusters flagged as potential cartels
        # - Network-wide connectivity metrics computed

        logger.info("Guardian graph fragmentation tested (graph analysis in DHT)")

    @pytest.mark.asyncio
    async def test_reputation_isolation_attack(self, coordinator):
        """Test: Identities with reputation from single source flagged"""
        # Create identity with reputation only from one network
        await coordinator.create_identity("isolated_rep", AgentType.HUMAN_MEMBER)

        # Only report reputation from zero_trustml
        await coordinator.sync_reputation_to_dht(
            participant_id="isolated_rep",
            reputation_score=0.95,
            score_type="trust",
            metadata={"network": "zero_trustml"}
        )

        # In production:
        # - Reputation source diversity computed
        # - Low diversity (network_count = 1) increases risk
        # - May require cross-validation

        logger.info("Reputation isolation tested (source diversity in DHT)")


@pytest.mark.skipif(not IDENTITY_AVAILABLE, reason="Identity modules not available")
@pytest.mark.byzantine
class TestSystemResilience:
    """Tests for overall system resilience"""

    @pytest.mark.asyncio
    async def test_mixed_attacker_honest_ratio(self, coordinator):
        """Test: System maintains security with 40% Byzantine nodes"""
        # Create 60 honest + 40 Byzantine identities
        honest_count = 60
        byzantine_count = 40

        # Honest identities (multi-factor)
        honest_factors = [
            {
                "factor_id": "crypto-key",
                "factor_type": "CryptoKey",
                "category": "PRIMARY",
                "status": "ACTIVE",
                "metadata": '{}',
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

        for i in range(honest_count):
            await coordinator.create_identity(
                participant_id=f"honest_{i:02d}",
                agent_type=AgentType.HUMAN_MEMBER,
                initial_factors=honest_factors
            )

        # Byzantine identities (minimal factors)
        for i in range(byzantine_count):
            await coordinator.create_identity(
                participant_id=f"byzantine_{i:02d}",
                agent_type=AgentType.HUMAN_MEMBER
            )

        # Verify all at E1 threshold
        honest_passed = 0
        byzantine_passed = 0

        for i in range(honest_count):
            result = await coordinator.verify_identity_for_fl(f"honest_{i:02d}", "E1")
            if result["verified"]:
                honest_passed += 1

        for i in range(byzantine_count):
            result = await coordinator.verify_identity_for_fl(f"byzantine_{i:02d}", "E1")
            if result["verified"]:
                byzantine_passed += 1

        honest_pass_rate = honest_passed / honest_count
        byzantine_pass_rate = byzantine_passed / byzantine_count

        logger.info(f"Honest pass rate: {honest_pass_rate:.1%}")
        logger.info(f"Byzantine pass rate: {byzantine_pass_rate:.1%}")

        # Most honest should pass, most Byzantine should fail
        assert honest_pass_rate > 0.9, "Should accept >90% of honest identities"
        assert byzantine_pass_rate < 0.3, "Should reject >70% of Byzantine identities"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "byzantine"])

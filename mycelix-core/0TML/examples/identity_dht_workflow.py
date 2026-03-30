#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Complete Identity DHT Workflow Example

Demonstrates end-to-end usage of the DHT-aware identity system:
1. Identity creation and registration
2. Multi-factor identity storage
3. Reputation synchronization
4. Guardian network setup
5. Identity verification for FL
6. Recovery authorization

Week 5-6 Phase 7: End-to-End Testing
"""

import asyncio
import logging
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import identity modules
try:
    from zerotrustml.identity import (
        DHT_IdentityCoordinator,
        IdentityCoordinatorConfig,
        CryptoKeyFactor,
        GitcoinPassportFactor,
        SocialRecoveryFactor,
        HardwareKeyFactor,
        FactorStatus,
        FactorCategory,
        AgentType
    )
    IDENTITY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Identity modules not available: {e}")
    IDENTITY_AVAILABLE = False


async def example_complete_workflow():
    """
    Complete workflow demonstrating all DHT identity features
    """
    if not IDENTITY_AVAILABLE:
        logger.error("Identity modules not available - cannot run example")
        return

    logger.info("=" * 80)
    logger.info("DHT Identity Workflow Example")
    logger.info("=" * 80)

    # ========================================
    # 1. Initialize Coordinator
    # ========================================
    logger.info("\n[1] Initializing DHT-Aware Identity Coordinator...")

    config = IdentityCoordinatorConfig(
        dht_enabled=True,
        auto_sync_to_dht=True,
        cache_dht_queries=True,
        cache_ttl=600,
        enable_local_fallback=True
    )

    coordinator = DHT_IdentityCoordinator(config=config)

    # Try to connect to DHT (will fallback to local if unavailable)
    dht_connected = await coordinator.initialize_dht()

    if dht_connected:
        logger.info("✅ DHT connection established")
    else:
        logger.warning("⚠️  DHT unavailable - running in local-only mode")

    # ========================================
    # 2. Create Identities
    # ========================================
    logger.info("\n[2] Creating participant identities...")

    participants = ["alice", "bob", "carol", "dave"]
    participant_dids: Dict[str, str] = {}

    for participant in participants:
        # Create identity factors
        initial_factors = [
            # CryptoKey factor (PRIMARY)
            {
                "factor_id": f"crypto-key-{participant}",
                "factor_type": "CryptoKey",
                "category": "PRIMARY",
                "status": "ACTIVE",
                "metadata": '{"algorithm": "Ed25519"}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            }
        ]

        # Add additional factors for some participants
        if participant in ["alice", "bob"]:
            initial_factors.append({
                "factor_id": f"gitcoin-{participant}",
                "factor_type": "GitcoinPassport",
                "category": "REPUTATION",
                "status": "ACTIVE",
                "metadata": '{"score": 75.5}',
                "added": int(time.time() * 1_000_000),
                "last_verified": int(time.time() * 1_000_000)
            })

        try:
            did = await coordinator.create_identity(
                participant_id=participant,
                agent_type=AgentType.HUMAN_MEMBER,
                initial_factors=initial_factors
            )

            participant_dids[participant] = did
            logger.info(f"✅ Created identity: {participant} -> {did[:40]}...")

        except Exception as e:
            logger.error(f"❌ Failed to create identity for {participant}: {e}")

    # ========================================
    # 3. Verify Identities for FL
    # ========================================
    logger.info("\n[3] Verifying identities for FL participation...")

    for participant in participants:
        try:
            verification = await coordinator.verify_identity_for_fl(
                participant_id=participant,
                required_assurance="E1"  # Minimum E1 for this example
            )

            if verification["verified"]:
                logger.info(
                    f"✅ {participant}: VERIFIED "
                    f"(Assurance: {verification.get('assurance_level', 'N/A')}, "
                    f"Sybil: {verification.get('sybil_resistance', 0.0):.2f})"
                )
            else:
                logger.warning(
                    f"❌ {participant}: REJECTED - {verification.get('reasons', ['Unknown'])}"
                )

        except Exception as e:
            logger.error(f"Error verifying {participant}: {e}")

    # ========================================
    # 4. Simulate FL Round and Reputation Sync
    # ========================================
    logger.info("\n[4] Simulating FL round and reputation sync...")

    # Simulated reputation scores after FL round
    reputation_scores = {
        "alice": 0.85,  # High performer
        "bob": 0.92,    # Excellent performer
        "carol": 0.68,  # Average performer
        "dave": 0.45    # Low performer (potential Byzantine)
    }

    for participant, score in reputation_scores.items():
        try:
            await coordinator.sync_reputation_to_dht(
                participant_id=participant,
                reputation_score=score,
                score_type="trust",
                metadata={"rounds_participated": 10, "byzantine_detected": score < 0.5}
            )

            logger.info(f"✅ Reputation synced: {participant} -> {score:.2f}")

        except Exception as e:
            logger.error(f"Error syncing reputation for {participant}: {e}")

    # ========================================
    # 5. Setup Guardian Networks
    # ========================================
    logger.info("\n[5] Setting up guardian networks...")

    # Alice's guardians: Bob (40%), Carol (30%), Dave (30%)
    alice_guardians = [
        ("bob", 0.4),
        ("carol", 0.3),
        ("dave", 0.3)
    ]

    for guardian, weight in alice_guardians:
        try:
            success = await coordinator.add_guardian(
                subject_participant_id="alice",
                guardian_participant_id=guardian,
                relationship_type="RECOVERY",
                weight=weight
            )

            if success:
                logger.info(f"✅ Guardian added: {guardian} guards alice (weight: {weight})")
            else:
                logger.warning(f"⚠️  Failed to add guardian: {guardian}")

        except Exception as e:
            logger.error(f"Error adding guardian {guardian}: {e}")

    # ========================================
    # 6. Test Recovery Authorization
    # ========================================
    logger.info("\n[6] Testing recovery authorization...")

    # Scenario 1: Bob + Carol approve (70% weight) - SHOULD PASS (threshold 60%)
    logger.info("Scenario 1: Bob + Carol approve recovery (70% weight)")
    try:
        result = await coordinator.authorize_recovery(
            subject_participant_id="alice",
            approving_guardian_ids=["bob", "carol"],
            required_threshold=0.6
        )

        if result["authorized"]:
            logger.info(
                f"✅ Recovery AUTHORIZED: "
                f"{result['approval_weight']:.2f} / {result['required_weight']:.2f}"
            )
        else:
            logger.warning(f"❌ Recovery DENIED: {result['reason']}")

    except Exception as e:
        logger.error(f"Error in recovery authorization: {e}")

    # Scenario 2: Only Dave approves (30% weight) - SHOULD FAIL
    logger.info("\nScenario 2: Only Dave approves recovery (30% weight)")
    try:
        result = await coordinator.authorize_recovery(
            subject_participant_id="alice",
            approving_guardian_ids=["dave"],
            required_threshold=0.6
        )

        if result["authorized"]:
            logger.warning("⚠️  Recovery AUTHORIZED (unexpected)")
        else:
            logger.info(
                f"✅ Recovery correctly DENIED: "
                f"{result['approval_weight']:.2f} / {result['required_weight']:.2f}"
            )

    except Exception as e:
        logger.error(f"Error in recovery authorization: {e}")

    # ========================================
    # 7. Query Complete Identities
    # ========================================
    logger.info("\n[7] Querying complete identity profiles...")

    for participant in ["alice", "bob"]:
        try:
            identity = await coordinator.get_identity(participant_id=participant)

            if identity and identity.get("complete"):
                signals = identity.get("identity_signals")
                reputation = identity.get("aggregated_reputation")
                guardians = identity.get("guardians", [])

                logger.info(f"\n{participant.upper()} Identity Profile:")
                logger.info(f"  DID: {identity['did'][:50]}...")

                if signals:
                    logger.info(f"  Assurance Level: {signals.assurance_level}")
                    logger.info(f"  Sybil Resistance: {signals.sybil_resistance:.2f}")
                    logger.info(f"  Risk Level: {signals.risk_level}")
                    logger.info(f"  Verified Human: {signals.verified_human}")

                if reputation:
                    logger.info(f"  Global Reputation: {reputation.global_score:.2f}")
                    logger.info(f"  Trust Score: {reputation.trust_score:.2f}")
                    logger.info(f"  Networks: {reputation.network_count}")

                logger.info(f"  Guardians: {len(guardians)}")

            else:
                logger.warning(f"Incomplete identity data for {participant}")

        except Exception as e:
            logger.error(f"Error querying identity for {participant}: {e}")

    # ========================================
    # 8. Identity Statistics
    # ========================================
    logger.info("\n[8] Identity system statistics...")

    try:
        stats = await coordinator.get_identity_statistics()

        logger.info(f"\nSystem Statistics:")
        logger.info(f"  Total Participants: {stats['total_participants']}")
        logger.info(f"  DHT Connected: {stats['dht_connected']}")
        logger.info(f"  Cache Size: {stats['cache_size']}")

        if stats.get('participants'):
            logger.info(f"\n  Participant Details:")
            for participant in stats['participants']:
                logger.info(
                    f"    {participant['participant_id']}: "
                    f"{participant['assurance_level']} "
                    f"(Sybil: {participant['sybil_resistance']:.2f})"
                )

    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")

    # ========================================
    # 9. Cache Performance Test
    # ========================================
    logger.info("\n[9] Testing cache performance...")

    try:
        # First query (uncached)
        start = time.time()
        identity1 = await coordinator.get_identity("alice", use_cache=False)
        uncached_time = time.time() - start

        # Second query (cached)
        start = time.time()
        identity2 = await coordinator.get_identity("alice", use_cache=True)
        cached_time = time.time() - start

        logger.info(f"  Uncached query: {uncached_time * 1000:.2f}ms")
        logger.info(f"  Cached query: {cached_time * 1000:.2f}ms")
        logger.info(f"  Speedup: {uncached_time / cached_time:.1f}x")

    except Exception as e:
        logger.error(f"Error in cache performance test: {e}")

    # ========================================
    # Cleanup
    # ========================================
    logger.info("\n[10] Cleaning up...")

    await coordinator.close_dht()
    logger.info("✅ DHT connection closed")

    logger.info("\n" + "=" * 80)
    logger.info("Workflow Complete!")
    logger.info("=" * 80)


async def example_fl_integration():
    """
    Example demonstrating FL coordinator integration with DHT identity
    """
    logger.info("\n" + "=" * 80)
    logger.info("FL Integration Example")
    logger.info("=" * 80)

    # Initialize coordinator
    coordinator = DHT_IdentityCoordinator()
    await coordinator.initialize_dht()

    # Simulated FL participants
    participants = ["node1", "node2", "node3", "node4", "node5"]

    # Create identities
    logger.info("\n[1] Registering FL participants...")
    for participant in participants:
        await coordinator.create_identity(participant)
        logger.info(f"✅ Registered: {participant}")

    # Pre-round verification
    logger.info("\n[2] Pre-round identity verification...")
    verified_participants = []

    for participant in participants:
        verification = await coordinator.verify_identity_for_fl(participant, "E1")

        if verification["verified"]:
            verified_participants.append(participant)
            logger.info(f"✅ {participant}: VERIFIED")
        else:
            logger.warning(f"❌ {participant}: REJECTED")

    logger.info(f"\nVerified: {len(verified_participants)}/{len(participants)} participants")

    # Simulate FL round
    logger.info("\n[3] Simulating FL round...")
    logger.info("(Training happens here...)")

    # Post-round reputation update
    logger.info("\n[4] Updating reputation scores...")
    reputation_scores = {p: 0.75 + (hash(p) % 20) / 100 for p in verified_participants}

    await coordinator.sync_all_reputations(reputation_scores)
    logger.info(f"✅ Synced {len(reputation_scores)} reputation scores")

    # Cleanup
    await coordinator.close_dht()
    logger.info("\n✅ FL integration example complete")


async def main():
    """Run all examples"""
    try:
        # Complete workflow
        await example_complete_workflow()

        # FL integration
        await example_fl_integration()

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"\n\nError in workflow: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

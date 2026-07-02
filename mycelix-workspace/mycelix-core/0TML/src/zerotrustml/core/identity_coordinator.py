#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity-Enhanced Zero-TrustML Coordinator
Integrates Multi-Factor Decentralized Identity with Byzantine-Resistant FL

Week 3-4 Implementation: Identity → MATL Trust Score Enhancement
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Import base coordinator
from .phase10_coordinator import Phase10Coordinator, Phase10Config

# Import identity system
from zerotrustml.identity import (
    IdentityMATLBridge,
    IdentityTrustSignal,
    IdentityFactor,
    VerifiableCredential,
    AssuranceLevel,
)

logger = logging.getLogger(__name__)


class IdentityCoordinator(Phase10Coordinator):
    """
    Identity-Enhanced FL Coordinator

    Extends Phase 10 Coordinator with:
    1. Multi-factor decentralized identity verification
    2. Identity-enhanced MATL trust scoring
    3. Sybil resistance through assurance levels
    4. Guardian network diversity analysis

    Byzantine Tolerance: 45% (vs 33% classical BFT) through identity-weighted reputation
    """

    def __init__(self, config: Phase10Config):
        """Initialize identity-enhanced coordinator"""
        super().__init__(config)

        # Identity integration
        self.identity_bridge = IdentityMATLBridge()
        self.node_identities: Dict[str, IdentityTrustSignal] = {}
        self.node_did_mapping: Dict[str, str] = {}  # node_id → DID

        # Enhanced metrics
        self.identity_metrics = {
            "nodes_registered": 0,
            "e0_nodes": 0,  # Anonymous
            "e1_nodes": 0,  # Testimonial
            "e2_nodes": 0,  # Privately Verifiable
            "e3_nodes": 0,  # Cryptographically Proven
            "e4_nodes": 0,  # Constitutionally Critical
            "sybil_attacks_detected": 0,
            "cartel_warnings": 0,
            "identity_boost_applied": 0,
        }

        logger.info("✅ Identity-Enhanced Coordinator initialized")

    async def register_node_identity(
        self,
        node_id: str,
        did: str,
        factors: List[IdentityFactor],
        credentials: List[VerifiableCredential],
        guardian_graph: Optional[Dict] = None
    ) -> IdentityTrustSignal:
        """
        Register node identity and compute trust signals

        This is the entry point for nodes to establish their identity
        on the federated learning network.

        Args:
            node_id: Node identifier (can be different from DID)
            did: W3C Decentralized Identifier
            factors: List of identity verification factors
            credentials: List of verifiable credentials
            guardian_graph: Optional guardian network graph for cartel detection

        Returns:
            Computed identity trust signal
        """
        logger.info(f"Registering node identity: {node_id} (DID: {did})")

        # Compute identity signals
        signals = self.identity_bridge.compute_identity_signals(
            did=did,
            factors=factors,
            credentials=credentials,
            guardian_graph=guardian_graph
        )

        # Store in coordinator memory
        self.node_identities[node_id] = signals
        self.node_did_mapping[node_id] = did

        # Store in backends
        try:
            await self._store_with_strategy(
                "store_identity",
                {
                    "node_id": node_id,
                    "did": did,
                    "identity_signals": signals.to_dict(),
                    "registered_at": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Could not store identity in backend: {e}")
            # Continue anyway - we have in-memory copy

        # Set initial reputation based on identity
        initial_rep = self.identity_bridge.get_initial_reputation(did)

        try:
            await self._store_with_strategy(
                "store_reputation",
                {
                    "node_id": node_id,
                    "score": initial_rep,
                    "source": "identity_verified",
                    "assurance_level": signals.assurance_level.value,
                    "sybil_resistance": signals.sybil_resistance_score,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Could not store initial reputation: {e}")

        # Update metrics
        self.identity_metrics["nodes_registered"] += 1
        self._update_assurance_metrics(signals.assurance_level)

        # Check for potential issues
        if signals.risk_level.value in ["Critical", "High"]:
            logger.warning(
                f"Node {node_id}: High risk identity - "
                f"{signals.assurance_level.value}, "
                f"Sybil resistance: {signals.sybil_resistance_score:.2f}"
            )

        if guardian_graph and signals.guardian_graph_diversity < 0.3:
            self.identity_metrics["cartel_warnings"] += 1
            logger.warning(
                f"Node {node_id}: Low guardian diversity ({signals.guardian_graph_diversity:.2f}) "
                f"- potential cartel"
            )

        logger.info(
            f"✅ Node {node_id} registered: "
            f"{signals.assurance_level.value}, "
            f"initial reputation: {initial_rep:.2f}, "
            f"Sybil resistance: {signals.sybil_resistance_score:.2f}"
        )

        return signals

    async def _get_reputation(self, node_id: str) -> float:
        """
        Get node reputation with identity enhancement

        Overrides base method to apply identity boost.

        Flow:
        1. Get identity signals for node
        2. Try to get base reputation from storage (gradient history)
        3. If no meaningful gradient history exists, use identity-based initial reputation
        4. Calculate identity boost
        5. Apply boost (clamped to [0.3, 1.0])
        6. Return enhanced score

        Args:
            node_id: Node to get reputation for

        Returns:
            Identity-enhanced reputation score (0.3-1.0)
        """
        # Get identity signals first
        identity_signals = self.node_identities.get(node_id)

        if not identity_signals:
            # No identity verification - use base only
            base_reputation = 0.5  # Default neutral
            logger.warning(
                f"Node {node_id}: No identity verification, "
                f"using default reputation: {base_reputation:.2f}"
            )
            return base_reputation

        # Try to get base reputation from storage (gradient history)
        base_rep_data = None
        try:
            base_rep_data = await self._get_with_strategy("get_reputation", node_id)
        except Exception as e:
            logger.debug(f"Could not get reputation from backend: {e}")

        # Check if we have meaningful gradient history
        has_gradient_history = (
            base_rep_data and
            "gradients_submitted" in base_rep_data and
            base_rep_data["gradients_submitted"] > 0
        )

        if has_gradient_history:
            # Use gradient-based reputation as base
            base_reputation = base_rep_data["score"]
            logger.debug(
                f"Node {node_id}: Using gradient-based reputation: {base_reputation:.2f} "
                f"({base_rep_data['gradients_submitted']} gradients)"
            )
        else:
            # No gradient history yet - use identity-based initial reputation
            did = self.node_did_mapping.get(node_id)
            if did:
                base_reputation = self.identity_bridge.get_initial_reputation(did)
                logger.debug(
                    f"Node {node_id}: Using identity-based initial reputation: {base_reputation:.2f} "
                    f"(Assurance: {identity_signals.assurance_level.value})"
                )
            else:
                base_reputation = 0.5  # Fallback
                logger.warning(f"Node {node_id}: No DID found, using default reputation")

        # Calculate identity boost (-0.2 to +0.2)
        identity_boost = self.identity_bridge._calculate_identity_boost(identity_signals)

        # Apply boost (clamped to [0.3, 1.0])
        enhanced_reputation = max(0.3, min(1.0, base_reputation + identity_boost))

        # Track metrics
        if abs(identity_boost) > 0.01:
            self.identity_metrics["identity_boost_applied"] += 1

        logger.debug(
            f"Node {node_id}: "
            f"base={base_reputation:.2f}, "
            f"identity_boost={identity_boost:+.2f} "
            f"({identity_signals.assurance_level.value}), "
            f"enhanced={enhanced_reputation:.2f}"
        )

        return enhanced_reputation

    async def handle_gradient_submission(
        self,
        node_id: str,
        encrypted_gradient: bytes,
        zkpoc_proof=None,
        pogq_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle gradient submission with identity-enhanced validation

        Overrides base method to add identity logging and enhanced Byzantine detection.

        Args:
            node_id: Node submitting gradient
            encrypted_gradient: Encrypted gradient data
            zkpoc_proof: Optional ZK proof
            pogq_score: Optional PoGQ score

        Returns:
            Result dict with identity metadata
        """
        # Call base implementation
        result = await super().handle_gradient_submission(
            node_id, encrypted_gradient, zkpoc_proof, pogq_score
        )

        # Add identity information to result
        identity_signals = self.node_identities.get(node_id)

        if identity_signals:
            result["identity_assurance"] = identity_signals.assurance_level.value
            result["sybil_resistance"] = identity_signals.sybil_resistance_score
            result["risk_level"] = identity_signals.risk_level.value
            result["verified_human"] = identity_signals.verified_human
        else:
            result["identity_assurance"] = "E0_Anonymous"
            result["sybil_resistance"] = 0.0
            result["risk_level"] = "Critical"
            result["verified_human"] = False

        # Enhanced Byzantine detection for low-identity nodes
        if not result["accepted"] and identity_signals:
            if identity_signals.risk_level.value in ["Critical", "High"]:
                self.identity_metrics["sybil_attacks_detected"] += 1
                logger.warning(
                    f"Potential Sybil attack from {node_id}: "
                    f"{identity_signals.assurance_level.value}, "
                    f"gradient rejected"
                )

        return result

    def _update_assurance_metrics(self, level: AssuranceLevel):
        """Update assurance level distribution metrics"""
        metric_map = {
            AssuranceLevel.E0_ANONYMOUS: "e0_nodes",
            AssuranceLevel.E1_TESTIMONIAL: "e1_nodes",
            AssuranceLevel.E2_PRIVATELY_VERIFIABLE: "e2_nodes",
            AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN: "e3_nodes",
            AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL: "e4_nodes",
        }

        metric_key = metric_map.get(level)
        if metric_key:
            self.identity_metrics[metric_key] += 1

    def get_identity_metrics(self) -> Dict[str, Any]:
        """
        Get identity-specific metrics

        Returns:
            Dictionary of identity metrics including assurance distribution
        """
        total_nodes = self.identity_metrics["nodes_registered"]

        return {
            **self.identity_metrics,
            "assurance_distribution": {
                "E0": self.identity_metrics["e0_nodes"],
                "E1": self.identity_metrics["e1_nodes"],
                "E2": self.identity_metrics["e2_nodes"],
                "E3": self.identity_metrics["e3_nodes"],
                "E4": self.identity_metrics["e4_nodes"],
            },
            "total_nodes": total_nodes,
            "average_sybil_resistance": self._calculate_average_sybil_resistance(),
        }

    def _calculate_average_sybil_resistance(self) -> float:
        """Calculate average Sybil resistance across all nodes"""
        if not self.node_identities:
            return 0.0

        total = sum(
            signal.sybil_resistance_score
            for signal in self.node_identities.values()
        )
        return total / len(self.node_identities)

    async def get_node_identity_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete identity information for a node

        Args:
            node_id: Node to query

        Returns:
            Identity information dict or None if not found
        """
        signals = self.node_identities.get(node_id)
        did = self.node_did_mapping.get(node_id)

        if not signals or not did:
            return None

        # Get current reputation
        reputation = await self._get_reputation(node_id)

        return {
            "node_id": node_id,
            "did": did,
            "assurance_level": signals.assurance_level.value,
            "assurance_score": signals.assurance_score,
            "risk_level": signals.risk_level.value,
            "sybil_resistance": signals.sybil_resistance_score,
            "verified_human": signals.verified_human,
            "gitcoin_score": signals.gitcoin_score,
            "guardian_count": signals.guardian_count,
            "guardian_diversity": signals.guardian_graph_diversity,
            "current_reputation": reputation,
            "registered_at": signals.computed_at.isoformat(),
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_identity_coordinator():
        """Test identity coordinator with mock data"""
        from zerotrustml.identity import (
            DIDManager,
            CryptoKeyFactor,
            GitcoinPassportFactor,
            SocialRecoveryFactor,
            FactorCategory,
            FactorStatus,
            VCManager,
            VCType,
        )
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        print("=== Identity-Enhanced Coordinator Test ===\n")

        # Create coordinator
        config = Phase10Config(
            postgres_enabled=False,
            localfile_enabled=True
        )
        coordinator = IdentityCoordinator(config)
        await coordinator.initialize()

        # Create test identities
        did_manager = DIDManager()
        alice_did = did_manager.create_did()

        # Create factors for Alice (E3 level)
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        alice_factors = [
            CryptoKeyFactor(
                factor_id="crypto-001",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            ),
            GitcoinPassportFactor(
                factor_id="gitcoin-001",
                factor_type="GitcoinPassport",
                category=FactorCategory.REPUTATION,
                status=FactorStatus.ACTIVE,
                passport_address="0xAlice",
                score=45.0
            ),
            SocialRecoveryFactor(
                factor_id="recovery-001",
                factor_type="SocialRecovery",
                category=FactorCategory.SOCIAL,
                status=FactorStatus.ACTIVE,
                threshold=3,
                guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                             "did:mycelix:g4", "did:mycelix:g5"]
            )
        ]

        # Register Alice
        print("Registering Alice (E3 level)...")
        alice_signals = await coordinator.register_node_identity(
            node_id="alice_node",
            did=alice_did.to_string(),
            factors=alice_factors,
            credentials=[]
        )

        print(f"  Assurance: {alice_signals.assurance_level.value}")
        print(f"  Sybil Resistance: {alice_signals.sybil_resistance_score:.2f}")
        print(f"  Initial Reputation: {coordinator.identity_bridge.get_initial_reputation(alice_did.to_string()):.2f}")

        # Get Alice's reputation
        alice_rep = await coordinator._get_reputation("alice_node")
        print(f"  Enhanced Reputation: {alice_rep:.2f}\n")

        # Create anonymous Bob (E0)
        bob_did = did_manager.create_did()
        print("Registering Bob (E0 anonymous)...")
        bob_signals = await coordinator.register_node_identity(
            node_id="bob_node",
            did=bob_did.to_string(),
            factors=[],  # No factors = E0
            credentials=[]
        )

        print(f"  Assurance: {bob_signals.assurance_level.value}")
        print(f"  Sybil Resistance: {bob_signals.sybil_resistance_score:.2f}")
        bob_rep = await coordinator._get_reputation("bob_node")
        print(f"  Enhanced Reputation: {bob_rep:.2f}\n")

        # Show metrics
        print("Identity Metrics:")
        metrics = coordinator.get_identity_metrics()
        print(f"  Total Nodes: {metrics['total_nodes']}")
        print(f"  E0 (Anonymous): {metrics['e0_nodes']}")
        print(f"  E3 (Crypto Proven): {metrics['e3_nodes']}")
        print(f"  Average Sybil Resistance: {metrics['average_sybil_resistance']:.2f}")

        print("\n✅ Identity-Enhanced Coordinator test complete!")

    # Run test
    asyncio.run(test_identity_coordinator())

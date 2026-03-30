# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integration tests for Identity-Enhanced Coordinator
Tests identity registration and reputation enhancement
"""

import pytest
import asyncio
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from zerotrustml.core.phase10_coordinator import Phase10Config
from zerotrustml.core.identity_coordinator import IdentityCoordinator
from zerotrustml.identity import (
    DIDManager,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    FactorCategory,
    FactorStatus,
    VCManager,
    VCType,
    AssuranceLevel,
)


@pytest.fixture
async def coordinator():
    """Create identity coordinator for testing"""
    config = Phase10Config(
        postgres_enabled=False,
        localfile_enabled=True,
        localfile_data_dir="/tmp/test_identity_coordinator"
    )
    coord = IdentityCoordinator(config)
    await coord.initialize()
    yield coord
    await coord.shutdown()


@pytest.fixture
def did_manager():
    """Create DID manager"""
    return DIDManager()


@pytest.fixture
def vc_manager():
    """Create VC manager"""
    return VCManager()


class TestIdentityRegistration:
    """Test node identity registration"""

    @pytest.mark.asyncio
    async def test_register_e0_node(self, coordinator, did_manager):
        """Test registering anonymous (E0) node"""
        did = did_manager.create_did()

        signals = await coordinator.register_node_identity(
            node_id="node_e0",
            did=did.to_string(),
            factors=[],  # No factors = E0
            credentials=[]
        )

        assert signals.assurance_level == AssuranceLevel.E0_ANONYMOUS
        assert signals.sybil_resistance_score == 0.0
        assert signals.risk_level.value == "Critical"

    @pytest.mark.asyncio
    async def test_register_e1_node(self, coordinator, did_manager):
        """Test registering testimonial (E1) node"""
        did = did_manager.create_did()

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        factors = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        signals = await coordinator.register_node_identity(
            node_id="node_e1",
            did=did.to_string(),
            factors=factors,
            credentials=[]
        )

        assert signals.assurance_score >= 0.5
        assert signals.active_factors == 1

    @pytest.mark.asyncio
    async def test_register_e3_node(self, coordinator, did_manager):
        """Test registering cryptographically proven (E3) node"""
        did = did_manager.create_did()

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        factors = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            ),
            GitcoinPassportFactor(
                factor_id="gitcoin",
                factor_type="GitcoinPassport",
                category=FactorCategory.REPUTATION,
                status=FactorStatus.ACTIVE,
                passport_address="0xtest",
                score=45.0
            ),
            SocialRecoveryFactor(
                factor_id="recovery",
                factor_type="SocialRecovery",
                category=FactorCategory.SOCIAL,
                status=FactorStatus.ACTIVE,
                threshold=3,
                guardian_dids=["g1", "g2", "g3", "g4", "g5"]
            )
        ]

        signals = await coordinator.register_node_identity(
            node_id="node_e3",
            did=did.to_string(),
            factors=factors,
            credentials=[]
        )

        assert signals.assurance_score >= 0.6
        assert signals.active_factors == 3
        assert signals.factor_categories == 3

    @pytest.mark.asyncio
    async def test_register_with_verified_human(self, coordinator, did_manager, vc_manager):
        """Test registering node with VerifiedHuman credential"""
        did = did_manager.create_did()

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        factors = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        # Issue VerifiedHuman credential
        issuer_private = ed25519.Ed25519PrivateKey.generate()
        issuer_private_bytes = issuer_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

        vh_cred = vc_manager.issue_credential(
            issuer_did="did:mycelix:verifier",
            issuer_private_key=issuer_private_bytes,
            subject_did=did.to_string(),
            vc_type=VCType.VERIFIED_HUMAN,
            claims={"verified": True}
        )

        signals = await coordinator.register_node_identity(
            node_id="node_verified",
            did=did.to_string(),
            factors=factors,
            credentials=[vh_cred]
        )

        assert signals.verified_human is True
        assert signals.credentials_valid == 1


class TestReputationEnhancement:
    """Test identity-enhanced reputation"""

    @pytest.mark.asyncio
    async def test_reputation_for_e0(self, coordinator, did_manager):
        """Test reputation for E0 (anonymous) node"""
        did = did_manager.create_did()

        await coordinator.register_node_identity(
            node_id="node_e0",
            did=did.to_string(),
            factors=[],
            credentials=[]
        )

        reputation = await coordinator._get_reputation("node_e0")

        # E0 should have low initial reputation (0.30)
        assert 0.25 <= reputation <= 0.35

    @pytest.mark.asyncio
    async def test_reputation_for_e4(self, coordinator, did_manager):
        """Test reputation for E4 (maximum verification) node"""
        from zerotrustml.identity import HardwareKeyFactor

        did = did_manager.create_did()

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Create maximum verification factors (need 4 factors across 4 categories for E4)
        factors = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            ),
            GitcoinPassportFactor(
                factor_id="gitcoin",
                factor_type="GitcoinPassport",
                category=FactorCategory.REPUTATION,
                status=FactorStatus.ACTIVE,
                passport_address="0xtest",
                score=55.0  # >= 50
            ),
            SocialRecoveryFactor(
                factor_id="recovery",
                factor_type="SocialRecovery",
                category=FactorCategory.SOCIAL,
                status=FactorStatus.ACTIVE,
                threshold=3,
                guardian_dids=["g1", "g2", "g3", "g4", "g5"]
            ),
            HardwareKeyFactor(
                factor_id="hardware",
                factor_type="HardwareKey",
                category=FactorCategory.BACKUP,
                status=FactorStatus.ACTIVE,
                device_type="yubikey",
                device_id="YK-001",
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        await coordinator.register_node_identity(
            node_id="node_e4",
            did=did.to_string(),
            factors=factors,
            credentials=[]
        )

        reputation = await coordinator._get_reputation("node_e4")

        # E4 should have high initial reputation (0.70 base + possible boost)
        assert reputation >= 0.65, f"E4 reputation too low: {reputation}"

    @pytest.mark.asyncio
    async def test_reputation_differential(self, coordinator, did_manager):
        """Test reputation differential between E0 and E4"""
        from zerotrustml.identity import HardwareKeyFactor

        # Register E0 node
        did_e0 = did_manager.create_did()
        await coordinator.register_node_identity(
            node_id="node_e0",
            did=did_e0.to_string(),
            factors=[],
            credentials=[]
        )

        # Register E4 node (with 4 factors across 4 categories)
        did_e4 = did_manager.create_did()
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        factors_e4 = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            ),
            GitcoinPassportFactor(
                factor_id="gitcoin",
                factor_type="GitcoinPassport",
                category=FactorCategory.REPUTATION,
                status=FactorStatus.ACTIVE,
                passport_address="0xtest",
                score=55.0
            ),
            SocialRecoveryFactor(
                factor_id="recovery",
                factor_type="SocialRecovery",
                category=FactorCategory.SOCIAL,
                status=FactorStatus.ACTIVE,
                threshold=3,
                guardian_dids=["g1", "g2", "g3", "g4", "g5"]
            ),
            HardwareKeyFactor(
                factor_id="hardware",
                factor_type="HardwareKey",
                category=FactorCategory.BACKUP,
                status=FactorStatus.ACTIVE,
                device_type="yubikey",
                device_id="YK-002",
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        await coordinator.register_node_identity(
            node_id="node_e4",
            did=did_e4.to_string(),
            factors=factors_e4,
            credentials=[]
        )

        rep_e0 = await coordinator._get_reputation("node_e0")
        rep_e4 = await coordinator._get_reputation("node_e4")

        # E4 should have significantly higher reputation
        assert rep_e4 > rep_e0, f"E4 ({rep_e4:.2f}) should be > E0 ({rep_e0:.2f})"
        assert (rep_e4 - rep_e0) >= 0.3, f"Differential ({rep_e4 - rep_e0:.2f}) should be >= 0.3"


class TestIdentityMetrics:
    """Test identity metrics tracking"""

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, coordinator, did_manager):
        """Test that metrics are tracked correctly"""
        # Register nodes of different levels
        for i, factors_count in enumerate([0, 1, 2, 3]):
            did = did_manager.create_did()

            factors = []
            if factors_count >= 1:
                private_key = ed25519.Ed25519PrivateKey.generate()
                public_key = private_key.public_key()
                factors.append(
                    CryptoKeyFactor(
                        factor_id=f"crypto-{i}",
                        factor_type="CryptoKey",
                        category=FactorCategory.PRIMARY,
                        status=FactorStatus.ACTIVE,
                        public_key=public_key.public_bytes(
                            encoding=serialization.Encoding.Raw,
                            format=serialization.PublicFormat.Raw
                        )
                    )
                )

            await coordinator.register_node_identity(
                node_id=f"node_{i}",
                did=did.to_string(),
                factors=factors,
                credentials=[]
            )

        metrics = coordinator.get_identity_metrics()

        assert metrics["nodes_registered"] == 4
        assert metrics["total_nodes"] == 4
        assert metrics["e0_nodes"] >= 1  # At least one E0


class TestNodeIdentityInfo:
    """Test getting node identity information"""

    @pytest.mark.asyncio
    async def test_get_node_identity_info(self, coordinator, did_manager):
        """Test retrieving complete node identity info"""
        did = did_manager.create_did()

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        factors = [
            CryptoKeyFactor(
                factor_id="crypto",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        await coordinator.register_node_identity(
            node_id="test_node",
            did=did.to_string(),
            factors=factors,
            credentials=[]
        )

        info = await coordinator.get_node_identity_info("test_node")

        assert info is not None
        assert info["node_id"] == "test_node"
        assert info["did"] == did.to_string()
        assert "assurance_level" in info
        assert "current_reputation" in info

    @pytest.mark.asyncio
    async def test_get_nonexistent_node_info(self, coordinator):
        """Test getting info for non-existent node"""
        info = await coordinator.get_node_identity_info("nonexistent")
        assert info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

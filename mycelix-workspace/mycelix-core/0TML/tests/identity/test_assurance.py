# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit tests for Assurance Level Calculation
Tests E0-E4 level calculation and capability assignment
"""

import pytest

from zerotrustml.identity import (
    calculate_assurance_level,
    AssuranceLevel,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    HardwareKeyFactor,
    FactorStatus,
    FactorCategory,
)
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


class TestAssuranceLevelCalculation:
    """Test assurance level calculation logic"""

    def test_e0_no_factors(self):
        """Test E0 level with no factors"""
        result = calculate_assurance_level([])

        assert result.level == AssuranceLevel.E0_ANONYMOUS
        assert result.score == 0.0
        assert "read_public_data" in result.capabilities

    def test_e1_crypto_key_only(self):
        """Test E1 level with just crypto key"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="test-crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        result = calculate_assurance_level([crypto_factor])

        # Crypto key gives 0.5 contribution
        assert result.score >= 0.5
        assert "create_content" in result.capabilities
        assert "send_messages" in result.capabilities

    def test_e2_crypto_plus_passport(self):
        """Test E2 level with crypto + Gitcoin Passport"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="test-crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        passport_factor = GitcoinPassportFactor(
            factor_id="test-passport",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address="0x123",
            score=25.0  # >= 20
        )

        result = calculate_assurance_level([crypto_factor, passport_factor])

        # 0.5 (crypto) + 0.3 (passport >=20) + 0.1 (2 categories * 0.05) = 0.9
        assert result.score >= 0.8  # E4 territory
        assert "participate_community" in result.capabilities

    def test_e3_multi_factor_with_recovery(self):
        """Test E3 level with multiple factors including recovery"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="test-crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        passport_factor = GitcoinPassportFactor(
            factor_id="test-passport",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address="0x123",
            score=45.0
        )

        recovery_factor = SocialRecoveryFactor(
            factor_id="test-recovery",
            factor_type="SocialRecovery",
            category=FactorCategory.SOCIAL,
            status=FactorStatus.ACTIVE,
            threshold=3,
            guardian_dids=["g1", "g2", "g3", "g4", "g5"]
        )

        result = calculate_assurance_level([crypto_factor, passport_factor, recovery_factor])

        # Score should be high enough for E3+
        assert result.score >= 0.6
        assert "vote_governance" in result.capabilities

    def test_e4_maximum_verification(self):
        """Test E4 level with maximum verification"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="test-crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        passport_factor = GitcoinPassportFactor(
            factor_id="test-passport",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address="0x123",
            score=55.0  # >= 50
        )

        recovery_factor = SocialRecoveryFactor(
            factor_id="test-recovery",
            factor_type="SocialRecovery",
            category=FactorCategory.SOCIAL,
            status=FactorStatus.ACTIVE,
            threshold=3,
            guardian_dids=["g1", "g2", "g3", "g4", "g5"]
        )

        hw_key = ed25519.Ed25519PrivateKey.generate()
        hardware_factor = HardwareKeyFactor(
            factor_id="test-hw",
            factor_type="HardwareKey",
            category=FactorCategory.BACKUP,
            status=FactorStatus.ACTIVE,
            device_type="yubikey",
            device_id="YK-123",
            public_key=hw_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        result = calculate_assurance_level([crypto_factor, passport_factor, recovery_factor, hardware_factor])

        # Score should be at maximum or near
        assert result.score >= 0.8
        assert result.level in [AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL]
        assert "propose_constitutional_amendment" in result.capabilities

    def test_inactive_factors_ignored(self):
        """Test inactive factors don't contribute"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        active_factor = CryptoKeyFactor(
            factor_id="active",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        inactive_factor = GitcoinPassportFactor(
            factor_id="inactive",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.INACTIVE,  # INACTIVE!
            passport_address="0x123",
            score=100.0
        )

        result = calculate_assurance_level([active_factor, inactive_factor])

        # Should only count active factor
        assert len(result.active_factors) == 1
        assert "active" in result.active_factors

    def test_diversity_bonus(self):
        """Test factor diversity increases score"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Single category
        crypto_only = [CryptoKeyFactor(
            factor_id="crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )]

        result_single = calculate_assurance_level(crypto_only)

        # Multiple categories
        crypto_plus_passport = crypto_only + [GitcoinPassportFactor(
            factor_id="passport",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,  # Different category
            status=FactorStatus.ACTIVE,
            passport_address="0x123",
            score=25.0
        )]

        result_multi = calculate_assurance_level(crypto_plus_passport)

        # Multi-category should have higher score due to diversity bonus
        assert result_multi.score > result_single.score

    def test_score_clamped_to_one(self):
        """Test score never exceeds 1.0"""
        # Create many high-contributing factors
        factors = []

        for i in range(10):
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()

            factors.append(CryptoKeyFactor(
                factor_id=f"crypto-{i}",
                factor_type="CryptoKey",
                category=FactorCategory.PRIMARY,
                status=FactorStatus.ACTIVE,
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            ))

        result = calculate_assurance_level(factors)

        # Score should be clamped to 1.0
        assert result.score <= 1.0

    def test_recommendations_for_e0(self):
        """Test E0 gets appropriate recommendations"""
        result = calculate_assurance_level([])

        # Should recommend creating DID
        assert any("DID" in rec or "crypto" in rec.lower() for rec in result.recommendations)

    def test_recommendations_for_e1(self):
        """Test E1 gets upgrade recommendations"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        result = calculate_assurance_level([crypto_factor])

        # Should recommend adding more factors
        assert len(result.recommendations) > 0


class TestCapabilities:
    """Test capability system"""

    def test_capability_progression(self):
        """Test capabilities increase with assurance level"""
        # E0
        e0_result = calculate_assurance_level([])
        e0_caps = len(e0_result.capabilities)

        # E1 - add crypto key
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        e1_result = calculate_assurance_level([crypto_factor])
        e1_caps = len(e1_result.capabilities)

        # E1+ should have more capabilities than E0
        assert e1_caps >= e0_caps

    def test_governance_voting_requires_e3(self):
        """Test governance voting requires E3 or higher"""
        # E2 should NOT have governance voting
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        e2_result = calculate_assurance_level([crypto_factor])

        # Check if we're actually at E2 or below
        if e2_result.level.value < AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN.value:
            assert "vote_governance" not in e2_result.capabilities

    def test_constitutional_proposals_require_e4(self):
        """Test constitutional proposals require E4"""
        # Lower levels should NOT have constitutional proposal capability
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        crypto_factor = CryptoKeyFactor(
            factor_id="crypto",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        e1_result = calculate_assurance_level([crypto_factor])

        if e1_result.level != AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL:
            assert "propose_constitutional_amendment" not in e1_result.capabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

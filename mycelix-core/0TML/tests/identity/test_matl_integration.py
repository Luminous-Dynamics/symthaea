# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit tests for MATL Integration
Tests identity trust signal computation and MATL score enhancement
"""

import pytest
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from zerotrustml.identity import (
    IdentityMATLBridge,
    IdentityTrustSignal,
    EnhancedMATLScore,
    IdentityRiskLevel,
)
from zerotrustml.identity.matl_integration import IdentityMATLBridge as MATLBridge
from zerotrustml.identity import (
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    BiometricFactor,
    HardwareKeyFactor,
    FactorCategory,
    FactorStatus,
    AssuranceLevel,
)
from zerotrustml.identity import VCManager, VCType


class TestIdentityTrustSignal:
    """Test IdentityTrustSignal dataclass"""

    def test_create_trust_signal(self):
        """Test creating an identity trust signal"""
        signal = IdentityTrustSignal(
            did="did:mycelix:alice",
            assurance_level=AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN,
            assurance_score=0.75,
            active_factors=3,
            factor_categories=3,
            factor_diversity_score=0.6,
            verified_human=True,
            gitcoin_score=42.5,
            credentials_count=2,
            credentials_valid=2,
            guardian_count=5,
            guardian_graph_diversity=0.8,
            risk_level=IdentityRiskLevel.LOW,
            sybil_resistance_score=0.85
        )

        assert signal.did == "did:mycelix:alice"
        assert signal.assurance_level == AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN
        assert signal.verified_human is True
        assert signal.sybil_resistance_score == 0.85

    def test_signal_to_dict(self):
        """Test serializing trust signal to dict"""
        signal = IdentityTrustSignal(
            did="did:mycelix:test",
            assurance_level=AssuranceLevel.E2_PRIVATELY_VERIFIABLE,
            assurance_score=0.5,
            active_factors=2,
            factor_categories=2,
            factor_diversity_score=0.4,
            verified_human=False,
            gitcoin_score=15.0,
            credentials_count=1,
            credentials_valid=1,
            guardian_count=0,
            guardian_graph_diversity=0.0,
            risk_level=IdentityRiskLevel.MEDIUM,
            sybil_resistance_score=0.45
        )

        data = signal.to_dict()

        assert data["did"] == "did:mycelix:test"
        assert data["assurance_level"] == "E2_PrivatelyVerifiable"
        assert data["risk_level"] == "Medium"
        assert "computed_at" in data


class TestEnhancedMATLScore:
    """Test EnhancedMATLScore dataclass"""

    def test_create_enhanced_score(self):
        """Test creating an enhanced MATL score"""
        signal = IdentityTrustSignal(
            did="did:mycelix:alice",
            assurance_level=AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN,
            assurance_score=0.75,
            active_factors=3,
            factor_categories=3,
            factor_diversity_score=0.6,
            verified_human=True,
            gitcoin_score=42.5,
            credentials_count=2,
            credentials_valid=2,
            guardian_count=5,
            guardian_graph_diversity=0.8,
            risk_level=IdentityRiskLevel.LOW,
            sybil_resistance_score=0.85
        )

        score = EnhancedMATLScore(
            pogq_score=0.80,
            tcdm_score=0.75,
            entropy_score=0.70,
            identity_boost=0.15,
            identity_signals=signal,
            base_score=0.755,  # (0.8*0.4 + 0.75*0.3 + 0.7*0.3)
            enhanced_score=0.905  # 0.755 + 0.15
        )

        assert score.base_score == 0.755
        assert score.identity_boost == 0.15
        assert score.enhanced_score == 0.905

    def test_enhanced_score_to_dict(self):
        """Test serializing enhanced score to dict"""
        signal = IdentityTrustSignal(
            did="did:mycelix:test",
            assurance_level=AssuranceLevel.E2_PRIVATELY_VERIFIABLE,
            assurance_score=0.5,
            active_factors=1,
            factor_categories=1,
            factor_diversity_score=0.2,
            verified_human=False,
            gitcoin_score=0.0,
            credentials_count=0,
            credentials_valid=0,
            guardian_count=0,
            guardian_graph_diversity=0.0,
            risk_level=IdentityRiskLevel.MEDIUM,
            sybil_resistance_score=0.3
        )

        score = EnhancedMATLScore(
            pogq_score=0.60,
            tcdm_score=0.55,
            entropy_score=0.50,
            identity_boost=0.05,
            identity_signals=signal,
            base_score=0.560,
            enhanced_score=0.610
        )

        data = score.to_dict()

        assert data["pogq_score"] == 0.60
        assert data["identity_boost"] == 0.05
        assert "identity_signals" in data
        assert "computed_at" in data


class TestIdentityMATLBridge:
    """Test IdentityMATLBridge functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.bridge = IdentityMATLBridge()

        # Create common test factors
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        self.crypto_factor = CryptoKeyFactor(
            factor_id="crypto-test",
            factor_type="CryptoKey",
            category=FactorCategory.PRIMARY,
            status=FactorStatus.ACTIVE,
            public_key=public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        )

        self.passport_factor = GitcoinPassportFactor(
            factor_id="gitcoin-test",
            factor_type="GitcoinPassport",
            category=FactorCategory.REPUTATION,
            status=FactorStatus.ACTIVE,
            passport_address="0xtest",
            score=45.0
        )

        self.recovery_factor = SocialRecoveryFactor(
            factor_id="recovery-test",
            factor_type="SocialRecovery",
            category=FactorCategory.SOCIAL,
            status=FactorStatus.ACTIVE,
            threshold=3,
            guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                          "did:mycelix:g4", "did:mycelix:g5"]
        )

        # Create VC manager for credentials
        self.vc_manager = VCManager()
        self.issuer_private = ed25519.Ed25519PrivateKey.generate()
        self.issuer_private_bytes = self.issuer_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

    def test_compute_basic_signals(self):
        """Test computing identity signals with basic factors"""
        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:test",
            factors=[self.crypto_factor],
            credentials=[]
        )

        assert signals.did == "did:mycelix:test"
        assert signals.active_factors == 1
        assert signals.factor_categories == 1
        assert signals.verified_human is False
        assert signals.gitcoin_score == 0.0

    def test_compute_signals_with_multiple_factors(self):
        """Test computing signals with multiple factors"""
        factors = [
            self.crypto_factor,
            self.passport_factor,
            self.recovery_factor
        ]

        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:alice",
            factors=factors,
            credentials=[]
        )

        assert signals.active_factors == 3
        assert signals.factor_categories == 3  # Primary, Reputation, Social
        assert signals.factor_diversity_score == 0.6  # 3/5 categories
        assert signals.guardian_count == 5

    def test_compute_signals_with_verified_human(self):
        """Test signals with VerifiedHuman credential"""
        vh_credential = self.vc_manager.issue_credential(
            issuer_did="did:mycelix:issuer",
            issuer_private_key=self.issuer_private_bytes,
            subject_did="did:mycelix:test",
            vc_type=VCType.VERIFIED_HUMAN,
            claims={"verified": True}
        )

        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:test",
            factors=[self.crypto_factor],
            credentials=[vh_credential]
        )

        assert signals.verified_human is True
        assert signals.credentials_count == 1
        assert signals.credentials_valid == 1

    def test_compute_signals_with_gitcoin_passport(self):
        """Test signals with Gitcoin Passport credential"""
        passport_cred = self.vc_manager.issue_credential(
            issuer_did="did:mycelix:issuer",
            issuer_private_key=self.issuer_private_bytes,
            subject_did="did:mycelix:test",
            vc_type=VCType.GITCOIN_PASSPORT,
            claims={"passportScore": 55.5}
        )

        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:test",
            factors=[self.crypto_factor],
            credentials=[passport_cred]
        )

        assert signals.gitcoin_score == 55.5

    def test_sybil_resistance_calculation(self):
        """Test Sybil resistance scoring"""
        # Low resistance (anonymous)
        low_score = self.bridge._calculate_sybil_resistance(
            assurance_score=0.0,
            gitcoin_score=0.0,
            verified_human=False,
            factor_diversity=0.0
        )
        assert low_score == 0.0

        # High resistance (fully verified)
        high_score = self.bridge._calculate_sybil_resistance(
            assurance_score=1.0,
            gitcoin_score=100.0,
            verified_human=True,
            factor_diversity=1.0
        )
        assert abs(high_score - 1.0) < 0.001  # Use approximate equality for float

        # Medium resistance
        medium_score = self.bridge._calculate_sybil_resistance(
            assurance_score=0.5,
            gitcoin_score=50.0,
            verified_human=True,
            factor_diversity=0.5
        )
        assert 0.4 < medium_score < 0.7

    def test_risk_level_determination(self):
        """Test risk level determination"""
        # E0 = Critical
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E0_ANONYMOUS,
            0.0
        )
        assert risk == IdentityRiskLevel.CRITICAL

        # E1 = High
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E1_TESTIMONIAL,
            0.3
        )
        assert risk == IdentityRiskLevel.HIGH

        # E2 with low sybil resistance = Medium
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E2_PRIVATELY_VERIFIABLE,
            0.5
        )
        assert risk == IdentityRiskLevel.MEDIUM

        # E2 with high sybil resistance = Low
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E2_PRIVATELY_VERIFIABLE,
            0.8
        )
        assert risk == IdentityRiskLevel.LOW

        # E3 with high sybil resistance = Minimal
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN,
            0.85
        )
        assert risk == IdentityRiskLevel.MINIMAL

        # E4 = Minimal
        risk = self.bridge._determine_risk_level(
            AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL,
            0.7
        )
        assert risk == IdentityRiskLevel.MINIMAL

    def test_identity_boost_calculation(self):
        """Test identity boost calculation"""
        # E0 (anonymous) = -0.20
        signal_e0 = IdentityTrustSignal(
            did="did:mycelix:test",
            assurance_level=AssuranceLevel.E0_ANONYMOUS,
            assurance_score=0.0,
            active_factors=0,
            factor_categories=0,
            factor_diversity_score=0.0,
            verified_human=False,
            gitcoin_score=0.0,
            credentials_count=0,
            credentials_valid=0,
            guardian_count=0,
            guardian_graph_diversity=0.0,
            risk_level=IdentityRiskLevel.CRITICAL,
            sybil_resistance_score=0.0
        )
        boost = self.bridge._calculate_identity_boost(signal_e0)
        assert boost == -0.20

        # E4 + all bonuses = +0.20 (capped)
        signal_e4 = IdentityTrustSignal(
            did="did:mycelix:test",
            assurance_level=AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL,
            assurance_score=1.0,
            active_factors=4,
            factor_categories=4,
            factor_diversity_score=0.8,
            verified_human=True,
            gitcoin_score=55.0,
            credentials_count=2,
            credentials_valid=2,
            guardian_count=5,
            guardian_graph_diversity=0.9,
            risk_level=IdentityRiskLevel.MINIMAL,
            sybil_resistance_score=0.95
        )
        boost = self.bridge._calculate_identity_boost(signal_e4)
        assert boost == 0.20  # Capped at max

    def test_enhance_matl_score(self):
        """Test MATL score enhancement"""
        # Compute signals first
        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:alice",
            factors=[self.crypto_factor, self.passport_factor],
            credentials=[]
        )

        # Enhance MATL score
        enhanced = self.bridge.enhance_matl_score(
            did="did:mycelix:alice",
            pogq_score=0.75,
            tcdm_score=0.80,
            entropy_score=0.70,
            identity_signals=signals
        )

        # Base score = (0.75*0.4) + (0.80*0.3) + (0.70*0.3) = 0.75
        assert abs(enhanced.base_score - 0.75) < 0.001

        # Enhanced score should have identity boost applied
        assert enhanced.enhanced_score != enhanced.base_score
        assert 0.0 <= enhanced.enhanced_score <= 1.0

    def test_enhance_matl_score_with_caching(self):
        """Test MATL score enhancement using cached signals"""
        # Compute and cache signals
        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:bob",
            factors=[self.crypto_factor],
            credentials=[]
        )

        # Enhance without providing signals (should use cache)
        enhanced = self.bridge.enhance_matl_score(
            did="did:mycelix:bob",
            pogq_score=0.60,
            tcdm_score=0.65,
            entropy_score=0.55
        )

        assert enhanced is not None
        assert enhanced.identity_signals.did == "did:mycelix:bob"

    def test_enhance_matl_score_clamping(self):
        """Test enhanced score is clamped to [0, 1]"""
        # Create signal with high boost
        signal = IdentityTrustSignal(
            did="did:mycelix:test",
            assurance_level=AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL,
            assurance_score=1.0,
            active_factors=4,
            factor_categories=4,
            factor_diversity_score=0.8,
            verified_human=True,
            gitcoin_score=100.0,
            credentials_count=3,
            credentials_valid=3,
            guardian_count=7,
            guardian_graph_diversity=0.9,
            risk_level=IdentityRiskLevel.MINIMAL,
            sybil_resistance_score=1.0
        )

        # Very high base score + positive boost
        enhanced = self.bridge.enhance_matl_score(
            did="did:mycelix:test",
            pogq_score=0.95,
            tcdm_score=0.95,
            entropy_score=0.95,
            identity_signals=signal
        )

        # Should be clamped to 1.0
        assert enhanced.enhanced_score <= 1.0

    def test_guardian_diversity_independent(self):
        """Test guardian diversity with independent guardians"""
        # No connections between guardians
        guardian_graph = {
            "did:mycelix:g1": [],
            "did:mycelix:g2": [],
            "did:mycelix:g3": [],
            "did:mycelix:g4": [],
            "did:mycelix:g5": []
        }

        guardian_dids = list(guardian_graph.keys())

        diversity = self.bridge._calculate_guardian_diversity(
            guardian_dids, guardian_graph
        )

        # High diversity (1.0) = no connections
        assert diversity == 1.0

    def test_guardian_diversity_cartel(self):
        """Test guardian diversity with fully connected cartel"""
        # All guardians know each other
        guardian_graph = {
            "did:mycelix:g1": ["did:mycelix:g2", "did:mycelix:g3"],
            "did:mycelix:g2": ["did:mycelix:g1", "did:mycelix:g3"],
            "did:mycelix:g3": ["did:mycelix:g1", "did:mycelix:g2"]
        }

        guardian_dids = list(guardian_graph.keys())

        diversity = self.bridge._calculate_guardian_diversity(
            guardian_dids, guardian_graph
        )

        # Low diversity (0.0) = fully connected
        assert diversity == 0.0

    def test_guardian_diversity_partial(self):
        """Test guardian diversity with partial connections"""
        # Some connections but not all
        guardian_graph = {
            "did:mycelix:g1": ["did:mycelix:g2"],
            "did:mycelix:g2": ["did:mycelix:g1"],
            "did:mycelix:g3": [],
            "did:mycelix:g4": [],
            "did:mycelix:g5": []
        }

        guardian_dids = list(guardian_graph.keys())

        diversity = self.bridge._calculate_guardian_diversity(
            guardian_dids, guardian_graph
        )

        # Medium diversity (1 connection out of 10 possible = 0.9)
        assert 0.8 <= diversity <= 1.0

    def test_initial_reputation_e0(self):
        """Test initial reputation for E0 (anonymous)"""
        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:anon",
            factors=[],  # No factors = E0
            credentials=[]
        )

        reputation = self.bridge.get_initial_reputation("did:mycelix:anon")

        # E0 should get 0.30
        assert reputation == 0.30

    def test_initial_reputation_e4(self):
        """Test initial reputation for E4 (maximum verification)"""
        # Create maximum verification
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
                passport_address="0x123",
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
                factor_id="hw",
                factor_type="HardwareKey",
                category=FactorCategory.BACKUP,
                status=FactorStatus.ACTIVE,
                device_type="yubikey",
                device_id="YK-123",
                public_key=public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            )
        ]

        vh_cred = self.vc_manager.issue_credential(
            issuer_did="did:mycelix:issuer",
            issuer_private_key=self.issuer_private_bytes,
            subject_did="did:mycelix:verified",
            vc_type=VCType.VERIFIED_HUMAN,
            claims={"verified": True}
        )

        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:verified",
            factors=factors,
            credentials=[vh_cred]
        )

        reputation = self.bridge.get_initial_reputation("did:mycelix:verified")

        # E4 (0.70) + verified human (0.05) + gitcoin >= 50 (0.05) = 0.70 (capped)
        assert reputation == 0.70

    def test_signal_caching(self):
        """Test that signals are cached after computation"""
        signals = self.bridge.compute_identity_signals(
            did="did:mycelix:cached",
            factors=[self.crypto_factor],
            credentials=[]
        )

        # Check cache
        assert "did:mycelix:cached" in self.bridge.signal_cache
        cached = self.bridge.signal_cache["did:mycelix:cached"]
        assert cached.did == "did:mycelix:cached"

    def test_enhance_without_cached_signals(self):
        """Test enhancing MATL score without cached signals raises error"""
        with pytest.raises(ValueError, match="No identity signals found"):
            self.bridge.enhance_matl_score(
                did="did:mycelix:nonexistent",
                pogq_score=0.5,
                tcdm_score=0.5,
                entropy_score=0.5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

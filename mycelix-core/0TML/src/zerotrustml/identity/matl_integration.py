# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MATL Integration for Multi-Factor Identity System
Connects identity assurance to MATL trust scoring

Enhances MATL's Composite Trust Score with identity verification:
- Identity assurance → Initial reputation boosting
- Factor diversity → TCDM (Temporal/Community Diversity) enhancement
- Verifiable credentials → PoGQ (Proof of Quality) attestations
- Guardian graph → Cartel detection signals
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from enum import Enum

from .assurance import AssuranceLevel, AssuranceResult, calculate_assurance_level
from .factors import IdentityFactor, FactorStatus
from .verifiable_credentials import VerifiableCredential, VCStatus, VCType


class IdentityRiskLevel(Enum):
    """Risk level based on identity verification"""
    CRITICAL = "Critical"  # Anonymous, no verification
    HIGH = "High"  # Basic verification only
    MEDIUM = "Medium"  # Multi-factor, some diversity
    LOW = "Low"  # Strong verification, good diversity
    MINIMAL = "Minimal"  # Maximum verification achieved


@dataclass
class IdentityTrustSignal:
    """
    Identity-derived trust signals for MATL

    These signals enhance MATL's composite trust scoring
    """
    # Core identity metrics
    did: str
    assurance_level: AssuranceLevel
    assurance_score: float  # 0.0-1.0

    # Factor diversity (enhances TCDM)
    active_factors: int
    factor_categories: int  # Number of unique factor categories
    factor_diversity_score: float  # 0.0-1.0

    # Credential attestations (enhances PoGQ)
    verified_human: bool
    gitcoin_score: float  # 0.0-100.0
    credentials_count: int
    credentials_valid: int

    # Social graph (enhances cartel detection)
    guardian_count: int
    guardian_graph_diversity: float  # 0.0-1.0 (how diverse are guardians)

    # Risk assessment
    risk_level: IdentityRiskLevel
    sybil_resistance_score: float  # 0.0-1.0

    # Timestamps
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verification: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Serialize trust signals"""
        return {
            "did": self.did,
            "assurance_level": self.assurance_level.value,
            "assurance_score": self.assurance_score,
            "active_factors": self.active_factors,
            "factor_categories": self.factor_categories,
            "factor_diversity_score": self.factor_diversity_score,
            "verified_human": self.verified_human,
            "gitcoin_score": self.gitcoin_score,
            "credentials_count": self.credentials_count,
            "credentials_valid": self.credentials_valid,
            "guardian_count": self.guardian_count,
            "guardian_graph_diversity": self.guardian_graph_diversity,
            "risk_level": self.risk_level.value,
            "sybil_resistance_score": self.sybil_resistance_score,
            "computed_at": self.computed_at.isoformat(),
            "last_verification": self.last_verification.isoformat() if self.last_verification else None
        }


@dataclass
class EnhancedMATLScore:
    """
    MATL Composite Trust Score enhanced with identity verification

    Original MATL: Score = (PoGQ × 0.4) + (TCDM × 0.3) + (Entropy × 0.3)
    Enhanced: Applies identity boost/penalty to initial score
    """
    # Original MATL components
    pogq_score: float  # 0.0-1.0 (Proof of Quality)
    tcdm_score: float  # 0.0-1.0 (Temporal/Community Diversity)
    entropy_score: float  # 0.0-1.0 (Behavioral Randomness)

    # Identity enhancement
    identity_boost: float  # -0.2 to +0.2 based on assurance level
    identity_signals: IdentityTrustSignal

    # Final scores
    base_score: float  # Original MATL score
    enhanced_score: float  # With identity boost applied

    # Metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        """Serialize enhanced score"""
        return {
            "pogq_score": self.pogq_score,
            "tcdm_score": self.tcdm_score,
            "entropy_score": self.entropy_score,
            "identity_boost": self.identity_boost,
            "base_score": self.base_score,
            "enhanced_score": self.enhanced_score,
            "identity_signals": self.identity_signals.to_dict(),
            "computed_at": self.computed_at.isoformat()
        }


class IdentityMATLBridge:
    """
    Bridge between Multi-Factor Identity System and MATL Trust Engine

    Converts identity verification into MATL-compatible trust signals
    """

    def __init__(self):
        """Initialize the bridge"""
        self.signal_cache: Dict[str, IdentityTrustSignal] = {}

    def compute_identity_signals(
        self,
        did: str,
        factors: List[IdentityFactor],
        credentials: List[VerifiableCredential],
        guardian_graph: Optional[Dict] = None
    ) -> IdentityTrustSignal:
        """
        Compute identity trust signals from factors and credentials

        Args:
            did: DID to compute signals for
            factors: List of identity factors
            credentials: List of verifiable credentials
            guardian_graph: Optional guardian network graph

        Returns:
            IdentityTrustSignal with all computed metrics
        """
        # Calculate assurance level
        assurance = calculate_assurance_level(factors)

        # Count active factors and categories
        active_factors_list = [f for f in factors if f.status == FactorStatus.ACTIVE]
        factor_categories = len({f.category for f in active_factors_list})

        # Calculate factor diversity (normalized)
        # Maximum possible categories = 5 (Primary, Backup, Social, Reputation, Biometric)
        factor_diversity = factor_categories / 5.0

        # Check for VerifiedHuman credential
        verified_human = any(
            vc.type[1] == VCType.VERIFIED_HUMAN.value and vc.status == VCStatus.ACTIVE
            for vc in credentials
        )

        # Extract Gitcoin score from credentials
        gitcoin_score = 0.0
        for vc in credentials:
            if vc.type[1] == VCType.GITCOIN_PASSPORT.value and vc.status == VCStatus.ACTIVE:
                gitcoin_score = vc.credential_subject.claims.get("passportScore", 0.0)
                break

        # Count valid credentials
        valid_credentials = [vc for vc in credentials if vc.is_valid()]

        # Guardian metrics
        guardian_count = 0
        guardian_diversity = 0.0

        for factor in active_factors_list:
            if hasattr(factor, 'guardian_dids'):
                guardian_count = len(factor.guardian_dids)
                # Calculate guardian diversity if graph provided
                if guardian_graph:
                    guardian_diversity = self._calculate_guardian_diversity(
                        factor.guardian_dids,
                        guardian_graph
                    )
                break

        # Calculate Sybil resistance
        sybil_resistance = self._calculate_sybil_resistance(
            assurance_score=assurance.score,
            gitcoin_score=gitcoin_score,
            verified_human=verified_human,
            factor_diversity=factor_diversity
        )

        # Determine risk level
        risk_level = self._determine_risk_level(assurance.level, sybil_resistance)

        # Create signal
        signal = IdentityTrustSignal(
            did=did,
            assurance_level=assurance.level,
            assurance_score=assurance.score,
            active_factors=len(active_factors_list),
            factor_categories=factor_categories,
            factor_diversity_score=factor_diversity,
            verified_human=verified_human,
            gitcoin_score=gitcoin_score,
            credentials_count=len(credentials),
            credentials_valid=len(valid_credentials),
            guardian_count=guardian_count,
            guardian_graph_diversity=guardian_diversity,
            risk_level=risk_level,
            sybil_resistance_score=sybil_resistance,
            last_verification=datetime.now(timezone.utc)
        )

        # Cache signal
        self.signal_cache[did] = signal

        return signal

    def enhance_matl_score(
        self,
        did: str,
        pogq_score: float,
        tcdm_score: float,
        entropy_score: float,
        identity_signals: Optional[IdentityTrustSignal] = None
    ) -> EnhancedMATLScore:
        """
        Enhance MATL trust score with identity verification

        Args:
            did: DID being scored
            pogq_score: Proof of Quality score (0.0-1.0)
            tcdm_score: Temporal/Community Diversity score (0.0-1.0)
            entropy_score: Behavioral entropy score (0.0-1.0)
            identity_signals: Pre-computed identity signals (optional)

        Returns:
            EnhancedMATLScore with identity boost applied
        """
        # Get identity signals from cache if not provided
        if not identity_signals:
            identity_signals = self.signal_cache.get(did)
            if not identity_signals:
                raise ValueError(f"No identity signals found for {did}")

        # Calculate base MATL score
        base_score = (pogq_score * 0.4) + (tcdm_score * 0.3) + (entropy_score * 0.3)

        # Calculate identity boost (-0.2 to +0.2)
        identity_boost = self._calculate_identity_boost(identity_signals)

        # Apply boost (clamped to [0.0, 1.0])
        enhanced_score = max(0.0, min(1.0, base_score + identity_boost))

        return EnhancedMATLScore(
            pogq_score=pogq_score,
            tcdm_score=tcdm_score,
            entropy_score=entropy_score,
            identity_boost=identity_boost,
            identity_signals=identity_signals,
            base_score=base_score,
            enhanced_score=enhanced_score
        )

    def _calculate_identity_boost(self, signals: IdentityTrustSignal) -> float:
        """
        Calculate identity boost/penalty for MATL score

        Boost range: -0.2 (anonymous) to +0.2 (maximum verification)

        Args:
            signals: Identity trust signals

        Returns:
            Boost value (-0.2 to +0.2)
        """
        # Base boost from assurance level
        assurance_boost = {
            AssuranceLevel.E0_ANONYMOUS: -0.20,
            AssuranceLevel.E1_TESTIMONIAL: -0.05,
            AssuranceLevel.E2_PRIVATELY_VERIFIABLE: 0.05,
            AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN: 0.12,
            AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL: 0.20,
        }[signals.assurance_level]

        # Additional boost from specific factors
        gitcoin_boost = 0.0
        if signals.gitcoin_score >= 50:
            gitcoin_boost = 0.03
        elif signals.gitcoin_score >= 20:
            gitcoin_boost = 0.02

        verified_human_boost = 0.03 if signals.verified_human else 0.0

        guardian_boost = 0.0
        if signals.guardian_count >= 5:
            guardian_boost = 0.02

        # Total boost (clamped to -0.2 to +0.2)
        total_boost = assurance_boost + gitcoin_boost + verified_human_boost + guardian_boost
        return max(-0.2, min(0.2, total_boost))

    def _calculate_sybil_resistance(
        self,
        assurance_score: float,
        gitcoin_score: float,
        verified_human: bool,
        factor_diversity: float
    ) -> float:
        """
        Calculate Sybil resistance score

        Combines multiple signals to estimate resistance to Sybil attacks

        Args:
            assurance_score: Identity assurance score (0.0-1.0)
            gitcoin_score: Gitcoin Passport score (0.0-100.0)
            verified_human: Has VerifiedHuman credential
            factor_diversity: Factor category diversity (0.0-1.0)

        Returns:
            Sybil resistance score (0.0-1.0)
        """
        # Normalize Gitcoin score to 0.0-1.0 (assuming max useful score is 100)
        gitcoin_normalized = min(gitcoin_score / 100.0, 1.0)

        # Weighted combination
        score = (
            assurance_score * 0.35 +
            gitcoin_normalized * 0.30 +
            (1.0 if verified_human else 0.0) * 0.20 +
            factor_diversity * 0.15
        )

        return min(score, 1.0)

    def _determine_risk_level(
        self,
        assurance_level: AssuranceLevel,
        sybil_resistance: float
    ) -> IdentityRiskLevel:
        """
        Determine identity risk level

        Args:
            assurance_level: Identity assurance level
            sybil_resistance: Sybil resistance score (0.0-1.0)

        Returns:
            IdentityRiskLevel
        """
        if assurance_level == AssuranceLevel.E0_ANONYMOUS:
            return IdentityRiskLevel.CRITICAL

        if assurance_level == AssuranceLevel.E1_TESTIMONIAL:
            return IdentityRiskLevel.HIGH

        if assurance_level == AssuranceLevel.E2_PRIVATELY_VERIFIABLE:
            if sybil_resistance >= 0.7:
                return IdentityRiskLevel.LOW
            return IdentityRiskLevel.MEDIUM

        if assurance_level == AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN:
            if sybil_resistance >= 0.8:
                return IdentityRiskLevel.MINIMAL
            return IdentityRiskLevel.LOW

        # E4: Constitutionally Critical
        return IdentityRiskLevel.MINIMAL

    def _calculate_guardian_diversity(
        self,
        guardian_dids: List[str],
        guardian_graph: Dict
    ) -> float:
        """
        Calculate diversity of guardian network

        Higher diversity = guardians from different clusters
        Lower diversity = guardians all know each other (potential cartel)

        Args:
            guardian_dids: List of guardian DIDs
            guardian_graph: Graph of guardian relationships

        Returns:
            Diversity score (0.0-1.0)
        """
        if not guardian_dids or len(guardian_dids) < 2:
            return 0.0

        # Calculate pairwise connections
        connections = 0
        max_connections = (len(guardian_dids) * (len(guardian_dids) - 1)) / 2

        for i, did1 in enumerate(guardian_dids):
            for did2 in guardian_dids[i+1:]:
                # Check if these guardians validate each other
                if self._are_connected(did1, did2, guardian_graph):
                    connections += 1

        # Diversity is inverse of connection density
        # High connections = low diversity (potential cartel)
        # Low connections = high diversity (independent guardians)
        if max_connections == 0:
            return 1.0

        connection_density = connections / max_connections
        diversity = 1.0 - connection_density

        return diversity

    @staticmethod
    def _are_connected(did1: str, did2: str, graph: Dict) -> bool:
        """Check if two DIDs are connected in the graph"""
        return did2 in graph.get(did1, []) or did1 in graph.get(did2, [])

    def get_initial_reputation(self, did: str) -> float:
        """
        Get initial MATL reputation based on identity verification

        New members start with reputation based on their identity assurance

        Args:
            did: DID to get initial reputation for

        Returns:
            Initial reputation score (0.3-0.7)
        """
        signals = self.signal_cache.get(did)
        if not signals:
            return 0.5  # Default neutral reputation

        # Map assurance level to initial reputation
        reputation_map = {
            AssuranceLevel.E0_ANONYMOUS: 0.30,  # Low starting reputation
            AssuranceLevel.E1_TESTIMONIAL: 0.40,
            AssuranceLevel.E2_PRIVATELY_VERIFIABLE: 0.50,  # Neutral
            AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN: 0.60,
            AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL: 0.70,  # High starting reputation
        }

        base_reputation = reputation_map[signals.assurance_level]

        # Boost for verified human
        if signals.verified_human:
            base_reputation += 0.05

        # Boost for high Gitcoin score
        if signals.gitcoin_score >= 50:
            base_reputation += 0.05

        return min(base_reputation, 0.7)  # Cap at 0.7


# Example usage
if __name__ == "__main__":
    from .factors import CryptoKeyFactor, GitcoinPassportFactor, SocialRecoveryFactor
    from .factors import FactorCategory
    from .verifiable_credentials import VCManager, CredentialSubject
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization

    print("=== MATL Integration Demo ===\n")

    # Create identity factors
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    crypto_factor = CryptoKeyFactor(
        factor_id="crypto-001",
        factor_type="CryptoKey",
        category=FactorCategory.PRIMARY,
        status=FactorStatus.ACTIVE,
        public_key=public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    )

    passport_factor = GitcoinPassportFactor(
        factor_id="gitcoin-001",
        factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION,
        status=FactorStatus.ACTIVE,
        passport_address="0x1234",
        score=45.0
    )

    recovery_factor = SocialRecoveryFactor(
        factor_id="recovery-001",
        factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL,
        status=FactorStatus.ACTIVE,
        threshold=3,
        guardian_dids=[
            "did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
            "did:mycelix:g4", "did:mycelix:g5"
        ]
    )

    factors = [crypto_factor, passport_factor, recovery_factor]

    # Create credentials
    vc_manager = VCManager()
    issuer_private = ed25519.Ed25519PrivateKey.generate()
    issuer_private_bytes = issuer_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    vh_credential = vc_manager.issue_credential(
        issuer_did="did:mycelix:vh_org",
        issuer_private_key=issuer_private_bytes,
        subject_did="did:mycelix:alice",
        vc_type=VCType.VERIFIED_HUMAN,
        claims={"verifiedHuman": True},
        validity_days=365
    )

    # Initialize bridge
    bridge = IdentityMATLBridge()

    # Compute identity signals
    print("Computing identity trust signals...")
    signals = bridge.compute_identity_signals(
        did="did:mycelix:alice",
        factors=factors,
        credentials=[vh_credential]
    )

    print(f"  Assurance Level: {signals.assurance_level.value}")
    print(f"  Assurance Score: {signals.assurance_score:.2f}")
    print(f"  Active Factors: {signals.active_factors}")
    print(f"  Factor Diversity: {signals.factor_diversity_score:.2f}")
    print(f"  Sybil Resistance: {signals.sybil_resistance_score:.2f}")
    print(f"  Risk Level: {signals.risk_level.value}")

    # Simulate MATL scores
    print("\nEnhancing MATL trust score...")
    enhanced = bridge.enhance_matl_score(
        did="did:mycelix:alice",
        pogq_score=0.75,  # Simulated PoGQ
        tcdm_score=0.80,  # Simulated TCDM
        entropy_score=0.70,  # Simulated Entropy
        identity_signals=signals
    )

    print(f"  Base MATL Score: {enhanced.base_score:.3f}")
    print(f"  Identity Boost: {enhanced.identity_boost:+.3f}")
    print(f"  Enhanced Score: {enhanced.enhanced_score:.3f}")

    print(f"\n  PoGQ (40%): {enhanced.pogq_score:.2f}")
    print(f"  TCDM (30%): {enhanced.tcdm_score:.2f}")
    print(f"  Entropy (30%): {enhanced.entropy_score:.2f}")

    # Get initial reputation
    print("\nInitial reputation for new member:")
    initial_rep = bridge.get_initial_reputation("did:mycelix:alice")
    print(f"  Starting reputation: {initial_rep:.2f}")
    print(f"  (vs default 0.50 for anonymous)")

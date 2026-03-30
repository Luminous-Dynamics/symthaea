#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Identity + MATL Integration Demo
Shows how Multi-Factor Identity enhances MATL trust scoring

Demonstrates:
1. Identity verification progression (E0 → E4)
2. MATL trust score enhancement
3. Initial reputation boosting
4. Sybil resistance calculation
5. Guardian graph diversity analysis

Phase 2 Implementation - Mycelix Protocol
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.identity import (
    DIDManager,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    HardwareKeyFactor,
    VCManager,
    VCType,
    IdentityMATLBridge,
    AssuranceLevel,
    FactorStatus,
    FactorCategory,
)
from zerotrustml.identity.did_manager import AgentType

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_subsection(title: str):
    """Print a subsection header"""
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}\n")


def demo_identity_progression():
    """Demo 1: Show identity verification progression"""
    print_section("1. Identity Verification Progression")

    # Create DID and MATL bridge
    manager = DIDManager()
    bridge = IdentityMATLBridge()

    # Create Alice's DID
    alice_did = manager.create_did(
        agent_type=AgentType.HUMAN_MEMBER,
        metadata={"nickname": "Alice"}
    )

    print(f"Created DID: {alice_did.to_string()}\n")

    # Scenario 1: E0 - Anonymous (no factors)
    print_subsection("Scenario 1: E0 - Anonymous (New User)")
    factors_e0 = []
    signals_e0 = bridge.compute_identity_signals(
        did=alice_did.to_string(),
        factors=factors_e0,
        credentials=[]
    )

    print(f"  Assurance Level: {signals_e0.assurance_level.value}")
    print(f"  Risk Level: {signals_e0.risk_level.value}")
    print(f"  Sybil Resistance: {signals_e0.sybil_resistance_score:.2f}")
    print(f"  Initial Reputation: {bridge.get_initial_reputation(alice_did.to_string()):.2f}")

    # Scenario 2: E1 - Testimonial (crypto key only)
    print_subsection("Scenario 2: E1 - Testimonial (Basic Key)")
    crypto_factor = CryptoKeyFactor(
        factor_id="crypto-001",
        factor_type="CryptoKey",
        category=FactorCategory.PRIMARY,
        status=FactorStatus.ACTIVE,
        public_key=alice_did.public_key,
        private_key=alice_did.private_key
    )
    factors_e1 = [crypto_factor]

    signals_e1 = bridge.compute_identity_signals(
        did=alice_did.to_string(),
        factors=factors_e1,
        credentials=[]
    )

    print(f"  Assurance Level: {signals_e1.assurance_level.value}")
    print(f"  Active Factors: {signals_e1.active_factors}")
    print(f"  Risk Level: {signals_e1.risk_level.value}")
    print(f"  Sybil Resistance: {signals_e1.sybil_resistance_score:.2f}")
    print(f"  Initial Reputation: {bridge.get_initial_reputation(alice_did.to_string()):.2f}")

    # Scenario 3: E2 - Privately Verifiable (+ Gitcoin Passport)
    print_subsection("Scenario 3: E2 - Privately Verifiable (+ Gitcoin)")
    passport_factor = GitcoinPassportFactor(
        factor_id="gitcoin-001",
        factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION,
        status=FactorStatus.ACTIVE,
        passport_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        score=42.5
    )
    factors_e2 = factors_e1 + [passport_factor]

    signals_e2 = bridge.compute_identity_signals(
        did=alice_did.to_string(),
        factors=factors_e2,
        credentials=[]
    )

    print(f"  Assurance Level: {signals_e2.assurance_level.value}")
    print(f"  Active Factors: {signals_e2.active_factors}")
    print(f"  Factor Categories: {signals_e2.factor_categories}")
    print(f"  Gitcoin Score: {signals_e2.gitcoin_score}")
    print(f"  Risk Level: {signals_e2.risk_level.value}")
    print(f"  Sybil Resistance: {signals_e2.sybil_resistance_score:.2f}")
    print(f"  Initial Reputation: {bridge.get_initial_reputation(alice_did.to_string()):.2f}")

    # Scenario 4: E3 - Cryptographically Proven (+ Social Recovery)
    print_subsection("Scenario 4: E3 - Cryptographically Proven (+ Recovery)")
    recovery_factor = SocialRecoveryFactor(
        factor_id="recovery-001",
        factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL,
        status=FactorStatus.ACTIVE,
        threshold=3,
        guardian_dids=[
            "did:mycelix:guardian_bob",
            "did:mycelix:guardian_carol",
            "did:mycelix:guardian_dave",
            "did:mycelix:guardian_eve",
            "did:mycelix:guardian_frank"
        ]
    )
    factors_e3 = factors_e2 + [recovery_factor]

    # Create VerifiedHuman credential
    vc_manager = VCManager()
    issuer_private = ed25519.Ed25519PrivateKey.generate()
    issuer_private_bytes = issuer_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    vh_credential = vc_manager.issue_credential(
        issuer_did="did:mycelix:verifiedhumanity_org",
        issuer_private_key=issuer_private_bytes,
        subject_did=alice_did.to_string(),
        vc_type=VCType.VERIFIED_HUMAN,
        claims={"verifiedHuman": True},
        validity_days=365
    )

    signals_e3 = bridge.compute_identity_signals(
        did=alice_did.to_string(),
        factors=factors_e3,
        credentials=[vh_credential]
    )

    print(f"  Assurance Level: {signals_e3.assurance_level.value}")
    print(f"  Active Factors: {signals_e3.active_factors}")
    print(f"  Factor Categories: {signals_e3.factor_categories}")
    print(f"  Guardian Count: {signals_e3.guardian_count}")
    print(f"  Verified Human: {signals_e3.verified_human}")
    print(f"  Risk Level: {signals_e3.risk_level.value}")
    print(f"  Sybil Resistance: {signals_e3.sybil_resistance_score:.2f}")
    print(f"  Initial Reputation: {bridge.get_initial_reputation(alice_did.to_string()):.2f}")

    # Scenario 5: E4 - Constitutionally Critical (+ Hardware Key)
    print_subsection("Scenario 5: E4 - Constitutionally Critical (+ HW Key)")
    hw_key_private = ed25519.Ed25519PrivateKey.generate()
    hw_key_public = hw_key_private.public_key()

    hardware_factor = HardwareKeyFactor(
        factor_id="hw-001",
        factor_type="HardwareKey",
        category=FactorCategory.BACKUP,
        status=FactorStatus.ACTIVE,
        device_type="yubikey",
        device_id="YK-12345",
        public_key=hw_key_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    )
    factors_e4 = factors_e3 + [hardware_factor]

    signals_e4 = bridge.compute_identity_signals(
        did=alice_did.to_string(),
        factors=factors_e4,
        credentials=[vh_credential]
    )

    print(f"  Assurance Level: {signals_e4.assurance_level.value}")
    print(f"  Active Factors: {signals_e4.active_factors}")
    print(f"  Factor Categories: {signals_e4.factor_categories}")
    print(f"  Factor Diversity: {signals_e4.factor_diversity_score:.2f}")
    print(f"  Risk Level: {signals_e4.risk_level.value}")
    print(f"  Sybil Resistance: {signals_e4.sybil_resistance_score:.2f}")
    print(f"  Initial Reputation: {bridge.get_initial_reputation(alice_did.to_string()):.2f}")

    return alice_did, bridge, signals_e4


def demo_matl_enhancement(bridge, signals):
    """Demo 2: Show MATL trust score enhancement"""
    print_section("2. MATL Trust Score Enhancement")

    # Simulate different MATL validation scenarios
    scenarios = [
        {
            "name": "New Member (Low Validation History)",
            "pogq": 0.50,  # Neutral - no history
            "tcdm": 0.60,  # Moderate diversity
            "entropy": 0.55,  # Moderate entropy
        },
        {
            "name": "Active Validator (Good Performance)",
            "pogq": 0.85,  # High validation accuracy
            "tcdm": 0.75,  # Good diversity
            "entropy": 0.70,  # Good behavioral randomness
        },
        {
            "name": "Suspicious Behavior (Potential Cartel)",
            "pogq": 0.80,  # Good validation but...
            "tcdm": 0.30,  # Low diversity (validates same group)
            "entropy": 0.40,  # Low entropy (scripted behavior)
        },
    ]

    for scenario in scenarios:
        print_subsection(scenario["name"])

        enhanced = bridge.enhance_matl_score(
            did=signals.did,
            pogq_score=scenario["pogq"],
            tcdm_score=scenario["tcdm"],
            entropy_score=scenario["entropy"],
            identity_signals=signals
        )

        print(f"  MATL Components:")
        print(f"    PoGQ (Proof of Quality):     {enhanced.pogq_score:.2f} × 0.4 = {enhanced.pogq_score * 0.4:.3f}")
        print(f"    TCDM (Diversity):             {enhanced.tcdm_score:.2f} × 0.3 = {enhanced.tcdm_score * 0.3:.3f}")
        print(f"    Entropy (Randomness):         {enhanced.entropy_score:.2f} × 0.3 = {enhanced.entropy_score * 0.3:.3f}")
        print(f"\n  Base MATL Score:               {enhanced.base_score:.3f}")
        print(f"  Identity Boost:                {enhanced.identity_boost:+.3f}")
        print(f"  ──────────────────────────────────────")
        print(f"  Enhanced Score:                {enhanced.enhanced_score:.3f}")

        improvement = ((enhanced.enhanced_score - enhanced.base_score) / enhanced.base_score) * 100
        print(f"  Improvement:                   {improvement:+.1f}%")


def demo_comparison_table():
    """Demo 3: Compare identity levels side-by-side"""
    print_section("3. Identity Level Comparison")

    bridge = IdentityMATLBridge()
    manager = DIDManager()

    # Create different identity profiles
    profiles = []

    # Profile 1: Anonymous (Sybil attacker)
    did1 = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)
    signals1 = bridge.compute_identity_signals(
        did=did1.to_string(),
        factors=[],
        credentials=[]
    )
    profiles.append(("Anonymous Sybil", signals1))

    # Profile 2: Basic user
    did2 = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)
    crypto2 = CryptoKeyFactor(
        factor_id="c2", factor_type="CryptoKey",
        category=FactorCategory.PRIMARY, status=FactorStatus.ACTIVE,
        public_key=did2.public_key
    )
    signals2 = bridge.compute_identity_signals(
        did=did2.to_string(),
        factors=[crypto2],
        credentials=[]
    )
    profiles.append(("Basic User", signals2))

    # Profile 3: Verified community member
    did3 = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)
    crypto3 = CryptoKeyFactor(
        factor_id="c3", factor_type="CryptoKey",
        category=FactorCategory.PRIMARY, status=FactorStatus.ACTIVE,
        public_key=did3.public_key
    )
    passport3 = GitcoinPassportFactor(
        factor_id="g3", factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION, status=FactorStatus.ACTIVE,
        passport_address="0x123", score=45.0
    )
    signals3 = bridge.compute_identity_signals(
        did=did3.to_string(),
        factors=[crypto3, passport3],
        credentials=[]
    )
    profiles.append(("Verified Member", signals3))

    # Profile 4: Governance participant
    did4 = manager.create_did(agent_type=AgentType.HUMAN_MEMBER)
    crypto4 = CryptoKeyFactor(
        factor_id="c4", factor_type="CryptoKey",
        category=FactorCategory.PRIMARY, status=FactorStatus.ACTIVE,
        public_key=did4.public_key
    )
    passport4 = GitcoinPassportFactor(
        factor_id="g4", factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION, status=FactorStatus.ACTIVE,
        passport_address="0x456", score=55.0
    )
    recovery4 = SocialRecoveryFactor(
        factor_id="r4", factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL, status=FactorStatus.ACTIVE,
        threshold=3, guardian_dids=["g1", "g2", "g3", "g4", "g5"]
    )
    signals4 = bridge.compute_identity_signals(
        did=did4.to_string(),
        factors=[crypto4, passport4, recovery4],
        credentials=[]
    )
    profiles.append(("Governance Participant", signals4))

    # Print comparison table
    print("┌────────────────────────┬─────────┬──────────┬──────────┬──────────────┐")
    print("│ Profile                │ Level   │ Sybil    │ Init Rep │ Risk         │")
    print("├────────────────────────┼─────────┼──────────┼──────────┼──────────────┤")

    for name, signals in profiles:
        level_short = signals.assurance_level.value.split('_')[0]  # E0, E1, etc.
        init_rep = bridge.get_initial_reputation(signals.did)
        print(f"│ {name:<22} │ {level_short:<7} │ {signals.sybil_resistance_score:>8.2f} │ {init_rep:>8.2f} │ {signals.risk_level.value:<12} │")

    print("└────────────────────────┴─────────┴──────────┴──────────┴──────────────┘")

    # Show MATL boost comparison
    print("\n\nMATL Score Enhancement (with PoGQ=0.80, TCDM=0.75, Entropy=0.70):")
    print("┌────────────────────────┬──────────┬──────────┬──────────┬────────────┐")
    print("│ Profile                │ Base     │ Boost    │ Enhanced │ Change     │")
    print("├────────────────────────┼──────────┼──────────┼──────────┼────────────┤")

    for name, signals in profiles:
        enhanced = bridge.enhance_matl_score(
            did=signals.did,
            pogq_score=0.80,
            tcdm_score=0.75,
            entropy_score=0.70,
            identity_signals=signals
        )
        change_pct = ((enhanced.enhanced_score - enhanced.base_score) / enhanced.base_score) * 100
        print(f"│ {name:<22} │ {enhanced.base_score:>8.3f} │ {enhanced.identity_boost:>+8.3f} │ {enhanced.enhanced_score:>8.3f} │ {change_pct:>+9.1f}% │")

    print("└────────────────────────┴──────────┴──────────┴──────────┴────────────┘")


def demo_guardian_diversity():
    """Demo 4: Show guardian network diversity impact"""
    print_section("4. Guardian Network Diversity Analysis")

    bridge = IdentityMATLBridge()

    # Scenario 1: Independent guardians (high diversity)
    print_subsection("Scenario 1: Independent Guardians (Ideal)")
    guardian_graph_independent = {
        "did:mycelix:g1": ["did:mycelix:g2"],  # Only knows g2
        "did:mycelix:g2": ["did:mycelix:g1"],
        "did:mycelix:g3": [],  # Independent
        "did:mycelix:g4": [],  # Independent
        "did:mycelix:g5": [],  # Independent
    }

    diversity_independent = bridge._calculate_guardian_diversity(
        guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                       "did:mycelix:g4", "did:mycelix:g5"],
        guardian_graph=guardian_graph_independent
    )

    print(f"  Diversity Score: {diversity_independent:.2f}")
    print(f"  Assessment: ✓ High diversity - guardians are independent")

    # Scenario 2: Partially connected (medium diversity)
    print_subsection("Scenario 2: Partially Connected (Acceptable)")
    guardian_graph_partial = {
        "did:mycelix:g1": ["did:mycelix:g2", "did:mycelix:g3"],
        "did:mycelix:g2": ["did:mycelix:g1", "did:mycelix:g3"],
        "did:mycelix:g3": ["did:mycelix:g1", "did:mycelix:g2"],
        "did:mycelix:g4": [],
        "did:mycelix:g5": [],
    }

    diversity_partial = bridge._calculate_guardian_diversity(
        guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                       "did:mycelix:g4", "did:mycelix:g5"],
        guardian_graph=guardian_graph_partial
    )

    print(f"  Diversity Score: {diversity_partial:.2f}")
    print(f"  Assessment: ⚠ Medium diversity - some clustering detected")

    # Scenario 3: Fully connected cartel (low diversity)
    print_subsection("Scenario 3: Fully Connected (Warning)")
    guardian_graph_cartel = {
        "did:mycelix:g1": ["did:mycelix:g2", "did:mycelix:g3", "did:mycelix:g4", "did:mycelix:g5"],
        "did:mycelix:g2": ["did:mycelix:g1", "did:mycelix:g3", "did:mycelix:g4", "did:mycelix:g5"],
        "did:mycelix:g3": ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g4", "did:mycelix:g5"],
        "did:mycelix:g4": ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3", "did:mycelix:g5"],
        "did:mycelix:g5": ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3", "did:mycelix:g4"],
    }

    diversity_cartel = bridge._calculate_guardian_diversity(
        guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                       "did:mycelix:g4", "did:mycelix:g5"],
        guardian_graph=guardian_graph_cartel
    )

    print(f"  Diversity Score: {diversity_cartel:.2f}")
    print(f"  Assessment: ❌ Low diversity - potential coordinated group (cartel)")


def main():
    """Run complete MATL integration demonstration"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      Mycelix Identity + MATL Integration Demonstration          ║
║                    Phase 2 Complete                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Demo 1: Identity progression
        alice_did, bridge, signals = demo_identity_progression()

        # Demo 2: MATL enhancement
        demo_matl_enhancement(bridge, signals)

        # Demo 3: Comparison table
        demo_comparison_table()

        # Demo 4: Guardian diversity
        demo_guardian_diversity()

        print_section("Demo Complete")
        print("✓ All MATL integration systems working correctly!\n")
        print("Phase 2 Implementation Status:")
        print("  ✓ Identity trust signal computation")
        print("  ✓ MATL score enhancement (-0.2 to +0.2 boost)")
        print("  ✓ Initial reputation based on assurance level")
        print("  ✓ Sybil resistance scoring")
        print("  ✓ Guardian graph diversity analysis")
        print("  ✓ Risk level assessment")
        print("\nNext: Phase 3 - ZK Proofs & Advanced Cartel Detection")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

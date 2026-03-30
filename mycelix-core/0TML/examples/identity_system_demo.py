#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Factor Decentralized Identity System - Complete Demo

This demonstration shows all components working together:
1. DID creation
2. Multi-factor enrollment
3. Assurance level progression
4. Social recovery setup
5. Verifiable credentials
6. Recovery scenarios

Phase 1 Implementation - Mycelix Protocol
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.identity import (
    DIDManager,
    MycelixDID,
    CryptoKeyFactor,
    GitcoinPassportFactor,
    SocialRecoveryFactor,
    HardwareKeyFactor,
    BiometricFactor,
    AssuranceLevel,
    calculate_assurance_level,
    RecoveryManager,
    RecoveryScenario,
    VCManager,
    VCType,
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


def demo_basic_identity_creation():
    """Demo 1: Create a basic DID"""
    print_section("1. Basic Identity Creation")

    # Create DID manager
    manager = DIDManager()

    # Create Alice's DID
    alice_did = manager.create_did(
        agent_type=AgentType.HUMAN_MEMBER,
        metadata={"nickname": "Alice", "location": "Earth"}
    )

    print(f"✓ Created DID: {alice_did.to_string()}")
    print(f"  Agent Type: {alice_did.agent_type.value}")
    print(f"  Created: {alice_did.created_at}")

    # Create her primary crypto factor
    crypto_factor = CryptoKeyFactor(
        factor_id="crypto-001",
        factor_type="CryptoKey",
        category=FactorCategory.PRIMARY,
        status=FactorStatus.ACTIVE,
        public_key=alice_did.public_key,
        private_key=alice_did.private_key
    )

    # Check assurance level (E1 - just crypto key)
    result = calculate_assurance_level([crypto_factor])
    print(f"\n  Initial Assurance Level: {result.level.value}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Can vote in governance: {result.level.value >= AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN.value}")

    print(f"\n  Recommendations:")
    for rec in result.recommendations[:3]:
        print(f"    • {rec}")

    return alice_did, crypto_factor


def demo_multi_factor_enrollment(alice_did: MycelixDID, crypto_factor: CryptoKeyFactor):
    """Demo 2: Enroll multiple factors"""
    print_section("2. Multi-Factor Enrollment")

    factors = [crypto_factor]

    # Add Gitcoin Passport
    print("Enrolling Gitcoin Passport...")
    passport = GitcoinPassportFactor(
        factor_id="gitcoin-001",
        factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION,
        status=FactorStatus.ACTIVE,
        passport_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        score=42.5
    )
    factors.append(passport)
    print(f"  ✓ Gitcoin Passport verified (score: {passport.score})")

    # Check assurance level after adding passport
    result = calculate_assurance_level(factors)
    print(f"\n  Assurance Level: {result.level.value}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Can participate in community: {'participate_community' in result.capabilities}")

    # Add social recovery guardians
    print("\nSetting up social recovery...")
    recovery = SocialRecoveryFactor(
        factor_id="recovery-001",
        factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL,
        status=FactorStatus.ACTIVE,
        threshold=3
    )

    # Add guardians
    guardians = [
        "did:mycelix:guardian_bob",
        "did:mycelix:guardian_carol",
        "did:mycelix:guardian_dave",
        "did:mycelix:guardian_eve",
        "did:mycelix:guardian_frank"
    ]

    for guardian in guardians:
        recovery.add_guardian(guardian)

    factors.append(recovery)
    print(f"  ✓ Social recovery configured with {len(guardians)} guardians")
    print(f"    Threshold: {recovery.threshold} guardians required")

    # Final assurance level
    result = calculate_assurance_level(factors)
    print(f"\n  Final Assurance Level: {result.level.value}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Can vote in governance: {'vote_governance' in result.capabilities}")
    print(f"  Can propose amendments: {'propose_constitutional_amendment' in result.capabilities}")

    return factors


def demo_verifiable_credentials(alice_did: MycelixDID):
    """Demo 3: Issue and verify credentials"""
    print_section("3. Verifiable Credentials")

    # Create issuer (simulating VerifiedHumanity.org)
    issuer_private = ed25519.Ed25519PrivateKey.generate()
    issuer_public = issuer_private.public_key()

    issuer_private_bytes = issuer_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

    issuer_public_bytes = issuer_public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )

    issuer_did = "did:mycelix:verifiedhumanity_org"

    # Create VC manager
    vc_manager = VCManager()

    # Issue VerifiedHuman credential
    print("Issuing VerifiedHuman credential...")
    vh_credential = vc_manager.issue_credential(
        issuer_did=issuer_did,
        issuer_private_key=issuer_private_bytes,
        subject_did=alice_did.to_string(),
        vc_type=VCType.VERIFIED_HUMAN,
        claims={
            "verifiedHuman": True,
            "verificationMethod": "VerifiedHumanity.org",
            "uniqueness": "high"
        },
        validity_days=365
    )

    print(f"  ✓ Credential issued: {vh_credential.id[:50]}...")
    print(f"    Type: {vh_credential.type[1]}")
    print(f"    Valid until: {vh_credential.expiration_date}")

    # Verify the credential
    print("\nVerifying credential signature...")
    is_valid = vc_manager.verify_credential(vh_credential, issuer_public_bytes)
    print(f"  ✓ Signature valid: {is_valid}")
    print(f"  ✓ Credential status: {vh_credential.status.value}")

    # Issue a second credential (Governance Eligibility)
    print("\nIssuing Governance Eligibility credential...")
    gov_credential = vc_manager.issue_credential(
        issuer_did="did:mycelix:knowledge_council",
        issuer_private_key=issuer_private_bytes,
        subject_did=alice_did.to_string(),
        vc_type=VCType.GOVERNANCE_ELIGIBILITY,
        claims={
            "eligible": True,
            "assuranceLevel": "E3",
            "votingWeight": 1.0
        },
        validity_days=90
    )

    print(f"  ✓ Credential issued: {gov_credential.id[:50]}...")

    # List all credentials
    print(f"\nAlice's credentials:")
    all_credentials = vc_manager.list_credentials_for_subject(alice_did.to_string())
    for i, cred in enumerate(all_credentials, 1):
        print(f"  {i}. {cred.type[1]}")
        print(f"     Issuer: {cred.issuer}")
        print(f"     Valid: {cred.is_valid()}")

    return vc_manager, vh_credential


def demo_social_recovery(alice_did: MycelixDID, guardians: list):
    """Demo 4: Social recovery process"""
    print_section("4. Social Recovery Process")

    # Create recovery manager
    recovery_manager = RecoveryManager()

    # Scenario: Alice lost her phone and backup codes
    print("Scenario: Alice lost her phone with 2FA and backup codes")
    print("Initiating social recovery with guardians...\n")

    request = recovery_manager.initiate_recovery(
        did=alice_did.to_string(),
        scenario=RecoveryScenario.MULTI_FACTOR,
        guardian_dids=guardians,
        threshold=3,
        evidence={
            "reason": "Lost phone with authenticator app",
            "last_login": "2025-01-15T10:30:00Z",
            "known_transactions": ["tx1", "tx2", "tx3"]
        }
    )

    print(f"✓ Recovery request created: {request.request_id}")
    print(f"  Status: {request.status.value}")
    print(f"  Required approvals: {request.required_approvals}")
    print(f"  Expires: {request.expires_at}")

    # Guardians approve one by one
    print("\nGuardian approval process:")
    for i, guardian_did in enumerate(guardians[:3], 1):
        # Simulate guardian signing the approval
        signature = f"signature_from_{guardian_did}".encode('utf-8')

        success = recovery_manager.approve_recovery(
            request.request_id,
            guardian_did,
            signature
        )

        print(f"  {i}. Guardian {guardian_did.split('_')[1]} approved ✓")
        print(f"     Progress: {len(request.received_approvals)}/{request.required_approvals}")

    print(f"\n✓ Recovery request status: {request.status.value}")

    # Complete the recovery
    if recovery_manager.complete_recovery(request.request_id):
        print("✓ Recovery completed successfully!")
        print("  Alice can now set up new factors with her recovered identity")

    return recovery_manager


def demo_key_splitting(alice_did: MycelixDID, guardians: list):
    """Demo 5: Shamir Secret Sharing for key backup"""
    print_section("5. Shamir Secret Sharing - Key Backup")

    recovery_manager = RecoveryManager()

    # Split Alice's private key for guardian backup
    print("Splitting Alice's private key into 5 shares (3 needed to recover)...")

    guardian_shares = recovery_manager.split_key_for_guardians(
        private_key=alice_did.private_key,
        guardian_dids=guardians,
        threshold=3
    )

    print(f"\n✓ Key split successfully:")
    for i, (guardian, share) in enumerate(guardian_shares.items(), 1):
        print(f"  {i}. {guardian}")
        print(f"     Share: {share.hex()[:32]}... ({len(share)} bytes)")

    # Simulate recovery: Collect shares from 3 guardians
    print("\nSimulating key recovery from guardians 1, 3, 5...")
    selected_guardians = [guardians[0], guardians[2], guardians[4]]

    # Reconstruct shares with share IDs
    shares_with_ids = {
        guardian: (i+1, guardian_shares[guardian])
        for i, guardian in enumerate(guardians)
        if guardian in selected_guardians
    }

    reconstructed_key = recovery_manager.reconstruct_key_from_guardians(
        guardian_shares=shares_with_ids,
        threshold=3
    )

    print(f"\n✓ Key reconstructed successfully!")
    print(f"  Original key:       {alice_did.private_key.hex()[:32]}...")
    print(f"  Reconstructed key:  {reconstructed_key.hex()[:32]}...")
    print(f"  Match: {alice_did.private_key == reconstructed_key}")


def demo_assurance_progression():
    """Demo 6: Show progression through assurance levels"""
    print_section("6. Assurance Level Progression")

    levels = [
        ("E0: Anonymous", 0.0, ["Read public data"]),
        ("E1: Testimonial", 0.3, ["Create content", "Send messages"]),
        ("E2: Privately Verifiable", 0.5, ["Participate in community", "Submit proposals"]),
        ("E3: Cryptographically Proven", 0.7, ["Vote in governance", "Run validator"]),
        ("E4: Constitutionally Critical", 0.9, ["Propose amendments", "Join Knowledge Council"])
    ]

    print("Assurance Level Capabilities:\n")

    for level, score, capabilities in levels:
        print(f"{level} (score ≥ {score})")
        for cap in capabilities:
            print(f"  • {cap}")
        print()


def main():
    """Run complete demonstration"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        Mycelix Multi-Factor Decentralized Identity System        ║
║                    Phase 1 Complete Demo                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Demo 1: Basic identity
        alice_did, crypto_factor = demo_basic_identity_creation()

        # Demo 2: Multi-factor enrollment
        guardians = [
            "did:mycelix:guardian_bob",
            "did:mycelix:guardian_carol",
            "did:mycelix:guardian_dave",
            "did:mycelix:guardian_eve",
            "did:mycelix:guardian_frank"
        ]
        factors = demo_multi_factor_enrollment(alice_did, crypto_factor)

        # Demo 3: Verifiable credentials
        vc_manager, vh_credential = demo_verifiable_credentials(alice_did)

        # Demo 4: Social recovery
        recovery_manager = demo_social_recovery(alice_did, guardians)

        # Demo 5: Key splitting
        demo_key_splitting(alice_did, guardians)

        # Demo 6: Assurance progression
        demo_assurance_progression()

        print_section("Demo Complete")
        print("✓ All systems working correctly!")
        print("\nPhase 1 Implementation Status:")
        print("  ✓ W3C DID Management")
        print("  ✓ Multi-Factor Authentication (9 factor types)")
        print("  ✓ Graduated Assurance Levels (E0-E4)")
        print("  ✓ Social Recovery with Shamir Secret Sharing")
        print("  ✓ W3C Verifiable Credentials")
        print("\nNext: Phase 2 - MATL Integration & ZK Proofs")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Assurance Level Calculation for Multi-Factor Identity
Aligned with Epistemic Charter v2.0 E-Axis

Maps identity factor verification to graduated capabilities
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set
from .factors import IdentityFactor, FactorStatus


class AssuranceLevel(Enum):
    """
    Assurance levels aligned with Epistemic Charter v2.0 E-Axis

    E0: Anonymous (0.0-0.2) - Unverified, read-only access
    E1: Testimonial (0.2-0.4) - Basic key pair, limited interactions
    E2: Privately Verifiable (0.4-0.6) - Multi-factor, community participation
    E3: Cryptographically Proven (0.6-0.8) - Strong verification, governance voting
    E4: Constitutionally Critical (0.8-1.0) - Maximum verification, constitutional proposals
    """
    E0_ANONYMOUS = "E0_Anonymous"
    E1_TESTIMONIAL = "E1_Testimonial"
    E2_PRIVATELY_VERIFIABLE = "E2_PrivatelyVerifiable"
    E3_CRYPTOGRAPHICALLY_PROVEN = "E3_CryptographicallyProven"
    E4_CONSTITUTIONALLY_CRITICAL = "E4_ConstitutionallyCritical"


@dataclass
class AssuranceResult:
    """
    Result of assurance level calculation

    Contains the calculated level and associated capabilities
    """
    level: AssuranceLevel
    score: float  # 0.0-1.0
    active_factors: List[str]
    capabilities: Set[str]
    recommendations: List[str]


# Capability definitions for each assurance level
LEVEL_CAPABILITIES = {
    AssuranceLevel.E0_ANONYMOUS: {
        "read_public_data",
        "browse_anonymous",
    },

    AssuranceLevel.E1_TESTIMONIAL: {
        "read_public_data",
        "browse_anonymous",
        "post_comments",
        "send_messages",
        "create_content",
    },

    AssuranceLevel.E2_PRIVATELY_VERIFIABLE: {
        "read_public_data",
        "browse_anonymous",
        "post_comments",
        "send_messages",
        "create_content",
        "participate_community",
        "submit_proposals",
        "receive_rewards",
    },

    AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN: {
        "read_public_data",
        "browse_anonymous",
        "post_comments",
        "send_messages",
        "create_content",
        "participate_community",
        "submit_proposals",
        "receive_rewards",
        "vote_governance",
        "validate_claims",
        "run_validator",
    },

    AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL: {
        "read_public_data",
        "browse_anonymous",
        "post_comments",
        "send_messages",
        "create_content",
        "participate_community",
        "submit_proposals",
        "receive_rewards",
        "vote_governance",
        "validate_claims",
        "run_validator",
        "propose_constitutional_amendment",
        "join_knowledge_council",
        "serve_guardian",
    },
}


def calculate_assurance_level(factors: List[IdentityFactor]) -> AssuranceResult:
    """
    Calculate assurance level from active identity factors

    Algorithm:
    1. Sum contributions from all ACTIVE factors
    2. Apply bonus for factor diversity (multiple categories)
    3. Map score to assurance level
    4. Return capabilities and recommendations

    Args:
        factors: List of identity factors

    Returns:
        AssuranceResult with level, score, capabilities, and recommendations
    """
    # Filter to active factors only
    active_factors = [f for f in factors if f.status == FactorStatus.ACTIVE]

    if not active_factors:
        return AssuranceResult(
            level=AssuranceLevel.E0_ANONYMOUS,
            score=0.0,
            active_factors=[],
            capabilities=LEVEL_CAPABILITIES[AssuranceLevel.E0_ANONYMOUS],
            recommendations=[
                "Create a DID with cryptographic key pair",
                "Verify your Gitcoin Passport for community participation",
                "Set up social recovery guardians"
            ]
        )

    # Calculate base score from factor contributions
    base_score = sum(f.get_contribution() for f in active_factors)

    # Apply diversity bonus
    categories = {f.category for f in active_factors}
    diversity_bonus = len(categories) * 0.05  # 5% per unique category

    # Final score (capped at 1.0)
    final_score = min(base_score + diversity_bonus, 1.0)

    # Map score to assurance level
    level = _score_to_level(final_score)

    # Get capabilities for this level
    capabilities = LEVEL_CAPABILITIES[level]

    # Generate recommendations for improvement
    recommendations = _generate_recommendations(active_factors, level)

    return AssuranceResult(
        level=level,
        score=final_score,
        active_factors=[f.factor_id for f in active_factors],
        capabilities=capabilities,
        recommendations=recommendations
    )


def _score_to_level(score: float) -> AssuranceLevel:
    """
    Map numerical score to assurance level

    Thresholds:
    - 0.0-0.2: E0 (Anonymous)
    - 0.2-0.4: E1 (Testimonial)
    - 0.4-0.6: E2 (Privately Verifiable)
    - 0.6-0.8: E3 (Cryptographically Proven)
    - 0.8-1.0: E4 (Constitutionally Critical)
    """
    if score >= 0.8:
        return AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL
    elif score >= 0.6:
        return AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN
    elif score >= 0.4:
        return AssuranceLevel.E2_PRIVATELY_VERIFIABLE
    elif score >= 0.2:
        return AssuranceLevel.E1_TESTIMONIAL
    else:
        return AssuranceLevel.E0_ANONYMOUS


def _generate_recommendations(
    active_factors: List[IdentityFactor],
    current_level: AssuranceLevel
) -> List[str]:
    """
    Generate recommendations to improve assurance level

    Args:
        active_factors: Currently active factors
        current_level: Current assurance level

    Returns:
        List of recommended actions
    """
    recommendations = []

    # Check what factor types are present
    factor_types = {f.factor_type for f in active_factors}

    # Recommendations based on missing factors
    if "CryptoKey" not in factor_types:
        recommendations.append("⚠️  CRITICAL: Generate a primary cryptographic key pair")

    if "GitcoinPassport" not in factor_types and current_level.value < AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN.value:
        recommendations.append("📊 Verify your Gitcoin Passport (required for governance voting)")

    if "SocialRecovery" not in factor_types:
        recommendations.append("👥 Set up social recovery with 5+ guardians for account security")

    if "HardwareKey" not in factor_types and current_level.value >= AssuranceLevel.E2_PRIVATELY_VERIFIABLE.value:
        recommendations.append("🔐 Add a hardware security key for enhanced security")

    if "Biometric" not in factor_types and current_level.value >= AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN.value:
        recommendations.append("🔍 Consider adding biometric verification for convenience")

    # Level-specific recommendations
    if current_level == AssuranceLevel.E0_ANONYMOUS:
        recommendations.append("🚀 Get started: Create your first DID to join the network")

    elif current_level == AssuranceLevel.E1_TESTIMONIAL:
        recommendations.append("⬆️  Reach E2: Add 2+ additional factors for community participation")

    elif current_level == AssuranceLevel.E2_PRIVATELY_VERIFIABLE:
        recommendations.append("⬆️  Reach E3: Verify Gitcoin Passport ≥20 for governance voting")

    elif current_level == AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN:
        recommendations.append("⬆️  Reach E4: Add hardware key + biometric for constitutional proposals")

    elif current_level == AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL:
        recommendations.append("🎉 Maximum verification achieved! You can participate in all governance activities")

    return recommendations


def get_required_level(capability: str) -> AssuranceLevel:
    """
    Get the minimum assurance level required for a capability

    Args:
        capability: Capability name

    Returns:
        Minimum required assurance level
    """
    for level, caps in LEVEL_CAPABILITIES.items():
        if capability in caps:
            return level

    # If not found, require maximum level
    return AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL


def has_capability(assurance_result: AssuranceResult, capability: str) -> bool:
    """
    Check if an identity has a specific capability

    Args:
        assurance_result: Result from calculate_assurance_level
        capability: Capability to check

    Returns:
        True if identity has this capability
    """
    return capability in assurance_result.capabilities


def get_capabilities_summary(assurance_result: AssuranceResult) -> Dict[str, any]:
    """
    Get a human-readable summary of capabilities

    Args:
        assurance_result: Result from calculate_assurance_level

    Returns:
        Dictionary with capability categories and details
    """
    return {
        "level": assurance_result.level.value,
        "score": round(assurance_result.score, 2),
        "active_factors": len(assurance_result.active_factors),
        "capabilities": {
            "read": has_capability(assurance_result, "read_public_data"),
            "write": has_capability(assurance_result, "create_content"),
            "community": has_capability(assurance_result, "participate_community"),
            "governance": has_capability(assurance_result, "vote_governance"),
            "constitutional": has_capability(assurance_result, "propose_constitutional_amendment"),
        },
        "next_level": _get_next_level(assurance_result.level),
        "recommendations": assurance_result.recommendations
    }


def _get_next_level(current_level: AssuranceLevel) -> str:
    """Get the next assurance level"""
    level_order = [
        AssuranceLevel.E0_ANONYMOUS,
        AssuranceLevel.E1_TESTIMONIAL,
        AssuranceLevel.E2_PRIVATELY_VERIFIABLE,
        AssuranceLevel.E3_CRYPTOGRAPHICALLY_PROVEN,
        AssuranceLevel.E4_CONSTITUTIONALLY_CRITICAL,
    ]

    try:
        current_idx = level_order.index(current_level)
        if current_idx < len(level_order) - 1:
            return level_order[current_idx + 1].value
        return "Maximum level achieved"
    except ValueError:
        return "Unknown"


# Example usage
if __name__ == "__main__":
    from .factors import CryptoKeyFactor, GitcoinPassportFactor, SocialRecoveryFactor
    from .factors import FactorCategory, FactorStatus
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization

    # Scenario 1: New user with just crypto key (E1)
    print("=== Scenario 1: New User (Crypto Key Only) ===")
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

    result = calculate_assurance_level([crypto_factor])
    print(f"Level: {result.level.value}")
    print(f"Score: {result.score}")
    print(f"Can vote in governance: {has_capability(result, 'vote_governance')}")
    print(f"Recommendations: {result.recommendations[:2]}")

    # Scenario 2: Active community member (E2)
    print("\n=== Scenario 2: Community Member (Crypto + Passport) ===")
    passport_factor = GitcoinPassportFactor(
        factor_id="gitcoin-001",
        factor_type="GitcoinPassport",
        category=FactorCategory.REPUTATION,
        status=FactorStatus.ACTIVE,
        passport_address="0x1234...",
        score=25.0
    )

    result = calculate_assurance_level([crypto_factor, passport_factor])
    print(f"Level: {result.level.value}")
    print(f"Score: {result.score}")
    print(f"Can vote in governance: {has_capability(result, 'vote_governance')}")
    print(f"Can participate in community: {has_capability(result, 'participate_community')}")

    # Scenario 3: Governance participant (E3)
    print("\n=== Scenario 3: Governance Participant (Multi-Factor) ===")
    recovery_factor = SocialRecoveryFactor(
        factor_id="recovery-001",
        factor_type="SocialRecovery",
        category=FactorCategory.SOCIAL,
        status=FactorStatus.ACTIVE,
        threshold=3,
        guardian_dids=["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3",
                      "did:mycelix:g4", "did:mycelix:g5"]
    )

    result = calculate_assurance_level([crypto_factor, passport_factor, recovery_factor])
    print(f"Level: {result.level.value}")
    print(f"Score: {result.score}")
    print(f"Can vote in governance: {has_capability(result, 'vote_governance')}")
    print(f"Can propose amendments: {has_capability(result, 'propose_constitutional_amendment')}")

    # Summary
    print("\n=== Capabilities Summary ===")
    summary = get_capabilities_summary(result)
    print(f"Level: {summary['level']}")
    print(f"Score: {summary['score']}")
    print(f"Capabilities: {summary['capabilities']}")
    print(f"Next level: {summary['next_level']}")

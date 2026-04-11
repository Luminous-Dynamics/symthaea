// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consciousness Gating Types (Legacy)
 *
 * @deprecated Use `@mycelix/sdk/core/sovereign-gate` for the 8D sovereign
 * profile system. This module is retained for backward compatibility.
 *
 * The 4D consciousness profile has been superseded by the 8D sovereign
 * profile (CivicTier, SovereignProfile, CivicRequirement).
 */

// =============================================================================
// Consciousness Tiers (matching Rust: consciousness_profile.rs)
// =============================================================================

/**
 * @deprecated Use `CivicTier` from `@mycelix/sdk/core/sovereign-gate` instead.
 * The tier values are identical (Observer→Guardian).
 */
export enum ConsciousnessTier {
  /** Read-only access, no governance participation */
  Observer = 'Observer',
  /** Can create basic proposals */
  Participant = 'Participant',
  /** Full voting rights */
  Citizen = 'Citizen',
  /** Constitutional actions, council participation */
  Steward = 'Steward',
  /** Emergency powers, system parameter changes */
  Guardian = 'Guardian',
}

/** Tier threshold constants */
export const TIER_THRESHOLDS = {
  Observer: 0.0,
  Participant: 0.30,
  Citizen: 0.40,
  Steward: 0.60,
  Guardian: 0.80,
} as const;

/** Hysteresis margin to prevent tier oscillation */
export const HYSTERESIS_MARGIN = 0.05;

// =============================================================================
// 4D Consciousness Profile
// =============================================================================

/** Dimension weights for combined score computation */
export const DIMENSION_WEIGHTS = {
  identity: 0.25,
  reputation: 0.25,
  community: 0.30,
  engagement: 0.20,
} as const;

/**
 * 4-dimensional consciousness profile.
 *
 * Each dimension is a score in [0.0, 1.0]:
 * - identity:   MFA assurance level (Anonymous=0.0, Critical=1.0)
 * - reputation:  Cross-hApp aggregated reputation with exponential decay
 * - community:   Peer trust attestations, weighted by attestor tier
 * - engagement:  Domain-specific participation (computed locally per bridge)
 */
export interface ConsciousnessProfile {
  /** MFA assurance level [0.0, 1.0] */
  identity: number;
  /** Cross-hApp aggregated reputation [0.0, 1.0] */
  reputation: number;
  /** Peer trust attestations [0.0, 1.0] */
  community: number;
  /** Domain-specific participation [0.0, 1.0] */
  engagement: number;
}

/**
 * Computed consciousness result combining profile with derived tier.
 */
export interface ConsciousnessResult {
  profile: ConsciousnessProfile;
  combinedScore: number;
  tier: ConsciousnessTier;
  /** Vote weight in basis points (0-10000) */
  voteWeight: number;
}

// =============================================================================
// Governance Requirements
// =============================================================================

/**
 * Governance requirement for an action.
 * Specifies the minimum tier and optional per-dimension minimums.
 */
export interface GovernanceRequirement {
  /** Minimum tier required */
  minTier: ConsciousnessTier;
  /** Optional minimum identity score (e.g., constitutional changes require identity verification) */
  minIdentity?: number;
  /** Optional minimum community score */
  minCommunity?: number;
}

/**
 * Result of a governance eligibility check.
 */
export interface GovernanceEligibility {
  eligible: boolean;
  currentTier: ConsciousnessTier;
  requiredTier: ConsciousnessTier;
  reason?: string;
}

// =============================================================================
// Consciousness Credential
// =============================================================================

/**
 * On-chain consciousness credential with TTL.
 * Issued by bridge coordinators, cached for 24 hours.
 */
export interface ConsciousnessCredential {
  /** DID of the agent */
  did: string;
  /** Profile dimensions at issuance time */
  profile: ConsciousnessProfile;
  /** Derived tier */
  tier: ConsciousnessTier;
  /** Issuance timestamp (microseconds) */
  issuedAt: number;
  /** Expiry timestamp (microseconds) — default TTL is 24 hours */
  expiresAt: number;
}

// =============================================================================
// Preset Requirements (matching Rust VoteWeightConfig presets)
// =============================================================================

/** Preset governance requirements for common action types */
export const GOVERNANCE_PRESETS = {
  /** Basic actions: Participant tier minimum */
  basic: { minTier: ConsciousnessTier.Participant } as GovernanceRequirement,
  /** Voting: Citizen tier minimum */
  voting: { minTier: ConsciousnessTier.Citizen } as GovernanceRequirement,
  /** Constitutional: Steward tier + identity verification */
  constitutional: { minTier: ConsciousnessTier.Steward, minIdentity: 0.5 } as GovernanceRequirement,
  /** Budget: Steward tier */
  budget: { minTier: ConsciousnessTier.Steward } as GovernanceRequirement,
  /** Emergency: Guardian tier + community trust */
  emergency: { minTier: ConsciousnessTier.Guardian, minCommunity: 0.6 } as GovernanceRequirement,
} as const;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Compute the combined consciousness score from a profile.
 */
export function computeCombinedScore(profile: ConsciousnessProfile): number {
  return (
    DIMENSION_WEIGHTS.identity * profile.identity +
    DIMENSION_WEIGHTS.reputation * profile.reputation +
    DIMENSION_WEIGHTS.community * profile.community +
    DIMENSION_WEIGHTS.engagement * profile.engagement
  );
}

/**
 * Derive the consciousness tier from a combined score.
 */
export function deriveTier(combinedScore: number): ConsciousnessTier {
  if (combinedScore >= TIER_THRESHOLDS.Guardian) return ConsciousnessTier.Guardian;
  if (combinedScore >= TIER_THRESHOLDS.Steward) return ConsciousnessTier.Steward;
  if (combinedScore >= TIER_THRESHOLDS.Citizen) return ConsciousnessTier.Citizen;
  if (combinedScore >= TIER_THRESHOLDS.Participant) return ConsciousnessTier.Participant;
  return ConsciousnessTier.Observer;
}

/**
 * Compute vote weight in basis points for a given tier.
 */
export function voteWeightForTier(tier: ConsciousnessTier): number {
  switch (tier) {
    case ConsciousnessTier.Observer: return 0;
    case ConsciousnessTier.Participant: return 5000;
    case ConsciousnessTier.Citizen: return 7500;
    case ConsciousnessTier.Steward: return 10000;
    case ConsciousnessTier.Guardian: return 10000;
  }
}

/**
 * Evaluate a full consciousness result from a profile.
 */
export function evaluateConsciousness(profile: ConsciousnessProfile): ConsciousnessResult {
  const combinedScore = computeCombinedScore(profile);
  const tier = deriveTier(combinedScore);
  return {
    profile,
    combinedScore,
    tier,
    voteWeight: voteWeightForTier(tier),
  };
}

/**
 * Check if a profile meets a governance requirement.
 */
export function checkEligibility(
  profile: ConsciousnessProfile,
  requirement: GovernanceRequirement,
): GovernanceEligibility {
  const result = evaluateConsciousness(profile);
  const tierOrder = [
    ConsciousnessTier.Observer,
    ConsciousnessTier.Participant,
    ConsciousnessTier.Citizen,
    ConsciousnessTier.Steward,
    ConsciousnessTier.Guardian,
  ];

  const currentIdx = tierOrder.indexOf(result.tier);
  const requiredIdx = tierOrder.indexOf(requirement.minTier);

  if (currentIdx < requiredIdx) {
    return {
      eligible: false,
      currentTier: result.tier,
      requiredTier: requirement.minTier,
      reason: `Requires ${requirement.minTier} tier (current: ${result.tier})`,
    };
  }

  if (requirement.minIdentity !== undefined && profile.identity < requirement.minIdentity) {
    return {
      eligible: false,
      currentTier: result.tier,
      requiredTier: requirement.minTier,
      reason: `Identity score ${profile.identity.toFixed(2)} below required ${requirement.minIdentity}`,
    };
  }

  if (requirement.minCommunity !== undefined && profile.community < requirement.minCommunity) {
    return {
      eligible: false,
      currentTier: result.tier,
      requiredTier: requirement.minTier,
      reason: `Community score ${profile.community.toFixed(2)} below required ${requirement.minCommunity}`,
    };
  }

  return {
    eligible: true,
    currentTier: result.tier,
    requiredTier: requirement.minTier,
  };
}

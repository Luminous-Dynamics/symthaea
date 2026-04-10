// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * 8D Sovereign Profile — Anti-tyranny civic identity for Mycelix governance.
 *
 * TypeScript mirror of the Rust `sovereign-profile` crate.
 *
 * When you reduce a human being to a single number, you create a system
 * that is infinitely gamifiable and inherently oppressive. The 8D Sovereign
 * Profile maps identity as a holographic, multi-faceted geometry.
 *
 * @module sovereign-profile
 */

// ============================================================================
// Types
// ============================================================================

/** The 8 axes of civic identity. */
export type SovereignDimension =
  | "EpistemicIntegrity"
  | "ThermodynamicYield"
  | "NetworkResilience"
  | "EconomicVelocity"
  | "CivicParticipation"
  | "StewardshipCare"
  | "SemanticResonance"
  | "DomainCompetence";

/** All 8 dimensions in canonical order. */
export const SOVEREIGN_DIMENSIONS: SovereignDimension[] = [
  "EpistemicIntegrity",
  "ThermodynamicYield",
  "NetworkResilience",
  "EconomicVelocity",
  "CivicParticipation",
  "StewardshipCare",
  "SemanticResonance",
  "DomainCompetence",
];

/** 8-dimensional sovereign civic profile. All values [0.0, 1.0]. */
export interface SovereignProfile {
  /** D0: Truth/claim verification history. */
  epistemic_integrity: number;
  /** D1: Physical energy contribution to local commons. */
  thermodynamic_yield: number;
  /** D2: Node uptime and bandwidth provision. */
  network_resilience: number;
  /** D3: Compliance with anti-hoarding demurrage. */
  economic_velocity: number;
  /** D4: Sortition jury duty, voting, dispute resolution. */
  civic_participation: number;
  /** D5: Verified physical labor on the commons. */
  stewardship_care: number;
  /** D6: Entanglement with community consensus. */
  semantic_resonance: number;
  /** D7: Peer-verified specialization. */
  domain_competence: number;
}

/** Progressive civic tier derived from the combined 8D score. */
export type CivicTier = "Observer" | "Participant" | "Citizen" | "Steward" | "Guardian";

/** All tiers in ascending order. */
export const CIVIC_TIERS: CivicTier[] = [
  "Observer",
  "Participant",
  "Citizen",
  "Steward",
  "Guardian",
];

/** Minimum combined score for each tier. */
export const TIER_THRESHOLDS: Record<CivicTier, number> = {
  Observer: 0.0,
  Participant: 0.3,
  Citizen: 0.4,
  Steward: 0.6,
  Guardian: 0.8,
};

/** Vote weight in basis points for each tier. */
export const TIER_VOTE_WEIGHT_BP: Record<CivicTier, number> = {
  Observer: 0,
  Participant: 5000,
  Citizen: 7500,
  Steward: 10000,
  Guardian: 10000,
};

/** Community-configurable dimension weights. Must sum to 1.0. */
export interface DimensionWeights {
  weights: [number, number, number, number, number, number, number, number];
}

/** A governance requirement: minimum tier + optional per-dimension minimums. */
export interface CivicRequirement {
  min_tier: CivicTier;
  min_dimensions: Array<[SovereignDimension, number]>;
}

/** Time-limited sovereign credential. */
export interface SovereignCredential {
  did: string;
  profile: SovereignProfile;
  tier: CivicTier;
  issued_at: number;
  expires_at: number;
  issuer: string;
}

// ============================================================================
// Weight presets
// ============================================================================

/** Equal weights: each dimension 12.5%. */
export const WEIGHTS_EQUAL: DimensionWeights = {
  weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
};

/** Default governance weights. */
export const WEIGHTS_GOVERNANCE: DimensionWeights = {
  weights: [0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10],
};

/** Energy cooperative weights. */
export const WEIGHTS_ENERGY: DimensionWeights = {
  weights: [0.10, 0.22, 0.18, 0.10, 0.10, 0.12, 0.10, 0.08],
};

/** Knowledge commons weights. */
export const WEIGHTS_KNOWLEDGE: DimensionWeights = {
  weights: [0.22, 0.06, 0.08, 0.08, 0.12, 0.10, 0.14, 0.20],
};

/** Care community weights. */
export const WEIGHTS_CARE: DimensionWeights = {
  weights: [0.08, 0.08, 0.08, 0.10, 0.14, 0.22, 0.20, 0.10],
};

// ============================================================================
// Dimension labels (i18n keys + English defaults)
// ============================================================================

export interface DimensionLabel {
  key: string;
  name: string;
  description: string;
  icon: string;
}

export const DIMENSION_LABELS: DimensionLabel[] = [
  { key: "epistemic_integrity", name: "Epistemic Integrity", description: "Your truth-telling track record", icon: "search" },
  { key: "thermodynamic_yield", name: "Thermodynamic Yield", description: "Energy contributed to the commons", icon: "bolt" },
  { key: "network_resilience", name: "Network Resilience", description: "Infrastructure uptime and reliability", icon: "wifi" },
  { key: "economic_velocity", name: "Economic Velocity", description: "Healthy circulation of mutual credit", icon: "refresh" },
  { key: "civic_participation", name: "Civic Participation", description: "Governance engagement and jury service", icon: "vote" },
  { key: "stewardship_care", name: "Stewardship & Care", description: "Verified labor maintaining the commons", icon: "hands" },
  { key: "semantic_resonance", name: "Semantic Resonance", description: "Alignment with community values", icon: "heart" },
  { key: "domain_competence", name: "Domain Competence", description: "Peer-verified expertise in your field", icon: "star" },
];

// ============================================================================
// Functions
// ============================================================================

/** Sanitize a dimension value: NaN/Infinity → 0, clamp [0, 1]. */
function sanitize(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/** Get all 8 dimension values as an array in canonical order. */
export function profileToArray(profile: SovereignProfile): number[] {
  return [
    profile.epistemic_integrity,
    profile.thermodynamic_yield,
    profile.network_resilience,
    profile.economic_velocity,
    profile.civic_participation,
    profile.stewardship_care,
    profile.semantic_resonance,
    profile.domain_competence,
  ];
}

/** Combined score using dimension weights. */
export function combinedScore(
  profile: SovereignProfile,
  weights: DimensionWeights = WEIGHTS_GOVERNANCE,
): number {
  const dims = profileToArray(profile);
  let score = 0;
  for (let i = 0; i < 8; i++) {
    score += sanitize(dims[i]) * weights.weights[i];
  }
  return Math.max(0, Math.min(1, score));
}

/** Derive civic tier from combined score. */
export function civicTierFromScore(score: number): CivicTier {
  if (score >= 0.8) return "Guardian";
  if (score >= 0.6) return "Steward";
  if (score >= 0.4) return "Citizen";
  if (score >= 0.3) return "Participant";
  return "Observer";
}

/** Derive civic tier from a profile. */
export function civicTier(
  profile: SovereignProfile,
  weights: DimensionWeights = WEIGHTS_GOVERNANCE,
): CivicTier {
  return civicTierFromScore(combinedScore(profile, weights));
}

/** Check whether a profile meets a civic requirement. */
export function meetsRequirement(
  profile: SovereignProfile,
  requirement: CivicRequirement,
  weights: DimensionWeights = WEIGHTS_GOVERNANCE,
): boolean {
  const tier = civicTier(profile, weights);
  const tierRank = CIVIC_TIERS.indexOf(tier);
  const requiredRank = CIVIC_TIERS.indexOf(requirement.min_tier);
  if (tierRank < requiredRank) return false;

  const dims = profileToArray(profile);
  for (const [dim, minValue] of requirement.min_dimensions) {
    const idx = SOVEREIGN_DIMENSIONS.indexOf(dim);
    if (idx >= 0 && sanitize(dims[idx]) < minValue) return false;
  }
  return true;
}

// ============================================================================
// Decay
// ============================================================================

/** Constitutional bounds for decay rate lambda. */
export const LAMBDA_MIN = 0.001; // half-life ≈ 693 days
export const LAMBDA_MAX = 0.020; // half-life ≈ 35 days

/** Apply exponential decay to a score. */
export function decayScore(rawScore: number, lambda: number, elapsedDays: number): number {
  if (!Number.isFinite(lambda) || !Number.isFinite(elapsedDays) || elapsedDays < 0) return 0;
  const decayed = rawScore * Math.exp(-lambda * elapsedDays);
  return Math.max(0, Math.min(1, decayed));
}

/** Half-life in days for a given lambda. */
export function halfLifeDays(lambda: number): number {
  return lambda > 0 ? Math.LN2 / lambda : Infinity;
}

/** Days until a score drops below a threshold. */
export function daysUntilThreshold(
  currentScore: number,
  lambda: number,
  threshold: number,
): number {
  if (currentScore <= threshold) return 0;
  if (threshold <= 0 || lambda <= 0) return Infinity;
  return -Math.log(threshold / currentScore) / lambda;
}

// ============================================================================
// Backward compatibility
// ============================================================================

/**
 * Legacy 4D ConsciousnessProfile — backward-compatible alias.
 *
 * @deprecated Use SovereignProfile instead.
 */
export interface ConsciousnessProfile {
  identity: number;
  reputation: number;
  community: number;
  engagement: number;
}

/**
 * Convert a legacy 4D profile to an 8D SovereignProfile.
 *
 * Maps: identity → epistemic+network, reputation → economic+stewardship,
 * community → civic+resonance, engagement → thermodynamic+competence.
 */
export function sovereignFromLegacy(legacy: ConsciousnessProfile): SovereignProfile {
  return {
    epistemic_integrity: sanitize(legacy.identity),
    thermodynamic_yield: sanitize(legacy.engagement),
    network_resilience: sanitize(legacy.identity),
    economic_velocity: sanitize(legacy.reputation),
    civic_participation: sanitize(legacy.community),
    stewardship_care: sanitize(legacy.reputation),
    semantic_resonance: sanitize(legacy.community),
    domain_competence: sanitize(legacy.engagement),
  };
}

/**
 * Legacy combined score using old 4D weights.
 *
 * @deprecated Use combinedScore() with SovereignProfile instead.
 */
export function legacyCombinedScore(profile: ConsciousnessProfile): number {
  const i = sanitize(profile.identity);
  const r = sanitize(profile.reputation);
  const c = sanitize(profile.community);
  const e = sanitize(profile.engagement);
  return Math.max(0, Math.min(1, i * 0.25 + r * 0.25 + c * 0.30 + e * 0.20));
}

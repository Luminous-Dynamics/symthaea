// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * 8D Sovereign Profile Gate — anti-tyranny civic identity middleware.
 *
 * Drop-in replacement for consciousness-gate.ts with 8 dimensions instead of 4.
 * Backward compatible: ConsciousnessProfile imports still work via type aliases.
 *
 * @module @mycelix/sdk/core/sovereign-gate
 */

import { SdkError, SdkErrorCode } from './errors.js';

// ============================================================================
// Types
// ============================================================================

/** The 8 axes of civic identity. */
export type SovereignDimension =
  | 'EpistemicIntegrity'
  | 'ThermodynamicYield'
  | 'NetworkResilience'
  | 'EconomicVelocity'
  | 'CivicParticipation'
  | 'StewardshipCare'
  | 'SemanticResonance'
  | 'DomainCompetence';

/** Progressive civic tier. */
export type CivicTier = 'Observer' | 'Participant' | 'Citizen' | 'Steward' | 'Guardian';

/** 8-dimensional sovereign profile. All values [0, 1]. */
export interface SovereignProfile {
  epistemic_integrity: number;
  thermodynamic_yield: number;
  network_resilience: number;
  economic_velocity: number;
  civic_participation: number;
  stewardship_care: number;
  semantic_resonance: number;
  domain_competence: number;
}

/** Community-configurable dimension weights. */
export interface DimensionWeights {
  weights: [number, number, number, number, number, number, number, number];
}

/** Civic requirement: minimum tier + optional per-dimension minimums. */
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

/** Pre-flight eligibility check result. */
export interface CivicEligibility {
  tier: CivicTier;
  eligible: boolean;
  combinedScore: number;
  reasons: string[];
  nearingExpiry: boolean;
  daysUntilDemotion: number;
}

/** Structured rejection from a civic gate. */
export interface CivicGateRejection {
  action: string;
  zome: string;
  actualTier: CivicTier;
  requiredTier: CivicTier;
  weightBp: number;
  reasons: string[];
  expired: boolean;
}

// ============================================================================
// Constants
// ============================================================================

export const CIVIC_TIERS: CivicTier[] = ['Observer', 'Participant', 'Citizen', 'Steward', 'Guardian'];

export const TIER_THRESHOLDS: Record<CivicTier, number> = {
  Observer: 0.0, Participant: 0.3, Citizen: 0.4, Steward: 0.6, Guardian: 0.8,
};

export const TIER_VOTE_WEIGHT_BP: Record<CivicTier, number> = {
  Observer: 0, Participant: 5000, Citizen: 7500, Steward: 10000, Guardian: 10000,
};

/** Default governance weights. */
export const WEIGHTS_GOVERNANCE: DimensionWeights = {
  weights: [0.15, 0.10, 0.10, 0.12, 0.18, 0.13, 0.12, 0.10],
};

/** Dimension labels for UI. */
export const DIMENSION_LABELS = [
  { key: 'epistemic_integrity', name: 'Epistemic Integrity', description: 'Your truth-telling track record', icon: 'search' },
  { key: 'thermodynamic_yield', name: 'Thermodynamic Yield', description: 'Energy contributed to the commons', icon: 'bolt' },
  { key: 'network_resilience', name: 'Network Resilience', description: 'Infrastructure uptime and reliability', icon: 'wifi' },
  { key: 'economic_velocity', name: 'Economic Velocity', description: 'Healthy circulation of mutual credit', icon: 'refresh' },
  { key: 'civic_participation', name: 'Civic Participation', description: 'Governance engagement and jury service', icon: 'vote' },
  { key: 'stewardship_care', name: 'Stewardship & Care', description: 'Verified labor maintaining the commons', icon: 'hands' },
  { key: 'semantic_resonance', name: 'Semantic Resonance', description: 'Alignment with community values', icon: 'heart' },
  { key: 'domain_competence', name: 'Domain Competence', description: 'Peer-verified expertise in your field', icon: 'star' },
] as const;

// ============================================================================
// Functions
// ============================================================================

function sanitize(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/** Combined score using dimension weights. */
export function combinedScore(
  profile: SovereignProfile,
  weights: DimensionWeights = WEIGHTS_GOVERNANCE,
): number {
  const dims = [
    profile.epistemic_integrity, profile.thermodynamic_yield,
    profile.network_resilience, profile.economic_velocity,
    profile.civic_participation, profile.stewardship_care,
    profile.semantic_resonance, profile.domain_competence,
  ];
  let score = 0;
  for (let i = 0; i < 8; i++) score += sanitize(dims[i]) * weights.weights[i];
  return Math.max(0, Math.min(1, score));
}

/** Derive civic tier from score. */
export function tierFromScore(score: number): CivicTier {
  if (score >= 0.8) return 'Guardian';
  if (score >= 0.6) return 'Steward';
  if (score >= 0.4) return 'Citizen';
  if (score >= 0.3) return 'Participant';
  return 'Observer';
}

/** Check requirement. */
export function meetsRequirement(
  profile: SovereignProfile,
  requirement: CivicRequirement,
  weights: DimensionWeights = WEIGHTS_GOVERNANCE,
): boolean {
  const tier = tierFromScore(combinedScore(profile, weights));
  if (CIVIC_TIERS.indexOf(tier) < CIVIC_TIERS.indexOf(requirement.min_tier)) return false;
  const dims = [
    profile.epistemic_integrity, profile.thermodynamic_yield,
    profile.network_resilience, profile.economic_velocity,
    profile.civic_participation, profile.stewardship_care,
    profile.semantic_resonance, profile.domain_competence,
  ];
  const dimNames: SovereignDimension[] = [
    'EpistemicIntegrity', 'ThermodynamicYield', 'NetworkResilience', 'EconomicVelocity',
    'CivicParticipation', 'StewardshipCare', 'SemanticResonance', 'DomainCompetence',
  ];
  for (const [dim, minVal] of requirement.min_dimensions) {
    const idx = dimNames.indexOf(dim);
    if (idx >= 0 && sanitize(dims[idx]) < minVal) return false;
  }
  return true;
}

// ============================================================================
// Gate Error Parsing
// ============================================================================

/** Parsed civic gate error from WASM. */
export class CivicGateError extends SdkError {
  public readonly rejection: CivicGateRejection;

  constructor(rejection: CivicGateRejection) {
    super(
      SdkErrorCode.UNAUTHORIZED,
      `Civic gate: ${rejection.actualTier} insufficient for ${rejection.requiredTier}`,
      { rejection },
    );
    this.rejection = rejection;
  }

  /** Try to parse a WASM error string into a CivicGateError. */
  static fromWasmError(error: unknown): CivicGateError | null {
    if (!error || typeof error !== 'object') return null;
    const msg = 'message' in error ? String((error as { message: string }).message) : '';
    if (!msg.includes('Civic gate:') && !msg.includes('gate_civic')) return null;

    const tierMatch = msg.match(/tier (\w+) insufficient/);
    const actualTier = (tierMatch?.[1] ?? 'Observer') as CivicTier;
    const reasons = msg.split('Reasons:')[1]?.split(',').map(s => s.trim()) ?? [];

    return new CivicGateError({
      action: 'unknown',
      zome: 'unknown',
      actualTier,
      requiredTier: 'Citizen',
      weightBp: TIER_VOTE_WEIGHT_BP[actualTier],
      reasons,
      expired: msg.includes('expired'),
    });
  }
}

// ============================================================================
// Decay
// ============================================================================

export const LAMBDA_MIN = 0.001;
export const LAMBDA_MAX = 0.020;

export function decayScore(rawScore: number, lambda: number, elapsedDays: number): number {
  if (!Number.isFinite(lambda) || !Number.isFinite(elapsedDays) || elapsedDays < 0) return 0;
  return Math.max(0, Math.min(1, rawScore * Math.exp(-lambda * elapsedDays)));
}

export function daysUntilThreshold(score: number, lambda: number, threshold: number): number {
  if (score <= threshold) return 0;
  if (threshold <= 0 || lambda <= 0) return Infinity;
  return -Math.log(threshold / score) / lambda;
}

// ============================================================================
// Backward compatibility
// ============================================================================

/** @deprecated Use SovereignProfile */
export type ConsciousnessProfile = {
  identity: number;
  reputation: number;
  community: number;
  engagement: number;
};

/** @deprecated Use CivicTier */
export type ConsciousnessTier = CivicTier;

/** @deprecated Use combinedScore with SovereignProfile */
export function legacyCombinedScore(profile: ConsciousnessProfile): number {
  const i = sanitize(profile.identity);
  const r = sanitize(profile.reputation);
  const c = sanitize(profile.community);
  const e = sanitize(profile.engagement);
  return Math.max(0, Math.min(1, i * 0.25 + r * 0.25 + c * 0.30 + e * 0.20));
}

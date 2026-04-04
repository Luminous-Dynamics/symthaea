// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Consciousness profile and trust tier types.
 *
 * Maps to the Rust `ConsciousnessProfile` in
 * `crates/mycelix-bridge-common/src/consciousness_profile.rs`.
 *
 * The 4D consciousness model gates governance actions with
 * progressive vote weighting via sigmoid authorization.
 */

/**
 * Trust tier derived from the combined consciousness score.
 *
 * Thresholds (with hysteresis margin of 0.05):
 * - Observer:    combined < 0.3
 * - Participant: combined >= 0.3
 * - Citizen:     combined >= 0.4
 * - Steward:     combined >= 0.6
 * - Guardian:    combined >= 0.8
 */
export type TrustTier =
  | "Observer"
  | "Participant"
  | "Citizen"
  | "Steward"
  | "Guardian";

/** Numeric values for tier comparison. */
export const TIER_RANK: Record<TrustTier, number> = {
  Observer: 0,
  Participant: 1,
  Citizen: 2,
  Steward: 3,
  Guardian: 4,
};

/**
 * 4-dimensional consciousness profile for governance gating.
 *
 * All dimensions are normalized to [0.0, 1.0].
 */
export interface ConsciousnessProfile {
  /**
   * Identity verification strength (from MFA AssuranceLevel).
   * Anonymous=0.0, Basic=0.25, Verified=0.5, HighlyAssured=0.75, Critical=1.0
   */
  identity: number;

  /**
   * Cross-hApp reputation with exponential decay (30-day half-life).
   * Multi-source weighted average from the identity bridge.
   */
  reputation: number;

  /**
   * Community trust attestations from aggregated peer credentials.
   * Weighted by attestor's own tier.
   */
  community: number;

  /**
   * Domain-specific engagement, computed locally by each cluster bridge.
   * Formula: min(1.0, raw_events / baseline_events)
   */
  engagement: number;
}

/**
 * Compute the combined score from a consciousness profile.
 *
 * Uses equal weighting across all four dimensions:
 *   combined = (identity + reputation + community + engagement) / 4
 *
 * @param profile - The 4D consciousness profile
 * @returns Combined score in [0.0, 1.0]
 */
export function combinedScore(profile: ConsciousnessProfile): number {
  const sum =
    profile.identity +
    profile.reputation +
    profile.community +
    profile.engagement;
  return Math.min(1.0, Math.max(0.0, sum / 4.0));
}

/**
 * Derive the trust tier from a consciousness profile.
 *
 * @param profile - The 4D consciousness profile
 * @returns The computed trust tier
 */
export function deriveTier(profile: ConsciousnessProfile): TrustTier {
  const score = combinedScore(profile);
  if (score >= 0.8) return "Guardian";
  if (score >= 0.6) return "Steward";
  if (score >= 0.4) return "Citizen";
  if (score >= 0.3) return "Participant";
  return "Observer";
}

/**
 * Check if a profile meets a minimum tier requirement.
 *
 * @param profile - The 4D consciousness profile
 * @param requiredTier - Minimum tier required
 * @returns True if the profile meets or exceeds the required tier
 */
export function meetsTier(
  profile: ConsciousnessProfile,
  requiredTier: TrustTier,
): boolean {
  return TIER_RANK[deriveTier(profile)] >= TIER_RANK[requiredTier];
}

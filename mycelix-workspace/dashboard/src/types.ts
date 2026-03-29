// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TypeScript types matching the Rust types from mycelix-bridge-common
 * and the SDK consciousness-gate module.
 *
 * Source of truth:
 *   Rust:  crates/mycelix-bridge-common/src/consciousness_profile.rs
 *   TS:    sdk-ts/src/core/consciousness-gate.ts
 */

/** Consciousness tier levels — matches Rust ConsciousnessTier enum */
export type ConsciousnessTier =
  | 'Observer'
  | 'Participant'
  | 'Citizen'
  | 'Steward'
  | 'Guardian';

/** Ordered list for iteration and display */
export const TIER_ORDER: ConsciousnessTier[] = [
  'Observer',
  'Participant',
  'Citizen',
  'Steward',
  'Guardian',
];

/** Minimum combined score for each tier (from Rust ConsciousnessTier::min_score) */
export const TIER_THRESHOLDS: Record<ConsciousnessTier, number> = {
  Observer: 0.0,
  Participant: 0.3,
  Citizen: 0.4,
  Steward: 0.6,
  Guardian: 0.8,
};

/** Vote weight in basis points per tier (from Rust ConsciousnessTier::vote_weight_bp) */
export const TIER_VOTE_WEIGHT_BP: Record<ConsciousnessTier, number> = {
  Observer: 0,
  Participant: 5000,
  Citizen: 7500,
  Steward: 10000,
  Guardian: 10000,
};

/**
 * 4-dimensional consciousness profile.
 * Each dimension is 0.0-1.0.
 *
 * Matches Rust: ConsciousnessProfile
 */
export interface ConsciousnessProfile {
  /** Identity verification strength (MFA AssuranceLevel: 0.0-1.0) */
  identity: number;
  /** Cross-hApp aggregated reputation with exponential decay */
  reputation: number;
  /** Peer trust attestations, weighted by attestor tier */
  community: number;
  /** Domain-specific participation, computed locally per bridge */
  engagement: number;
}

/**
 * Time-limited consciousness credential.
 * Matches Rust: ConsciousnessCredential
 */
export interface ConsciousnessCredential {
  did: string;
  profile: ConsciousnessProfile;
  tier: ConsciousnessTier;
  issued_at: number;
  expires_at: number;
  issuer: string;
}

/**
 * Result of evaluating a profile against a governance requirement.
 * Matches Rust: GovernanceEligibility
 */
export interface GovernanceEligibility {
  eligible: boolean;
  weight_bp: number;
  tier: ConsciousnessTier;
  profile: ConsciousnessProfile;
  reasons: string[];
}

/**
 * Input for logging a governance gate decision.
 * Matches Rust: GateAuditInput
 */
export interface GateAuditEntry {
  /** The extern function that triggered the gate check */
  action_name: string;
  /** The zome that performed the check */
  zome_name: string;
  /** Whether the agent met all requirements */
  eligible: boolean;
  /** The agent's derived consciousness tier (Debug-formatted) */
  actual_tier: string;
  /** The minimum tier required (Debug-formatted) */
  required_tier: string;
  /** Progressive vote weight in basis points (0-10000) */
  weight_bp: number;
  /** Optional correlation ID for cross-cluster audit trail */
  correlation_id?: string;
  /** Timestamp (added by the dashboard for display) */
  timestamp: number;
  /** Agent DID (added by dashboard for display) */
  agent_did: string;
}

/**
 * Filter for querying governance gate audit events.
 * Matches SDK: GovernanceAuditFilter
 */
export interface GovernanceAuditFilter {
  action_name?: string;
  zome_name?: string;
  eligible?: boolean;
  from_us?: number;
  to_us?: number;
}

/** Tier distribution entry for the pie/bar chart */
export interface TierDistributionEntry {
  tier: ConsciousnessTier;
  count: number;
  percentage: number;
}

/** Time-series data point for gate decisions */
export interface GateDecisionPoint {
  /** ISO timestamp (truncated to minute) */
  timestamp: string;
  approvals: number;
  rejections: number;
  action_type: string;
}

/**
 * Combined score formula:
 *   identity * 0.25 + reputation * 0.25 + community * 0.30 + engagement * 0.20
 */
export function combinedScore(profile: ConsciousnessProfile): number {
  return (
    profile.identity * 0.25 +
    profile.reputation * 0.25 +
    profile.community * 0.3 +
    profile.engagement * 0.2
  );
}

/** Derive tier from combined score */
export function tierFromScore(score: number): ConsciousnessTier {
  if (score >= 0.8) return 'Guardian';
  if (score >= 0.6) return 'Steward';
  if (score >= 0.4) return 'Citizen';
  if (score >= 0.3) return 'Participant';
  return 'Observer';
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Metabolism Types
 *
 * Types for the Knowledge Metabolism system - managing claim lifecycles.
 * Claims are born, grow with evidence, mature, and eventually die.
 * Dead claims leave tombstones to preserve lessons learned.
 *
 * @packageDocumentation
 * @module metabolism/types
 */

import type { EpistemicClassification, Evidence } from '../epistemic/index.js';

// ============================================================================
// LIFECYCLE PHASES
// ============================================================================

/**
 * Lifecycle phase of a claim
 *
 * Claims progress through phases like organisms:
 * - nascent: Just created, minimal evidence
 * - growing: Accumulating evidence and citations
 * - mature: Stable, well-evidenced
 * - stagnant: No recent activity, may be outdated
 * - challenged: Under active dispute
 * - dying: Evidence is being refuted
 * - dead: Conclusively refuted or obsolete
 * - decomposing: Being broken down for useful components
 */
export type LifecyclePhase =
  | 'nascent'
  | 'growing'
  | 'mature'
  | 'stagnant'
  | 'challenged'
  | 'dying'
  | 'dead'
  | 'decomposing';

/**
 * Reason a claim died
 */
export type DeathReason =
  | 'refuted'           // Evidence proved it wrong
  | 'superseded'        // Replaced by better claim
  | 'obsolete'          // No longer relevant
  | 'retracted'         // Author withdrew it
  | 'merged'            // Combined into another claim
  | 'decomposed'        // Broken into component parts
  | 'abandoned';        // No engagement, expired

// ============================================================================
// HEALTH METRICS
// ============================================================================

/**
 * Health metrics for a claim
 *
 * Tracks the "vital signs" of a claim to determine lifecycle transitions.
 */
export interface ClaimHealth {
  /** Confidence level (0-1) */
  confidence: number;

  /** Evidence strength (0-1) */
  evidenceStrength: number;

  /** Citation count (how many other claims reference this) */
  citationCount: number;

  /** Challenge count (active disputes) */
  challengeCount: number;

  /** Days since last activity */
  daysSinceActivity: number;

  /** Corroboration score (independent confirmations) */
  corroborationScore: number;

  /** Contradiction score (conflicting evidence) */
  contradictionScore: number;

  /** Author reputation at time of claim */
  authorReputation: number;

  /** Calculated overall health (0-1) */
  overallHealth: number;
}

/**
 * Activity event on a claim
 */
export interface ClaimActivity {
  /** Activity type */
  type: 'evidence_added' | 'cited' | 'challenged' | 'corroborated' | 'contradicted' | 'updated' | 'viewed';

  /** When it happened */
  timestamp: number;

  /** Who did it */
  actorId: string;

  /** Additional context */
  details?: Record<string, unknown>;
}

// ============================================================================
// TOMBSTONES
// ============================================================================

/**
 * Tombstone - The memorial of a dead claim
 *
 * When claims die, we don't delete them. We create tombstones that:
 * - Preserve what we learned from the claim's death
 * - Prevent the same mistake from being repeated
 * - Enable counterfactual reasoning
 * - Maintain audit trail
 */
export interface Tombstone {
  /** Original claim ID */
  claimId: string;

  /** Original claim content (preserved) */
  originalContent: string;

  /** Original classification */
  originalClassification: EpistemicClassification;

  /** When the claim was created */
  bornAt: number;

  /** When the claim died */
  diedAt: number;

  /** Why it died */
  deathReason: DeathReason;

  /** What we learned from this death */
  lessonLearned: string;

  /** Evidence that led to death */
  refutingEvidence: Evidence[];

  /** Claims that were updated because this died */
  affectedClaims: string[];

  /** If superseded, what replaced it */
  supersededBy?: string;

  /** Tags for searchability */
  tags: string[];

  /** Counterfactual: What would be different if this had been true? */
  counterfactual?: string;

  /** Warning patterns to detect similar claims */
  warningPatterns?: string[];
}

/**
 * Tombstone summary for quick lookup
 */
export interface TombstoneSummary {
  claimId: string;
  originalContent: string;
  deathReason: DeathReason;
  lessonLearned: string;
  diedAt: number;
}

// ============================================================================
// LIFECYCLE TRANSITIONS
// ============================================================================

/**
 * Lifecycle transition event
 */
export interface LifecycleTransition {
  /** Claim ID */
  claimId: string;

  /** Previous phase */
  fromPhase: LifecyclePhase;

  /** New phase */
  toPhase: LifecyclePhase;

  /** When transition occurred */
  timestamp: number;

  /** Why transition happened */
  reason: string;

  /** Health metrics at transition */
  healthAtTransition: ClaimHealth;

  /** Who/what triggered transition */
  triggeredBy: 'system' | 'evidence' | 'challenge' | 'human' | 'decay';
}

/**
 * Rules for lifecycle transitions
 */
export interface TransitionRule {
  /** From phase */
  from: LifecyclePhase;

  /** To phase */
  to: LifecyclePhase;

  /** Condition to trigger (returns true if should transition) */
  condition: (health: ClaimHealth, claim: MetabolizingClaim) => boolean;

  /** Priority (higher = checked first) */
  priority: number;

  /** Human-readable description */
  description: string;
}

// ============================================================================
// METABOLIZING CLAIM
// ============================================================================

/**
 * A claim with full metabolism tracking
 *
 * This wraps an epistemic claim with lifecycle management.
 */
export interface MetabolizingClaim {
  /** Unique identifier */
  id: string;

  /** Claim content */
  content: string;

  /** Epistemic classification */
  classification: EpistemicClassification;

  /** Current confidence */
  confidence: number;

  /** Current lifecycle phase */
  phase: LifecyclePhase;

  /** Health metrics */
  health: ClaimHealth;

  /** Evidence supporting the claim */
  evidence: Evidence[];

  /** Activity history */
  activityHistory: ClaimActivity[];

  /** Transition history */
  transitionHistory: LifecycleTransition[];

  /** Claims this depends on */
  dependencies: string[];

  /** Claims that depend on this */
  dependents: string[];

  /** Author ID */
  authorId: string;

  /** Created timestamp */
  createdAt: number;

  /** Last updated timestamp */
  updatedAt: number;

  /** Last activity timestamp */
  lastActivityAt: number;

  /** Domain */
  domain: string;

  /** Tags */
  tags: string[];
}

// ============================================================================
// DECOMPOSITION
// ============================================================================

/**
 * Result of decomposing a dead claim
 *
 * When claims die, they can be "decomposed" to extract useful parts:
 * - Still-valid sub-claims
 * - Evidence that applies elsewhere
 * - Methodological insights
 */
export interface DecompositionResult {
  /** Original claim ID */
  originalClaimId: string;

  /** Extracted sub-claims (may still be valid) */
  extractedClaims: Array<{
    content: string;
    confidence: number;
    rationale: string;
  }>;

  /** Evidence that can be reused */
  reusableEvidence: Evidence[];

  /** Methodological lessons */
  methodologicalInsights: string[];

  /** What to avoid next time */
  antiPatterns: string[];

  /** When decomposition occurred */
  decomposedAt: number;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Configuration for the metabolism system
 */
export interface MetabolismConfig {
  /** Days of inactivity before claim becomes stagnant */
  stagnationThresholdDays: number;

  /** Minimum evidence strength to exit nascent phase */
  nascentExitThreshold: number;

  /** Challenge ratio to enter challenged phase */
  challengeThreshold: number;

  /** Contradiction score to start dying */
  dyingThreshold: number;

  /** Health score below which claim dies */
  deathThreshold: number;

  /** Days to keep tombstones */
  tombstoneRetentionDays: number;

  /** Enable automatic lifecycle transitions */
  autoTransition: boolean;

  /** Enable automatic decomposition of dead claims */
  autoDecompose: boolean;

  /** Maximum transition history to keep per claim */
  maxTransitionHistory: number;

  /**
   * Whether to persist claims and tombstones to SCEI persistence layer.
   * When enabled, claims and tombstones are automatically saved and can be restored.
   */
  enablePersistence: boolean;

  /** TTL for persisted claims in milliseconds (0 = no expiry) */
  claimPersistenceTtlMs: number;
}

/**
 * Default metabolism configuration
 */
export const DEFAULT_METABOLISM_CONFIG: MetabolismConfig = {
  stagnationThresholdDays: 90,
  nascentExitThreshold: 0.3,
  challengeThreshold: 0.3,
  dyingThreshold: 0.5,
  deathThreshold: 0.1,
  tombstoneRetentionDays: 365 * 5, // 5 years
  autoTransition: true,
  autoDecompose: false, // Requires human review
  maxTransitionHistory: 100,
  enablePersistence: false,
  claimPersistenceTtlMs: 0, // No expiry by default - claims persist until explicitly deleted
};

// ============================================================================
// EVENTS
// ============================================================================

/**
 * Events emitted by the metabolism system
 */
export type MetabolismEvent =
  | { type: 'claim_born'; claim: MetabolizingClaim }
  | { type: 'claim_transitioned'; transition: LifecycleTransition }
  | { type: 'claim_died'; claimId: string; tombstone: Tombstone }
  | { type: 'claim_decomposed'; result: DecompositionResult }
  | { type: 'health_updated'; claimId: string; health: ClaimHealth }
  | { type: 'tombstone_created'; tombstone: Tombstone };

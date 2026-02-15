/**
 * @mycelix/sdk Calibration Types
 *
 * Types for the Calibration Engine - the keystone of SCEI.
 * Calibration answers: "When we said 70%, were we right 70% of the time?"
 *
 * @packageDocumentation
 * @module calibration/types
 */

import type { SourceType } from '../knowledge/index.js';

// ============================================================================
// CALIBRATION BUCKETS
// ============================================================================

/**
 * Calibration bucket for reliability analysis
 *
 * Claims are grouped into confidence ranges (e.g., 60-70%)
 * and we track how often claims in that range were correct.
 */
export interface CalibrationBucket {
  /** Confidence range (e.g., [0.6, 0.7]) */
  confidenceRange: [number, number];

  /** Number of claims in this bucket */
  claimCount: number;

  /** Number that resolved as expected */
  correctCount: number;

  /** Actual accuracy = correctCount / claimCount */
  actualAccuracy: number;

  /** Expected accuracy = midpoint of range */
  expectedAccuracy: number;

  /** Calibration error = |actual - expected| */
  calibrationError: number;
}

// ============================================================================
// CALIBRATION SCOPE
// ============================================================================

/**
 * Scope for calibration analysis
 *
 * Calibration can be computed globally, per domain, per source type,
 * per agent, or per claim type.
 */
export type CalibrationScope =
  | { type: 'global' }
  | { type: 'domain'; domain: string }
  | { type: 'source'; sourceType: SourceType }
  | { type: 'agent'; agentId: string }
  | { type: 'claimType'; claimType: string };

// ============================================================================
// CALIBRATION REPORT
// ============================================================================

/**
 * Full calibration report for a scope
 */
export interface CalibrationReport {
  /** What this report covers */
  scope: CalibrationScope;

  /** Time window for this analysis */
  timeWindow: { start: number; end: number };

  /** Calibration buckets (typically 10: 0-10%, 10-20%, ..., 90-100%) */
  buckets: CalibrationBucket[];

  /** Brier score (lower is better, 0 = perfect) */
  brierScore: number;

  /** Log loss (lower is better) */
  logLoss: number;

  /** Overall calibration error (weighted by bucket size) */
  overallCalibrationError: number;

  /** Confidence inflation detector */
  inflationBias: number;

  /** Reliability by source type */
  reliabilityBySource: Map<SourceType, number>;

  /** Trend over time */
  trend: 'improving' | 'stable' | 'degrading';

  /** Number of resolved claims in this analysis */
  sampleSize: number;

  /** When this report was generated */
  generatedAt: number;
}

// ============================================================================
// RESOLUTION RECORDS
// ============================================================================

/**
 * Record of a claim resolution for calibration tracking
 */
export interface ResolutionRecord {
  /** Claim ID */
  claimId: string;

  /** Stated confidence at time of claim */
  statedConfidence: number;

  /** Outcome */
  outcome: 'correct' | 'incorrect' | 'ambiguous';

  /** Domain of the claim */
  domain: string;

  /** Source type */
  sourceType: SourceType;

  /** When resolved */
  resolvedAt: number;

  /** Evidence for resolution */
  evidence: ResolutionEvidence[];
}

/**
 * Evidence for a resolution
 */
export interface ResolutionEvidence {
  /** Type of evidence */
  type: string;

  /** Reference to evidence */
  reference: string;

  /** Brief description */
  description: string;
}

// ============================================================================
// ADJUSTED CONFIDENCE
// ============================================================================

/**
 * Result of confidence adjustment based on calibration
 */
export interface AdjustedConfidence {
  /** Adjusted confidence value */
  adjustedConfidence: number;

  /** Uncertainty in this adjustment */
  uncertainty: number;

  /** Whether calibration data was available */
  calibrationAvailable: boolean;

  /** Reason for adjustment */
  reason: string;

  /** Original confidence (if adjusted) */
  originalConfidence?: number;

  /** Bucket stats used for adjustment */
  bucketStats?: CalibrationBucket;
}

// ============================================================================
// CREDIBILITY CHECK
// ============================================================================

/**
 * Result of checking if a confidence claim is credible
 */
export interface CredibilityCheck {
  /** Is the confidence claim credible? */
  credible: boolean;

  /** Reason for determination */
  reason: string;

  /** Recommendation if not credible */
  recommendation?: string;

  /** Suggested confidence if available */
  suggestedConfidence?: number;

  /** Historical accuracy for this range */
  historicalAccuracy?: number;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Configuration for the Calibration Engine
 */
export interface CalibrationConfig {
  /** Minimum resolved claims needed for a valid report */
  minSampleSize: number;

  /** Number of buckets for calibration (default: 10) */
  bucketCount: number;

  /** Maximum acceptable calibration error for high-confidence claims */
  highConfidenceErrorThreshold: number;

  /** Time window for "recent" claims (ms) */
  recentWindowMs: number;

  /** Whether to auto-adjust confidence based on calibration */
  autoAdjust: boolean;

  /**
   * Whether to persist resolution records to SCEI persistence layer.
   * When enabled, records are automatically saved and can be restored.
   */
  enablePersistence: boolean;

  /** TTL for persisted records in milliseconds (0 = no expiry) */
  persistenceTtlMs: number;
}

/**
 * Default calibration configuration
 */
export const DEFAULT_CALIBRATION_CONFIG: CalibrationConfig = {
  minSampleSize: 30,
  bucketCount: 10,
  highConfidenceErrorThreshold: 0.15,
  recentWindowMs: 90 * 24 * 60 * 60 * 1000, // 90 days
  autoAdjust: false,
  enablePersistence: false,
  persistenceTtlMs: 365 * 24 * 60 * 60 * 1000, // 1 year default
};

// ============================================================================
// ERRORS
// ============================================================================

/**
 * Error thrown when there is insufficient data for calibration
 */
export class InsufficientCalibrationDataError extends Error {
  constructor(
    public readonly required: number,
    public readonly available: number
  ) {
    super(
      `Insufficient calibration data: need at least ${required} resolved claims, have ${available}`
    );
    this.name = 'InsufficientCalibrationDataError';
  }
}

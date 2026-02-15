/**
 * @mycelix/sdk Safe Belief Propagation Types
 *
 * Types for the Safe Belief Propagator - managing how belief updates
 * flow through the knowledge graph with circuit breakers to prevent
 * cascade failures.
 *
 * Key safety mechanisms:
 * - Rate limits per claim and globally
 * - Quarantine for suspicious patterns
 * - Human-in-the-loop for high-centrality nodes
 * - Depth limits to prevent infinite propagation
 *
 * @packageDocumentation
 * @module propagation/types
 */

// ============================================================================
// CIRCUIT BREAKER CONFIGURATION
// ============================================================================

/**
 * Circuit breaker configuration
 *
 * These limits prevent cascade failures where one bad update
 * causes a chain reaction through the entire knowledge graph.
 */
export interface CircuitBreakerConfig {
  /** Maximum depth of propagation from source claim */
  maxDepth: number;

  /** Maximum number of claims affected by single update */
  maxAffectedClaims: number;

  /** Maximum confidence change in single propagation step */
  maxConfidenceJump: number;

  /** Centrality threshold that triggers human review */
  centralityThresholdForReview: number;

  /** Maximum propagations per claim per hour */
  perClaimRateLimit: number;

  /** Maximum total propagations per hour globally */
  globalRateLimit: number;

  /** Minimum time between propagations to same claim (ms) */
  cooldownPeriodMs: number;

  /** Enable automatic quarantine */
  enableQuarantine: boolean;

  /** Quarantine threshold (suspicious pattern score) */
  quarantineThreshold: number;
}

/**
 * Default circuit breaker configuration
 */
export const DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfig = {
  maxDepth: 5,
  maxAffectedClaims: 100,
  maxConfidenceJump: 0.3,
  centralityThresholdForReview: 0.8,
  perClaimRateLimit: 10,
  globalRateLimit: 1000,
  cooldownPeriodMs: 60000, // 1 minute
  enableQuarantine: true,
  quarantineThreshold: 0.7,
};

// ============================================================================
// PROPAGATION STATE
// ============================================================================

/**
 * State of a propagation in progress
 */
export type PropagationState =
  | 'pending'      // Waiting to execute
  | 'in_progress'  // Currently propagating
  | 'completed'    // Successfully completed
  | 'blocked'      // Blocked by circuit breaker
  | 'quarantined'  // Quarantined for review
  | 'rejected'     // Rejected after review
  | 'approved';    // Approved after review

/**
 * Reason a propagation was blocked
 */
export type BlockReason =
  | 'max_depth_exceeded'
  | 'max_affected_exceeded'
  | 'confidence_jump_too_large'
  | 'rate_limit_exceeded'
  | 'cooldown_active'
  | 'quarantined'
  | 'calibration_failed'
  | 'human_review_required'
  | 'source_claim_invalid';

// ============================================================================
// PROPAGATION REQUEST
// ============================================================================

/**
 * Request to propagate a belief update
 */
export interface PropagationRequest {
  /** Unique request ID */
  id: string;

  /** Source claim that triggered the update */
  sourceClaimId: string;

  /** Type of update */
  updateType: 'confidence_increase' | 'confidence_decrease' | 'refutation' | 'corroboration';

  /** New confidence for source claim */
  newConfidence: number;

  /** Old confidence for source claim */
  oldConfidence: number;

  /** Who/what initiated this propagation */
  initiatedBy: string;

  /** When request was created */
  createdAt: number;

  /** Priority (higher = processed first) */
  priority: number;

  /** Optional: Force human review regardless of other factors */
  requiresHumanReview?: boolean;

  /** Optional: Evidence supporting this update */
  evidence?: Array<{
    type: string;
    reference: string;
    strength: number;
  }>;
}

/**
 * Result of attempting to propagate
 */
export interface PropagationResult {
  /** Request ID */
  requestId: string;

  /** Final state */
  state: PropagationState;

  /** If blocked, why */
  blockReason?: BlockReason;

  /** Claims that were affected */
  affectedClaims: AffectedClaim[];

  /** Depth reached */
  depthReached: number;

  /** Time taken (ms) */
  durationMs: number;

  /** Whether human review is pending */
  pendingHumanReview: boolean;

  /** Warnings generated */
  warnings: string[];

  /** Detailed trace for debugging */
  trace?: PropagationTrace;
}

/**
 * A claim affected by propagation
 */
export interface AffectedClaim {
  /** Claim ID */
  claimId: string;

  /** Old confidence */
  oldConfidence: number;

  /** New confidence */
  newConfidence: number;

  /** Depth from source */
  depth: number;

  /** How it was affected */
  relationship: 'dependent' | 'supporting' | 'contradicting';

  /** Was update applied? */
  applied: boolean;

  /** If not applied, why */
  reason?: string;
}

// ============================================================================
// PROPAGATION TRACE
// ============================================================================

/**
 * Detailed trace of a propagation for debugging and audit
 */
export interface PropagationTrace {
  /** Request that triggered this */
  request: PropagationRequest;

  /** Steps taken */
  steps: PropagationStep[];

  /** Circuit breaker checks performed */
  circuitBreakerChecks: CircuitBreakerCheck[];

  /** Total claims visited */
  totalVisited: number;

  /** Total claims modified */
  totalModified: number;

  /** Peak memory usage estimate */
  peakClaimsInMemory: number;
}

/**
 * Single step in propagation
 */
export interface PropagationStep {
  /** Claim being processed */
  claimId: string;

  /** Depth at this step */
  depth: number;

  /** Action taken */
  action: 'visit' | 'update' | 'skip' | 'block';

  /** Reason for action */
  reason: string;

  /** Confidence before/after */
  confidenceBefore?: number;
  confidenceAfter?: number;

  /** Time at this step */
  timestamp: number;
}

/**
 * Circuit breaker check result
 */
export interface CircuitBreakerCheck {
  /** Which check */
  check: string;

  /** Did it pass? */
  passed: boolean;

  /** Current value */
  currentValue: number;

  /** Threshold */
  threshold: number;

  /** Action if failed */
  actionIfFailed: 'block' | 'quarantine' | 'require_review';
}

// ============================================================================
// QUARANTINE
// ============================================================================

/**
 * A quarantined propagation awaiting review
 */
export interface QuarantinedPropagation {
  /** Propagation request */
  request: PropagationRequest;

  /** Why it was quarantined */
  quarantineReason: string;

  /** Suspicious pattern score (0-1) */
  suspicionScore: number;

  /** Patterns detected */
  detectedPatterns: SuspiciousPattern[];

  /** When quarantined */
  quarantinedAt: number;

  /** Reviewer assigned (if any) */
  assignedReviewer?: string;

  /** Review deadline */
  reviewDeadline: number;

  /** Potential impact if approved */
  potentialImpact: {
    estimatedAffectedClaims: number;
    highCentralityClaims: string[];
    maxConfidenceChange: number;
  };
}

/**
 * A suspicious pattern detected
 */
export interface SuspiciousPattern {
  /** Pattern type */
  type: 'rapid_updates' | 'coordinated_attack' | 'confidence_oscillation' | 'unusual_source' | 'circular_dependency';

  /** Confidence in this detection */
  confidence: number;

  /** Description */
  description: string;

  /** Evidence for this pattern */
  evidence: string[];
}

/**
 * Decision on a quarantined item
 */
export interface QuarantineDecision {
  /** Quarantine ID */
  quarantineId: string;

  /** Decision */
  decision: 'approve' | 'reject' | 'modify';

  /** Reviewer ID */
  reviewerId: string;

  /** Reason for decision */
  reason: string;

  /** If modified, new parameters */
  modifiedRequest?: Partial<PropagationRequest>;

  /** When decided */
  decidedAt: number;
}

// ============================================================================
// RATE LIMITING
// ============================================================================

/**
 * Rate limit state for a claim
 */
export interface ClaimRateLimitState {
  claimId: string;
  propagationsThisHour: number;
  lastPropagation: number;
  hourStarted: number;
}

/**
 * Global rate limit state
 */
export interface GlobalRateLimitState {
  propagationsThisHour: number;
  hourStarted: number;
  quarantinesThisHour: number;
  blocksThisHour: number;
}

// ============================================================================
// HUMAN REVIEW
// ============================================================================

/**
 * Human review request
 */
export interface HumanReviewRequest {
  /** Request ID */
  id: string;

  /** Propagation being reviewed */
  propagationRequest: PropagationRequest;

  /** Why review is needed */
  reviewReason: 'high_centrality' | 'large_impact' | 'suspicious_pattern' | 'explicit_request';

  /** Claims that need review */
  claimsToReview: string[];

  /** Estimated impact */
  estimatedImpact: {
    affectedClaims: number;
    maxConfidenceChange: number;
  };

  /** Created at */
  createdAt: number;

  /** Deadline for review */
  deadline: number;

  /** Current status */
  status: 'pending' | 'in_review' | 'approved' | 'rejected' | 'expired';
}

/**
 * Human review decision
 */
export interface HumanReviewDecision {
  /** Request ID */
  requestId: string;

  /** Decision */
  decision: 'approve' | 'reject' | 'approve_with_modifications';

  /** Reviewer */
  reviewerId: string;

  /** Reason */
  reason: string;

  /** Modifications if any */
  modifications?: {
    maxDepth?: number;
    maxAffectedClaims?: number;
    excludeClaims?: string[];
  };

  /** Decided at */
  decidedAt: number;
}

// ============================================================================
// EVENTS
// ============================================================================

/**
 * Events emitted by the propagation system
 */
export type PropagationEvent =
  | { type: 'propagation_started'; request: PropagationRequest }
  | { type: 'propagation_completed'; result: PropagationResult }
  | { type: 'propagation_blocked'; request: PropagationRequest; reason: BlockReason }
  | { type: 'claim_updated'; claimId: string; oldConfidence: number; newConfidence: number }
  | { type: 'quarantine_added'; quarantine: QuarantinedPropagation }
  | { type: 'quarantine_decided'; decision: QuarantineDecision }
  | { type: 'human_review_requested'; request: HumanReviewRequest }
  | { type: 'human_review_decided'; decision: HumanReviewDecision }
  | { type: 'circuit_breaker_tripped'; check: string; value: number; threshold: number }
  | { type: 'rate_limit_exceeded'; claimId?: string; global: boolean };

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Full propagation configuration
 */
export interface PropagationConfig {
  /** Circuit breaker settings */
  circuitBreaker: CircuitBreakerConfig;

  /** Enable detailed tracing */
  enableTracing: boolean;

  /** Enable calibration gating (requires CalibrationEngine) */
  enableCalibrationGating: boolean;

  /** Human review timeout (ms) */
  humanReviewTimeoutMs: number;

  /** Quarantine review timeout (ms) */
  quarantineTimeoutMs: number;

  /** Auto-expire pending reviews */
  autoExpireReviews: boolean;

  /** Batch size for processing */
  batchSize: number;
}

/**
 * Default propagation configuration
 */
export const DEFAULT_PROPAGATION_CONFIG: PropagationConfig = {
  circuitBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG,
  enableTracing: false,
  enableCalibrationGating: true,
  humanReviewTimeoutMs: 24 * 60 * 60 * 1000, // 24 hours
  quarantineTimeoutMs: 48 * 60 * 60 * 1000, // 48 hours
  autoExpireReviews: true,
  batchSize: 50,
};

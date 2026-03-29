// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MATL - Mycelix Adaptive Trust Layer
 *
 * TypeScript implementation of the MATL Byzantine fault tolerance protocol (34% validated).
 * Enhanced with comprehensive error handling and validation.
 */

import {
  validate,
  ValidationError,
  assertDefined,
} from '../errors';

// Default weights for composite score calculation
export const DEFAULT_QUALITY_WEIGHT = 0.4;
export const DEFAULT_CONSISTENCY_WEIGHT = 0.3;
export const DEFAULT_REPUTATION_WEIGHT = 0.3;

// Maximum validated Byzantine tolerance (34%)
export const MAX_BYZANTINE_TOLERANCE = 0.34;

// Default Byzantine detection threshold
export const DEFAULT_BYZANTINE_THRESHOLD = 0.5;

// Default entropy penalty weight
export const DEFAULT_ENTROPY_PENALTY = 0.1;

/**
 * Options for customizing composite score calculation
 */
export interface CompositeScoreOptions {
  /** Weight for quality contribution (default: 0.4) */
  qualityWeight?: number;
  /** Weight for consistency contribution (default: 0.3) */
  consistencyWeight?: number;
  /** Weight for reputation contribution (default: 0.3) */
  reputationWeight?: number;
  /** Penalty weight for entropy (default: 0.1) */
  entropyPenalty?: number;
}

/**
 * Proof of Gradient Quality (PoGQ)
 * Core trust measurement combining quality, consistency, and entropy.
 */
export interface ProofOfGradientQuality {
  quality: number;      // Gradient quality metric [0, 1]
  consistency: number;  // Temporal consistency [0, 1]
  entropy: number;      // Update entropy [0, 1]
  timestamp: number;    // Unix timestamp
}

/**
 * Validate PoGQ input values
 */
function validatePoGQInputs(quality: number, consistency: number, entropy: number): void {
  validate()
    .inRange('quality', quality, 0, 1)
    .inRange('consistency', consistency, 0, 1)
    .inRange('entropy', entropy, 0, 1)
    .throwIfInvalid();
}

/**
 * Create a new PoGQ measurement
 * @throws {ValidationError} If any input is outside [0, 1]
 */
export function createPoGQ(
  quality: number,
  consistency: number,
  entropy: number
): ProofOfGradientQuality {
  validatePoGQInputs(quality, consistency, entropy);

  return {
    quality: clamp(quality),
    consistency: clamp(consistency),
    entropy: clamp(entropy),
    timestamp: Date.now(),
  };
}

/**
 * Calculate composite score from PoGQ and reputation
 *
 * The composite score combines quality, consistency, and reputation contributions
 * minus an entropy penalty. All weights can be customized via options.
 *
 * @param pogq - Proof of Gradient Quality measurement
 * @param reputation - Reputation score (will be clamped to [0, 1])
 * @param options - Optional weights for customizing the calculation
 * @throws {ValidationError} If reputation is not a valid number
 *
 * @example
 * ```typescript
 * // Default weights
 * const score = compositeScore(pogq, 0.8);
 *
 * // Custom weights emphasizing quality
 * const customScore = compositeScore(pogq, 0.8, {
 *   qualityWeight: 0.6,
 *   consistencyWeight: 0.2,
 *   reputationWeight: 0.2,
 *   entropyPenalty: 0.05,
 * });
 * ```
 */
export function compositeScore(
  pogq: ProofOfGradientQuality,
  reputation: number,
  options: CompositeScoreOptions = {}
): number {
  assertDefined(pogq, 'pogq');

  if (typeof reputation !== 'number' || !Number.isFinite(reputation)) {
    throw new ValidationError('reputation', 'must be a valid finite number', reputation);
  }

  const {
    qualityWeight = DEFAULT_QUALITY_WEIGHT,
    consistencyWeight = DEFAULT_CONSISTENCY_WEIGHT,
    reputationWeight = DEFAULT_REPUTATION_WEIGHT,
    entropyPenalty = DEFAULT_ENTROPY_PENALTY,
  } = options;

  const qualityContribution = qualityWeight * pogq.quality;
  const consistencyContribution = consistencyWeight * pogq.consistency;
  const reputationContribution = reputationWeight * clamp(reputation);
  const entropyDeduction = entropyPenalty * pogq.entropy;

  return clamp(
    qualityContribution + consistencyContribution + reputationContribution - entropyDeduction
  );
}

/**
 * Check if a PoGQ indicates Byzantine behavior
 */
export function isByzantine(pogq: ProofOfGradientQuality, threshold = DEFAULT_BYZANTINE_THRESHOLD): boolean {
  assertDefined(pogq, 'pogq');

  // Low quality + high entropy = likely Byzantine
  const score = pogq.quality * (1 - pogq.entropy);
  return score < threshold;
}

/**
 * Check if a score value indicates Byzantine behavior
 */
export function isByzantineScore(score: number, threshold = DEFAULT_BYZANTINE_THRESHOLD): boolean {
  if (typeof score !== 'number' || isNaN(score)) {
    throw new ValidationError('score', 'must be a valid number', score);
  }
  return score < threshold;
}

/**
 * Reputation score with Bayesian smoothing
 */
export interface ReputationScore {
  agentId: string;
  positiveCount: number;
  negativeCount: number;
  lastUpdate: number;
}

/**
 * Create a new reputation score
 * @throws {ValidationError} If agentId is empty
 */
export function createReputation(agentId: string): ReputationScore {
  validate()
    .notEmpty('agentId', agentId)
    .throwIfInvalid();

  return {
    agentId,
    positiveCount: 1, // Prior (Laplace smoothing)
    negativeCount: 1, // Prior (Laplace smoothing)
    lastUpdate: Date.now(),
  };
}

/**
 * Calculate reputation value using Bayesian estimation
 * Returns (positive + 1) / (positive + negative + 2) for Laplace smoothing
 */
export function reputationValue(reputation: ReputationScore): number {
  assertDefined(reputation, 'reputation');

  const alpha = Math.max(0, reputation.positiveCount);
  const beta = Math.max(0, reputation.negativeCount);

  if (alpha + beta === 0) {
    return 0.5; // Default neutral score
  }

  return alpha / (alpha + beta);
}

/**
 * Record positive interaction
 * @throws {MATLError} If reputation is not defined
 */
export function recordPositive(reputation: ReputationScore): ReputationScore {
  assertDefined(reputation, 'reputation');

  return {
    ...reputation,
    positiveCount: reputation.positiveCount + 1,
    lastUpdate: Date.now(),
  };
}

/**
 * Record negative interaction
 * @throws {MATLError} If reputation is not defined
 */
export function recordNegative(reputation: ReputationScore): ReputationScore {
  assertDefined(reputation, 'reputation');

  return {
    ...reputation,
    negativeCount: reputation.negativeCount + 1,
    lastUpdate: Date.now(),
  };
}

/**
 * Check if reputation meets trust threshold
 */
export function isTrustworthyReputation(reputation: ReputationScore, threshold = 0.5): boolean {
  assertDefined(reputation, 'reputation');
  return reputationValue(reputation) >= threshold;
}

/**
 * Composite trust score combining PoGQ and reputation
 */
export interface CompositeScore {
  pogq: ProofOfGradientQuality;
  reputation: ReputationScore;
  pogqScore: number;
  reputationScore: number;
  finalScore: number;
  confidence: number;
  timestamp: number;
  isTrustworthy: boolean;
}

/**
 * Calculate composite trust score
 * @throws {MATLError} If inputs are invalid
 */
export function calculateComposite(
  pogq: ProofOfGradientQuality,
  reputation: ReputationScore,
  trustThreshold = DEFAULT_BYZANTINE_THRESHOLD
): CompositeScore {
  assertDefined(pogq, 'pogq');
  assertDefined(reputation, 'reputation');

  const repValue = reputationValue(reputation);
  const final = compositeScore(pogq, repValue);

  // PoGQ score (quality adjusted by entropy)
  const pogqScore = pogq.quality * (1 - 0.5 * pogq.entropy);

  // Confidence based on agreement between components
  const pogqAvg = (pogq.quality + pogq.consistency) / 2;
  const deviation = Math.abs(pogqAvg - repValue);
  const conf = 1.0 - deviation;

  return {
    pogq,
    reputation,
    pogqScore: clamp(pogqScore),
    reputationScore: repValue,
    finalScore: final,
    confidence: clamp(conf),
    timestamp: Date.now(),
    isTrustworthy: final >= trustThreshold && conf >= 0.5,
  };
}

/**
 * Check if composite score is trustworthy
 */
export function isTrustworthy(
  composite: CompositeScore | ReputationScore,
  threshold = DEFAULT_BYZANTINE_THRESHOLD,
  minConfidence = 0.5
): boolean {
  assertDefined(composite, 'composite');

  // Handle ReputationScore (for backward compatibility)
  if ('positiveCount' in composite) {
    return reputationValue(composite) >= threshold;
  }

  // Handle CompositeScore
  return composite.finalScore >= threshold && composite.confidence >= minConfidence;
}

/**
 * Adaptive threshold based on node history
 */
export interface AdaptiveThreshold {
  nodeId: string;
  observations: number[];
  windowSize: number;
  minThreshold: number;
  sigmaMultiplier: number;
}

/**
 * Create adaptive threshold tracker for a node
 * @throws {ValidationError} If nodeId is empty or parameters are invalid
 */
export function createAdaptiveThreshold(
  nodeId: string,
  windowSize = 100,
  minThreshold = DEFAULT_BYZANTINE_THRESHOLD,
  sigmaMultiplier = 2.0
): AdaptiveThreshold {
  validate()
    .notEmpty('nodeId', nodeId)
    .positive('windowSize', windowSize)
    .inRange('minThreshold', minThreshold, 0, 1)
    .positive('sigmaMultiplier', sigmaMultiplier)
    .throwIfInvalid();

  return {
    nodeId,
    observations: [],
    windowSize,
    minThreshold,
    sigmaMultiplier,
  };
}

/**
 * Add observation to adaptive threshold
 * @throws {ValidationError} If score is not in [0, 1]
 */
export function observe(at: AdaptiveThreshold, score: number): AdaptiveThreshold {
  assertDefined(at, 'adaptiveThreshold');
  validate().inRange('score', score, 0, 1).throwIfInvalid();

  const observations = [...at.observations, clamp(score)];
  if (observations.length > at.windowSize) {
    observations.shift();
  }
  return { ...at, observations };
}

/**
 * Get current threshold based on history
 */
export function getThreshold(at: AdaptiveThreshold): number {
  assertDefined(at, 'adaptiveThreshold');

  if (at.observations.length === 0) {
    return at.minThreshold;
  }

  const mean = at.observations.reduce((a, b) => a + b, 0) / at.observations.length;
  const variance = at.observations.reduce((acc, x) => acc + Math.pow(x - mean, 2), 0) / at.observations.length;
  const stdDev = Math.sqrt(variance);

  return Math.max(at.minThreshold, mean - at.sigmaMultiplier * stdDev);
}

/**
 * Check if score is anomalous for this node
 */
export function isAnomalous(at: AdaptiveThreshold, score: number): boolean {
  assertDefined(at, 'adaptiveThreshold');
  return score < getThreshold(at);
}

/**
 * Get statistics about the adaptive threshold
 */
export function getThresholdStats(at: AdaptiveThreshold): {
  currentThreshold: number;
  mean: number;
  stdDev: number;
  observationCount: number;
} {
  assertDefined(at, 'adaptiveThreshold');

  if (at.observations.length === 0) {
    return {
      currentThreshold: at.minThreshold,
      mean: 0,
      stdDev: 0,
      observationCount: 0,
    };
  }

  const mean = at.observations.reduce((a, b) => a + b, 0) / at.observations.length;
  const variance = at.observations.reduce((acc, x) => acc + Math.pow(x - mean, 2), 0) / at.observations.length;
  const stdDev = Math.sqrt(variance);

  return {
    currentThreshold: getThreshold(at),
    mean,
    stdDev,
    observationCount: at.observations.length,
  };
}

// Utility functions
function clamp(value: number, min = 0.0, max = 1.0): number {
  if (typeof value !== 'number' || isNaN(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

// Re-export all types and functions
export type {
  ProofOfGradientQuality as PoGQ,
  ReputationScore as Reputation,
  CompositeScore as Composite,
};

// Legacy alias for backward compatibility
export { observe as addObservation };
export { createAdaptiveThreshold as createThreshold };

// ============================================================================
// Reputation Cache
// ============================================================================

/**
 * Configuration for ReputationCache
 */
export interface ReputationCacheConfig {
  /** Maximum number of entries to cache (default: 1000) */
  maxSize?: number;
  /** Time-to-live for cache entries in ms (default: 60000 = 1 minute) */
  ttlMs?: number;
}

interface CacheEntry {
  value: number;
  timestamp: number;
  key: string;
}

/**
 * LRU cache for memoizing reputation calculations.
 * Prevents redundant calculations in high-frequency scenarios.
 *
 * @example
 * ```typescript
 * const cache = new ReputationCache({ maxSize: 500, ttlMs: 30000 });
 *
 * // First call computes and caches
 * const score1 = cache.getReputationValue(reputation);
 *
 * // Second call returns cached value
 * const score2 = cache.getReputationValue(reputation);
 *
 * // Check cache stats
 * const stats = cache.getStats();
 * console.log(`Hit rate: ${(stats.hits / (stats.hits + stats.misses) * 100).toFixed(1)}%`);
 * ```
 */
export class ReputationCache {
  private cache = new Map<string, CacheEntry>();
  private maxSize: number;
  private ttlMs: number;
  private hits = 0;
  private misses = 0;

  constructor(config: ReputationCacheConfig = {}) {
    this.maxSize = config.maxSize ?? 1000;
    this.ttlMs = config.ttlMs ?? 60000;
  }

  /**
   * Generate cache key from reputation score
   */
  private makeKey(rep: ReputationScore): string {
    return `${rep.agentId}:${rep.positiveCount}:${rep.negativeCount}`;
  }

  /**
   * Check if cache entry is still valid
   */
  private isValid(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp < this.ttlMs;
  }

  /**
   * Evict oldest entry if at max capacity
   */
  private evictIfNeeded(): void {
    if (this.cache.size >= this.maxSize) {
      // Find and remove oldest entry
      let oldestKey: string | null = null;
      let oldestTime = Infinity;

      for (const [key, entry] of this.cache) {
        if (entry.timestamp < oldestTime) {
          oldestTime = entry.timestamp;
          oldestKey = key;
        }
      }

      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
  }

  /**
   * Get cached reputation value or compute and cache it
   */
  getReputationValue(rep: ReputationScore): number {
    const key = this.makeKey(rep);
    const cached = this.cache.get(key);

    if (cached && this.isValid(cached)) {
      this.hits++;
      return cached.value;
    }

    // Compute and cache
    this.misses++;
    const value = reputationValue(rep);

    this.evictIfNeeded();
    this.cache.set(key, {
      value,
      timestamp: Date.now(),
      key,
    });

    return value;
  }

  /**
   * Get cached composite score or compute and cache it
   * Uses cached reputation value for efficiency
   */
  getCompositeScore(
    pogq: ProofOfGradientQuality,
    rep: ReputationScore,
    trustThreshold?: number
  ): CompositeScore {
    // First ensure reputation is cached
    this.getReputationValue(rep);

    // Compute full composite score (uses reputationValue internally)
    return calculateComposite(pogq, rep, trustThreshold);
  }

  /**
   * Invalidate cache entry for a specific agent
   */
  invalidate(agentId: string): number {
    let count = 0;
    for (const [key] of this.cache) {
      if (key.startsWith(`${agentId}:`)) {
        this.cache.delete(key);
        count++;
      }
    }
    return count;
  }

  /**
   * Clear all cached entries
   */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Remove expired entries
   */
  prune(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [key, entry] of this.cache) {
      if (now - entry.timestamp >= this.ttlMs) {
        this.cache.delete(key);
        pruned++;
      }
    }

    return pruned;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    hits: number;
    misses: number;
    hitRate: number;
    ttlMs: number;
  } {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
      ttlMs: this.ttlMs,
    };
  }
}

// =============================================================================
// Coercion Resistance Constants
// =============================================================================

/** Minimum floor for reputation decay (prevents total destruction) */
export const REPUTATION_DECAY_FLOOR = 0.05;

/** Maximum penalty per update to prevent coercive attacks (20%) */
export const MAX_COERCIVE_PENALTY = 0.2;

/** Time window for mobbing detection (24 hours) */
export const MOBBING_WINDOW_MS = 24 * 60 * 60 * 1000;

/** Minimum events required for mobbing analysis */
export const MIN_MOBBING_EVENTS = 3;

// =============================================================================
// Coercion Resistance Types
// =============================================================================

/**
 * Slash event for correlation analysis
 */
export interface SlashEvent {
  id: string;
  targetDid: string;
  slasherDid: string;
  reason: string;
  magnitude: number;
  sourceHapp: string;
  timestamp: number;
}

/**
 * Mobbing detection options
 */
export interface MobbingDetectionOptions {
  minEventsForAnalysis?: number;
  correlationThreshold?: number;
  windowMs?: number;
}

/**
 * Mobbing alert
 */
export interface MobbingAlert {
  targetDid: string;
  correlation: number;
  eventCount: number;
  slasherDids: string[];
  timeSpan: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  details: {
    slashCount: number;
    uniqueSlashers: number;
    avgMagnitude: number;
  };
}

/**
 * Reputation vector with multiple dimensions
 */
export interface ReputationVector {
  trust: number;
  skill: number;
  gifting: number;
  grounding: number;
  governance: number;
  justice: number;
}

/**
 * Non-coercive update result
 */
export interface NonCoerciveUpdateResult {
  adjustedScore: number;
  originalScore: number;
  requestedScore: number;
  wasCapped: boolean;
  warnings: string[];
}

// =============================================================================
// Coercion Resistance Functions
// =============================================================================

/**
 * Apply decay floor to prevent reputation destruction
 */
export function applyDecayFloor(value: number): number {
  if (value < REPUTATION_DECAY_FLOOR) {
    return REPUTATION_DECAY_FLOOR;
  }
  return value;
}

/**
 * Calculate slasher correlation to detect coordinated attacks
 * Returns correlation coefficient 0-1, where higher values indicate coordinated behavior
 */
export function calculateSlasherCorrelation(events: SlashEvent[]): number {
  // Need at least 2 events for correlation
  if (events.length < 2) {
    return 0;
  }

  // Sort by timestamp
  const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);

  // Calculate time intervals
  const intervals: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    intervals.push(sorted[i].timestamp - sorted[i - 1].timestamp);
  }

  // Calculate average interval
  const avgInterval = intervals.reduce((sum, i) => sum + i, 0) / intervals.length;

  // Calculate interval variance (normalized)
  const maxInterval = Math.max(...intervals);
  const intervalVariance =
    maxInterval > 0
      ? intervals.reduce((sum, i) => sum + Math.pow(i - avgInterval, 2), 0) /
        (intervals.length * maxInterval * maxInterval)
      : 0;

  // Count unique slashers
  const uniqueSlashers = new Set(events.map((e) => e.slasherDid)).size;
  const slasherRatio = uniqueSlashers / events.length;

  // Calculate magnitude variance
  const avgMagnitude = events.reduce((sum, e) => sum + e.magnitude, 0) / events.length;
  const magnitudeVariance =
    events.reduce((sum, e) => sum + Math.pow(e.magnitude - avgMagnitude, 2), 0) / events.length;

  // Count unique hApps - cross-hApp coordination increases suspicion
  const uniqueHapps = new Set(events.map((e) => e.sourceHapp)).size;
  const crossHappBonus = uniqueHapps > 1 ? 0.15 * Math.min(uniqueHapps / events.length, 1) : 0;

  // High correlation = low interval variance + few unique slashers + low magnitude variance
  // Normalized to 0-1 range
  const intervalScore = 1 - Math.min(intervalVariance, 1);
  const slasherScore = 1 - slasherRatio;
  const magnitudeScore = 1 - Math.min(magnitudeVariance, 1);

  // Weighted average with cross-hApp bonus
  const baseScore = intervalScore * 0.3 + slasherScore * 0.35 + magnitudeScore * 0.2;
  return Math.min(baseScore + crossHappBonus, 1);
}

/**
 * Detect reputation mobbing (coordinated attacks on a target)
 */
export function detectReputationMobbing(
  targetDid: string,
  events: SlashEvent[],
  options: MobbingDetectionOptions = {}
): MobbingAlert | null {
  const {
    minEventsForAnalysis = MIN_MOBBING_EVENTS,
    correlationThreshold = 0.5,
    windowMs = MOBBING_WINDOW_MS,
  } = options;

  // Filter events for this target within window
  const now = Date.now();
  const relevantEvents = events.filter(
    (e) => e.targetDid === targetDid && now - e.timestamp <= windowMs
  );

  if (relevantEvents.length < minEventsForAnalysis) {
    return null;
  }

  const correlation = calculateSlasherCorrelation(relevantEvents);

  if (correlation < correlationThreshold) {
    return null;
  }

  const slasherDids = [...new Set(relevantEvents.map((e) => e.slasherDid))];
  const timestamps = relevantEvents.map((e) => e.timestamp);
  const timeSpan = Math.max(...timestamps) - Math.min(...timestamps);
  const avgMagnitude =
    relevantEvents.reduce((sum, e) => sum + e.magnitude, 0) / relevantEvents.length;

  // Determine severity
  let severity: MobbingAlert['severity'];
  if (correlation >= 0.9) {
    severity = 'critical';
  } else if (correlation >= 0.7) {
    severity = 'high';
  } else if (correlation >= 0.5) {
    severity = 'medium';
  } else {
    severity = 'low';
  }

  return {
    targetDid,
    correlation,
    eventCount: relevantEvents.length,
    slasherDids,
    timeSpan,
    severity,
    details: {
      slashCount: relevantEvents.length,
      uniqueSlashers: slasherDids.length,
      avgMagnitude,
    },
  };
}

/**
 * Create a multi-dimensional reputation vector
 */
export function createReputationVector(
  dimensions: Partial<ReputationVector>
): ReputationVector {
  return {
    trust: dimensions.trust ?? 0.5,
    skill: dimensions.skill ?? 0.5,
    gifting: dimensions.gifting ?? 0.5,
    grounding: dimensions.grounding ?? 0.5,
    governance: dimensions.governance ?? 0.5,
    justice: dimensions.justice ?? 0.5,
  };
}

/**
 * Aggregate reputation vector to single score
 */
export function aggregateReputationVector(
  vector: ReputationVector,
  _strategy: 'default' | 'weighted' | 'geometric' = 'default'
): number {
  const values = [
    vector.trust,
    vector.skill,
    vector.gifting,
    vector.grounding,
    vector.governance,
    vector.justice,
  ];

  // Simple average for default strategy
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;

  // Apply decay floor to protect against zero reputation
  return applyDecayFloor(avg);
}

/**
 * Validate a reputation update doesn't exceed coercive limits
 */
export function validateNonCoerciveUpdate(
  currentScore: number,
  requestedScore: number,
  _context: { reason: string; isAutomated: boolean }
): NonCoerciveUpdateResult {
  const warnings: string[] = [];
  let adjustedScore = requestedScore;
  let wasCapped = false;

  // Calculate penalty (negative change)
  const penalty = currentScore - requestedScore;

  // Cap large penalties
  if (penalty > MAX_COERCIVE_PENALTY) {
    adjustedScore = currentScore - MAX_COERCIVE_PENALTY;
    wasCapped = true;
    warnings.push(
      `Penalty capped from ${penalty.toFixed(3)} to ${MAX_COERCIVE_PENALTY} to prevent coercive update`
    );
  }

  // Clamp to valid range
  if (adjustedScore > 1) {
    adjustedScore = 1;
    wasCapped = true;
    warnings.push('Score capped at maximum 1.0');
  }

  // Apply floor
  if (adjustedScore < REPUTATION_DECAY_FLOOR) {
    adjustedScore = REPUTATION_DECAY_FLOOR;
    wasCapped = true;
    warnings.push(`Score raised to minimum floor ${REPUTATION_DECAY_FLOOR}`);
  }

  return {
    adjustedScore,
    originalScore: currentScore,
    requestedScore,
    wasCapped,
    warnings,
  };
}

/**
 * Alert structure for coercion resistance
 */
export interface CoercionAlert {
  type: string;
  targetDid: string;
  slashCount: number;
  timespan: number;
  correlation: number;
  confidence: number;
  detectedAt: number;
}

/**
 * Manager for coercion resistance operations
 */
export class CoercionResistanceManager {
  private eventHistory: Map<string, SlashEvent[]> = new Map();
  private readonly maxHistoryPerTarget: number;
  private readonly windowMs: number;
  private alerts: CoercionAlert[] = [];

  /** Alias for alerts - for internal test access */
  private static readonly SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

  constructor(options: { maxHistoryPerTarget?: number; pruneIntervalMs?: number; windowMs?: number } = {}) {
    this.maxHistoryPerTarget = options.maxHistoryPerTarget ?? 1000;
    this.windowMs = options.windowMs ?? options.pruneIntervalMs ?? MOBBING_WINDOW_MS;
  }

  /**
   * Record a slash event
   */
  recordEvent(event: SlashEvent): void {
    const existing = this.eventHistory.get(event.targetDid) || [];
    existing.push(event);

    // Trim to max history
    if (existing.length > this.maxHistoryPerTarget) {
      existing.shift();
    }

    this.eventHistory.set(event.targetDid, existing);
  }

  /**
   * Record a slash event (alias for recordEvent)
   */
  recordSlash(event: SlashEvent): void {
    this.recordEvent(event);
  }

  /**
   * Get events for a target
   */
  getEvents(targetDid: string): SlashEvent[] {
    return this.eventHistory.get(targetDid) || [];
  }

  /**
   * Analyze for mobbing
   */
  analyzeTarget(
    targetDid: string,
    options?: MobbingDetectionOptions
  ): MobbingAlert | null {
    const events = this.getEvents(targetDid);
    return detectReputationMobbing(targetDid, events, options);
  }

  /**
   * Prune old events
   */
  prune(): { eventsRemoved: number; targetsRemoved: number } {
    const now = Date.now();
    let eventsRemoved = 0;
    let targetsRemoved = 0;

    for (const [targetDid, events] of this.eventHistory) {
      const fresh = events.filter((e) => now - e.timestamp < this.windowMs);
      eventsRemoved += events.length - fresh.length;

      if (fresh.length === 0) {
        this.eventHistory.delete(targetDid);
        targetsRemoved++;
      } else {
        this.eventHistory.set(targetDid, fresh);
      }
    }

    return { eventsRemoved, targetsRemoved };
  }

  /**
   * Get active alerts (within 7 days)
   */
  getActiveAlerts(): CoercionAlert[] {
    const now = Date.now();
    return this.alerts.filter((a) => now - a.detectedAt < CoercionResistanceManager.SEVEN_DAYS_MS);
  }

  /**
   * Get alerts for a specific target
   */
  getAlertsForTarget(targetDid: string): CoercionAlert[] {
    return this.alerts.filter((a) => a.targetDid === targetDid);
  }

  /**
   * Add an alert
   */
  addAlert(alert: CoercionAlert): void {
    this.alerts.push(alert);
  }

  /**
   * Get statistics
   */
  getStats(): {
    targetCount: number;
    uniqueTargets: number;
    totalEvents: number;
    totalEventsTracked: number;
    activeAlerts: number;
    alertsByType: Record<string, number>;
  } {
    let totalEvents = 0;
    const alertsByType: Record<string, number> = {};

    // Count from alerts
    for (const alert of this.alerts) {
      alertsByType[alert.type] = (alertsByType[alert.type] ?? 0) + 1;
    }

    for (const events of this.eventHistory.values()) {
      totalEvents += events.length;
    }

    return {
      targetCount: this.eventHistory.size,
      uniqueTargets: this.eventHistory.size,
      totalEvents,
      totalEventsTracked: totalEvents,
      activeAlerts: this.getActiveAlerts().length,
      alertsByType,
    };
  }
}

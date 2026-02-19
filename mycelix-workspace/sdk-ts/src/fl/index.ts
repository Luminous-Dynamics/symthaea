/**
 * Federated Learning Module
 *
 * Byzantine-resistant distributed machine learning with MATL integration.
 * Supports multiple aggregation algorithms with trust-weighted contributions.
 */

import { FederatedLearningError, ErrorCode } from '../errors.js';
import * as matl from '../matl/index.js';

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Validate gradient updates have consistent shapes
 */
function validateGradientConsistency(updates: GradientUpdate[]): void {
  if (updates.length === 0) {
    throw new FederatedLearningError(
      'No gradient updates provided for aggregation',
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { required: 1, received: 0 }
    );
  }

  const expectedSize = updates[0].gradients.length;
  if (expectedSize === 0) {
    throw new FederatedLearningError(
      'Gradient array cannot be empty',
      ErrorCode.FL_INVALID_GRADIENT_SHAPE,
      { participantId: updates[0].participantId, gradientLength: 0 }
    );
  }

  for (let i = 1; i < updates.length; i++) {
    if (updates[i].gradients.length !== expectedSize) {
      throw new FederatedLearningError(
        `Gradient size mismatch: expected ${expectedSize}, got ${updates[i].gradients.length}`,
        ErrorCode.FL_INVALID_GRADIENT_SHAPE,
        {
          participantId: updates[i].participantId,
          expectedSize,
          actualSize: updates[i].gradients.length,
        }
      );
    }
  }
}

/**
 * Validate FL configuration parameters
 */
function validateFLConfig(config: FLConfig): void {
  if (config.minParticipants < 1) {
    throw new FederatedLearningError(
      'minParticipants must be at least 1',
      ErrorCode.INVALID_ARGUMENT,
      { minParticipants: config.minParticipants }
    );
  }

  if (config.maxParticipants < config.minParticipants) {
    throw new FederatedLearningError(
      'maxParticipants cannot be less than minParticipants',
      ErrorCode.INVALID_ARGUMENT,
      { minParticipants: config.minParticipants, maxParticipants: config.maxParticipants }
    );
  }

  if (config.byzantineTolerance < 0 || config.byzantineTolerance > 0.34) {
    throw new FederatedLearningError(
      'byzantineTolerance must be between 0 and 0.34 (34% validated)',
      ErrorCode.INVALID_ARGUMENT,
      { byzantineTolerance: config.byzantineTolerance }
    );
  }

  if (config.trustThreshold < 0 || config.trustThreshold > 1) {
    throw new FederatedLearningError(
      'trustThreshold must be between 0 and 1',
      ErrorCode.INVALID_ARGUMENT,
      { trustThreshold: config.trustThreshold }
    );
  }

  if (config.roundTimeout < 1000) {
    throw new FederatedLearningError(
      'roundTimeout must be at least 1000ms',
      ErrorCode.INVALID_ARGUMENT,
      { roundTimeout: config.roundTimeout }
    );
  }
}

/**
 * Validate gradient update metadata
 */
function validateUpdateMetadata(update: GradientUpdate): void {
  if (update.metadata.batchSize <= 0) {
    throw new FederatedLearningError(
      'Batch size must be positive',
      ErrorCode.INVALID_ARGUMENT,
      { participantId: update.participantId, batchSize: update.metadata.batchSize }
    );
  }

  if (!Number.isFinite(update.metadata.loss)) {
    throw new FederatedLearningError(
      'Loss must be a finite number',
      ErrorCode.INVALID_ARGUMENT,
      { participantId: update.participantId, loss: update.metadata.loss }
    );
  }
}

// ============================================================================
// Types
// ============================================================================

/**
 * Gradient update from a participant
 */
export interface GradientUpdate {
  participantId: string;
  modelVersion: number;
  gradients: Float64Array;
  metadata: {
    batchSize: number;
    loss: number;
    accuracy?: number;
    timestamp: number;
  };
}

/**
 * Aggregated gradient result
 */
export interface AggregatedGradient {
  gradients: Float64Array;
  modelVersion: number;
  participantCount: number;
  excludedCount: number;
  aggregationMethod: AggregationMethod;
  timestamp: number;
}

/**
 * Participant state for FL round
 */
export interface Participant {
  id: string;
  reputation: matl.ReputationScore;
  pogq?: matl.ProofOfGradientQuality;
  lastContribution?: GradientUpdate;
  roundsParticipated: number;
}

/**
 * Federated learning round state
 */
export interface FLRound {
  roundId: number;
  modelVersion: number;
  participants: Map<string, Participant>;
  updates: GradientUpdate[];
  status: 'collecting' | 'aggregating' | 'completed';
  startTime: number;
  endTime?: number;
  aggregatedResult?: AggregatedGradient;
}

/**
 * Aggregation methods
 */
export enum AggregationMethod {
  FedAvg = 'fedavg',
  TrimmedMean = 'trimmed_mean',
  Median = 'median',
  Krum = 'krum',
  MultiKrum = 'multi_krum',
  GeometricMedian = 'geometric_median',
  TrustWeighted = 'trust_weighted',
}

/**
 * FL coordinator configuration
 */
export interface FLConfig {
  minParticipants: number;
  maxParticipants: number;
  roundTimeout: number; // ms
  byzantineTolerance: number; // 0-0.34 (34% validated)
  aggregationMethod: AggregationMethod;
  trustThreshold: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_CONFIG: FLConfig = {
  minParticipants: 3,
  maxParticipants: 100,
  roundTimeout: 60000, // 1 minute
  byzantineTolerance: 0.33, // 33% Byzantine tolerance
  aggregationMethod: AggregationMethod.TrustWeighted,
  trustThreshold: 0.5,
};

// ============================================================================
// Aggregation Algorithms
// ============================================================================

/**
 * Federated Averaging (FedAvg)
 * Standard weighted average based on batch sizes
 */
export function fedAvg(updates: GradientUpdate[]): Float64Array {
  validateGradientConsistency(updates);
  updates.forEach(validateUpdateMetadata);

  const gradientSize = updates[0].gradients.length;
  const result = new Float64Array(gradientSize);
  let totalSamples = 0;

  for (const update of updates) {
    totalSamples += update.metadata.batchSize;
  }

  for (const update of updates) {
    const weight = update.metadata.batchSize / totalSamples;
    for (let i = 0; i < gradientSize; i++) {
      result[i] += update.gradients[i] * weight;
    }
  }

  return result;
}

/**
 * Trimmed Mean Aggregation
 * Removes top and bottom percentile before averaging
 */
export function trimmedMean(
  updates: GradientUpdate[],
  trimPercentage: number = 0.1
): Float64Array {
  validateGradientConsistency(updates);

  if (trimPercentage < 0 || trimPercentage >= 0.5) {
    throw new FederatedLearningError(
      'trimPercentage must be between 0 and 0.5',
      ErrorCode.INVALID_ARGUMENT,
      { trimPercentage }
    );
  }

  const gradientSize = updates[0].gradients.length;
  const result = new Float64Array(gradientSize);
  const trimCount = Math.floor(updates.length * trimPercentage);

  for (let i = 0; i < gradientSize; i++) {
    const values = updates.map((u) => u.gradients[i]).sort((a, b) => a - b);

    // Trim top and bottom
    const trimmedValues = values.slice(trimCount, values.length - trimCount);

    // Average remaining
    result[i] =
      trimmedValues.reduce((a, b) => a + b, 0) / trimmedValues.length;
  }

  return result;
}

/**
 * Coordinate-wise Median
 * Robust to up to 50% Byzantine participants
 */
export function coordinateMedian(updates: GradientUpdate[]): Float64Array {
  validateGradientConsistency(updates);

  const gradientSize = updates[0].gradients.length;
  const result = new Float64Array(gradientSize);

  for (let i = 0; i < gradientSize; i++) {
    const values = updates.map((u) => u.gradients[i]).sort((a, b) => a - b);
    const mid = Math.floor(values.length / 2);

    if (values.length % 2 === 0) {
      result[i] = (values[mid - 1] + values[mid]) / 2;
    } else {
      result[i] = values[mid];
    }
  }

  return result;
}

/**
 * Krum Aggregation
 * Selects the gradient closest to its neighbors
 * Tolerates up to (n-2)/2 Byzantine participants
 */
export function krum(
  updates: GradientUpdate[],
  numSelect: number = 1
): Float64Array {
  validateGradientConsistency(updates);

  const n = updates.length;

  if (numSelect < 1 || numSelect > n) {
    throw new FederatedLearningError(
      `numSelect must be between 1 and ${n}`,
      ErrorCode.INVALID_ARGUMENT,
      { numSelect, numUpdates: n }
    );
  }

  if (n < 3) {
    throw new FederatedLearningError(
      'Krum requires at least 3 updates for Byzantine tolerance',
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { required: 3, received: n }
    );
  }
  const numNeighbors = n - 2; // Number of closest neighbors to consider

  // Calculate pairwise distances
  const distances: number[][] = [];
  for (let i = 0; i < n; i++) {
    distances[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        distances[i][j] = Infinity;
      } else {
        distances[i][j] = euclideanDistance(
          updates[i].gradients,
          updates[j].gradients
        );
      }
    }
  }

  // Calculate Krum scores (sum of distances to closest neighbors)
  const scores: { index: number; score: number }[] = [];
  for (let i = 0; i < n; i++) {
    const sortedDistances = [...distances[i]].sort((a, b) => a - b);
    const score = sortedDistances
      .slice(0, numNeighbors)
      .reduce((a, b) => a + b, 0);
    scores.push({ index: i, score });
  }

  // Select updates with lowest scores
  scores.sort((a, b) => a.score - b.score);
  const selectedUpdates = scores
    .slice(0, numSelect)
    .map((s) => updates[s.index]);

  // Average selected updates
  return fedAvg(selectedUpdates);
}

/**
 * Multi-Krum Aggregation
 *
 * Selects top-m gradients by Krum score and averages them.
 * More robust than single Krum when Byzantine fraction is lower.
 * Tolerates up to (n-m-2) Byzantine participants out of n.
 *
 * @param updates - Gradient updates to aggregate
 * @param numSelect - Number of top-scoring gradients to average (default: ceil(n/2))
 */
export function multiKrum(
  updates: GradientUpdate[],
  numSelect?: number,
): Float64Array {
  validateGradientConsistency(updates);

  const n = updates.length;
  const m = numSelect ?? Math.ceil(n / 2);

  if (m < 1 || m > n) {
    throw new FederatedLearningError(
      `numSelect must be between 1 and ${n}`,
      ErrorCode.INVALID_ARGUMENT,
      { numSelect: m, numUpdates: n },
    );
  }

  if (n < 3) {
    throw new FederatedLearningError(
      'Multi-Krum requires at least 3 updates',
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { required: 3, received: n },
    );
  }

  const numNeighbors = n - 2;

  // Calculate pairwise distances
  const distances: number[][] = [];
  for (let i = 0; i < n; i++) {
    distances[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        distances[i][j] = Infinity;
      } else {
        distances[i][j] = euclideanDistance(updates[i].gradients, updates[j].gradients);
      }
    }
  }

  // Calculate Krum scores
  const scores: { index: number; score: number }[] = [];
  for (let i = 0; i < n; i++) {
    const sortedDistances = [...distances[i]].sort((a, b) => a - b);
    const score = sortedDistances.slice(0, numNeighbors).reduce((a, b) => a + b, 0);
    scores.push({ index: i, score });
  }

  // Select top-m by lowest score
  scores.sort((a, b) => a.score - b.score);
  const selectedUpdates = scores.slice(0, m).map((s) => updates[s.index]);

  return fedAvg(selectedUpdates);
}

/**
 * Geometric Median Aggregation (Weiszfeld algorithm)
 *
 * Computes the L2 geometric median — the point minimizing the sum of
 * Euclidean distances to all inputs. More robust than coordinate-wise
 * median against coordinated attacks across dimensions.
 *
 * @param updates - Gradient updates to aggregate
 * @param maxIterations - Maximum Weiszfeld iterations (default: 100)
 * @param tolerance - Convergence tolerance (default: 1e-7)
 */
export function geometricMedian(
  updates: GradientUpdate[],
  maxIterations: number = 100,
  tolerance: number = 1e-7,
): Float64Array {
  validateGradientConsistency(updates);

  if (updates.length === 1) {
    return new Float64Array(updates[0].gradients);
  }

  const gradientSize = updates[0].gradients.length;
  const n = updates.length;

  // Initialize with coordinate-wise mean
  const estimate = new Float64Array(gradientSize);
  for (const u of updates) {
    for (let i = 0; i < gradientSize; i++) {
      estimate[i] += u.gradients[i] / n;
    }
  }

  // Weiszfeld iterations
  for (let iter = 0; iter < maxIterations; iter++) {
    const numerator = new Float64Array(gradientSize);
    let denominator = 0;

    for (const u of updates) {
      const dist = euclideanDistance(estimate, u.gradients);
      if (dist < 1e-10) continue; // Skip coincident points
      const w = 1.0 / dist;
      denominator += w;
      for (let i = 0; i < gradientSize; i++) {
        numerator[i] += u.gradients[i] * w;
      }
    }

    if (denominator < 1e-10) break;

    let maxDiff = 0;
    for (let i = 0; i < gradientSize; i++) {
      const newVal = numerator[i] / denominator;
      maxDiff = Math.max(maxDiff, Math.abs(newVal - estimate[i]));
      estimate[i] = newVal;
    }

    if (maxDiff < tolerance) break;
  }

  return estimate;
}

/**
 * Trust-Weighted Aggregation (MATL Integration)
 * Weights contributions by participant reputation and PoGQ
 */
export function trustWeightedAggregation(
  updates: GradientUpdate[],
  participants: Map<string, Participant>,
  trustThreshold: number = 0.5
): AggregatedGradient {
  validateGradientConsistency(updates);
  updates.forEach(validateUpdateMetadata);

  if (trustThreshold < 0 || trustThreshold > 1) {
    throw new FederatedLearningError(
      'trustThreshold must be between 0 and 1',
      ErrorCode.INVALID_ARGUMENT,
      { trustThreshold }
    );
  }

  const gradientSize = updates[0].gradients.length;
  const result = new Float64Array(gradientSize);
  let totalWeight = 0;
  let excludedCount = 0;

  // Calculate trust weights
  const weights: Map<string, number> = new Map();

  for (const update of updates) {
    const participant = participants.get(update.participantId);

    if (!participant) {
      excludedCount++;
      continue;
    }

    const repValue = matl.reputationValue(participant.reputation);

    // Check trust threshold
    if (repValue < trustThreshold) {
      excludedCount++;
      continue;
    }

    // Calculate weight from reputation and PoGQ
    let weight = repValue;

    if (participant.pogq) {
      const composite = matl.calculateComposite(
        participant.pogq,
        participant.reputation
      );
      weight = composite.finalScore;
    }

    // Scale by batch size
    weight *= update.metadata.batchSize;
    weights.set(update.participantId, weight);
    totalWeight += weight;
  }

  // Aggregate with trust weights
  for (const update of updates) {
    const weight = weights.get(update.participantId);
    if (weight === undefined) continue;

    const normalizedWeight = weight / totalWeight;
    for (let i = 0; i < gradientSize; i++) {
      result[i] += update.gradients[i] * normalizedWeight;
    }
  }

  return {
    gradients: result,
    modelVersion: 1,
    participantCount: updates.length - excludedCount,
    excludedCount,
    aggregationMethod: AggregationMethod.TrustWeighted,
    timestamp: Date.now(),
  };
}

// ============================================================================
// Byzantine Detection
// ============================================================================

/**
 * Byzantine detection result
 */
export interface ByzantineDetectionResult {
  /** Indices of detected Byzantine participants */
  byzantineIndices: number[];
  /** Detection confidence scores */
  confidenceScores: number[];
  /** Whether the round should be rejected entirely */
  shouldReject: boolean;
}

/**
 * Norm-based Byzantine detection
 *
 * Uses z-score outlier detection on L2 norms to identify obviously
 * Byzantine gradient updates before running aggregation.
 *
 * @param updates - Gradient updates to check
 * @param zThreshold - Z-score threshold for flagging (default: 3.0)
 */
export function detectByzantine(
  updates: GradientUpdate[],
  zThreshold: number = 3.0,
): ByzantineDetectionResult {
  if (updates.length < 3) {
    return { byzantineIndices: [], confidenceScores: [], shouldReject: false };
  }

  // Compute L2 norms
  const norms = updates.map(u => {
    let sum = 0;
    for (let i = 0; i < u.gradients.length; i++) {
      sum += u.gradients[i] * u.gradients[i];
    }
    return Math.sqrt(sum);
  });

  // Compute mean and std
  const meanNorm = norms.reduce((a, b) => a + b, 0) / norms.length;
  const variance = norms.reduce((sum, n) => sum + (n - meanNorm) ** 2, 0) / norms.length;
  const stdNorm = Math.sqrt(variance);

  const byzantineIndices: number[] = [];
  const confidenceScores: number[] = [];

  if (stdNorm > 1e-10) {
    for (let i = 0; i < norms.length; i++) {
      const zScore = Math.abs(norms[i] - meanNorm) / stdNorm;
      if (zScore > zThreshold) {
        byzantineIndices.push(i);
        confidenceScores.push(Math.min((zScore - zThreshold) / zThreshold, 1.0));
      }
    }
  }

  // Also flag extreme absolute norms (> 1000x median)
  const sortedNorms = [...norms].sort((a, b) => a - b);
  const medianNorm = sortedNorms[Math.floor(sortedNorms.length / 2)];
  for (let i = 0; i < norms.length; i++) {
    if (!byzantineIndices.includes(i) && norms[i] > 100 * Math.max(medianNorm, 1.0)) {
      byzantineIndices.push(i);
      confidenceScores.push(1.0);
    }
  }

  const byzantineFraction = byzantineIndices.length / updates.length;
  const shouldReject = byzantineFraction > 0.5;

  return { byzantineIndices, confidenceScores, shouldReject };
}

/**
 * Filter updates by removing detected Byzantine participants
 */
function filterByzantine(
  updates: GradientUpdate[],
  byzantineTolerance: number,
): { filtered: GradientUpdate[]; detection: ByzantineDetectionResult } {
  const detection = detectByzantine(updates);

  if (detection.shouldReject) {
    throw new FederatedLearningError(
      'Too many Byzantine participants detected (>50%)',
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { byzantineCount: detection.byzantineIndices.length, total: updates.length }
    );
  }

  const byzantineFraction = detection.byzantineIndices.length / updates.length;
  if (byzantineFraction > byzantineTolerance) {
    throw new FederatedLearningError(
      `Byzantine fraction ${(byzantineFraction * 100).toFixed(1)}% exceeds tolerance ${(byzantineTolerance * 100).toFixed(1)}%`,
      ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS,
      { byzantineFraction, byzantineTolerance }
    );
  }

  const filtered = updates.filter((_, i) => !detection.byzantineIndices.includes(i));
  return { filtered, detection };
}

// ============================================================================
// FL Coordinator
// ============================================================================

/**
 * Federated Learning Coordinator
 * Manages FL rounds and participant interactions
 */
export class FLCoordinator {
  private config: FLConfig;
  private currentRound: FLRound | null = null;
  private roundHistory: FLRound[] = [];
  private participants: Map<string, Participant> = new Map();
  private modelVersion: number = 0;

  constructor(config: Partial<FLConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    validateFLConfig(this.config);
  }

  /**
   * Register a participant
   */
  registerParticipant(id: string): Participant {
    const participant: Participant = {
      id,
      reputation: matl.createReputation(id),
      roundsParticipated: 0,
    };
    this.participants.set(id, participant);
    return participant;
  }

  /**
   * Start a new FL round
   */
  startRound(): FLRound {
    if (this.currentRound && this.currentRound.status !== 'completed') {
      throw new FederatedLearningError(
        'Previous round not completed',
        ErrorCode.FL_ROUND_NOT_STARTED,
        { currentStatus: this.currentRound.status, roundId: this.currentRound.roundId }
      );
    }

    this.modelVersion++;
    this.currentRound = {
      roundId: this.roundHistory.length + 1,
      modelVersion: this.modelVersion,
      participants: new Map(this.participants),
      updates: [],
      status: 'collecting',
      startTime: Date.now(),
    };

    return this.currentRound;
  }

  /**
   * Submit a gradient update
   */
  submitUpdate(update: GradientUpdate): boolean {
    if (!this.currentRound || this.currentRound.status !== 'collecting') {
      return false;
    }

    if (update.modelVersion !== this.currentRound.modelVersion) {
      return false;
    }

    // Validate update metadata
    try {
      validateUpdateMetadata(update);
    } catch {
      return false;
    }

    // Validate gradient consistency if we have previous updates
    if (this.currentRound.updates.length > 0) {
      const expectedSize = this.currentRound.updates[0].gradients.length;
      if (update.gradients.length !== expectedSize) {
        return false;
      }
    }

    // Validate participant
    const participant = this.participants.get(update.participantId);
    if (!participant) {
      return false;
    }

    // Check reputation threshold
    const repValue = matl.reputationValue(participant.reputation);
    if (repValue < this.config.trustThreshold) {
      return false;
    }

    // Create PoGQ for this update
    const quality = this.calculateUpdateQuality(update);
    participant.pogq = matl.createPoGQ(quality, 0.8, 0.1);
    participant.lastContribution = update;

    this.currentRound.updates.push(update);

    // Check if we have enough participants
    if (this.currentRound.updates.length >= this.config.maxParticipants) {
      return this.aggregateRound();
    }

    return true;
  }

  /**
   * Aggregate the current round
   */
  aggregateRound(): boolean {
    if (!this.currentRound || this.currentRound.status !== 'collecting') {
      return false;
    }

    if (this.currentRound.updates.length < this.config.minParticipants) {
      return false;
    }

    this.currentRound.status = 'aggregating';

    // Step 1: Apply Byzantine detection and filtering
    const { filtered } = filterByzantine(
      this.currentRound.updates,
      this.config.byzantineTolerance
    );
    const excludedCount = this.currentRound.updates.length - filtered.length;

    if (filtered.length < this.config.minParticipants) {
      this.currentRound.status = 'collecting';
      return false;
    }

    // Step 2: Perform aggregation on filtered updates
    let aggregatedResult: AggregatedGradient;

    switch (this.config.aggregationMethod) {
      case AggregationMethod.FedAvg:
        aggregatedResult = {
          gradients: fedAvg(filtered),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.FedAvg,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.TrimmedMean:
        // Use byzantineTolerance as minimum trim floor for defense-in-depth:
        // pre-filtering removes detected Byzantine nodes, trimming handles
        // undetected adversaries that passed the z-score filter.
        aggregatedResult = {
          gradients: trimmedMean(filtered, Math.max(0.1, this.config.byzantineTolerance)),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.TrimmedMean,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.Median:
        aggregatedResult = {
          gradients: coordinateMedian(filtered),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.Median,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.Krum:
        aggregatedResult = {
          gradients: krum(filtered),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.Krum,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.MultiKrum:
        aggregatedResult = {
          gradients: multiKrum(filtered),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.MultiKrum,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.GeometricMedian:
        aggregatedResult = {
          gradients: geometricMedian(filtered),
          modelVersion: this.roundHistory.length + 1,
          participantCount: filtered.length,
          excludedCount,
          aggregationMethod: AggregationMethod.GeometricMedian,
          timestamp: Date.now(),
        };
        break;

      case AggregationMethod.TrustWeighted:
      default:
        aggregatedResult = trustWeightedAggregation(
          filtered,
          this.participants,
          this.config.trustThreshold
        );
        // Add Byzantine-excluded count
        aggregatedResult.excludedCount += excludedCount;
        break;
    }

    // Store the aggregated result in the round
    this.currentRound.aggregatedResult = aggregatedResult;

    // Update participant reputations based on contribution quality
    this.updateReputations();

    this.currentRound.status = 'completed';
    this.currentRound.endTime = Date.now();
    this.roundHistory.push(this.currentRound);

    return true;
  }

  /**
   * Get round statistics
   */
  getRoundStats(): {
    totalRounds: number;
    currentRound: FLRound | null;
    participantCount: number;
    averageParticipation: number;
  } {
    const totalParticipation = this.roundHistory.reduce(
      (sum, r) => sum + r.updates.length,
      0
    );

    return {
      totalRounds: this.roundHistory.length,
      currentRound: this.currentRound,
      participantCount: this.participants.size,
      averageParticipation:
        this.roundHistory.length > 0
          ? totalParticipation / this.roundHistory.length
          : 0,
    };
  }

  /**
   * Calculate update quality for PoGQ
   */
  private calculateUpdateQuality(update: GradientUpdate): number {
    // Base quality on loss improvement and gradient magnitude
    const lossQuality = Math.exp(-update.metadata.loss);
    const gradientMag = Math.sqrt(
      update.gradients.reduce((sum, g) => sum + g * g, 0)
    );
    const magQuality = 1.0 / (1.0 + Math.log1p(gradientMag));

    return (lossQuality + magQuality) / 2;
  }

  /**
   * Update participant reputations after round
   */
  private updateReputations(): void {
    if (!this.currentRound) return;

    for (const update of this.currentRound.updates) {
      const participant = this.participants.get(update.participantId);
      if (!participant) continue;

      // Positive update for valid contribution
      participant.reputation = matl.recordPositive(participant.reputation);
      participant.roundsParticipated++;
    }
  }
}

// ============================================================================
// Utilities
// ============================================================================

/**
 * Euclidean distance between two gradient vectors
 */
function euclideanDistance(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Serialize gradients for transmission
 * @throws {FederatedLearningError} If gradients is empty
 */
export function serializeGradients(gradients: Float64Array): Uint8Array {
  if (gradients.length === 0) {
    throw new FederatedLearningError(
      'Cannot serialize empty gradients',
      ErrorCode.INVALID_ARGUMENT,
      { length: gradients.length }
    );
  }

  const buffer = new ArrayBuffer(gradients.length * 8);
  const view = new DataView(buffer);

  for (let i = 0; i < gradients.length; i++) {
    view.setFloat64(i * 8, gradients[i], true);
  }

  return new Uint8Array(buffer);
}

/**
 * Deserialize gradients from transmission
 * @throws {FederatedLearningError} If data length is not a multiple of 8
 */
export function deserializeGradients(data: Uint8Array): Float64Array {
  if (data.length % 8 !== 0) {
    throw new FederatedLearningError(
      'Invalid gradient data: length must be a multiple of 8 bytes',
      ErrorCode.INVALID_ARGUMENT,
      { length: data.length, remainder: data.length % 8 }
    );
  }

  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const gradients = new Float64Array(data.length / 8);

  for (let i = 0; i < gradients.length; i++) {
    gradients[i] = view.getFloat64(i * 8, true);
  }

  return gradients;
}

/**
 * Serialized gradient update for JSON transmission
 */
export interface SerializedGradientUpdate {
  participantId: string;
  modelVersion: number;
  gradients: string; // base64 encoded
  metadata: {
    batchSize: number;
    loss: number;
    accuracy?: number;
    timestamp: number;
  };
}

/**
 * Serialize a gradient update for JSON transmission
 */
export function serializeGradientUpdate(update: GradientUpdate): SerializedGradientUpdate {
  const bytes = serializeGradients(update.gradients);
  const base64 = btoa(String.fromCharCode(...bytes));

  return {
    participantId: update.participantId,
    modelVersion: update.modelVersion,
    gradients: base64,
    metadata: update.metadata,
  };
}

/**
 * Deserialize a gradient update from JSON transmission
 */
export function deserializeGradientUpdate(serialized: SerializedGradientUpdate): GradientUpdate {
  const binaryString = atob(serialized.gradients);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return {
    participantId: serialized.participantId,
    modelVersion: serialized.modelVersion,
    gradients: deserializeGradients(bytes),
    metadata: serialized.metadata,
  };
}

/**
 * Serialized aggregated gradient for JSON transmission
 */
export interface SerializedAggregatedGradient {
  gradients: string; // base64 encoded
  modelVersion: number;
  participantCount: number;
  excludedCount?: number;
  aggregationMethod: AggregationMethod;
  timestamp: number;
}

/**
 * Serialize an aggregated gradient for JSON transmission
 */
export function serializeAggregatedGradient(aggregated: AggregatedGradient): SerializedAggregatedGradient {
  const bytes = serializeGradients(aggregated.gradients);
  const base64 = btoa(String.fromCharCode(...bytes));

  return {
    gradients: base64,
    modelVersion: aggregated.modelVersion,
    participantCount: aggregated.participantCount,
    excludedCount: aggregated.excludedCount,
    aggregationMethod: aggregated.aggregationMethod,
    timestamp: aggregated.timestamp,
  };
}

/**
 * Deserialize an aggregated gradient from JSON transmission
 */
export function deserializeAggregatedGradient(serialized: SerializedAggregatedGradient): AggregatedGradient {
  const binaryString = atob(serialized.gradients);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return {
    gradients: deserializeGradients(bytes),
    modelVersion: serialized.modelVersion,
    participantCount: serialized.participantCount,
    excludedCount: serialized.excludedCount ?? 0,
    aggregationMethod: serialized.aggregationMethod,
    timestamp: serialized.timestamp,
  };
}

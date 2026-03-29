// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Adaptive Thresholds
 *
 * Online learning system for dynamically adjusting detection thresholds
 * using UCB1 multi-armed bandit and gradient estimation algorithms.
 *
 * @packageDocumentation
 * @module agentic/adaptive-thresholds
 */

// =============================================================================
// Configuration
// =============================================================================

/**
 * Adaptive threshold configuration
 */
export interface AdaptiveConfig {
  /** Initial threshold value */
  initialThreshold: number;
  /** Minimum allowed threshold */
  minThreshold: number;
  /** Maximum allowed threshold */
  maxThreshold: number;
  /** Learning rate for gradient updates */
  learningRate: number;
  /** UCB exploration coefficient */
  explorationCoeff: number;
  /** Momentum for gradient estimation */
  momentum: number;
}

/**
 * Create default adaptive configuration
 */
export function createDefaultAdaptiveConfig(): AdaptiveConfig {
  return {
    initialThreshold: 0.5,
    minThreshold: 0.1,
    maxThreshold: 0.9,
    learningRate: 0.01,
    explorationCoeff: 1.0,
    momentum: 0.9,
  };
}

// =============================================================================
// Threshold Types
// =============================================================================

/**
 * Types of thresholds that can be adapted
 */
export enum ThresholdType {
  /** Trust acceptance threshold */
  TrustAcceptance = 'TrustAcceptance',
  /** Gaming detection threshold */
  GamingDetection = 'GamingDetection',
  /** Quarantine threshold */
  Quarantine = 'Quarantine',
  /** Slashing threshold */
  Slashing = 'Slashing',
  /** Byzantine detection threshold */
  Byzantine = 'Byzantine',
}

// =============================================================================
// Feedback Types
// =============================================================================

/**
 * Outcome of threshold decision
 */
export enum FeedbackOutcome {
  /** True positive: correctly triggered */
  TruePositive = 'TruePositive',
  /** True negative: correctly didn't trigger */
  TrueNegative = 'TrueNegative',
  /** False positive: incorrectly triggered */
  FalsePositive = 'FalsePositive',
  /** False negative: should have triggered */
  FalseNegative = 'FalseNegative',
}

/**
 * Feedback context for threshold adjustment
 */
export interface FeedbackContext {
  /** Current network health */
  networkHealth: number;
  /** Number of active agents */
  activeAgents: number;
  /** Current threat level */
  threatLevel: number;
}

/**
 * Create default feedback context
 */
export function createDefaultFeedbackContext(): FeedbackContext {
  return {
    networkHealth: 1.0,
    activeAgents: 100,
    threatLevel: 0.0,
  };
}

/**
 * Threshold feedback for learning
 */
export interface ThresholdFeedback {
  /** Type of threshold */
  thresholdType: ThresholdType;
  /** Threshold value used */
  thresholdValue: number;
  /** Outcome of the decision */
  outcome: FeedbackOutcome;
  /** Context at time of decision */
  context: FeedbackContext;
  /** Timestamp */
  timestamp: number;
}

// =============================================================================
// UCB1 Bandit for Threshold Selection
// =============================================================================

/**
 * Arm statistics for UCB1
 */
interface ArmStats {
  count: number;
  totalReward: number;
  avgReward: number;
}

/**
 * UCB1 multi-armed bandit for threshold selection
 */
export class ThresholdBandit {
  private arms: Map<number, ArmStats> = new Map();
  private totalPulls = 0;
  private explorationCoeff: number;
  private thresholdValues: number[];

  constructor(
    thresholdValues: number[],
    explorationCoeff = 1.0
  ) {
    this.explorationCoeff = explorationCoeff;
    this.thresholdValues = thresholdValues;

    // Initialize arms
    for (const value of thresholdValues) {
      this.arms.set(value, { count: 0, totalReward: 0, avgReward: 0 });
    }
  }

  /**
   * Select next threshold using UCB1
   */
  selectThreshold(): number {
    // First, try any unexplored arms
    for (const [value, stats] of this.arms) {
      if (stats.count === 0) {
        return value;
      }
    }

    // UCB1 selection
    let bestValue = this.thresholdValues[0];
    let bestUcb = -Infinity;

    for (const [value, stats] of this.arms) {
      const ucb =
        stats.avgReward +
        this.explorationCoeff *
          Math.sqrt((2 * Math.log(this.totalPulls)) / stats.count);

      if (ucb > bestUcb) {
        bestUcb = ucb;
        bestValue = value;
      }
    }

    return bestValue;
  }

  /**
   * Update arm with reward
   */
  update(threshold: number, reward: number): void {
    const stats = this.arms.get(threshold);
    if (!stats) return;

    stats.count++;
    stats.totalReward += reward;
    stats.avgReward = stats.totalReward / stats.count;
    this.totalPulls++;
  }

  /**
   * Get best threshold based on average reward
   */
  getBestThreshold(): number {
    let best = this.thresholdValues[0];
    let bestAvg = -Infinity;

    for (const [value, stats] of this.arms) {
      if (stats.count > 0 && stats.avgReward > bestAvg) {
        bestAvg = stats.avgReward;
        best = value;
      }
    }

    return best;
  }

  /**
   * Get statistics for all arms
   */
  getStats(): Map<number, ArmStats> {
    return new Map(this.arms);
  }
}

// =============================================================================
// Gradient Estimator
// =============================================================================

/**
 * Gradient estimator with momentum
 */
export class GradientEstimator {
  private gradient = 0;
  private momentum: number;
  private velocity = 0;

  constructor(momentum = 0.9) {
    this.momentum = momentum;
  }

  /**
   * Update gradient estimate based on outcome
   */
  update(_threshold: number, outcome: FeedbackOutcome): number {
    // Compute local gradient based on outcome
    let localGradient: number;

    switch (outcome) {
      case FeedbackOutcome.TruePositive:
        // Correct detection - no adjustment needed
        localGradient = 0;
        break;
      case FeedbackOutcome.TrueNegative:
        // Correct non-detection - no adjustment needed
        localGradient = 0;
        break;
      case FeedbackOutcome.FalsePositive:
        // Too sensitive - increase threshold
        localGradient = 0.1;
        break;
      case FeedbackOutcome.FalseNegative:
        // Not sensitive enough - decrease threshold
        localGradient = -0.1;
        break;
    }

    // Apply momentum
    this.velocity = this.momentum * this.velocity + (1 - this.momentum) * localGradient;
    this.gradient = this.velocity;

    return this.gradient;
  }

  /**
   * Get current gradient
   */
  getGradient(): number {
    return this.gradient;
  }

  /**
   * Reset estimator
   */
  reset(): void {
    this.gradient = 0;
    this.velocity = 0;
  }
}

// =============================================================================
// Adaptive Threshold Engine
// =============================================================================

/**
 * Main adaptive threshold engine
 */
export class AdaptiveThresholdEngine {
  private config: AdaptiveConfig;
  private thresholds: Map<ThresholdType, number> = new Map();
  private bandits: Map<ThresholdType, ThresholdBandit> = new Map();
  private gradientEstimators: Map<ThresholdType, GradientEstimator> = new Map();
  private feedbackHistory: ThresholdFeedback[] = [];

  constructor(config: AdaptiveConfig = createDefaultAdaptiveConfig()) {
    this.config = config;

    // Initialize all threshold types
    for (const type of Object.values(ThresholdType)) {
      this.thresholds.set(type as ThresholdType, config.initialThreshold);

      // Create bandit with discrete threshold options
      const thresholdOptions = this.generateThresholdOptions();
      this.bandits.set(
        type as ThresholdType,
        new ThresholdBandit(thresholdOptions, config.explorationCoeff)
      );

      this.gradientEstimators.set(
        type as ThresholdType,
        new GradientEstimator(config.momentum)
      );
    }
  }

  /**
   * Generate discrete threshold options
   */
  private generateThresholdOptions(): number[] {
    const options: number[] = [];
    const step = (this.config.maxThreshold - this.config.minThreshold) / 10;
    for (
      let t = this.config.minThreshold;
      t <= this.config.maxThreshold;
      t += step
    ) {
      options.push(Math.round(t * 100) / 100);
    }
    return options;
  }

  /**
   * Get current threshold
   */
  getThreshold(type: ThresholdType): number {
    return this.thresholds.get(type) ?? this.config.initialThreshold;
  }

  /**
   * Process feedback and update thresholds
   */
  processFeedback(feedback: ThresholdFeedback): void {
    this.feedbackHistory.push(feedback);

    // Update gradient estimator
    const estimator = this.gradientEstimators.get(feedback.thresholdType);
    if (estimator) {
      const gradient = estimator.update(
        feedback.thresholdValue,
        feedback.outcome
      );

      // Apply gradient update
      const currentThreshold = this.getThreshold(feedback.thresholdType);
      const newThreshold = this.clamp(
        currentThreshold + this.config.learningRate * gradient
      );
      this.thresholds.set(feedback.thresholdType, newThreshold);
    }

    // Update bandit
    const bandit = this.bandits.get(feedback.thresholdType);
    if (bandit) {
      const reward = this.computeReward(feedback.outcome);
      bandit.update(feedback.thresholdValue, reward);
    }
  }

  /**
   * Compute reward for outcome
   */
  private computeReward(outcome: FeedbackOutcome): number {
    switch (outcome) {
      case FeedbackOutcome.TruePositive:
        return 1.0;
      case FeedbackOutcome.TrueNegative:
        return 1.0;
      case FeedbackOutcome.FalsePositive:
        return -0.5;
      case FeedbackOutcome.FalseNegative:
        return -1.0;
    }
  }

  /**
   * Clamp threshold to valid range
   */
  private clamp(value: number): number {
    return Math.max(
      this.config.minThreshold,
      Math.min(this.config.maxThreshold, value)
    );
  }

  /**
   * Get optimal threshold from bandit
   */
  getOptimalThreshold(type: ThresholdType): number {
    const bandit = this.bandits.get(type);
    if (bandit) {
      return bandit.getBestThreshold();
    }
    return this.config.initialThreshold;
  }

  /**
   * Get feedback history
   */
  getFeedbackHistory(): ThresholdFeedback[] {
    return [...this.feedbackHistory];
  }

  /**
   * Get all current thresholds
   */
  getAllThresholds(): Map<ThresholdType, number> {
    return new Map(this.thresholds);
  }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Differential Privacy for Trust Analytics
 *
 * Provides differential privacy guarantees for trust metrics aggregation.
 * Uses Laplace and Gaussian noise mechanisms to protect individual agent privacy
 * while enabling population-level analytics.
 *
 * @packageDocumentation
 * @module agentic/differential-privacy
 */

// =============================================================================
// Configuration
// =============================================================================

/**
 * Differential privacy configuration
 */
export interface DPConfig {
  /** Privacy budget epsilon (lower = more privacy) */
  epsilon: number;
  /** Privacy parameter delta for approximate DP */
  delta: number;
  /** Noise mechanism to use */
  mechanism: NoiseMechanism;
  /** Clipping bound for sensitivity calculation */
  clippingBound: number;
}

/**
 * Create default DP configuration
 */
export function createDefaultDPConfig(): DPConfig {
  return {
    epsilon: 1.0,
    delta: 1e-6,
    mechanism: NoiseMechanism.Laplace,
    clippingBound: 1.0,
  };
}

/**
 * Noise mechanism types
 */
export enum NoiseMechanism {
  /** Laplace mechanism for pure DP */
  Laplace = 'Laplace',
  /** Gaussian mechanism for approximate DP */
  Gaussian = 'Gaussian',
}

// =============================================================================
// Noise Generation
// =============================================================================

/**
 * Generate Laplace noise
 * Uses inverse CDF sampling: X = -b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
 */
export function sampleLaplace(scale: number): number {
  const u = Math.random();
  const shifted = u - 0.5;
  const sign = shifted >= 0 ? 1 : -1;
  return -scale * sign * Math.log(1 - 2 * Math.abs(shifted));
}

/**
 * Generate Gaussian noise using Box-Muller transform
 */
export function sampleGaussian(stdDev: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * stdDev;
}

/**
 * Add noise based on mechanism type
 */
export function addNoise(
  value: number,
  sensitivity: number,
  config: DPConfig
): number {
  const noise =
    config.mechanism === NoiseMechanism.Laplace
      ? sampleLaplace(sensitivity / config.epsilon)
      : sampleGaussian(
          (sensitivity * Math.sqrt(2 * Math.log(1.25 / config.delta))) /
            config.epsilon
        );

  return value + noise;
}

// =============================================================================
// Privacy Budget Tracking
// =============================================================================

/**
 * Privacy budget tracker
 */
export class PrivacyBudget {
  private totalEpsilon: number;
  private totalDelta: number;
  private consumedEpsilon = 0;
  private consumedDelta = 0;
  private queryCount = 0;

  constructor(epsilon: number, delta: number) {
    this.totalEpsilon = epsilon;
    this.totalDelta = delta;
  }

  /**
   * Check if budget allows a query
   */
  canQuery(epsilonCost: number, deltaCost = 0): boolean {
    return (
      this.consumedEpsilon + epsilonCost <= this.totalEpsilon &&
      this.consumedDelta + deltaCost <= this.totalDelta
    );
  }

  /**
   * Consume budget for a query
   */
  consume(epsilonCost: number, deltaCost = 0): boolean {
    if (!this.canQuery(epsilonCost, deltaCost)) {
      return false;
    }
    this.consumedEpsilon += epsilonCost;
    this.consumedDelta += deltaCost;
    this.queryCount++;
    return true;
  }

  /**
   * Get remaining budget
   */
  remaining(): { epsilon: number; delta: number } {
    return {
      epsilon: this.totalEpsilon - this.consumedEpsilon,
      delta: this.totalDelta - this.consumedDelta,
    };
  }

  /**
   * Reset budget for new epoch
   */
  reset(): void {
    this.consumedEpsilon = 0;
    this.consumedDelta = 0;
    this.queryCount = 0;
  }

  /**
   * Get query count
   */
  getQueryCount(): number {
    return this.queryCount;
  }
}

// =============================================================================
// Private Aggregator
// =============================================================================

/**
 * Differentially private aggregator for trust metrics
 */
export class PrivateAggregator {
  private config: DPConfig;
  private budget: PrivacyBudget;

  constructor(config: DPConfig) {
    this.config = config;
    this.budget = new PrivacyBudget(config.epsilon, config.delta);
  }

  /**
   * Compute private mean
   */
  privateMean(
    values: number[],
    epsilonPerQuery: number
  ): { value: number; success: boolean } {
    if (!this.budget.consume(epsilonPerQuery)) {
      return { value: 0, success: false };
    }

    if (values.length === 0) {
      return { value: 0, success: true };
    }

    // Clip values to sensitivity bound
    const clipped = values.map((v) =>
      Math.max(-this.config.clippingBound, Math.min(this.config.clippingBound, v))
    );

    // Compute true mean
    const sum = clipped.reduce((a, b) => a + b, 0);
    const mean = sum / values.length;

    // Add noise (sensitivity = clippingBound / n)
    const sensitivity = this.config.clippingBound / values.length;
    const noisyMean = addNoise(mean, sensitivity, {
      ...this.config,
      epsilon: epsilonPerQuery,
    });

    return { value: noisyMean, success: true };
  }

  /**
   * Compute private count
   */
  privateCount(
    values: number[],
    predicate: (v: number) => boolean,
    epsilonPerQuery: number
  ): { value: number; success: boolean } {
    if (!this.budget.consume(epsilonPerQuery)) {
      return { value: 0, success: false };
    }

    const count = values.filter(predicate).length;

    // Sensitivity = 1 for counting queries
    const noisyCount = addNoise(count, 1, {
      ...this.config,
      epsilon: epsilonPerQuery,
    });

    return { value: Math.max(0, Math.round(noisyCount)), success: true };
  }

  /**
   * Compute private histogram
   */
  privateHistogram(
    values: number[],
    bins: number,
    epsilonPerQuery: number
  ): { histogram: number[]; success: boolean } {
    if (!this.budget.consume(epsilonPerQuery)) {
      return { histogram: [], success: false };
    }

    // Create histogram bins [0, 1] divided into `bins` buckets
    const histogram = new Array(bins).fill(0);
    const binWidth = 1 / bins;

    for (const v of values) {
      const clipped = Math.max(0, Math.min(1, v));
      const binIndex = Math.min(Math.floor(clipped / binWidth), bins - 1);
      histogram[binIndex]++;
    }

    // Add noise to each bin (sensitivity = 1 per bin)
    const epsilonPerBin = epsilonPerQuery / bins;
    const noisyHistogram = histogram.map((count) =>
      Math.max(
        0,
        Math.round(addNoise(count, 1, { ...this.config, epsilon: epsilonPerBin }))
      )
    );

    return { histogram: noisyHistogram, success: true };
  }

  /**
   * Get remaining budget
   */
  remainingBudget(): { epsilon: number; delta: number } {
    return this.budget.remaining();
  }

  /**
   * Reset budget
   */
  resetBudget(): void {
    this.budget.reset();
  }
}

// =============================================================================
// Trust Distribution Analytics
// =============================================================================

/**
 * Private trust distribution result
 */
export interface TrustDistribution {
  /** Private mean trust score */
  mean: number;
  /** Private median (approximated) */
  median: number;
  /** Private count */
  count: number;
  /** Private histogram bins */
  histogram: number[];
}

/**
 * Private trust analytics system
 */
export class PrivateTrustAnalytics {
  private aggregator: PrivateAggregator;

  constructor(config: DPConfig) {
    this.aggregator = new PrivateAggregator(config);
  }

  /**
   * Analyze trust distribution with DP guarantees
   */
  analyzeTrustDistribution(
    trustScores: number[],
    epsilonPerQuery: number
  ): TrustDistribution | null {
    // Compute private mean
    const meanResult = this.aggregator.privateMean(trustScores, epsilonPerQuery);
    if (!meanResult.success) return null;

    // Compute private histogram (10 bins)
    const histResult = this.aggregator.privateHistogram(
      trustScores,
      10,
      epsilonPerQuery
    );
    if (!histResult.success) return null;

    // Approximate median from histogram
    const totalCount = histResult.histogram.reduce((a, b) => a + b, 0);
    let cumulativeCount = 0;
    let medianBin = 0;
    for (let i = 0; i < histResult.histogram.length; i++) {
      cumulativeCount += histResult.histogram[i];
      if (cumulativeCount >= totalCount / 2) {
        medianBin = i;
        break;
      }
    }
    const median = (medianBin + 0.5) / 10;

    // Compute private count
    const countResult = this.aggregator.privateCount(
      trustScores,
      () => true,
      epsilonPerQuery
    );
    if (!countResult.success) return null;

    return {
      mean: meanResult.value,
      median,
      count: countResult.value,
      histogram: histResult.histogram,
    };
  }

  /**
   * Get remaining privacy budget
   */
  remainingBudget(): { epsilon: number; delta: number } {
    return this.aggregator.remainingBudget();
  }

  /**
   * Reset budget for new epoch
   */
  resetBudget(): void {
    this.aggregator.resetBudget();
  }
}

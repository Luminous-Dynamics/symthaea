/**
 * Privacy Manager
 *
 * Implements differential privacy for federated learning including
 * privacy budget tracking, noise injection, and gradient clipping.
 */

import type {
  PrivacyBudget,
  PrivacyState,
  RoundPrivacyAllocation,
  DifferentialPrivacyConfig,
} from './types.js';

// =============================================================================
// Privacy Manager
// =============================================================================

/**
 * Manages privacy budget and differential privacy mechanisms
 */
export class PrivacyManager {
  private budget: PrivacyBudget;
  private state: PrivacyState;
  private config: DifferentialPrivacyConfig;

  constructor(budget: PrivacyBudget, config?: Partial<DifferentialPrivacyConfig>) {
    this.budget = budget;
    this.config = {
      enabled: true,
      mechanism: 'gaussian',
      clippingBound: 1.0,
      noiseMultiplier: 1.1,
      ...config,
    };
    this.state = {
      totalBudget: budget,
      usedEpsilon: 0,
      remainingEpsilon: budget.epsilon,
      roundAllocations: [],
    };
  }

  // ===========================================================================
  // Budget Management
  // ===========================================================================

  /**
   * Get remaining privacy budget
   */
  getRemainingBudget(): number {
    return this.state.remainingEpsilon;
  }

  /**
   * Check if there's sufficient budget for a round
   */
  hasBudgetForRound(estimatedEpsilon: number): boolean {
    return this.state.remainingEpsilon >= estimatedEpsilon;
  }

  /**
   * Allocate privacy budget for a round
   */
  allocateRoundBudget(round: number, epsilon: number): RoundPrivacyAllocation {
    if (epsilon > this.state.remainingEpsilon) {
      throw new Error(
        `Insufficient privacy budget: requested ${epsilon}, remaining ${this.state.remainingEpsilon}`
      );
    }

    const noiseScale = this.calculateNoiseScale(epsilon);
    const allocation: RoundPrivacyAllocation = {
      round,
      epsilonUsed: epsilon,
      noiseScale,
      clippingBound: this.config.clippingBound,
    };

    this.state.usedEpsilon += epsilon;
    this.state.remainingEpsilon -= epsilon;
    this.state.roundAllocations.push(allocation);

    return allocation;
  }

  /**
   * Get privacy state
   */
  getState(): Readonly<PrivacyState> {
    return { ...this.state };
  }

  /**
   * Calculate recommended per-round epsilon for a given number of rounds
   */
  calculatePerRoundBudget(totalRounds: number, reserveFraction: number = 0.1): number {
    const availableBudget = this.budget.epsilon * (1 - reserveFraction);
    return availableBudget / totalRounds;
  }

  // ===========================================================================
  // Noise Injection
  // ===========================================================================

  /**
   * Add noise to gradients for differential privacy
   */
  addNoise(gradients: number[], allocation: RoundPrivacyAllocation): number[] {
    if (!this.config.enabled) {
      return gradients;
    }

    // First clip the gradients
    const clipped = this.clipGradients(gradients, allocation.clippingBound);

    // Then add noise
    return clipped.map((g) => {
      const noise = this.sampleNoise(allocation.noiseScale);
      return g + noise;
    });
  }

  /**
   * Add noise to a single value
   */
  addNoiseToValue(value: number, allocation: RoundPrivacyAllocation): number {
    if (!this.config.enabled) {
      return value;
    }

    const clipped = Math.max(-allocation.clippingBound, Math.min(allocation.clippingBound, value));
    const noise = this.sampleNoise(allocation.noiseScale);
    return clipped + noise;
  }

  /**
   * Clip gradients to bound their sensitivity
   */
  clipGradients(gradients: number[], maxNorm: number): number[] {
    const norm = Math.sqrt(gradients.reduce((sum, g) => sum + g * g, 0));

    if (norm <= maxNorm) {
      return gradients;
    }

    const scale = maxNorm / norm;
    return gradients.map((g) => g * scale);
  }

  // ===========================================================================
  // Privacy Accounting
  // ===========================================================================

  /**
   * Calculate epsilon for given noise parameters
   * Uses the Gaussian mechanism formula
   */
  calculateEpsilonFromNoise(
    sensitivity: number,
    sigma: number,
    delta: number
  ): number {
    // Basic Gaussian mechanism: epsilon = sqrt(2 * ln(1.25/delta)) * sensitivity / sigma
    return Math.sqrt(2 * Math.log(1.25 / delta)) * (sensitivity / sigma);
  }

  /**
   * Calculate total epsilon using RDP (Renyi Differential Privacy) composition
   * This gives tighter bounds than simple composition
   */
  calculateComposedEpsilon(
    epsilons: number[],
    delta: number
  ): number {
    if (this.budget.accountingMethod === 'simple') {
      // Simple composition: just sum
      return epsilons.reduce((sum, e) => sum + e, 0);
    }

    // RDP composition (approximation)
    // In practice, you'd use a proper RDP accountant
    const sumSquaredEpsilons = epsilons.reduce((sum, e) => sum + e * e, 0);
    const basicSum = epsilons.reduce((sum, e) => sum + e, 0);

    // RDP gives better bounds for many compositions
    const rdpEpsilon = Math.sqrt(sumSquaredEpsilons + 2 * Math.log(1 / delta));

    return Math.min(basicSum, rdpEpsilon);
  }

  /**
   * Estimate privacy loss for a training run
   */
  estimatePrivacyLoss(
    rounds: number,
    samplingRate: number,
    noiseMultiplier: number
  ): PrivacyEstimate {
    const perRoundEpsilon = this.calculateEpsilonFromNoise(
      this.config.clippingBound,
      this.config.clippingBound * noiseMultiplier,
      this.budget.delta / rounds
    );

    // Account for subsampling amplification
    const amplifiedEpsilon = perRoundEpsilon * samplingRate;

    // Compose across rounds
    const totalEpsilon = this.calculateComposedEpsilon(
      Array(rounds).fill(amplifiedEpsilon),
      this.budget.delta
    );

    return {
      perRoundEpsilon: amplifiedEpsilon,
      totalEpsilon,
      delta: this.budget.delta,
      withinBudget: totalEpsilon <= this.budget.epsilon,
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  /**
   * Calculate noise scale for given epsilon
   */
  private calculateNoiseScale(epsilon: number): number {
    // sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    const sensitivity = this.config.clippingBound;
    return (sensitivity * Math.sqrt(2 * Math.log(1.25 / this.budget.delta))) / epsilon;
  }

  /**
   * Sample noise from the configured distribution
   */
  private sampleNoise(scale: number): number {
    if (this.config.mechanism === 'gaussian') {
      return this.sampleGaussian(0, scale);
    } else {
      return this.sampleLaplace(0, scale);
    }
  }

  /**
   * Sample from Gaussian distribution using Box-Muller transform
   */
  private sampleGaussian(mean: number, stdDev: number): number {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + z * stdDev;
  }

  /**
   * Sample from Laplace distribution
   */
  private sampleLaplace(mean: number, scale: number): number {
    const u = Math.random() - 0.5;
    return mean - scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  }
}

// =============================================================================
// Types
// =============================================================================

export interface PrivacyEstimate {
  perRoundEpsilon: number;
  totalEpsilon: number;
  delta: number;
  withinBudget: boolean;
}

// =============================================================================
// Secure Aggregation with Privacy
// =============================================================================

/**
 * Privacy-preserving aggregation that combines
 * secure aggregation with differential privacy
 */
export class PrivateAggregator {
  private privacyManager: PrivacyManager;

  constructor(budget: PrivacyBudget, config?: Partial<DifferentialPrivacyConfig>) {
    this.privacyManager = new PrivacyManager(budget, config);
  }

  /**
   * Aggregate gradients with differential privacy
   */
  aggregateWithPrivacy(
    gradients: number[][],
    round: number,
    perRoundEpsilon?: number
  ): AggregationResult {
    // Calculate per-round budget if not specified
    const epsilon = perRoundEpsilon ?? this.privacyManager.calculatePerRoundBudget(100);

    // Allocate privacy budget
    const allocation = this.privacyManager.allocateRoundBudget(round, epsilon);

    // Clip and average gradients
    const clippedGradients = gradients.map((g) =>
      this.privacyManager.clipGradients(g, allocation.clippingBound)
    );

    // Compute average
    const n = clippedGradients.length;
    const avgLength = clippedGradients[0]?.length ?? 0;
    const averaged = new Array(avgLength).fill(0);

    for (const gradient of clippedGradients) {
      for (let i = 0; i < gradient.length; i++) {
        averaged[i] += gradient[i] / n;
      }
    }

    // Add noise
    const privatized = this.privacyManager.addNoise(averaged, allocation);

    return {
      gradients: privatized,
      allocation,
      participantCount: n,
      remainingBudget: this.privacyManager.getRemainingBudget(),
    };
  }

  /**
   * Get privacy manager for advanced operations
   */
  getPrivacyManager(): PrivacyManager {
    return this.privacyManager;
  }
}

export interface AggregationResult {
  gradients: number[];
  allocation: RoundPrivacyAllocation;
  participantCount: number;
  remainingBudget: number;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a privacy manager
 */
export function createPrivacyManager(
  budget: PrivacyBudget,
  config?: Partial<DifferentialPrivacyConfig>
): PrivacyManager {
  return new PrivacyManager(budget, config);
}

/**
 * Create a privacy-preserving aggregator
 */
export function createPrivateAggregator(
  budget: PrivacyBudget,
  config?: Partial<DifferentialPrivacyConfig>
): PrivateAggregator {
  return new PrivateAggregator(budget, config);
}

/**
 * Create a default privacy budget for development
 */
export function createDevPrivacyBudget(): PrivacyBudget {
  return {
    epsilon: 10,
    delta: 1e-5,
    accountingMethod: 'simple',
  };
}

/**
 * Create a strict privacy budget for production
 */
export function createStrictPrivacyBudget(): PrivacyBudget {
  return {
    epsilon: 1,
    delta: 1e-7,
    accountingMethod: 'rdp',
  };
}

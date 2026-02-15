/**
 * Reputation Decay System
 *
 * Manages time-based reputation decay and activity bonuses.
 * Features:
 * - Configurable decay rates
 * - Activity-based reputation boosts
 * - Minimum reputation floors
 * - Decay history tracking
 */

// ============================================================================
// TYPES
// ============================================================================

export interface ReputationConfig {
  /** Base decay rate per day (0-1) */
  dailyDecayRate: number;
  /** Minimum reputation floor */
  minimumReputation: number;
  /** Maximum reputation ceiling */
  maximumReputation: number;
  /** Half-life in days for decay calculation */
  halfLifeDays: number;
  /** Activity bonus multiplier */
  activityBonusMultiplier: number;
}

export interface ReputationState {
  currentReputation: number;
  baseReputation: number;
  lastActivityAt: number;
  lastDecayAt: number;
  activityStreak: number;
  totalDecay: number;
  totalBonuses: number;
}

export interface ActivityEvent {
  type: 'vote' | 'share' | 'validate' | 'participate';
  timestamp: number;
  weight: number;
}

export interface DecayResult {
  previousReputation: number;
  newReputation: number;
  decayAmount: number;
  daysSinceLastDecay: number;
}

export interface ActivityBonusResult {
  previousReputation: number;
  newReputation: number;
  bonusAmount: number;
  streakBonus: number;
}

// ============================================================================
// DEFAULT CONFIG
// ============================================================================

export const DEFAULT_REPUTATION_CONFIG: ReputationConfig = {
  dailyDecayRate: 0.02, // 2% per day
  minimumReputation: 10,
  maximumReputation: 1000,
  halfLifeDays: 30,
  activityBonusMultiplier: 1.5,
};

// ============================================================================
// ACTIVITY WEIGHTS
// ============================================================================

const ACTIVITY_WEIGHTS: Record<ActivityEvent['type'], number> = {
  vote: 1.0,
  share: 2.0,
  validate: 1.5,
  participate: 0.5,
};

// ============================================================================
// REPUTATION DECAY SERVICE
// ============================================================================

export class ReputationDecayService {
  private config: ReputationConfig;
  private states = new Map<string, ReputationState>();
  private activityHistory = new Map<string, ActivityEvent[]>();

  constructor(config: Partial<ReputationConfig> = {}) {
    this.config = { ...DEFAULT_REPUTATION_CONFIG, ...config };
  }

  /**
   * Initialize reputation state for an agent
   */
  initializeAgent(agentId: string, initialReputation: number): ReputationState {
    const now = Date.now();
    const state: ReputationState = {
      currentReputation: this.clampReputation(initialReputation),
      baseReputation: initialReputation,
      lastActivityAt: now,
      lastDecayAt: now,
      activityStreak: 0,
      totalDecay: 0,
      totalBonuses: 0,
    };

    this.states.set(agentId, state);
    this.activityHistory.set(agentId, []);
    return state;
  }

  /**
   * Get current state for an agent (with decay applied)
   */
  getState(agentId: string): ReputationState | null {
    const state = this.states.get(agentId);
    if (!state) return null;

    // Apply any pending decay
    this.applyDecay(agentId);
    return this.states.get(agentId) || null;
  }

  /**
   * Apply time-based decay to an agent's reputation
   */
  applyDecay(agentId: string): DecayResult | null {
    const state = this.states.get(agentId);
    if (!state) return null;

    const now = Date.now();
    const msSinceLastDecay = now - state.lastDecayAt;
    const daysSinceLastDecay = msSinceLastDecay / (1000 * 60 * 60 * 24);

    if (daysSinceLastDecay < 0.1) {
      // Don't apply decay more than ~10 times per day
      return {
        previousReputation: state.currentReputation,
        newReputation: state.currentReputation,
        decayAmount: 0,
        daysSinceLastDecay,
      };
    }

    const previousReputation = state.currentReputation;

    // Calculate decay using exponential decay formula
    // R(t) = R(0) * e^(-kt) where k = ln(2) / halfLife
    const decayConstant = Math.log(2) / this.config.halfLifeDays;
    const decayMultiplier = Math.exp(-decayConstant * daysSinceLastDecay);

    // Apply activity streak modifier (reduce decay if active)
    const streakModifier = Math.max(0.2, 1 - state.activityStreak * 0.05);
    const effectiveDecayMultiplier = 1 - (1 - decayMultiplier) * streakModifier;

    let newReputation = previousReputation * effectiveDecayMultiplier;
    newReputation = this.clampReputation(newReputation);

    const decayAmount = previousReputation - newReputation;

    // Update state
    state.currentReputation = newReputation;
    state.lastDecayAt = now;
    state.totalDecay += decayAmount;

    // Decay activity streak
    if (daysSinceLastDecay > 1) {
      state.activityStreak = Math.max(0, state.activityStreak - Math.floor(daysSinceLastDecay));
    }

    this.states.set(agentId, state);

    return {
      previousReputation,
      newReputation,
      decayAmount,
      daysSinceLastDecay,
    };
  }

  /**
   * Record an activity and apply bonus
   */
  recordActivity(agentId: string, type: ActivityEvent['type']): ActivityBonusResult | null {
    let state = this.states.get(agentId);
    if (!state) {
      // Initialize with base reputation
      state = this.initializeAgent(agentId, 50);
    }

    const now = Date.now();
    const weight = ACTIVITY_WEIGHTS[type];

    // Record activity
    const activity: ActivityEvent = { type, timestamp: now, weight };
    const history = this.activityHistory.get(agentId) || [];
    history.push(activity);

    // Keep only last 100 activities
    if (history.length > 100) {
      history.shift();
    }
    this.activityHistory.set(agentId, history);

    // Apply decay first
    this.applyDecay(agentId);
    state = this.states.get(agentId)!;

    const previousReputation = state.currentReputation;

    // Calculate base bonus
    const baseBonus = weight * this.config.activityBonusMultiplier;

    // Calculate streak bonus
    const lastActivityDays = (now - state.lastActivityAt) / (1000 * 60 * 60 * 24);
    if (lastActivityDays < 1) {
      state.activityStreak = Math.min(30, state.activityStreak + 1);
    } else if (lastActivityDays < 2) {
      // Keep streak
    } else {
      // Reset streak
      state.activityStreak = 1;
    }

    const streakBonus = Math.min(state.activityStreak * 0.1, 1.0) * baseBonus;
    const totalBonus = baseBonus + streakBonus;

    // Apply bonus with diminishing returns at high reputation
    const diminishingFactor = Math.max(
      0.1,
      1 - state.currentReputation / this.config.maximumReputation
    );
    const effectiveBonus = totalBonus * diminishingFactor;

    let newReputation = state.currentReputation + effectiveBonus;
    newReputation = this.clampReputation(newReputation);

    const bonusAmount = newReputation - previousReputation;

    // Update state
    state.currentReputation = newReputation;
    state.lastActivityAt = now;
    state.totalBonuses += bonusAmount;

    this.states.set(agentId, state);

    return {
      previousReputation,
      newReputation,
      bonusAmount,
      streakBonus,
    };
  }

  /**
   * Get activity history for an agent
   */
  getActivityHistory(agentId: string, limit: number = 50): ActivityEvent[] {
    const history = this.activityHistory.get(agentId) || [];
    return history.slice(-limit);
  }

  /**
   * Calculate projected reputation after N days
   */
  projectReputation(agentId: string, daysAhead: number): number {
    const state = this.states.get(agentId);
    if (!state) return this.config.minimumReputation;

    const decayConstant = Math.log(2) / this.config.halfLifeDays;
    const decayMultiplier = Math.exp(-decayConstant * daysAhead);

    return this.clampReputation(state.currentReputation * decayMultiplier);
  }

  /**
   * Get reputation rank among all agents
   */
  getReputationRank(agentId: string): { rank: number; total: number; percentile: number } {
    const state = this.states.get(agentId);
    if (!state) return { rank: 0, total: 0, percentile: 0 };

    const allReputations = Array.from(this.states.values())
      .map((s) => s.currentReputation)
      .sort((a, b) => b - a);

    const rank = allReputations.findIndex((r) => r <= state.currentReputation) + 1;
    const total = allReputations.length;
    const percentile = total > 0 ? ((total - rank + 1) / total) * 100 : 0;

    return { rank, total, percentile };
  }

  /**
   * Get agents at risk of falling below threshold
   */
  getAgentsAtRisk(threshold: number, daysAhead: number = 7): string[] {
    const atRisk: string[] = [];

    for (const [agentId, state] of this.states.entries()) {
      if (state.currentReputation >= threshold) {
        const projected = this.projectReputation(agentId, daysAhead);
        if (projected < threshold) {
          atRisk.push(agentId);
        }
      }
    }

    return atRisk;
  }

  /**
   * Clamp reputation to configured bounds
   */
  private clampReputation(reputation: number): number {
    return Math.max(
      this.config.minimumReputation,
      Math.min(this.config.maximumReputation, reputation)
    );
  }

  /**
   * Reset all states (for testing)
   */
  reset(): void {
    this.states.clear();
    this.activityHistory.clear();
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const reputationDecay = new ReputationDecayService();

// Convenience functions
export const initializeReputation = reputationDecay.initializeAgent.bind(reputationDecay);
export const getReputation = reputationDecay.getState.bind(reputationDecay);
export const recordActivity = reputationDecay.recordActivity.bind(reputationDecay);
export const projectReputation = reputationDecay.projectReputation.bind(reputationDecay);
export const getReputationRank = reputationDecay.getReputationRank.bind(reputationDecay);
export const getAgentsAtRisk = reputationDecay.getAgentsAtRisk.bind(reputationDecay);

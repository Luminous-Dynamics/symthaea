/**
 * @mycelix/sdk End-to-End Integration Flows
 *
 * Integrates all agentic modules into cohesive workflows:
 * - Trust Pipeline: K-Vector -> Attestation -> Consensus -> Update
 * - Attack Response: Detection -> Thresholds -> Quarantine -> Slash
 * - Privacy Analytics: DP Aggregation -> Dashboard -> Alerts
 * - Agent Lifecycle: Create -> Output -> Classify -> Update Trust
 *
 * @packageDocumentation
 * @module agentic/integration
 */

import {
  type AdaptiveConfig,
  createDefaultAdaptiveConfig,
  AdaptiveThresholdEngine,
  ThresholdType,
  FeedbackOutcome,
  type ThresholdFeedback,
  createDefaultFeedbackContext,
} from './adaptive-thresholds.js';
import {
  type DashboardConfig,
  createDefaultDashboardConfig,
  Dashboard,
  type LiveMetrics,
  AlertSeverity,
} from './dashboard.js';
import {
  type DPConfig,
  createDefaultDPConfig,
  PrivateTrustAnalytics,
  type TrustDistribution,
} from './differential-privacy.js';
import { AgentClass, AgentStatus } from './types.js';
import {
  VerificationEngine,
  type SystemState,
  type InvariantCheckResult,
} from './verification.js';

import { computeTrustScore, calculateKreditFromTrust } from './index.js';

import type { KVectorValues } from './types.js';

// =============================================================================
// Agent Types (Simplified for Integration)
// =============================================================================

/**
 * Simplified agent for integration demos
 */
export interface IntegrationAgent {
  agentId: string;
  sponsorDid: string;
  agentClass: AgentClass;
  kreditBalance: number;
  kreditCap: number;
  status: AgentStatus;
  kVector: KVectorValues;
  totalOutputs: number;
  averageWeight: number;
  actionsThisHour: number;
}

/**
 * Create a new integration agent
 */
export function createIntegrationAgent(
  agentId: string,
  sponsorDid: string,
  agentClass: AgentClass
): IntegrationAgent {
  const kVector: KVectorValues = {
    k_r: 0.5,
    k_a: 0.0,
    k_i: 1.0,
    k_p: 0.5,
    k_m: 0.0,
    k_s: 0.0,
    k_h: 0.5,
    k_topo: 0.0,
    k_v: 0.0,
    k_coherence: 0.5,
  };

  const trustScore = computeTrustScore(kVector);
  const kreditCap = calculateKreditFromTrust(trustScore);

  return {
    agentId,
    sponsorDid,
    agentClass,
    kreditBalance: Math.floor(kreditCap / 2),
    kreditCap,
    status: AgentStatus.Active,
    kVector,
    totalOutputs: 0,
    averageWeight: 0,
    actionsThisHour: 0,
  };
}

// =============================================================================
// Trust Pipeline Integration
// =============================================================================

/**
 * Trust pipeline configuration
 */
export interface TrustPipelineConfig {
  /** Minimum trust to participate */
  minTrustThreshold: number;
  /** Consensus approval threshold */
  approvalThreshold: number;
  /** Enable quadratic voting */
  quadraticVoting: boolean;
  /** KREDIT base multiplier */
  kreditBase: number;
}

/**
 * Create default trust pipeline config
 */
export function createDefaultTrustPipelineConfig(): TrustPipelineConfig {
  return {
    minTrustThreshold: 0.3,
    approvalThreshold: 0.5,
    quadraticVoting: true,
    kreditBase: 10000,
  };
}

/**
 * Attestation result
 */
export interface AttestationResult {
  fromAgent: string;
  toAgent: string;
  weight: number;
  oldTrust: number;
  newTrust: number;
  newKreditCap: number;
}

/**
 * Cascade result
 */
export interface CascadeResult {
  initialAgent: string;
  shockMagnitude: number;
  agentsAffected: number;
  agentsFailed: number;
  totalTrustLoss: number;
  maxDepthReached: number;
}

/**
 * Integrated trust pipeline
 */
export class IntegratedTrustPipeline {
  private config: TrustPipelineConfig;
  private agents: Map<string, IntegrationAgent> = new Map();
  private edges: Array<{ from: string; to: string; weight: number }> = [];
  private verification: VerificationEngine;

  constructor(config: TrustPipelineConfig = createDefaultTrustPipelineConfig()) {
    this.config = config;
    this.verification = VerificationEngine.withDefaults();
  }

  /**
   * Register an agent
   */
  registerAgent(agent: IntegrationAgent): void {
    this.agents.set(agent.agentId, agent);
  }

  /**
   * Process attestation between agents
   */
  processAttestation(
    fromAgentId: string,
    toAgentId: string,
    attestationWeight: number
  ): AttestationResult | null {
    const fromAgent = this.agents.get(fromAgentId);
    const toAgent = this.agents.get(toAgentId);

    if (!fromAgent || !toAgent) return null;

    const fromTrust = computeTrustScore(fromAgent.kVector);
    if (fromTrust < this.config.minTrustThreshold) return null;

    // Add edge
    this.edges.push({ from: fromAgentId, to: toAgentId, weight: attestationWeight });

    // Calculate weight (quadratic if enabled)
    const weight = this.config.quadraticVoting
      ? Math.sqrt(fromTrust) * attestationWeight
      : fromTrust * attestationWeight;

    // Update toAgent trust
    const oldTrust = computeTrustScore(toAgent.kVector);
    toAgent.kVector.k_r = Math.max(0, Math.min(1, toAgent.kVector.k_r + weight * 0.1));
    const newTrust = computeTrustScore(toAgent.kVector);

    // Recalculate KREDIT
    const multiplier = calculateKreditFromTrust(newTrust);
    toAgent.kreditCap = multiplier * this.config.kreditBase;

    return {
      fromAgent: fromAgentId,
      toAgent: toAgentId,
      weight,
      oldTrust,
      newTrust,
      newKreditCap: toAgent.kreditCap,
    };
  }

  /**
   * Simulate trust cascade
   */
  simulateCascade(agentId: string, shockMagnitude: number): CascadeResult {
    const affected = new Set<string>();
    const queue = [{ id: agentId, magnitude: shockMagnitude, depth: 0 }];
    let maxDepth = 0;
    let totalTrustLoss = 0;
    let failed = 0;

    while (queue.length > 0) {
      const { id, magnitude, depth } = queue.shift()!;

      if (affected.has(id) || magnitude < 0.01) continue;
      affected.add(id);
      maxDepth = Math.max(maxDepth, depth);

      const agent = this.agents.get(id);
      if (!agent) continue;

      // Apply shock
      const trustBefore = computeTrustScore(agent.kVector);
      agent.kVector.k_r = Math.max(0, agent.kVector.k_r - magnitude * 0.5);
      const trustAfter = computeTrustScore(agent.kVector);
      totalTrustLoss += trustBefore - trustAfter;

      if (trustAfter < 0.1) failed++;

      // Propagate to connected agents
      for (const edge of this.edges) {
        if (edge.from === id || edge.to === id) {
          const neighbor = edge.from === id ? edge.to : edge.from;
          if (!affected.has(neighbor)) {
            queue.push({
              id: neighbor,
              magnitude: magnitude * edge.weight * 0.5,
              depth: depth + 1,
            });
          }
        }
      }
    }

    return {
      initialAgent: agentId,
      shockMagnitude,
      agentsAffected: affected.size,
      agentsFailed: failed,
      totalTrustLoss,
      maxDepthReached: maxDepth,
    };
  }

  /**
   * Verify system invariants
   */
  verifyInvariants(): InvariantCheckResult[] {
    const state: SystemState = {
      index: 0,
      timestamp: Date.now(),
      trustScores: new Map(
        Array.from(this.agents.entries()).map(([id, a]) => [
          id,
          computeTrustScore(a.kVector),
        ])
      ),
      byzantineCount: 0,
      networkHealth: 1.0,
      variables: new Map(),
    };

    return this.verification.checkInvariants(state);
  }

  /**
   * Get all agents
   */
  getAgents(): Map<string, IntegrationAgent> {
    return this.agents;
  }
}

// =============================================================================
// Attack Response Integration
// =============================================================================

/**
 * Attack response configuration
 */
export interface AttackResponseConfig {
  thresholds: AdaptiveConfig;
  gamingSuspicionThreshold: number;
  quarantineDurationMs: number;
}

/**
 * Create default attack response config
 */
export function createDefaultAttackResponseConfig(): AttackResponseConfig {
  return {
    thresholds: createDefaultAdaptiveConfig(),
    gamingSuspicionThreshold: 0.5,
    quarantineDurationMs: 24 * 60 * 60 * 1000,
  };
}

/**
 * Attack response result
 */
export interface AttackResponse {
  agentId: string;
  attackType: string;
  confidence: number;
  actionsTaken: string[];
  timestamp: number;
}

/**
 * Integrated attack response system
 */
export class IntegratedAttackResponse {
  private config: AttackResponseConfig;
  private thresholds: AdaptiveThresholdEngine;
  private quarantined: Set<string> = new Set();
  private responses: AttackResponse[] = [];

  constructor(config: AttackResponseConfig = createDefaultAttackResponseConfig()) {
    this.config = config;
    this.thresholds = new AdaptiveThresholdEngine(config.thresholds);
  }

  /**
   * Process agent behavior for attacks
   */
  processBehavior(agent: IntegrationAgent): AttackResponse | null {
    const timestamp = Date.now();

    // Compute suspicion score based on activity patterns
    const suspicionScore = this.computeSuspicionScore(agent);

    if (suspicionScore > this.config.gamingSuspicionThreshold) {
      // Quarantine agent
      this.quarantined.add(agent.agentId);

      // Update thresholds with feedback
      const feedback: ThresholdFeedback = {
        thresholdType: ThresholdType.GamingDetection,
        thresholdValue: this.thresholds.getThreshold(ThresholdType.GamingDetection),
        outcome: FeedbackOutcome.TruePositive,
        context: {
          ...createDefaultFeedbackContext(),
          threatLevel: suspicionScore,
        },
        timestamp,
      };
      this.thresholds.processFeedback(feedback);

      // Create response
      const response: AttackResponse = {
        agentId: agent.agentId,
        attackType: 'Gaming',
        confidence: suspicionScore,
        actionsTaken: ['Quarantined'],
        timestamp,
      };

      this.responses.push(response);
      return response;
    }

    return null;
  }

  /**
   * Compute suspicion score for gaming detection
   */
  private computeSuspicionScore(agent: IntegrationAgent): number {
    let score = 0;

    // High activity rate
    if (agent.actionsThisHour > 100) {
      score += 0.3;
    }

    // Many outputs with low weight (spam)
    if (agent.totalOutputs > 50 && agent.averageWeight < 0.2) {
      score += 0.3;
    }

    // Autonomous agents get more scrutiny
    if (agent.agentClass === AgentClass.Autonomous) {
      score += 0.1;
    }

    return Math.min(1, score);
  }

  /**
   * Check if agent is quarantined
   */
  isQuarantined(agentId: string): boolean {
    return this.quarantined.has(agentId);
  }

  /**
   * Get current threshold
   */
  getThreshold(type: ThresholdType): number {
    return this.thresholds.getThreshold(type);
  }

  /**
   * Get response history
   */
  getResponses(): AttackResponse[] {
    return [...this.responses];
  }
}

// =============================================================================
// Privacy Analytics Integration
// =============================================================================

/**
 * Privacy analytics configuration
 */
export interface PrivacyAnalyticsConfig {
  dp: DPConfig;
  dashboard: DashboardConfig;
}

/**
 * Create default privacy analytics config
 */
export function createDefaultPrivacyAnalyticsConfig(): PrivacyAnalyticsConfig {
  return {
    dp: createDefaultDPConfig(),
    dashboard: createDefaultDashboardConfig(),
  };
}

/**
 * Privacy analytics result
 */
export interface PrivateAnalyticsResult {
  distribution: TrustDistribution;
  liveMetrics: LiveMetrics;
  remainingBudget: { epsilon: number; delta: number };
}

/**
 * Integrated privacy-preserving analytics
 */
export class IntegratedPrivacyAnalytics {
  private analytics: PrivateTrustAnalytics;
  private dashboard: Dashboard;

  constructor(config: PrivacyAnalyticsConfig = createDefaultPrivacyAnalyticsConfig()) {
    this.analytics = new PrivateTrustAnalytics(config.dp);
    this.dashboard = new Dashboard(config.dashboard);
  }

  /**
   * Analyze and display with privacy guarantees
   */
  analyzeAndDisplay(
    trustScores: number[],
    phiValues: number[],
    threatLevel: number
  ): PrivateAnalyticsResult | null {
    const timestamp = Date.now();
    const epsilonPerQuery = 0.25;

    // Compute private distribution
    const distribution = this.analytics.analyzeTrustDistribution(
      trustScores,
      epsilonPerQuery
    );
    if (!distribution) return null;

    // Update dashboard
    const liveMetrics = this.dashboard.update(
      {
        trustScores,
        transactionCount: 0,
        alerts:
          distribution.mean < 0.3
            ? [
                {
                  severity: AlertSeverity.High,
                  title: 'Low Average Trust',
                  description: `Private mean trust is ${distribution.mean.toFixed(2)}`,
                  source: 'privacy_analytics',
                },
              ]
            : [],
        phiValues,
        threats: [threatLevel],
      },
      timestamp
    );

    return {
      distribution,
      liveMetrics,
      remainingBudget: this.analytics.remainingBudget(),
    };
  }

  /**
   * Get dashboard
   */
  getDashboard(): Dashboard {
    return this.dashboard;
  }

  /**
   * Reset epoch
   */
  resetEpoch(): void {
    this.analytics.resetBudget();
  }
}

// =============================================================================
// Epistemic Lifecycle Integration
// =============================================================================

/**
 * Epistemic lifecycle configuration
 */
export interface EpistemicLifecycleConfig {
  minEpistemicWeight: number;
  kreditBase: number;
}

/**
 * Create default epistemic lifecycle config
 */
export function createDefaultEpistemicLifecycleConfig(): EpistemicLifecycleConfig {
  return {
    minEpistemicWeight: 0.1,
    kreditBase: 10000,
  };
}

/**
 * Output processing result
 */
export interface OutputProcessingResult {
  outputId: string;
  epistemicWeight: number;
  trustDelta: number;
  newTrust: number;
  newKreditCap: number;
  reward: number;
}

/**
 * Integrated epistemic agent lifecycle
 */
export class IntegratedEpistemicLifecycle {
  private config: EpistemicLifecycleConfig;

  constructor(config: EpistemicLifecycleConfig = createDefaultEpistemicLifecycleConfig()) {
    this.config = config;
  }

  /**
   * Create a new agent
   */
  createAgent(sponsorDid: string, agentClass: AgentClass): IntegrationAgent {
    const agentId = `agent-${Date.now().toString(16)}${Math.random().toString(16).slice(2, 10)}`;
    return createIntegrationAgent(agentId, sponsorDid, agentClass);
  }

  /**
   * Process an agent output
   */
  processOutput(agent: IntegrationAgent, _content: string): OutputProcessingResult {
    const outputId = `out-${agent.agentId}-${Date.now().toString(16)}`;

    // Compute epistemic weight (simplified)
    const epistemicWeight = 0.084; // Default weight for basic output

    // Update stats
    agent.totalOutputs++;
    agent.averageWeight =
      (agent.averageWeight * (agent.totalOutputs - 1) + epistemicWeight) /
      agent.totalOutputs;

    // Update K-Vector if weight is sufficient
    let trustDelta = 0;
    if (epistemicWeight >= this.config.minEpistemicWeight) {
      trustDelta = epistemicWeight * 0.01;
      agent.kVector.k_r = Math.min(1, agent.kVector.k_r + trustDelta);
    }

    const newTrust = computeTrustScore(agent.kVector);
    const multiplier = calculateKreditFromTrust(newTrust);
    agent.kreditCap = multiplier * this.config.kreditBase;

    // Calculate reward
    const reward = Math.floor(newTrust * 30);

    return {
      outputId,
      epistemicWeight,
      trustDelta,
      newTrust,
      newKreditCap: agent.kreditCap,
      reward,
    };
  }

  /**
   * Verify an output
   */
  verifyOutput(agent: IntegrationAgent, _outputId: string, correct: boolean): void {
    if (correct) {
      agent.kVector.k_r = Math.min(1, agent.kVector.k_r + 0.02);
    } else {
      agent.kVector.k_r = Math.max(0, agent.kVector.k_r - 0.05);
    }

    const newTrust = computeTrustScore(agent.kVector);
    const multiplier = calculateKreditFromTrust(newTrust);
    agent.kreditCap = multiplier * this.config.kreditBase;
  }
}

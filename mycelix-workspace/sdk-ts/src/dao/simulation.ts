/**
 * Governance Simulation Sandbox
 *
 * Test governance scenarios without on-chain execution:
 * - Voting outcome predictions
 * - Delegation chain analysis
 * - Attack vector simulation
 * - Parameter sensitivity analysis
 * - Monte Carlo governance modeling
 */

import type {
  ProposalType,
  VotingMechanism,
  VoteChoice,
  EpistemicClassification,
  DelegationDomain,
} from './types';

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Configuration for simulation runs
 */
export interface SimulationConfig {
  /** Number of simulated agents */
  agentCount: number;
  /** Number of Monte Carlo iterations */
  iterations: number;
  /** Random seed for reproducibility */
  seed?: number;
  /** Agent behavior distribution */
  agentDistribution: AgentDistribution;
  /** Enable attack simulations */
  simulateAttacks: boolean;
  /** Time horizon in hours */
  timeHorizonHours: number;
}

/**
 * Distribution of agent behaviors
 */
export interface AgentDistribution {
  /** Percentage of honest random voters */
  randomVoters: number;
  /** Percentage of domain experts (vote based on proposal type) */
  experts: number;
  /** Percentage of delegators */
  delegators: number;
  /** Percentage of coordinated attackers (if simulateAttacks) */
  attackers: number;
  /** Percentage of passive observers (rarely vote) */
  passive: number;
}

const DEFAULT_CONFIG: SimulationConfig = {
  agentCount: 100,
  iterations: 1000,
  agentDistribution: {
    randomVoters: 0.3,
    experts: 0.2,
    delegators: 0.35,
    attackers: 0.05,
    passive: 0.1,
  },
  simulateAttacks: true,
  timeHorizonHours: 168, // 1 week
};

// ============================================================================
// TYPES
// ============================================================================

/**
 * A simulated agent
 */
export interface SimulatedAgent {
  id: string;
  type: 'random' | 'expert' | 'delegator' | 'attacker' | 'passive';
  reputation: number;
  sybilScore: number;
  expertDomains: DelegationDomain[];
  delegatesTo?: string;
  votingBias: number; // -1 to 1 (no bias, yes bias)
  activityLevel: number; // 0 to 1
}

/**
 * A simulated vote
 */
export interface SimulatedVote {
  agentId: string;
  choice: VoteChoice;
  weight: number;
  timestamp: number;
  isDelegated: boolean;
}

/**
 * Result of a single simulation run
 */
export interface SimulationRunResult {
  passed: boolean;
  yesVotes: number;
  noVotes: number;
  abstainVotes: number;
  totalWeight: number;
  quorumReached: boolean;
  participationRate: number;
  attackSucceeded?: boolean;
  finalConviction?: number;
}

/**
 * Aggregated simulation results
 */
export interface SimulationResults {
  /** Probability of proposal passing (0-1) */
  passRate: number;
  /** 95% confidence interval for pass rate */
  passRateCI: [number, number];
  /** Average participation rate */
  avgParticipation: number;
  /** Attack success rate (if attacks simulated) */
  attackSuccessRate?: number;
  /** Distribution of outcomes */
  outcomeDistribution: {
    passed: number;
    failed: number;
    noQuorum: number;
  };
  /** Time to reach quorum (avg hours) */
  avgTimeToQuorum?: number;
  /** Sensitivity analysis results */
  sensitivity?: SensitivityAnalysis;
  /** All individual runs */
  runs: SimulationRunResult[];
}

/**
 * Parameter sensitivity analysis
 */
export interface SensitivityAnalysis {
  /** How pass rate changes with quorum */
  quorumSensitivity: Array<{ quorum: number; passRate: number }>;
  /** How pass rate changes with approval threshold */
  thresholdSensitivity: Array<{ threshold: number; passRate: number }>;
  /** How pass rate changes with agent count */
  scaleSensitivity: Array<{ agents: number; passRate: number }>;
}

// ============================================================================
// SIMULATION ENGINE
// ============================================================================

/**
 * Governance Simulation Engine
 *
 * @example
 * ```typescript
 * const sim = new GovernanceSimulator();
 *
 * // Simulate a technical proposal
 * const result = await sim.simulateProposal({
 *   proposalType: 'TechnicalMIP',
 *   votingMechanism: {
 *     ReputationWeighted: { quorum: 0.3, approval_threshold: 0.6 }
 *   },
 *   classification: { e: 'E3', n: 'N1', m: 'M2', override_from_default: false },
 * });
 *
 * console.log(`Pass probability: ${(result.passRate * 100).toFixed(1)}%`);
 * console.log(`95% CI: [${result.passRateCI[0]}, ${result.passRateCI[1]}]`);
 * ```
 */
export class GovernanceSimulator {
  private config: SimulationConfig;
  private agents: SimulatedAgent[] = [];
  private rng: SeededRandom;

  constructor(config?: Partial<SimulationConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = new SeededRandom(this.config.seed || Date.now());
    this.initializeAgents();
  }

  /**
   * Initialize simulated agent population
   */
  private initializeAgents(): void {
    this.agents = [];
    const dist = this.config.agentDistribution;

    for (let i = 0; i < this.config.agentCount; i++) {
      const roll = this.rng.random();
      let type: SimulatedAgent['type'];
      let cumulative = 0;

      if (roll < (cumulative += dist.randomVoters)) {
        type = 'random';
      } else if (roll < (cumulative += dist.experts)) {
        type = 'expert';
      } else if (roll < (cumulative += dist.delegators)) {
        type = 'delegator';
      } else if (roll < (cumulative += dist.attackers) && this.config.simulateAttacks) {
        type = 'attacker';
      } else {
        type = 'passive';
      }

      const agent: SimulatedAgent = {
        id: `agent-${i}`,
        type,
        reputation: this.rng.random() * 100,
        sybilScore: this.generateSybilScore(type),
        expertDomains: this.generateExpertise(),
        votingBias: type === 'attacker' ? -1 : (this.rng.random() - 0.5) * 0.4,
        activityLevel: type === 'passive' ? 0.1 : this.rng.random() * 0.5 + 0.5,
      };

      // Assign delegation for delegators
      if (type === 'delegator' && this.agents.length > 0) {
        const eligibleDelegates = this.agents.filter(
          a => a.type === 'expert' || a.type === 'random'
        );
        if (eligibleDelegates.length > 0) {
          agent.delegatesTo = eligibleDelegates[
            Math.floor(this.rng.random() * eligibleDelegates.length)
          ].id;
        }
      }

      this.agents.push(agent);
    }
  }

  private generateSybilScore(type: SimulatedAgent['type']): number {
    // Attackers have lower sybil scores (less verified)
    if (type === 'attacker') {
      return 0.2 + this.rng.random() * 0.3;
    }
    // Experts and delegators tend to be well-verified
    if (type === 'expert' || type === 'delegator') {
      return 0.6 + this.rng.random() * 0.4;
    }
    return 0.4 + this.rng.random() * 0.5;
  }

  private generateExpertise(): DelegationDomain[] {
    const domains: DelegationDomain[] = [
      'Technical', 'Economic', 'Governance', 'Social', 'Cultural'
    ];
    const count = Math.floor(this.rng.random() * 3);
    const shuffled = domains.sort(() => this.rng.random() - 0.5);
    return shuffled.slice(0, count);
  }

  // =========================================================================
  // SIMULATION METHODS
  // =========================================================================

  /**
   * Simulate a proposal with given parameters
   */
  async simulateProposal(params: {
    proposalType: ProposalType;
    votingMechanism: VotingMechanism;
    classification: EpistemicClassification;
    fundingRequest?: number;
    customAgentCount?: number;
  }): Promise<SimulationResults> {
    const runs: SimulationRunResult[] = [];

    // Reset RNG for reproducibility
    this.rng = new SeededRandom(this.config.seed || Date.now());

    if (params.customAgentCount && params.customAgentCount !== this.config.agentCount) {
      this.config.agentCount = params.customAgentCount;
      this.initializeAgents();
    }

    // Run Monte Carlo simulations
    for (let i = 0; i < this.config.iterations; i++) {
      const result = this.runSingleSimulation(params);
      runs.push(result);
    }

    // Aggregate results
    return this.aggregateResults(runs, params.votingMechanism);
  }

  /**
   * Run a single simulation iteration
   */
  private runSingleSimulation(params: {
    proposalType: ProposalType;
    votingMechanism: VotingMechanism;
    classification: EpistemicClassification;
    fundingRequest?: number;
  }): SimulationRunResult {
    const votes: SimulatedVote[] = [];
    const votedAgents = new Set<string>();

    // Determine which agents vote
    for (const agent of this.agents) {
      // Skip if delegating
      if (agent.delegatesTo && !votedAgents.has(agent.delegatesTo)) {
        continue;
      }

      // Check activity level
      if (this.rng.random() > agent.activityLevel) {
        continue;
      }

      const vote = this.generateVote(agent, params);
      if (vote) {
        votes.push(vote);
        votedAgents.add(agent.id);
      }
    }

    // Process delegated votes
    for (const agent of this.agents) {
      if (agent.delegatesTo && votedAgents.has(agent.delegatesTo)) {
        // Find delegate's vote and mirror it
        const delegateVote = votes.find(v => v.agentId === agent.delegatesTo);
        if (delegateVote) {
          votes.push({
            agentId: agent.id,
            choice: delegateVote.choice,
            weight: this.calculateWeight(agent, params.votingMechanism),
            timestamp: delegateVote.timestamp + this.rng.random() * 1000,
            isDelegated: true,
          });
          votedAgents.add(agent.id);
        }
      }
    }

    // Tally results
    return this.tallyVotes(votes, params.votingMechanism);
  }

  /**
   * Generate a vote for an agent
   */
  private generateVote(
    agent: SimulatedAgent,
    params: {
      proposalType: ProposalType;
      classification: EpistemicClassification;
    }
  ): SimulatedVote | null {
    let choice: VoteChoice;

    switch (agent.type) {
      case 'random':
        // Random vote with slight bias
        const roll = this.rng.random() + agent.votingBias * 0.3;
        choice = roll > 0.6 ? 'Yes' : roll < 0.3 ? 'No' : 'Abstain';
        break;

      case 'expert':
        // Experts vote based on domain match and classification
        const hasExpertise = this.checkDomainMatch(agent, params.proposalType);
        if (hasExpertise) {
          // More likely to vote yes on well-classified proposals
          const classificationQuality = this.evaluateClassification(params.classification);
          choice = classificationQuality > 0.5 && this.rng.random() > 0.3 ? 'Yes' : 'No';
        } else {
          choice = 'Abstain';
        }
        break;

      case 'attacker':
        // Attackers vote no (trying to block)
        choice = 'No';
        break;

      case 'passive':
        // Usually abstains
        choice = this.rng.random() > 0.8 ? (this.rng.random() > 0.5 ? 'Yes' : 'No') : 'Abstain';
        break;

      default:
        choice = this.rng.random() > 0.5 ? 'Yes' : 'No';
    }

    return {
      agentId: agent.id,
      choice,
      weight: this.calculateWeight(agent, null),
      timestamp: this.rng.random() * this.config.timeHorizonHours * 3600,
      isDelegated: false,
    };
  }

  /**
   * Calculate vote weight based on mechanism
   */
  private calculateWeight(
    agent: SimulatedAgent,
    mechanism: VotingMechanism | null
  ): number {
    // Base weight from reputation and sybil score
    const base = 1 + (agent.reputation / 100) * 0.5;
    const sybilFactor = 0.5 + agent.sybilScore * 0.5;

    // Mechanism-specific adjustments
    if (mechanism && 'EqualWeight' in mechanism) {
      return 1.0 * sybilFactor;
    }

    if (mechanism && 'Quadratic' in mechanism) {
      // Simulate credit spending
      const credits = Math.floor(this.rng.random() * 25);
      return Math.sqrt(credits) * sybilFactor;
    }

    // Default: reputation-weighted
    return base * sybilFactor;
  }

  /**
   * Check if agent has expertise in proposal domain
   */
  private checkDomainMatch(agent: SimulatedAgent, proposalType: ProposalType): boolean {
    const domainMap: Record<string, DelegationDomain> = {
      TechnicalMIP: 'Technical',
      EconomicMIP: 'Economic',
      GovernanceMIP: 'Governance',
      SocialMIP: 'Social',
      CulturalMIP: 'Cultural',
      TreasuryAllocation: 'Economic',
      StreamingGrant: 'Economic',
    };

    const domain = domainMap[proposalType];
    return domain ? agent.expertDomains.includes(domain) : false;
  }

  /**
   * Evaluate classification quality
   */
  private evaluateClassification(classification: EpistemicClassification): number {
    const e = parseInt(classification.e[1]);
    const n = parseInt(classification.n[1]);
    const m = parseInt(classification.m[1]);

    // Higher E and appropriate N/M is better
    return (e * 0.5 + n * 0.25 + m * 0.25) / 4;
  }

  /**
   * Tally votes and determine outcome
   */
  private tallyVotes(
    votes: SimulatedVote[],
    mechanism: VotingMechanism
  ): SimulationRunResult {
    let yes = 0, no = 0, abstain = 0;
    let totalWeight = 0;

    for (const vote of votes) {
      totalWeight += vote.weight;
      switch (vote.choice) {
        case 'Yes': yes += vote.weight; break;
        case 'No': no += vote.weight; break;
        case 'Abstain': abstain += vote.weight; break;
      }
    }

    // Determine quorum and approval
    let quorumThreshold = 0.3;
    let approvalThreshold = 0.5;

    if ('ReputationWeighted' in mechanism) {
      quorumThreshold = mechanism.ReputationWeighted.quorum;
      approvalThreshold = mechanism.ReputationWeighted.approval_threshold;
    } else if ('EqualWeight' in mechanism) {
      quorumThreshold = mechanism.EqualWeight.quorum;
      approvalThreshold = mechanism.EqualWeight.approval_threshold;
    } else if ('Quadratic' in mechanism) {
      approvalThreshold = mechanism.Quadratic.approval_threshold;
    }

    const totalPossibleWeight = this.agents.reduce((sum, a) => sum + this.calculateWeight(a, mechanism), 0);
    const participationRate = totalWeight / totalPossibleWeight;
    const quorumReached = participationRate >= quorumThreshold;

    const votingWeight = yes + no; // Exclude abstains from approval calculation
    const approvalRate = votingWeight > 0 ? yes / votingWeight : 0;
    const passed = quorumReached && approvalRate >= approvalThreshold;

    // Check if attackers succeeded
    const attackerVotes = votes.filter(v => {
      const agent = this.agents.find(a => a.id === v.agentId);
      return agent?.type === 'attacker';
    });
    const attackSucceeded = !passed && attackerVotes.length > 0 &&
      attackerVotes.reduce((sum, v) => sum + v.weight, 0) > yes * 0.5;

    return {
      passed,
      yesVotes: yes,
      noVotes: no,
      abstainVotes: abstain,
      totalWeight,
      quorumReached,
      participationRate,
      attackSucceeded,
    };
  }

  /**
   * Aggregate results from all simulation runs
   */
  private aggregateResults(
    runs: SimulationRunResult[],
    _mechanism: VotingMechanism
  ): SimulationResults {
    const n = runs.length;
    const passed = runs.filter(r => r.passed).length;
    const noQuorum = runs.filter(r => !r.quorumReached).length;
    const failed = n - passed - noQuorum;

    const passRate = passed / n;

    // Calculate 95% confidence interval using Wilson score
    const z = 1.96;
    const phat = passRate;
    const denominator = 1 + z * z / n;
    const center = phat + z * z / (2 * n);
    const spread = z * Math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n);
    const ci: [number, number] = [
      Math.max(0, (center - spread) / denominator),
      Math.min(1, (center + spread) / denominator),
    ];

    const avgParticipation = runs.reduce((sum, r) => sum + r.participationRate, 0) / n;

    const attackRuns = runs.filter(r => r.attackSucceeded !== undefined);
    const attackSuccessRate = attackRuns.length > 0
      ? attackRuns.filter(r => r.attackSucceeded).length / attackRuns.length
      : undefined;

    return {
      passRate,
      passRateCI: ci,
      avgParticipation,
      attackSuccessRate,
      outcomeDistribution: {
        passed,
        failed,
        noQuorum,
      },
      runs,
    };
  }

  // =========================================================================
  // ANALYSIS METHODS
  // =========================================================================

  /**
   * Run sensitivity analysis on parameters
   */
  async runSensitivityAnalysis(params: {
    proposalType: ProposalType;
    classification: EpistemicClassification;
    baseQuorum: number;
    baseThreshold: number;
  }): Promise<SensitivityAnalysis> {
    const quorumSensitivity: Array<{ quorum: number; passRate: number }> = [];
    const thresholdSensitivity: Array<{ threshold: number; passRate: number }> = [];
    const scaleSensitivity: Array<{ agents: number; passRate: number }> = [];

    // Quorum sensitivity
    for (const quorum of [0.1, 0.2, 0.3, 0.4, 0.5]) {
      const result = await this.simulateProposal({
        ...params,
        votingMechanism: {
          ReputationWeighted: { quorum, approval_threshold: params.baseThreshold },
        },
      });
      quorumSensitivity.push({ quorum, passRate: result.passRate });
    }

    // Threshold sensitivity
    for (const threshold of [0.4, 0.5, 0.6, 0.67, 0.75]) {
      const result = await this.simulateProposal({
        ...params,
        votingMechanism: {
          ReputationWeighted: { quorum: params.baseQuorum, approval_threshold: threshold },
        },
      });
      thresholdSensitivity.push({ threshold, passRate: result.passRate });
    }

    // Scale sensitivity
    for (const agents of [50, 100, 200, 500, 1000]) {
      const result = await this.simulateProposal({
        ...params,
        votingMechanism: {
          ReputationWeighted: { quorum: params.baseQuorum, approval_threshold: params.baseThreshold },
        },
        customAgentCount: agents,
      });
      scaleSensitivity.push({ agents, passRate: result.passRate });
    }

    return {
      quorumSensitivity,
      thresholdSensitivity,
      scaleSensitivity,
    };
  }

  /**
   * Simulate a Sybil attack
   */
  async simulateSybilAttack(params: {
    proposalType: ProposalType;
    votingMechanism: VotingMechanism;
    classification: EpistemicClassification;
    attackerCount: number;
    attackerSybilScore: number;
  }): Promise<{
    basePassRate: number;
    attackPassRate: number;
    attackEffectiveness: number;
  }> {
    // Run base simulation
    const baseResult = await this.simulateProposal(params);

    // Add attackers and re-run
    const attackConfig: SimulationConfig = {
      ...this.config,
      agentDistribution: {
        ...this.config.agentDistribution,
        attackers: params.attackerCount / this.config.agentCount,
      },
    };

    const attackSim = new GovernanceSimulator(attackConfig);

    // Override attacker sybil scores
    for (const agent of attackSim.agents) {
      if (agent.type === 'attacker') {
        agent.sybilScore = params.attackerSybilScore;
      }
    }

    const attackResult = await attackSim.simulateProposal(params);

    return {
      basePassRate: baseResult.passRate,
      attackPassRate: attackResult.passRate,
      attackEffectiveness: baseResult.passRate - attackResult.passRate,
    };
  }

  /**
   * Simulate conviction voting over time
   */
  async simulateConvictionVoting(params: {
    fundingRequest: number;
    availableTreasury: number;
    halfLifeHours: number;
    supporters: number;
    avgWeight: number;
  }): Promise<{
    timeToPass: number;
    passes: boolean;
    convictionCurve: Array<{ hour: number; conviction: number; threshold: number }>;
  }> {
    const { fundingRequest, availableTreasury, halfLifeHours, supporters, avgWeight } = params;

    // Threshold formula from Conviction Voting paper
    const alpha = fundingRequest / availableTreasury;
    const threshold = avgWeight * supporters * alpha / (1 - alpha);

    const curve: Array<{ hour: number; conviction: number; threshold: number }> = [];
    const maxHours = this.config.timeHorizonHours;

    let timeToPass = -1;

    for (let hour = 0; hour <= maxHours; hour += 6) {
      // Conviction accumulation: weight × (1 - e^(-t/τ))
      const conviction = avgWeight * supporters * (1 - Math.exp(-hour / halfLifeHours));
      curve.push({ hour, conviction, threshold });

      if (conviction >= threshold && timeToPass < 0) {
        timeToPass = hour;
      }
    }

    return {
      timeToPass: timeToPass > 0 ? timeToPass : Infinity,
      passes: timeToPass > 0 && timeToPass <= maxHours,
      convictionCurve: curve,
    };
  }

  /**
   * Get current agent distribution
   */
  getAgentDistribution(): Record<SimulatedAgent['type'], number> {
    const distribution: Record<string, number> = {
      random: 0,
      expert: 0,
      delegator: 0,
      attacker: 0,
      passive: 0,
    };

    for (const agent of this.agents) {
      distribution[agent.type]++;
    }

    return distribution as Record<SimulatedAgent['type'], number>;
  }

  /**
   * Get simulation configuration
   */
  getConfig(): SimulationConfig {
    return { ...this.config };
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Seeded random number generator for reproducibility
 */
class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  random(): number {
    // Mulberry32 PRNG
    let t = (this.seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

// Export helper factory function
export function createSimulator(config?: Partial<SimulationConfig>): GovernanceSimulator {
  return new GovernanceSimulator(config);
}

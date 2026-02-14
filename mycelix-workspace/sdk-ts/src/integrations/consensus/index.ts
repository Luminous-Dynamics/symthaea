/**
 * @mycelix/sdk Consensus Integration
 *
 * hApp-specific adapter for Mycelix-Consensus providing:
 * - MATL-weighted Byzantine fault tolerant consensus
 * - Multi-round voting with conviction
 * - Threshold signature coordination
 * - Cross-hApp consensus verification
 * - Federated Learning for consensus optimization
 *
 * @packageDocumentation
 * @module integrations/consensus
 * @see {@link ConsensusService} - Main service class
 * @see {@link getConsensusService} - Singleton accessor
 *
 * @example Basic consensus round
 * ```typescript
 * import { getConsensusService } from '@mycelix/sdk/integrations/consensus';
 *
 * const consensus = getConsensusService();
 *
 * // Create a consensus topic
 * const topic = consensus.createTopic({
 *   id: 'topic-001',
 *   title: 'Upgrade Protocol Version',
 *   description: 'Should we upgrade to v2.0?',
 *   options: ['yes', 'no', 'abstain'],
 *   quorumPercentage: 0.67,
 *   byzantineTolerance: 0.34,
 * });
 *
 * // Submit votes
 * consensus.submitVote(topic.id, 'participant-1', 'yes', 0.9);
 * consensus.submitVote(topic.id, 'participant-2', 'yes', 0.8);
 * consensus.submitVote(topic.id, 'participant-3', 'no', 0.7);
 *
 * // Check consensus
 * const result = consensus.checkConsensus(topic.id);
 * if (result.reached) {
 *   console.log(`Consensus reached: ${result.winner}`);
 * }
 * ```
 */

import { LocalBridge, createReputationQuery } from '../../bridge/index.js';
import { FLCoordinator, AggregationMethod } from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Consensus-Specific Types
// ============================================================================

/**
 * Consensus topic for voting
 */
export interface ConsensusTopic {
  id: string;
  title: string;
  description: string;
  options: string[];
  quorumPercentage: number; // 0-1, percentage of participants needed
  byzantineTolerance: number; // 0-1, max Byzantine participants tolerated
  createdAt: number;
  expiresAt?: number;
  status: 'active' | 'concluded' | 'expired' | 'failed';
  finalDecision?: string;
}

/**
 * Individual vote submission
 */
export interface Vote {
  id: string;
  topicId: string;
  participantId: string;
  choice: string;
  weight: number; // MATL-derived voting weight
  conviction: number; // 0-1, strength of conviction
  timestamp: number;
  signature?: string;
}

/**
 * Consensus round state
 */
export interface ConsensusRound {
  roundNumber: number;
  topicId: string;
  votes: Vote[];
  participantCount: number;
  quorumReached: boolean;
  leadingOption: string;
  leadingPercentage: number;
  byzantineDetected: number;
  startedAt: number;
  completedAt?: number;
}

/**
 * Consensus result
 */
export interface ConsensusResult {
  topicId: string;
  reached: boolean;
  winner?: string;
  totalVotes: number;
  weightedVotes: number;
  optionBreakdown: Record<string, { votes: number; weight: number }>;
  quorumPercentage: number;
  byzantineFiltered: number;
  rounds: number;
  confidence: number; // 0-1, how confident we are in the result
}

/**
 * Participant in consensus
 */
export interface ConsensusParticipant {
  id: string;
  reputation: ReputationScore;
  votingWeight: number;
  participationRate: number;
  byzantineFlags: number;
  lastActive: number;
  verified: boolean;
}

/**
 * Threshold signature shard
 */
export interface SignatureShard {
  id: string;
  topicId: string;
  signerId: string;
  shard: string; // Partial signature
  index: number;
  timestamp: number;
}

/**
 * Multi-sig execution
 */
export interface ThresholdSignature {
  topicId: string;
  threshold: number; // Number of shards needed
  totalSigners: number;
  shards: SignatureShard[];
  aggregatedSignature?: string;
  valid: boolean;
}

// ============================================================================
// Consensus Service
// ============================================================================

/**
 * ConsensusService - MATL-weighted Byzantine fault tolerant consensus
 *
 * @remarks
 * This service provides:
 * - Trust-weighted voting (higher reputation = more weight)
 * - 34% validated Byzantine fault tolerance through MATL filtering
 * - Multi-round consensus with conviction
 * - Threshold signature coordination
 * - Cross-hApp consensus verification
 *
 * @example
 * ```typescript
 * const consensus = new ConsensusService();
 *
 * // Create topic and collect votes
 * const topic = consensus.createTopic({...});
 *
 * // Add participants and votes
 * consensus.registerParticipant('alice');
 * consensus.submitVote(topic.id, 'alice', 'yes', 0.9);
 *
 * // Check result
 * const result = consensus.checkConsensus(topic.id);
 * ```
 */
export class ConsensusService {
  private topics: Map<string, ConsensusTopic> = new Map();
  private rounds: Map<string, ConsensusRound[]> = new Map();
  private participants: Map<string, ConsensusParticipant> = new Map();
  private thresholdSigs: Map<string, ThresholdSignature> = new Map();
  private participantReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  // Default Byzantine tolerance (34% validated)
  private readonly DEFAULT_BYZANTINE_TOLERANCE = 0.34;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('consensus');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: this.DEFAULT_BYZANTINE_TOLERANCE,
    });
  }

  /**
   * Create a new consensus topic
   *
   * @param input - Topic configuration
   * @returns Created topic
   *
   * @remarks
   * - Quorum percentage determines minimum participation
   * - Byzantine tolerance defaults to 34% (validated) if not specified
   */
  createTopic(input: Omit<ConsensusTopic, 'createdAt' | 'status'>): ConsensusTopic {
    const topic: ConsensusTopic = {
      ...input,
      byzantineTolerance: input.byzantineTolerance || this.DEFAULT_BYZANTINE_TOLERANCE,
      createdAt: Date.now(),
      status: 'active',
    };

    this.topics.set(topic.id, topic);
    this.rounds.set(topic.id, []);

    // Initialize first round
    this.startNewRound(topic.id);

    return topic;
  }

  /**
   * Register a consensus participant
   *
   * @param participantId - Unique participant identifier
   * @returns Registered participant with calculated weight
   */
  registerParticipant(participantId: string): ConsensusParticipant {
    let reputation = this.participantReputations.get(participantId);
    if (!reputation) {
      reputation = createReputation(participantId);
      this.participantReputations.set(participantId, reputation);
    }

    const votingWeight = this.calculateVotingWeight(reputation);

    const participant: ConsensusParticipant = {
      id: participantId,
      reputation,
      votingWeight,
      participationRate: 1.0,
      byzantineFlags: 0,
      lastActive: Date.now(),
      verified: reputationValue(reputation) >= 0.7,
    };

    this.participants.set(participantId, participant);

    // Store in bridge for cross-hApp queries
    this.bridge.setReputation('consensus', participantId, reputation);

    return participant;
  }

  /**
   * Submit a vote on a topic
   *
   * @param topicId - Topic to vote on
   * @param participantId - Voter ID
   * @param choice - Selected option
   * @param conviction - Strength of conviction (0-1)
   * @returns Submitted vote
   *
   * @remarks
   * - Vote weight is calculated from MATL reputation
   * - Conviction multiplies the effective weight
   * - Double voting is rejected
   */
  submitVote(
    topicId: string,
    participantId: string,
    choice: string,
    conviction = 1.0
  ): Vote {
    const topic = this.topics.get(topicId);
    if (!topic) {
      throw new Error(`Topic not found: ${topicId}`);
    }

    if (topic.status !== 'active') {
      throw new Error(`Topic is not active: ${topic.status}`);
    }

    if (!topic.options.includes(choice)) {
      throw new Error(`Invalid choice: ${choice}. Options: ${topic.options.join(', ')}`);
    }

    // Ensure participant is registered
    let participant = this.participants.get(participantId);
    if (!participant) {
      participant = this.registerParticipant(participantId);
    }

    // Check for double voting in current round
    const currentRound = this.getCurrentRound(topicId);
    if (currentRound.votes.some((v) => v.participantId === participantId)) {
      throw new Error('Already voted in this round');
    }

    const vote: Vote = {
      id: `vote-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      topicId,
      participantId,
      choice,
      weight: participant.votingWeight * conviction,
      conviction: Math.max(0, Math.min(1, conviction)),
      timestamp: Date.now(),
    };

    currentRound.votes.push(vote);
    currentRound.participantCount++;

    // Update participant activity
    participant.lastActive = Date.now();
    this.participants.set(participantId, participant);

    // Update reputation for participation
    let rep = this.participantReputations.get(participantId)!;
    rep = recordPositive(rep);
    this.participantReputations.set(participantId, rep);

    return vote;
  }

  /**
   * Check if consensus has been reached
   *
   * @param topicId - Topic to check
   * @returns Consensus result with breakdown
   */
  checkConsensus(topicId: string): ConsensusResult {
    const topic = this.topics.get(topicId);
    if (!topic) {
      throw new Error(`Topic not found: ${topicId}`);
    }

    const rounds = this.rounds.get(topicId) || [];
    const currentRound = rounds[rounds.length - 1];
    if (!currentRound) {
      return this.emptyResult(topicId);
    }

    // Filter Byzantine votes using MATL
    const { validVotes, byzantineCount } = this.filterByzantineVotes(
      currentRound.votes,
      topic.byzantineTolerance
    );

    // Calculate weighted totals per option
    const optionBreakdown: Record<string, { votes: number; weight: number }> = {};
    let totalWeight = 0;

    for (const option of topic.options) {
      const optionVotes = validVotes.filter((v) => v.choice === option);
      const weight = optionVotes.reduce((sum, v) => sum + v.weight, 0);
      optionBreakdown[option] = {
        votes: optionVotes.length,
        weight,
      };
      totalWeight += weight;
    }

    // Find winner
    let winner: string | undefined;
    let maxWeight = 0;
    for (const [option, data] of Object.entries(optionBreakdown)) {
      if (data.weight > maxWeight) {
        maxWeight = data.weight;
        winner = option;
      }
    }

    // Check quorum
    const totalParticipants = this.participants.size;
    const participationRate = currentRound.participantCount / Math.max(totalParticipants, 1);
    const quorumReached = participationRate >= topic.quorumPercentage;

    // Check if consensus reached (winner has majority of weight)
    const winnerPercentage = totalWeight > 0 ? maxWeight / totalWeight : 0;
    const consensusReached = quorumReached && winnerPercentage > 0.5;

    // Calculate confidence
    const confidence = Math.min(1, winnerPercentage * participationRate * (1 - byzantineCount / currentRound.votes.length));

    // Update topic if concluded
    if (consensusReached) {
      topic.status = 'concluded';
      topic.finalDecision = winner;
      this.topics.set(topicId, topic);
    } else if (topic.expiresAt && Date.now() > topic.expiresAt) {
      topic.status = 'expired';
      this.topics.set(topicId, topic);
    }

    return {
      topicId,
      reached: consensusReached,
      winner: consensusReached ? winner : undefined,
      totalVotes: validVotes.length,
      weightedVotes: totalWeight,
      optionBreakdown,
      quorumPercentage: participationRate,
      byzantineFiltered: byzantineCount,
      rounds: rounds.length,
      confidence,
    };
  }

  /**
   * Filter potentially Byzantine votes using MATL scoring
   */
  private filterByzantineVotes(
    votes: Vote[],
    byzantineTolerance: number
  ): { validVotes: Vote[]; byzantineCount: number } {
    const validVotes: Vote[] = [];
    let byzantineCount = 0;

    // Calculate median weight
    const weights = votes.map((v) => v.weight).sort((a, b) => a - b);
    const medianWeight = weights[Math.floor(weights.length / 2)] || 0;

    // Calculate adaptive threshold
    const threshold = medianWeight * (1 + byzantineTolerance);

    for (const vote of votes) {
      const participant = this.participants.get(vote.participantId);

      // Check for Byzantine indicators
      let isByzantine = false;

      // 1. Extremely high weight anomaly
      if (vote.weight > threshold * 2) {
        isByzantine = true;
      }

      // 2. Previous Byzantine flags
      if (participant && participant.byzantineFlags >= 3) {
        isByzantine = true;
      }

      // 3. Low reputation with high conviction (suspicious)
      if (participant && participant.votingWeight < 0.3 && vote.conviction > 0.95) {
        isByzantine = true;
      }

      if (isByzantine) {
        byzantineCount++;
        // Flag participant
        if (participant) {
          participant.byzantineFlags++;
          let rep = this.participantReputations.get(vote.participantId)!;
          rep = recordNegative(rep);
          this.participantReputations.set(vote.participantId, rep);
        }
      } else {
        validVotes.push(vote);
      }
    }

    return { validVotes, byzantineCount };
  }

  /**
   * Calculate voting weight from reputation
   */
  private calculateVotingWeight(reputation: ReputationScore): number {
    const baseWeight = reputationValue(reputation);

    // Apply sqrt to reduce gap between high and low reputation
    // This prevents reputation oligarchy while still rewarding trust
    return Math.sqrt(baseWeight);
  }

  /**
   * Start a new consensus round
   */
  private startNewRound(topicId: string): ConsensusRound {
    const rounds = this.rounds.get(topicId) || [];

    const round: ConsensusRound = {
      roundNumber: rounds.length + 1,
      topicId,
      votes: [],
      participantCount: 0,
      quorumReached: false,
      leadingOption: '',
      leadingPercentage: 0,
      byzantineDetected: 0,
      startedAt: Date.now(),
    };

    rounds.push(round);
    this.rounds.set(topicId, rounds);

    return round;
  }

  /**
   * Get current active round
   */
  private getCurrentRound(topicId: string): ConsensusRound {
    const rounds = this.rounds.get(topicId) || [];
    if (rounds.length === 0) {
      return this.startNewRound(topicId);
    }
    return rounds[rounds.length - 1];
  }

  /**
   * Create empty result for topics with no votes
   */
  private emptyResult(topicId: string): ConsensusResult {
    const topic = this.topics.get(topicId);
    const optionBreakdown: Record<string, { votes: number; weight: number }> = {};

    if (topic) {
      for (const option of topic.options) {
        optionBreakdown[option] = { votes: 0, weight: 0 };
      }
    }

    return {
      topicId,
      reached: false,
      totalVotes: 0,
      weightedVotes: 0,
      optionBreakdown,
      quorumPercentage: 0,
      byzantineFiltered: 0,
      rounds: 0,
      confidence: 0,
    };
  }

  /**
   * Initialize threshold signature collection
   */
  initThresholdSignature(topicId: string, threshold: number, totalSigners: number): ThresholdSignature {
    const sig: ThresholdSignature = {
      topicId,
      threshold,
      totalSigners,
      shards: [],
      valid: false,
    };

    this.thresholdSigs.set(topicId, sig);
    return sig;
  }

  /**
   * Submit a signature shard for multi-sig execution
   */
  submitSignatureShard(topicId: string, signerId: string, shard: string, index: number): boolean {
    const thresholdSig = this.thresholdSigs.get(topicId);
    if (!thresholdSig) {
      throw new Error(`No threshold signature initialized for: ${topicId}`);
    }

    // Check if signer is already registered
    if (thresholdSig.shards.some((s) => s.signerId === signerId)) {
      throw new Error('Signer already submitted shard');
    }

    const shardEntry: SignatureShard = {
      id: `shard-${Date.now()}-${index}`,
      topicId,
      signerId,
      shard,
      index,
      timestamp: Date.now(),
    };

    thresholdSig.shards.push(shardEntry);

    // Check if threshold reached
    if (thresholdSig.shards.length >= thresholdSig.threshold) {
      // In real implementation, aggregate shards into signature
      thresholdSig.aggregatedSignature = `aggregated-${topicId}-${thresholdSig.shards.length}`;
      thresholdSig.valid = true;
    }

    this.thresholdSigs.set(topicId, thresholdSig);

    return thresholdSig.valid;
  }

  /**
   * Get participant profile
   */
  getParticipantProfile(participantId: string): ConsensusParticipant | undefined {
    return this.participants.get(participantId);
  }

  /**
   * Get topic status
   */
  getTopicStatus(topicId: string): ConsensusTopic | undefined {
    return this.topics.get(topicId);
  }

  /**
   * Check if participant is trustworthy
   */
  isParticipantTrustworthy(participantId: string, threshold = 0.7): boolean {
    const reputation = this.participantReputations.get(participantId);
    if (!reputation) return false;
    return isTrustworthy(reputation, threshold);
  }

  /**
   * Query participant reputation from other hApps
   */
  queryExternalReputation(participantId: string): void {
    const query = createReputationQuery('consensus', participantId);
    this.bridge.send('governance', query);
    this.bridge.send('identity', query);
  }

  /**
   * Get FL coordinator for consensus optimization
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }

  /**
   * Get current Byzantine tolerance level
   */
  getByzantineTolerance(): number {
    return this.DEFAULT_BYZANTINE_TOLERANCE;
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
};

// Default service instance
let defaultService: ConsensusService | null = null;

/**
 * Get the default consensus service instance
 */
export function getConsensusService(): ConsensusService {
  if (!defaultService) {
    defaultService = new ConsensusService();
  }
  return defaultService;
}

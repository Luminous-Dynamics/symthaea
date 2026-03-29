// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Collective Zome Client
 *
 * Collective intelligence: belief sharing, voting, Phi-weighted consensus,
 * pattern detection, epistemic reputation, trust decay, and trend analysis.
 *
 * @module @mycelix/sdk/clients/lucid/collective
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  ShareBeliefInput,
  CastVoteInput,
  DetectPatternsInput,
  PatternCluster,
  PhiWeightedConsensusInput,
  RecordPatternInput,
  UpdateReputationInput,
  DomainExpertiseInput,
  DomainExpertiseResult,
  ConsensusExplanation,
  CollectiveStats,
  UpdateRelationshipInput,
  AgentRelationship,
  WeightedConsensusResult,
  TrendAnalysisInput,
  TrendAnalysisResult,
  BeliefLifecycle,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface CollectiveClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: CollectiveClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Collective zome
 *
 * Provides Phi-weighted collective intelligence, epistemic reputation,
 * trust relationships, and trend analysis.
 */
export class CollectiveClient extends ZomeClient {
  protected readonly zomeName = 'lucid_collective';

  constructor(client: AppClient, config: CollectiveClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Belief Sharing
  // ============================================================================

  /** Share a belief to the collective */
  async shareBelief(input: ShareBeliefInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('share_belief', input);
  }

  /** Get beliefs by tag */
  async getBeliefsByTag(tag: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_beliefs_by_tag', tag);
  }

  /** Get all belief shares */
  async getAllBeliefShares(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_all_belief_shares', null);
  }

  // ============================================================================
  // Voting
  // ============================================================================

  /** Cast a vote on a shared belief */
  async castVote(input: CastVoteInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cast_vote', input);
  }

  /** Get votes for a belief share */
  async getBeliefVotes(beliefHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_belief_votes', beliefHash);
  }

  /** Get consensus for a belief share */
  async getBeliefConsensus(beliefHash: ActionHash): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_belief_consensus', beliefHash);
  }

  // ============================================================================
  // Phi-Weighted Consensus
  // ============================================================================

  /**
   * Calculate Phi-weighted consensus for a belief.
   *
   * Weights each voter's agreement by their Phi score, giving more influence
   * to agents with higher integrated information.
   */
  async calculatePhiWeightedConsensus(input: PhiWeightedConsensusInput): Promise<HolochainRecord | null> {
    return this.callZome<HolochainRecord | null>('calculate_phi_weighted_consensus', input);
  }

  /**
   * Create a Phi-weighted consensus entry for a belief.
   *
   * Fetches Phi scores from the bridge zome automatically and stores
   * the consensus result on the DHT.
   */
  async createPhiConsensus(input: PhiWeightedConsensusInput): Promise<HolochainRecord | null> {
    return this.callZomeOnce<HolochainRecord | null>('create_phi_consensus', input);
  }

  /** Calculate weighted consensus factoring in agent relationships */
  async calculateWeightedConsensus(beliefHash: ActionHash): Promise<WeightedConsensusResult> {
    return this.callZome<WeightedConsensusResult>('calculate_weighted_consensus', beliefHash);
  }

  /** Explain why a consensus was reached */
  async explainConsensus(beliefHash: ActionHash): Promise<ConsensusExplanation> {
    return this.callZome<ConsensusExplanation>('explain_consensus', beliefHash);
  }

  // ============================================================================
  // Pattern Detection
  // ============================================================================

  /** Detect epistemic patterns in collective beliefs */
  async detectPatterns(input: DetectPatternsInput): Promise<PatternCluster[]> {
    return this.callZome<PatternCluster[]>('detect_patterns', input);
  }

  /** Get existing detected patterns */
  async getRecordedPatterns(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_recorded_patterns', null);
  }

  /** Record an emergent pattern */
  async recordPattern(input: RecordPatternInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_pattern', input);
  }

  // ============================================================================
  // Epistemic Reputation
  // ============================================================================

  /** Update agent's epistemic reputation */
  async updateReputation(input: UpdateReputationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_reputation', input);
  }

  /** Get domain expertise score for an agent */
  async getDomainExpertise(input: DomainExpertiseInput): Promise<DomainExpertiseResult> {
    return this.callZome<DomainExpertiseResult>('get_domain_expertise', input);
  }

  // ============================================================================
  // Trust & Relationships
  // ============================================================================

  /** Apply trust decay to all relationships */
  async applyTrustDecay(): Promise<number> {
    return this.callZomeOnce<number>('apply_trust_decay', null);
  }

  /** Update or create a relationship with another agent */
  async updateRelationship(input: UpdateRelationshipInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_relationship', input);
  }

  /** Get relationship with a specific agent */
  async getRelationship(otherAgent: Uint8Array): Promise<AgentRelationship | null> {
    return this.callZomeOrNull<AgentRelationship>('get_relationship', otherAgent);
  }

  /** Get all my relationships */
  async getMyRelationships(): Promise<AgentRelationship[]> {
    return this.callZome<AgentRelationship[]>('get_my_relationships', null);
  }

  // ============================================================================
  // Stats & Trends
  // ============================================================================

  /** Get collective statistics */
  async getCollectiveStats(): Promise<CollectiveStats> {
    return this.callZome<CollectiveStats>('get_collective_stats', null);
  }

  /** Analyze belief trends (direction, velocity, herding warnings) */
  async analyzeBeliefTrends(input: TrendAnalysisInput): Promise<TrendAnalysisResult> {
    return this.callZome<TrendAnalysisResult>('analyze_belief_trends', input);
  }

  /** Get belief lifecycle analysis for a specific topic */
  async getBeliefLifecycle(tag: string): Promise<BeliefLifecycle> {
    return this.callZome<BeliefLifecycle>('get_belief_lifecycle', tag);
  }
}

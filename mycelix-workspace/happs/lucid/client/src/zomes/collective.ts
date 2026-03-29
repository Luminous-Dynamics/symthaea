// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collective Sensemaking Zome Client
 *
 * Distributed belief sharing, validation, consensus discovery, and emergent truth.
 */

import type { AppClient, Record as HolochainRecord, AgentPubKey, ActionHash } from '@holochain/client';
import type {
  BeliefShare,
  ShareBeliefInput,
  ValidationVote,
  CastVoteInput,
  ConsensusRecord,
  EmergentPattern,
  RecordPatternInput,
  DetectPatternsInput,
  PatternCluster,
  EpistemicReputation,
  UpdateReputationInput,
  AgentRelationship,
  UpdateRelationshipInput,
  WeightedConsensusResult,
  CollectiveStats,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class CollectiveZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'collective'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ============================================================================
  // BELIEF SHARING
  // ============================================================================

  /** Share a belief to the collective */
  async shareBelief(input: ShareBeliefInput): Promise<BeliefShare> {
    const record = await this.callZome<HolochainRecord>('share_belief', input);
    return decodeRecord<BeliefShare>(record);
  }

  /** Get all shared beliefs */
  async getAllBeliefShares(): Promise<BeliefShare[]> {
    const records = await this.callZome<HolochainRecord[]>('get_all_belief_shares', null);
    return decodeRecords<BeliefShare>(records);
  }

  /** Get beliefs by tag */
  async getBeliefsByTag(tag: string): Promise<BeliefShare[]> {
    const records = await this.callZome<HolochainRecord[]>('get_beliefs_by_tag', tag);
    return decodeRecords<BeliefShare>(records);
  }

  // ============================================================================
  // VOTING
  // ============================================================================

  /** Cast a vote on a shared belief */
  async castVote(input: CastVoteInput): Promise<ValidationVote> {
    const record = await this.callZome<HolochainRecord>('cast_vote', input);
    return decodeRecord<ValidationVote>(record);
  }

  /** Get votes for a belief share */
  async getBeliefVotes(beliefHash: ActionHash): Promise<ValidationVote[]> {
    const records = await this.callZome<HolochainRecord[]>('get_belief_votes', beliefHash);
    return decodeRecords<ValidationVote>(records);
  }

  // ============================================================================
  // CONSENSUS
  // ============================================================================

  /** Get consensus for a belief */
  async getBeliefConsensus(beliefHash: ActionHash): Promise<ConsensusRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_belief_consensus', beliefHash);
    return record ? decodeRecord<ConsensusRecord>(record) : null;
  }

  /** Calculate weighted consensus using relationships */
  async calculateWeightedConsensus(beliefHash: ActionHash): Promise<WeightedConsensusResult> {
    return this.callZome<WeightedConsensusResult>('calculate_weighted_consensus', beliefHash);
  }

  // ============================================================================
  // PATTERN DETECTION
  // ============================================================================

  /** Detect emergent patterns across beliefs */
  async detectPatterns(input?: DetectPatternsInput): Promise<PatternCluster[]> {
    return this.callZome<PatternCluster[]>('detect_patterns', input ?? {});
  }

  /** Record an emergent pattern */
  async recordPattern(input: RecordPatternInput): Promise<EmergentPattern> {
    const record = await this.callZome<HolochainRecord>('record_pattern', input);
    return decodeRecord<EmergentPattern>(record);
  }

  /** Get all recorded patterns */
  async getRecordedPatterns(): Promise<EmergentPattern[]> {
    const records = await this.callZome<HolochainRecord[]>('get_recorded_patterns', null);
    return decodeRecords<EmergentPattern>(records);
  }

  // ============================================================================
  // REPUTATION
  // ============================================================================

  /** Update agent's epistemic reputation */
  async updateReputation(input: UpdateReputationInput): Promise<EpistemicReputation> {
    const record = await this.callZome<HolochainRecord>('update_reputation', input);
    return decodeRecord<EpistemicReputation>(record);
  }

  // ============================================================================
  // RELATIONSHIPS
  // ============================================================================

  /** Update or create a relationship with another agent */
  async updateRelationship(input: UpdateRelationshipInput): Promise<AgentRelationship> {
    const record = await this.callZome<HolochainRecord>('update_relationship', input);
    return decodeRecord<AgentRelationship>(record);
  }

  /** Get relationship with a specific agent */
  async getRelationship(otherAgent: AgentPubKey): Promise<AgentRelationship | null> {
    const result = await this.callZome<AgentRelationship | null>('get_relationship', otherAgent);
    return result;
  }

  /** Get all my relationships */
  async getMyRelationships(): Promise<AgentRelationship[]> {
    return this.callZome<AgentRelationship[]>('get_my_relationships', null);
  }

  // ============================================================================
  // STATISTICS
  // ============================================================================

  /** Get collective statistics */
  async getCollectiveStats(): Promise<CollectiveStats> {
    return this.callZome<CollectiveStats>('get_collective_stats', null);
  }
}

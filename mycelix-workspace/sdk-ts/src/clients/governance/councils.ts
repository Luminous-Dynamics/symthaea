// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Councils Zome Client
 *
 * Handles holonic council management, membership, reflection,
 * and decision tracking for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/councils
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

export type CouncilType =
  | { type: 'Root' }
  | { type: 'Domain'; domain: string }
  | { type: 'Regional'; region: string }
  | { type: 'WorkingGroup'; focus: string; expires?: number }
  | { type: 'Advisory' }
  | { type: 'Emergency'; expires: number };

export type CouncilStatus = 'Active' | 'Dormant' | 'Dissolved' | 'Suspended';

export type MemberRole =
  | 'Member'
  | 'Facilitator'
  | 'Steward'
  | 'Observer'
  | { Delegate: { from_council: string } };

export type MembershipStatus = 'Active' | 'OnLeave' | 'Suspended' | 'Removed';

export interface Council {
  id: string;
  name: string;
  purpose: string;
  councilType: CouncilType;
  parentCouncilId?: string;
  phiThreshold: number;
  quorum: number;
  supermajority: number;
  canSpawnChildren: boolean;
  maxDelegationDepth: number;
  signingCommitteeId?: string;
  status: CouncilStatus;
  createdAt: number;
  lastActivity: number;
}

export interface CouncilMembership {
  id: string;
  councilId: string;
  memberDid: string;
  role: MemberRole;
  phiScore: number;
  votingWeight: number;
  canDelegate: boolean;
  status: MembershipStatus;
  joinedAt: number;
  lastParticipation: number;
}

export interface CouncilReflection {
  id: string;
  councilId: string;
  reflectorDid: string;
  harmonyScores: Record<string, number>;
  narrative: string;
  insights: string[];
  timestamp: number;
}

export type DecisionType =
  | 'Operational'
  | 'Policy'
  | 'Resource'
  | 'Membership'
  | 'SubCouncil'
  | 'Constitutional';

export type DecisionStatus =
  | 'Pending'
  | 'Approved'
  | 'Rejected'
  | 'Executed'
  | 'Vetoed';

export interface CouncilDecision {
  id: string;
  councilId: string;
  proposalId?: string;
  title: string;
  content: string;
  decisionType: DecisionType;
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  phiWeightedResult: number;
  passed: boolean;
  status: DecisionStatus;
  createdAt: number;
  executedAt?: number;
}

export interface CreateCouncilInput {
  name: string;
  purpose: string;
  councilType: CouncilType;
  parentCouncilId?: string;
  phiThreshold: number;
  quorum: number;
  supermajority: number;
  canSpawnChildren: boolean;
  maxDelegationDepth: number;
  signingCommitteeId?: string;
}

export interface JoinCouncilInput {
  councilId: string;
  memberDid: string;
  role: MemberRole;
  phiScore: number;
}

export interface ReflectOnCouncilInput {
  councilId: string;
  reflectorDid: string;
  harmonyScores: Record<string, number>;
  narrative: string;
  insights: string[];
}

export interface RecordDecisionInput {
  councilId: string;
  proposalId?: string;
  title: string;
  content: string;
  decisionType: DecisionType;
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  phiWeightedResult: number;
}

// ============================================================================
// Configuration
// ============================================================================

export interface CouncilsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: CouncilsClientConfig = {
  roleName: 'governance',
};

// ============================================================================
// Client
// ============================================================================

/**
 * Client for Council operations
 *
 * Manages holonic governance councils with membership, reflection,
 * and decision tracking.
 *
 * @example
 * ```typescript
 * const councils = new CouncilsClient(appClient);
 *
 * const council = await councils.createCouncil({
 *   name: 'Water Commons Council',
 *   purpose: 'Steward local water resources',
 *   councilType: { type: 'Domain', domain: 'water' },
 *   phiThreshold: 0.3,
 *   quorum: 0.5,
 *   supermajority: 0.67,
 *   canSpawnChildren: true,
 *   maxDelegationDepth: 3,
 * });
 * ```
 */
export class CouncilsClient extends ZomeClient {
  protected readonly zomeName = 'councils';

  constructor(client: AppClient, config: CouncilsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Council CRUD
  // ============================================================================

  async createCouncil(input: CreateCouncilInput): Promise<Council> {
    const record = await this.callZomeOnce<HolochainRecord>('create_council', {
      name: input.name,
      purpose: input.purpose,
      council_type: input.councilType,
      parent_council_id: input.parentCouncilId,
      phi_threshold: input.phiThreshold,
      quorum: input.quorum,
      supermajority: input.supermajority,
      can_spawn_children: input.canSpawnChildren,
      max_delegation_depth: input.maxDelegationDepth,
      signing_committee_id: input.signingCommitteeId,
    });
    return this.mapCouncil(record);
  }

  async getCouncil(councilId: ActionHash): Promise<Council | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_council_by_id', councilId);
    if (!record) return null;
    return this.mapCouncil(record);
  }

  async getAllCouncils(): Promise<Council[]> {
    const records = await this.callZome<HolochainRecord[]>('get_all_councils', null);
    return records.map((r) => this.mapCouncil(r));
  }

  async getChildCouncils(councilId: string): Promise<Council[]> {
    const records = await this.callZome<HolochainRecord[]>('get_child_councils', councilId);
    return records.map((r) => this.mapCouncil(r));
  }

  // ============================================================================
  // Membership
  // ============================================================================

  async joinCouncil(input: JoinCouncilInput): Promise<CouncilMembership> {
    const record = await this.callZomeOnce<HolochainRecord>('join_council', {
      council_id: input.councilId,
      member_did: input.memberDid,
      role: input.role,
      phi_score: input.phiScore,
    });
    return this.mapMembership(record);
  }

  async getCouncilMembers(councilId: string): Promise<CouncilMembership[]> {
    const records = await this.callZome<HolochainRecord[]>('get_council_members', councilId);
    return records.map((r) => this.mapMembership(r));
  }

  async getMemberCouncils(memberDid: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_member_councils', memberDid);
  }

  // ============================================================================
  // Reflection & Decisions
  // ============================================================================

  async reflectOnCouncil(input: ReflectOnCouncilInput): Promise<CouncilReflection> {
    const record = await this.callZomeOnce<HolochainRecord>('reflect_on_council', {
      council_id: input.councilId,
      reflector_did: input.reflectorDid,
      harmony_scores: input.harmonyScores,
      narrative: input.narrative,
      insights: input.insights,
    });
    return this.mapReflection(record);
  }

  async getCouncilReflections(councilId: string): Promise<CouncilReflection[]> {
    const records = await this.callZome<HolochainRecord[]>('get_council_reflections', councilId);
    return records.map((r) => this.mapReflection(r));
  }

  async getLatestReflection(councilId: string): Promise<CouncilReflection | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_latest_council_reflection', councilId);
    if (!record) return null;
    return this.mapReflection(record);
  }

  async getHolonicTree(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_holonic_tree', null);
  }

  async recordDecision(input: RecordDecisionInput): Promise<CouncilDecision> {
    const record = await this.callZomeOnce<HolochainRecord>('record_decision', {
      council_id: input.councilId,
      proposal_id: input.proposalId,
      title: input.title,
      content: input.content,
      decision_type: input.decisionType,
      votes_for: input.votesFor,
      votes_against: input.votesAgainst,
      abstentions: input.abstentions,
      phi_weighted_result: input.phiWeightedResult,
    });
    return this.mapDecision(record);
  }

  async getCouncilDecisions(councilId: string): Promise<CouncilDecision[]> {
    const records = await this.callZome<HolochainRecord[]>('get_council_decisions', councilId);
    return records.map((r) => this.mapDecision(r));
  }

  // ============================================================================
  // Mappers
  // ============================================================================

  private mapCouncil(record: HolochainRecord): Council {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      name: entry.name,
      purpose: entry.purpose,
      councilType: entry.council_type,
      parentCouncilId: entry.parent_council_id,
      phiThreshold: entry.phi_threshold,
      quorum: entry.quorum,
      supermajority: entry.supermajority,
      canSpawnChildren: entry.can_spawn_children,
      maxDelegationDepth: entry.max_delegation_depth,
      signingCommitteeId: entry.signing_committee_id,
      status: entry.status,
      createdAt: entry.created_at,
      lastActivity: entry.last_activity,
    };
  }

  private mapMembership(record: HolochainRecord): CouncilMembership {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      councilId: entry.council_id,
      memberDid: entry.member_did,
      role: entry.role,
      phiScore: entry.phi_score,
      votingWeight: entry.voting_weight,
      canDelegate: entry.can_delegate,
      status: entry.status,
      joinedAt: entry.joined_at,
      lastParticipation: entry.last_participation,
    };
  }

  private mapReflection(record: HolochainRecord): CouncilReflection {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      councilId: entry.council_id,
      reflectorDid: entry.reflector_did,
      harmonyScores: entry.harmony_scores,
      narrative: entry.narrative,
      insights: entry.insights,
      timestamp: entry.timestamp,
    };
  }

  private mapDecision(record: HolochainRecord): CouncilDecision {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      councilId: entry.council_id,
      proposalId: entry.proposal_id,
      title: entry.title,
      content: entry.content,
      decisionType: entry.decision_type,
      votesFor: entry.votes_for,
      votesAgainst: entry.votes_against,
      abstentions: entry.abstentions,
      phiWeightedResult: entry.phi_weighted_result,
      passed: entry.passed,
      status: entry.status,
      createdAt: entry.created_at,
      executedAt: entry.executed_at,
    };
  }
}

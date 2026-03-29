// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DAO Zome Client
 *
 * Handles DAO creation, membership, and management operations
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/dao
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  DAO,
  CreateDAOInput,
  UpdateDAOInput,
  DAOMembership,
  JoinDAOInput,
  DAOFilter,
  MemberRole,
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the DAO client
 */
export interface DAOClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: DAOClientConfig = {
  roleName: 'governance',
};

/**
 * Client for DAO operations
 *
 * Provides methods for creating and managing DAOs, handling membership,
 * and querying DAO information.
 *
 * @example
 * ```typescript
 * import { DAOClient } from '@mycelix/sdk/clients/governance';
 *
 * const dao = new DAOClient(appClient);
 *
 * // Create a new DAO
 * const newDao = await dao.createDAO({
 *   name: 'Community Governance',
 *   description: 'Decentralized decision-making for our community',
 *   charter: 'All decisions require 2/3 majority approval...',
 *   defaultVotingPeriodHours: 168,
 *   defaultQuorum: 0.33,
 *   defaultThreshold: 0.51,
 * });
 *
 * // Join the DAO
 * const membership = await dao.joinDAO({ daoId: newDao.id });
 *
 * // Get all members
 * const members = await dao.getMembers(newDao.id);
 * ```
 */
export class DAOClient extends ZomeClient {
  protected readonly zomeName = 'dao';
  

  constructor(client: AppClient, config: DAOClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
    
  }

  // ============================================================================
  // DAO CRUD Operations
  // ============================================================================

  /**
   * Create a new DAO
   *
   * The caller becomes the founder and first member with full admin privileges.
   *
   * @param input - DAO creation parameters
   * @returns The created DAO
   */
  async createDAO(input: CreateDAOInput): Promise<DAO> {
    const record = await this.callZomeOnce<HolochainRecord>('create_dao', {
      name: input.name,
      description: input.description,
      charter: input.charter,
      charter_hash: input.charterHash,
      default_voting_period_hours: input.defaultVotingPeriodHours ?? 168,
      default_quorum: input.defaultQuorum ?? 0.33,
      default_threshold: input.defaultThreshold ?? 0.51,
    });
    return this.mapDAO(record);
  }

  /**
   * Get a DAO by ID
   *
   * @param daoId - DAO action hash
   * @returns The DAO or null if not found
   */
  async getDAO(daoId: ActionHash): Promise<DAO | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_dao', daoId);
    if (!record) return null;
    return this.mapDAO(record);
  }

  /**
   * Update DAO settings
   *
   * Only admins and founders can update DAO settings. Some changes
   * (like charter updates) may require governance approval.
   *
   * @param input - Update parameters
   * @returns The updated DAO
   */
  async updateDAO(input: UpdateDAOInput): Promise<DAO> {
    const record = await this.callZomeOnce<HolochainRecord>('update_dao', {
      dao_id: input.daoId,
      name: input.name,
      description: input.description,
      charter: input.charter,
      treasury_id: input.treasuryId,
      default_voting_period_hours: input.defaultVotingPeriodHours,
      default_quorum: input.defaultQuorum,
      default_threshold: input.defaultThreshold,
    });
    return this.mapDAO(record);
  }

  /**
   * List all DAOs
   *
   * @param filter - Optional filter parameters
   * @returns Array of DAOs
   */
  async listDAOs(filter?: DAOFilter): Promise<DAO[]> {
    const records = await this.callZome<HolochainRecord[]>('list_daos', {
      founder_did: filter?.founderDid,
      active: filter?.active,
      name_contains: filter?.nameContains,
      limit: filter?.limit,
      offset: filter?.offset,
    });
    return records.map(r => this.mapDAO(r));
  }

  /**
   * Get DAOs by member
   *
   * @param memberDid - Member's DID
   * @returns Array of DAOs the member belongs to
   */
  async getDAOsByMember(memberDid: string): Promise<DAO[]> {
    const records = await this.callZome<HolochainRecord[]>('get_daos_by_member', memberDid);
    return records.map(r => this.mapDAO(r));
  }

  /**
   * Archive a DAO (soft delete)
   *
   * @param daoId - DAO to archive
   */
  async archiveDAO(daoId: ActionHash): Promise<void> {
    await this.callZomeOnce('archive_dao', daoId);
  }

  // ============================================================================
  // Membership Operations
  // ============================================================================

  /**
   * Join a DAO as a member
   *
   * New members start with Member role and base voting power.
   *
   * @param input - Join parameters
   * @returns The membership record
   */
  async joinDAO(input: JoinDAOInput): Promise<DAOMembership> {
    const record = await this.callZomeOnce<HolochainRecord>('join_dao', {
      dao_id: input.daoId,
      initial_voting_power: input.initialVotingPower ?? 0,
    });
    return this.mapMembership(record);
  }

  /**
   * Leave a DAO
   *
   * @param daoId - DAO to leave
   */
  async leaveDAO(daoId: ActionHash): Promise<void> {
    await this.callZomeOnce('leave_dao', daoId);
  }

  /**
   * Get all members of a DAO
   *
   * @param daoId - DAO identifier
   * @returns Array of membership records
   */
  async getMembers(daoId: ActionHash): Promise<DAOMembership[]> {
    const records = await this.callZome<HolochainRecord[]>('get_dao_members', daoId);
    return records.map(r => this.mapMembership(r));
  }

  /**
   * Get membership details for a specific member
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns Membership or null if not a member
   */
  async getMembership(daoId: ActionHash, memberDid: string): Promise<DAOMembership | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_membership', {
      dao_id: daoId,
      member_did: memberDid,
    });
    if (!record) return null;
    return this.mapMembership(record);
  }

  /**
   * Get my membership in a DAO
   *
   * @param daoId - DAO identifier
   * @returns My membership or null
   */
  async getMyMembership(daoId: ActionHash): Promise<DAOMembership | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_my_membership', daoId);
    if (!record) return null;
    return this.mapMembership(record);
  }

  /**
   * Check if a DID is a member of a DAO
   *
   * @param daoId - DAO identifier
   * @param memberDid - DID to check
   * @returns True if member
   */
  async isMember(daoId: ActionHash, memberDid: string): Promise<boolean> {
    return this.callZome<boolean>('is_member', {
      dao_id: daoId,
      member_did: memberDid,
    });
  }

  /**
   * Update member role
   *
   * Only admins/founders can change roles. Cannot demote founders.
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @param newRole - New role to assign
   * @returns Updated membership
   */
  async updateMemberRole(
    daoId: ActionHash,
    memberDid: string,
    newRole: MemberRole
  ): Promise<DAOMembership> {
    const record = await this.callZomeOnce<HolochainRecord>('update_member_role', {
      dao_id: daoId,
      member_did: memberDid,
      new_role: newRole,
    });
    return this.mapMembership(record);
  }

  /**
   * Update member voting power
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @param votingPower - New voting power
   * @returns Updated membership
   */
  async updateMemberVotingPower(
    daoId: ActionHash,
    memberDid: string,
    votingPower: number
  ): Promise<DAOMembership> {
    const record = await this.callZomeOnce<HolochainRecord>('update_member_voting_power', {
      dao_id: daoId,
      member_did: memberDid,
      voting_power: votingPower,
    });
    return this.mapMembership(record);
  }

  /**
   * Remove a member from DAO
   *
   * Only admins/founders can remove members.
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member to remove
   */
  async removeMember(daoId: ActionHash, memberDid: string): Promise<void> {
    await this.callZomeOnce('remove_member', {
      dao_id: daoId,
      member_did: memberDid,
    });
  }

  // ============================================================================
  // Permission Checks
  // ============================================================================

  /**
   * Check if a member has admin privileges
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns True if admin or founder
   */
  async isAdmin(daoId: ActionHash, memberDid: string): Promise<boolean> {
    const membership = await this.getMembership(daoId, memberDid);
    if (!membership) return false;
    return membership.role === 'Admin' || membership.role === 'Founder';
  }

  /**
   * Check if a member is the founder
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns True if founder
   */
  async isFounder(daoId: ActionHash, memberDid: string): Promise<boolean> {
    const membership = await this.getMembership(daoId, memberDid);
    return membership?.role === 'Founder';
  }

  // ============================================================================
  // Statistics
  // ============================================================================

  /**
   * Get DAO statistics
   *
   * @param daoId - DAO identifier
   * @returns DAO statistics
   */
  async getDAOStats(daoId: ActionHash): Promise<{
    memberCount: number;
    totalVotingPower: number;
    activeProposals: number;
    totalProposals: number;
    treasuryBalance?: number;
  }> {
    return this.callZome('get_dao_stats', daoId);
  }

  /**
   * Get member activity summary
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns Activity summary
   */
  async getMemberActivity(
    daoId: ActionHash,
    memberDid: string
  ): Promise<{
    proposalsCreated: number;
    votesCast: number;
    participationRate: number;
    lastActiveAt: number;
  }> {
    return this.callZome('get_member_activity', {
      dao_id: daoId,
      member_did: memberDid,
    });
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to DAO type
   */
  private mapDAO(record: HolochainRecord): DAO {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      name: entry.name,
      description: entry.description,
      charter: entry.charter,
      charterHash: entry.charter_hash,
      founderDid: entry.founder_did,
      treasuryId: entry.treasury_id,
      defaultVotingPeriodHours: entry.default_voting_period_hours,
      defaultQuorum: entry.default_quorum,
      defaultThreshold: entry.default_threshold,
      memberCount: entry.member_count,
      totalVotingPower: entry.total_voting_power,
      active: entry.active,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  /**
   * Map Holochain record to DAOMembership type
   */
  private mapMembership(record: HolochainRecord): DAOMembership {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      daoId: entry.dao_id,
      memberDid: entry.member_did,
      memberPubKey: entry.member_pub_key,
      role: entry.role,
      votingPower: entry.voting_power,
      reputationScore: entry.reputation_score,
      active: entry.active,
      joinedAt: entry.joined_at,
      lastActiveAt: entry.last_active_at,
    };
  }
}

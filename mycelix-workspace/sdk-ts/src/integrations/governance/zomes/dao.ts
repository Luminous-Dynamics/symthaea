// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DAO Zome Client
 *
 * Handles DAO creation, membership, and management operations.
 *
 * @module @mycelix/sdk/integrations/governance/zomes/dao
 */

import { GovernanceSdkError } from '../types';

import type {
  Dao,
  CreateDaoInput,
  UpdateDaoInput,
  Membership,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the DAO client
 */
export interface DaoClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: DaoClientConfig = {
  roleId: 'governance',
  zomeName: 'governance',
};

/**
 * Client for DAO operations
 *
 * @example
 * ```typescript
 * const dao = new DaoClient(holochainClient);
 *
 * // Create a new DAO
 * const newDao = await dao.createDao({
 *   id: 'mycelix-core',
 *   name: 'Mycelix Core Governance',
 *   description: 'Core protocol governance',
 *   constitution: 'All decisions require 51% approval...',
 *   founder_did: 'did:mycelix:abc123',
 * });
 *
 * // Join the DAO
 * await dao.joinDao({
 *   dao_id: 'mycelix-core',
 *   member_did: 'did:mycelix:def456',
 *   role: 'Member',
 *   voting_power: 100,
 *   joined_at: Date.now() * 1000,
 * });
 * ```
 */
export class DaoClient {
  private readonly config: DaoClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<DaoClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new GovernanceSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // DAO Operations
  // ============================================================================

  /**
   * Create a new DAO
   *
   * @param input - DAO creation parameters
   * @returns The created DAO
   * @throws {GovernanceSdkError} If founder_did doesn't match caller
   */
  async createDao(input: CreateDaoInput): Promise<Dao> {
    const now = Date.now() * 1000; // Convert to microseconds
    const daoInput: Dao = {
      ...input,
      member_count: 1, // Founder is first member
      created: now,
      updated: now,
    };

    const record = await this.call<HolochainRecord>('create_dao', daoInput);
    return this.extractEntry<Dao>(record);
  }

  /**
   * Get a DAO by ID
   *
   * @param daoId - DAO identifier
   * @returns The DAO or null if not found
   */
  async getDao(daoId: string): Promise<Dao | null> {
    const record = await this.call<HolochainRecord | null>('get_dao', daoId);
    if (!record) return null;
    return this.extractEntry<Dao>(record);
  }

  /**
   * Update DAO settings
   *
   * @param input - Update parameters
   * @returns The updated DAO
   * @throws {GovernanceSdkError} If caller is not admin/founder
   */
  async updateDao(input: UpdateDaoInput): Promise<Dao> {
    const record = await this.call<HolochainRecord>('update_dao', input);
    return this.extractEntry<Dao>(record);
  }

  // ============================================================================
  // Membership Operations
  // ============================================================================

  /**
   * Join a DAO as a member
   *
   * @param membership - Membership details
   * @returns The membership record
   * @throws {GovernanceSdkError} If member_did doesn't match caller
   */
  async joinDao(membership: Membership): Promise<Membership> {
    const record = await this.call<HolochainRecord>('join_dao', membership);
    return this.extractEntry<Membership>(record);
  }

  /**
   * Get all members of a DAO
   *
   * @param daoId - DAO identifier
   * @returns Array of membership records
   */
  async getMembers(daoId: string): Promise<Membership[]> {
    return this.call<Membership[]>('get_dao_members', daoId);
  }

  /**
   * Check if a DID is a member of a DAO
   *
   * @param daoId - DAO identifier
   * @param memberDid - DID to check
   * @returns True if member, false otherwise
   */
  async isMember(daoId: string, memberDid: string): Promise<boolean> {
    const members = await this.getMembers(daoId);
    return members.some(m => m.member_did === memberDid);
  }

  /**
   * Get membership details for a DID in a DAO
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns Membership or null if not a member
   */
  async getMembership(daoId: string, memberDid: string): Promise<Membership | null> {
    const members = await this.getMembers(daoId);
    return members.find(m => m.member_did === memberDid) || null;
  }

  /**
   * Check if a DID has admin privileges
   *
   * @param daoId - DAO identifier
   * @param memberDid - DID to check
   * @returns True if admin or founder
   */
  async isAdmin(daoId: string, memberDid: string): Promise<boolean> {
    const membership = await this.getMembership(daoId, memberDid);
    if (!membership) return false;
    return membership.role === 'Admin' || membership.role === 'Founder';
  }
}

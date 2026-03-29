// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Delegation Zome Client
 *
 * Handles vote power delegation operations for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/delegation
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Delegation,
  CreateDelegationInput,
  DelegationFilter,
  DelegationChain,
  
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Delegation client
 */
export interface DelegationClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: DelegationClientConfig = {
  roleName: 'governance',
};

/**
 * Client for Delegation operations
 *
 * Vote delegation allows members to delegate their voting power to
 * trusted representatives. Delegations can be:
 * - Full (All): Delegate all voting power
 * - Categorical: Only delegate for specific proposal categories
 *
 * @example
 * ```typescript
 * import { DelegationClient } from '@mycelix/sdk/clients/governance';
 *
 * const delegation = new DelegationClient(appClient);
 *
 * // Delegate all voting power
 * const del = await delegation.createDelegation({
 *   delegateDid: 'did:mycelix:trusted-delegate',
 *   daoId: 'uhCkkp...',
 *   scope: 'All',
 *   powerPercentage: 1.0,
 * });
 *
 * // Delegate only for treasury proposals
 * await delegation.createDelegation({
 *   delegateDid: 'did:mycelix:treasury-expert',
 *   daoId: 'uhCkkp...',
 *   scope: 'Category',
 *   category: 'treasury',
 *   powerPercentage: 0.5,
 * });
 *
 * // Check delegated power received
 * const power = await delegation.getDelegatedPowerReceived('uhCkkp...', 'did:mycelix:me');
 * ```
 */
export class DelegationClient extends ZomeClient {
  protected readonly zomeName = 'delegation';
  

  constructor(client: AppClient, config: DelegationClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
    
  }

  // ============================================================================
  // Delegation CRUD Operations
  // ============================================================================

  /**
   * Create a vote delegation
   *
   * Allows a member to delegate their voting power to another member.
   * Validates:
   * - Delegator is a DAO member
   * - Delegate is a DAO member
   * - No circular delegation would be created
   * - Power percentage is valid (0-1)
   *
   * @param input - Delegation parameters
   * @returns The created delegation
   */
  async createDelegation(input: CreateDelegationInput): Promise<Delegation> {
    const record = await this.callZomeOnce<HolochainRecord>('create_delegation', {
      delegate_did: input.delegateDid,
      dao_id: input.daoId,
      scope: input.scope,
      category: input.category,
      power_percentage: input.powerPercentage ?? 1.0,
      expires_at: input.expiresAt,
    });
    return this.mapDelegation(record);
  }

  /**
   * Revoke an existing delegation
   *
   * @param delegationId - Delegation identifier
   */
  async revokeDelegation(delegationId: ActionHash): Promise<void> {
    await this.callZomeOnce('revoke_delegation', delegationId);
  }

  /**
   * Update delegation settings
   *
   * @param delegationId - Delegation identifier
   * @param updates - Fields to update
   * @returns Updated delegation
   */
  async updateDelegation(
    delegationId: ActionHash,
    updates: {
      powerPercentage?: number;
      expiresAt?: number;
      category?: string;
    }
  ): Promise<Delegation> {
    const record = await this.callZomeOnce<HolochainRecord>('update_delegation', {
      delegation_id: delegationId,
      power_percentage: updates.powerPercentage,
      expires_at: updates.expiresAt,
      category: updates.category,
    });
    return this.mapDelegation(record);
  }

  /**
   * Get a delegation by ID
   *
   * @param delegationId - Delegation identifier
   * @returns The delegation or null
   */
  async getDelegation(delegationId: ActionHash): Promise<Delegation | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_delegation', delegationId);
    if (!record) return null;
    return this.mapDelegation(record);
  }

  /**
   * List delegations with filters
   *
   * @param filter - Filter parameters
   * @returns Array of delegations
   */
  async listDelegations(filter?: DelegationFilter): Promise<Delegation[]> {
    const records = await this.callZome<HolochainRecord[]>('list_delegations', {
      dao_id: filter?.daoId,
      delegator_did: filter?.delegatorDid,
      delegate_did: filter?.delegateDid,
      active_only: filter?.activeOnly ?? true,
      limit: filter?.limit,
      offset: filter?.offset,
    });
    return records.map(r => this.mapDelegation(r));
  }

  // ============================================================================
  // Delegation Queries
  // ============================================================================

  /**
   * Get delegations created by a delegator
   *
   * @param daoId - DAO identifier
   * @param delegatorDid - Delegator's DID
   * @returns Array of delegations
   */
  async getDelegationsFrom(daoId: ActionHash, delegatorDid: string): Promise<Delegation[]> {
    return this.listDelegations({
      daoId,
      delegatorDid,
      activeOnly: true,
    });
  }

  /**
   * Get delegations received by a delegate
   *
   * @param daoId - DAO identifier
   * @param delegateDid - Delegate's DID
   * @returns Array of delegations
   */
  async getDelegationsTo(daoId: ActionHash, delegateDid: string): Promise<Delegation[]> {
    return this.listDelegations({
      daoId,
      delegateDid,
      activeOnly: true,
    });
  }

  /**
   * Get my outgoing delegations
   *
   * @param daoId - DAO identifier
   * @returns Array of my delegations
   */
  async getMyDelegations(daoId: ActionHash): Promise<Delegation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_delegations', daoId);
    return records.map(r => this.mapDelegation(r));
  }

  /**
   * Get delegations I've received
   *
   * @param daoId - DAO identifier
   * @returns Array of delegations to me
   */
  async getDelegationsToMe(daoId: ActionHash): Promise<Delegation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_delegations_to_me', daoId);
    return records.map(r => this.mapDelegation(r));
  }

  /**
   * Check if a member has delegated their vote
   *
   * @param daoId - DAO identifier
   * @param delegatorDid - Delegator's DID
   * @returns DID of the delegate, or null if not delegated
   */
  async hasDelegated(daoId: ActionHash, delegatorDid: string): Promise<string | null> {
    return this.callZome<string | null>('has_delegated', {
      dao_id: daoId,
      delegator_did: delegatorDid,
    });
  }

  /**
   * Get total power delegated TO a delegate
   *
   * @param daoId - DAO identifier
   * @param delegateDid - Delegate's DID
   * @returns Total delegated voting power
   */
  async getDelegatedPowerReceived(daoId: ActionHash, delegateDid: string): Promise<number> {
    return this.callZome<number>('get_delegated_power_received', {
      dao_id: daoId,
      delegate_did: delegateDid,
    });
  }

  /**
   * Get power delegated OUT by a delegator
   *
   * @param daoId - DAO identifier
   * @param delegatorDid - Delegator's DID
   * @returns Total power delegated out
   */
  async getDelegatedPowerGiven(daoId: ActionHash, delegatorDid: string): Promise<number> {
    return this.callZome<number>('get_delegated_power_given', {
      dao_id: daoId,
      delegator_did: delegatorDid,
    });
  }

  // ============================================================================
  // Delegation Chain & Validation
  // ============================================================================

  /**
   * Get the delegation chain for a DID
   *
   * Useful for detecting circular delegations and understanding
   * how voting power flows.
   *
   * @param daoId - DAO identifier
   * @param startDid - Starting DID
   * @returns Delegation chain
   */
  async getDelegationChain(daoId: ActionHash, startDid: string): Promise<DelegationChain> {
    const result = await this.callZome<any>('get_delegation_chain', {
      dao_id: daoId,
      start_did: startDid,
    });
    return {
      startDid: result.start_did,
      chain: result.chain,
      hasCycle: result.has_cycle,
    };
  }

  /**
   * Check if creating a delegation would cause a cycle
   *
   * @param daoId - DAO identifier
   * @param delegatorDid - Potential delegator
   * @param delegateDid - Potential delegate
   * @returns True if would create a cycle
   */
  async wouldCreateCycle(
    daoId: ActionHash,
    delegatorDid: string,
    delegateDid: string
  ): Promise<boolean> {
    return this.callZome<boolean>('would_create_cycle', {
      dao_id: daoId,
      delegator_did: delegatorDid,
      delegate_did: delegateDid,
    });
  }

  /**
   * Check if delegation is still valid (active and not expired)
   *
   * @param delegation - Delegation to check
   * @returns True if delegation is valid
   */
  isDelegationValid(delegation: Delegation): boolean {
    if (!delegation.active) return false;
    if (delegation.expiresAt) {
      const now = Date.now() * 1000; // microseconds
      return now < delegation.expiresAt;
    }
    return true;
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Create a full delegation (all voting power)
   *
   * @param daoId - DAO identifier
   * @param delegateDid - Delegate's DID
   * @param expiresAt - Optional expiration timestamp
   * @returns The created delegation
   */
  async delegateAll(
    daoId: ActionHash,
    delegateDid: string,
    expiresAt?: number
  ): Promise<Delegation> {
    return this.createDelegation({
      daoId,
      delegateDid,
      scope: 'All',
      powerPercentage: 1.0,
      expiresAt,
    });
  }

  /**
   * Create a category-specific delegation
   *
   * @param daoId - DAO identifier
   * @param delegateDid - Delegate's DID
   * @param category - Proposal category
   * @param powerPercentage - Percentage of power to delegate (0-1)
   * @param expiresAt - Optional expiration
   * @returns The created delegation
   */
  async delegateForCategory(
    daoId: ActionHash,
    delegateDid: string,
    category: string,
    powerPercentage: number = 1.0,
    expiresAt?: number
  ): Promise<Delegation> {
    return this.createDelegation({
      daoId,
      delegateDid,
      scope: 'Category',
      category,
      powerPercentage,
      expiresAt,
    });
  }

  /**
   * Revoke all delegations for a DAO
   *
   * @param daoId - DAO identifier
   */
  async revokeAllDelegations(daoId: ActionHash): Promise<void> {
    const delegations = await this.getMyDelegations(daoId);
    await Promise.all(delegations.map(d => this.revokeDelegation(d.id)));
  }

  /**
   * Get delegators count (number of people who delegated to a delegate)
   *
   * @param daoId - DAO identifier
   * @param delegateDid - Delegate's DID
   * @returns Number of delegators
   */
  async getDelegatorsCount(daoId: ActionHash, delegateDid: string): Promise<number> {
    const delegations = await this.getDelegationsTo(daoId, delegateDid);
    return delegations.length;
  }

  /**
   * Get top delegates by received power
   *
   * @param daoId - DAO identifier
   * @param limit - Maximum results
   * @returns Array of delegates with power amounts
   */
  async getTopDelegates(
    daoId: ActionHash,
    limit: number = 10
  ): Promise<Array<{ delegateDid: string; totalPower: number; delegatorCount: number }>> {
    return this.callZome('get_top_delegates', {
      dao_id: daoId,
      limit,
    });
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to Delegation type
   */
  private mapDelegation(record: HolochainRecord): Delegation {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      delegatorDid: entry.delegator_did,
      delegateDid: entry.delegate_did,
      daoId: entry.dao_id,
      scope: entry.scope,
      category: entry.category,
      powerPercentage: entry.power_percentage,
      expiresAt: entry.expires_at,
      active: entry.active,
      createdAt: entry.created_at,
    };
  }
}

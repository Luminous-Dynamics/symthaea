// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Delegation Zome Client
 *
 * Handles vote power delegation operations.
 *
 * @module @mycelix/sdk/integrations/governance/zomes/delegation
 */

import { GovernanceSdkError } from '../types';

import type {
  Delegation,
  DelegatedPowerQuery,
  HasDelegatedQuery,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Delegation client
 */
export interface DelegationClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: DelegationClientConfig = {
  roleId: 'governance',
  zomeName: 'governance',
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
 * const delegation = new DelegationClient(holochainClient);
 *
 * // Delegate all voting power
 * await delegation.delegate({
 *   id: 'del-001',
 *   delegator_did: 'did:mycelix:abc123',
 *   delegate_did: 'did:mycelix:def456',
 *   dao_id: 'mycelix-core',
 *   scope: 'All',
 *   active: true,
 *   created: Date.now() * 1000,
 * });
 *
 * // Check delegated power
 * const power = await delegation.getDelegatedPower({
 *   dao_id: 'mycelix-core',
 *   delegate_did: 'did:mycelix:def456',
 * });
 * ```
 */
export class DelegationClient {
  private readonly config: DelegationClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<DelegationClientConfig> = {}
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
  // Delegation Operations
  // ============================================================================

  /**
   * Create a vote delegation
   *
   * Allows a member to delegate their voting power to another member.
   * The delegate can then vote on behalf of the delegator.
   *
   * @param delegation - Delegation parameters
   * @returns The created delegation
   */
  async delegate(delegation: Delegation): Promise<Delegation> {
    const record = await this.call<HolochainRecord>('delegate', delegation);
    return this.extractEntry<Delegation>(record);
  }

  /**
   * Revoke an existing delegation
   *
   * @param delegationId - Delegation identifier
   */
  async revokeDelegation(delegationId: string): Promise<void> {
    await this.call<void>('revoke_delegation', delegationId);
  }

  /**
   * Get total delegated power for a delegate
   *
   * Returns the sum of voting power delegated to this delegate.
   *
   * @param query - Delegated power query
   * @returns Total delegated voting power
   */
  async getDelegatedPower(query: DelegatedPowerQuery): Promise<number> {
    return this.call<number>('get_delegated_power', query);
  }

  /**
   * Check if a member has delegated their vote
   *
   * @param query - Has delegated query
   * @returns DID of the delegate, or null if not delegated
   */
  async hasDelegated(query: HasDelegatedQuery): Promise<string | null> {
    return this.call<string | null>('has_delegated', query);
  }

  // ============================================================================
  // Delegation Helpers
  // ============================================================================

  /**
   * Create a full delegation (all voting power)
   *
   * @param delegatorDid - DID of the person delegating
   * @param delegateDid - DID of the delegate
   * @param daoId - DAO identifier
   * @param expiresAt - Optional expiration timestamp
   * @returns The created delegation
   */
  async delegateAll(
    delegatorDid: string,
    delegateDid: string,
    daoId: string,
    expiresAt?: number
  ): Promise<Delegation> {
    const delegation: Delegation = {
      id: `del-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      delegator_did: delegatorDid,
      delegate_did: delegateDid,
      dao_id: daoId,
      scope: 'All',
      active: true,
      expires_at: expiresAt,
      created: Date.now() * 1000,
    };

    return this.delegate(delegation);
  }

  /**
   * Create a category-specific delegation
   *
   * @param delegatorDid - DID of the person delegating
   * @param delegateDid - DID of the delegate
   * @param daoId - DAO identifier
   * @param category - Proposal category to delegate for
   * @param expiresAt - Optional expiration timestamp
   * @returns The created delegation
   */
  async delegateCategory(
    delegatorDid: string,
    delegateDid: string,
    daoId: string,
    category: string,
    expiresAt?: number
  ): Promise<Delegation> {
    const delegation: Delegation = {
      id: `del-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      delegator_did: delegatorDid,
      delegate_did: delegateDid,
      dao_id: daoId,
      scope: 'Category',
      category,
      active: true,
      expires_at: expiresAt,
      created: Date.now() * 1000,
    };

    return this.delegate(delegation);
  }

  /**
   * Check if a delegation is still valid
   *
   * @param delegation - Delegation to check
   * @returns True if delegation is active and not expired
   */
  isDelegationValid(delegation: Delegation): boolean {
    if (!delegation.active) return false;
    if (delegation.expires_at) {
      const now = Date.now() * 1000;
      return now < delegation.expires_at;
    }
    return true;
  }
}

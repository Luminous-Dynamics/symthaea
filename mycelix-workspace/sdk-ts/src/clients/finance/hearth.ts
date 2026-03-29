// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * HEARTH (Commons Pool) Zome Client
 *
 * Community-governed resource pools implementing Commons Charter Article II, Section 2.
 *
 * Philosophy: "From each according to ability, to each according to need"
 * but with democratic accountability.
 *
 * Key Features:
 * - Anyone can contribute
 * - Community votes on allocation requests
 * - Transparent governance (all activity visible)
 *
 * @module @mycelix/sdk/clients/finance/hearth
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  CommonsPool,
  CreateCommonsPoolInput,
  PoolSummary,
  PoolContribution,
  ContributeToPoolInput,
  
  RequestAllocationInput,
  RequestSummary,
  AllocationVote,
  VoteInput,
} from './types.js';
import type { AppClient } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'hearth';

/**
 * HEARTH Client for Commons Pool governance
 *
 * HEARTH pools are community-governed resources where anyone can contribute
 * and the community votes on allocation requests.
 *
 * @example
 * ```typescript
 * const hearth = new HearthClient(client);
 *
 * // Create a commons pool
 * const pool = await hearth.createPool({
 *   name: 'Emergency Fund',
 *   description: 'Community emergency assistance',
 *   dao_did: 'did:mycelix:my-dao',
 *   resource_type: 'Money',
 * });
 *
 * // Contribute to pool
 * await hearth.contribute({
 *   pool_id: pool.id,
 *   amount: 100,
 *   message: 'Happy to help!',
 *   anonymous: false,
 * });
 *
 * // Request an allocation
 * await hearth.requestAllocation({
 *   pool_id: pool.id,
 *   amount: 50,
 *   justification: 'Medical emergency',
 *   need_category: 'Healthcare',
 * });
 *
 * // Vote on a request
 * await hearth.voteOnRequest({
 *   request_id: 'request:...',
 *   vote: true,
 *   comment: 'Verified need',
 * });
 * ```
 */
export class HearthClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Pool Management
  // ===========================================================================

  /**
   * Create a new commons pool
   */
  async createPool(input: CreateCommonsPoolInput): Promise<PoolSummary> {
    return this.callZomeOnce<PoolSummary>('create_pool', input);
  }

  /**
   * Get a specific pool
   */
  async getPool(poolId: string): Promise<CommonsPool | null> {
    return this.callZomeOrNull<CommonsPool>('get_pool', poolId);
  }

  /**
   * Get all pools for a DAO
   */
  async getDaoPools(daoDid: string): Promise<PoolSummary[]> {
    return this.callZome<PoolSummary[]>('get_dao_pools', daoDid);
  }

  // ===========================================================================
  // Contributions
  // ===========================================================================

  /**
   * Contribute to a commons pool
   */
  async contribute(input: ContributeToPoolInput): Promise<PoolContribution> {
    return this.callZomeOnce<PoolContribution>('contribute', input);
  }

  /**
   * Get all contributions to a pool
   */
  async getPoolContributions(poolId: string): Promise<PoolContribution[]> {
    return this.callZome<PoolContribution[]>('get_pool_contributions', poolId);
  }

  /**
   * Get my contributions across all pools
   */
  async getMyContributions(): Promise<PoolContribution[]> {
    return this.callZome<PoolContribution[]>('get_my_contributions', null);
  }

  /**
   * Get commons contribution for reputation calculation
   *
   * Commons contribution factors into reputation with max 5% weight.
   */
  async getCommonsReputationInput(memberDid: string): Promise<number> {
    return this.callZome<number>('get_commons_reputation_input', memberDid);
  }

  // ===========================================================================
  // Allocation Requests
  // ===========================================================================

  /**
   * Request an allocation from a pool
   *
   * Creates a request that the community votes on.
   */
  async requestAllocation(input: RequestAllocationInput): Promise<RequestSummary> {
    return this.callZomeOnce<RequestSummary>('request_allocation', input);
  }

  /**
   * Get all pending requests for a pool
   */
  async getPoolRequests(poolId: string): Promise<RequestSummary[]> {
    return this.callZome<RequestSummary[]>('get_pool_requests', poolId);
  }

  // ===========================================================================
  // Voting
  // ===========================================================================

  /**
   * Vote on an allocation request
   */
  async voteOnRequest(input: VoteInput): Promise<AllocationVote> {
    return this.callZomeOnce<AllocationVote>('vote_on_request', input);
  }

  /**
   * Get all votes for a request
   */
  async getRequestVotes(requestId: string): Promise<AllocationVote[]> {
    return this.callZome<AllocationVote[]>('get_request_votes', requestId);
  }

  /**
   * Finalize a request (check if voting period ended and apply result)
   */
  async finalizeRequest(requestId: string): Promise<RequestSummary> {
    return this.callZomeOnce<RequestSummary>('finalize_request', requestId);
  }
}

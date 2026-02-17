/**
 * Treasury Zome Client
 *
 * Treasury management for DAOs and organizations.
 *
 * @module @mycelix/sdk/clients/finance/treasury
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  Treasury,
  CreateTreasuryInput,
  Contribution,
  ContributeInput,
  Allocation,
  
  ProposeAllocationInput,
  ApproveAllocationInput,
  RejectAllocationInput,
  CancelAllocationInput,
  AddManagerInput,
  RemoveManagerInput,
  AllocationStatusQuery,
  SavingsPool,
  CreatePoolInput,
  JoinPoolInput,
  PoolContributionInput,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'treasury';

/**
 * Treasury Client for DAO treasury management
 *
 * @example
 * ```typescript
 * const treasury = new TreasuryClient(client);
 *
 * // Create a treasury
 * const t = await treasury.createTreasury({
 *   name: 'Community Fund',
 *   description: 'Main community treasury',
 *   currency: 'MYC',
 *   reserve_ratio: 0.2,
 *   managers: ['did:mycelix:alice', 'did:mycelix:bob'],
 * });
 *
 * // Contribute to treasury
 * await treasury.contribute({
 *   treasury_id: t.id,
 *   contributor_did: 'did:mycelix:carol',
 *   amount: 1000,
 *   currency: 'MYC',
 *   contribution_type: 'Donation',
 * });
 * ```
 */
export class TreasuryClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Treasury Management
  // ===========================================================================

  /**
   * Create a new treasury
   */
  async createTreasury(input: CreateTreasuryInput): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('create_treasury', input);
    return this.extractEntry<Treasury>(record);
  }

  /**
   * Get a treasury by ID
   */
  async getTreasury(treasuryId: string): Promise<Treasury | null> {
    return this.callZomeOrNull<Treasury>('get_treasury', treasuryId);
  }

  /**
   * Get all treasuries managed by a DID
   */
  async getManagerTreasuries(managerDid: string): Promise<Treasury[]> {
    const records = await this.callZome<HolochainRecord[]>('get_manager_treasuries', managerDid);
    return records.map((r) => this.extractEntry<Treasury>(r));
  }

  /**
   * Add a manager to treasury
   */
  async addManager(input: AddManagerInput): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('add_manager', input);
    return this.extractEntry<Treasury>(record);
  }

  /**
   * Remove a manager from treasury
   */
  async removeManager(input: RemoveManagerInput): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('remove_manager', input);
    return this.extractEntry<Treasury>(record);
  }

  // ===========================================================================
  // Contributions
  // ===========================================================================

  /**
   * Contribute to a treasury
   */
  async contribute(input: ContributeInput): Promise<Contribution> {
    const record = await this.callZomeOnce<HolochainRecord>('contribute', input);
    return this.extractEntry<Contribution>(record);
  }

  /**
   * Get all contributions for a treasury
   */
  async getTreasuryContributions(treasuryId: string): Promise<Contribution[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_treasury_contributions',
      treasuryId
    );
    return records.map((r) => this.extractEntry<Contribution>(r));
  }

  /**
   * Get contributor's contribution history
   */
  async getContributorHistory(contributorDid: string): Promise<Contribution[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_contributor_history',
      contributorDid
    );
    return records.map((r) => this.extractEntry<Contribution>(r));
  }

  // ===========================================================================
  // Allocations
  // ===========================================================================

  /**
   * Propose an allocation from treasury
   */
  async proposeAllocation(input: ProposeAllocationInput): Promise<Allocation> {
    const record = await this.callZomeOnce<HolochainRecord>('propose_allocation', input);
    return this.extractEntry<Allocation>(record);
  }

  /**
   * Approve an allocation (manager only)
   */
  async approveAllocation(input: ApproveAllocationInput): Promise<Allocation> {
    const record = await this.callZomeOnce<HolochainRecord>('approve_allocation', input);
    return this.extractEntry<Allocation>(record);
  }

  /**
   * Reject an allocation (manager only)
   */
  async rejectAllocation(input: RejectAllocationInput): Promise<Allocation> {
    const record = await this.callZomeOnce<HolochainRecord>('reject_allocation', input);
    return this.extractEntry<Allocation>(record);
  }

  /**
   * Cancel a proposed allocation
   */
  async cancelAllocation(input: CancelAllocationInput): Promise<Allocation> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_allocation', input);
    return this.extractEntry<Allocation>(record);
  }

  /**
   * Execute an approved allocation
   */
  async executeAllocation(allocationId: string): Promise<Allocation> {
    const record = await this.callZomeOnce<HolochainRecord>('execute_allocation', allocationId);
    return this.extractEntry<Allocation>(record);
  }

  /**
   * Get all allocations for a treasury
   */
  async getTreasuryAllocations(treasuryId: string): Promise<Allocation[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_treasury_allocations',
      treasuryId
    );
    return records.map((r) => this.extractEntry<Allocation>(r));
  }

  /**
   * Get allocations by status
   */
  async getAllocationsByStatus(input: AllocationStatusQuery): Promise<Allocation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_allocations_by_status', input);
    return records.map((r) => this.extractEntry<Allocation>(r));
  }

  // ===========================================================================
  // Savings Pools
  // ===========================================================================

  /**
   * Create a savings pool within a treasury
   */
  async createSavingsPool(input: CreatePoolInput): Promise<SavingsPool> {
    const record = await this.callZomeOnce<HolochainRecord>('create_savings_pool', input);
    return this.extractEntry<SavingsPool>(record);
  }

  /**
   * Get a savings pool by ID
   */
  async getSavingsPool(poolId: string): Promise<SavingsPool | null> {
    return this.callZomeOrNull<SavingsPool>('get_savings_pool', poolId);
  }

  /**
   * Get all savings pools for a treasury
   */
  async getTreasuryPools(treasuryId: string): Promise<SavingsPool[]> {
    const records = await this.callZome<HolochainRecord[]>('get_treasury_pools', treasuryId);
    return records.map((r) => this.extractEntry<SavingsPool>(r));
  }

  /**
   * Join a savings pool
   */
  async joinSavingsPool(input: JoinPoolInput): Promise<SavingsPool> {
    const record = await this.callZomeOnce<HolochainRecord>('join_savings_pool', input);
    return this.extractEntry<SavingsPool>(record);
  }

  /**
   * Contribute to a savings pool
   */
  async contributeToPool(input: PoolContributionInput): Promise<SavingsPool> {
    const record = await this.callZomeOnce<HolochainRecord>('contribute_to_pool', input);
    return this.extractEntry<SavingsPool>(record);
  }

  /**
   * Get all pools a member belongs to
   */
  async getMemberPools(memberDid: string): Promise<SavingsPool[]> {
    const records = await this.callZome<HolochainRecord[]>('get_member_pools', memberDid);
    return records.map((r) => this.extractEntry<SavingsPool>(r));
  }
}

/**
 * Treasury Zome Client
 *
 * Handles community treasury and allocation operations.
 *
 * @module @mycelix/sdk/integrations/finance/zomes/treasury
 */

import { FinanceSdkError } from '../types';

import type {
  Treasury,
  CreateTreasuryInput,
  TreasuryAllocation,
  ProposeAllocationInput,
  AllocationStatus,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Treasury client
 */
export interface TreasuryClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: TreasuryClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Client for treasury management operations
 *
 * Treasuries are community pools that require governance approval
 * for allocations above certain thresholds.
 *
 * @example
 * ```typescript
 * const treasury = new TreasuryClient(holochainClient);
 *
 * // Create a DAO treasury
 * const daoTreasury = await treasury.createTreasury({
 *   id: 'community-treasury',
 *   name: 'Community Development Fund',
 *   description: 'Funds for community projects and initiatives',
 *   dao_id: 'community-dao',
 *   governance_threshold: 0.66,
 *   created_by: 'did:mycelix:founder',
 * });
 *
 * // Propose an allocation
 * const allocation = await treasury.proposeAllocation({
 *   id: 'alloc-001',
 *   treasury_id: 'community-treasury',
 *   purpose: 'Fund cooperative equipment purchase',
 *   amount: 5000,
 *   currency: 'MYC',
 *   recipient_did: 'did:mycelix:cooperative',
 *   proposal_id: 'prop-equipment-2024',
 * });
 * ```
 */
export class TreasuryClient {
  private readonly config: TreasuryClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<TreasuryClientConfig> = {}
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
      throw new FinanceSdkError(
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
      throw new FinanceSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Treasury Operations
  // ============================================================================

  /**
   * Create a community treasury
   *
   * @param input - Treasury creation parameters
   * @returns The created treasury
   */
  async createTreasury(input: CreateTreasuryInput): Promise<Treasury> {
    const record = await this.call<HolochainRecord>('create_treasury', input);
    return this.extractEntry<Treasury>(record);
  }

  /**
   * Get a treasury by ID
   *
   * @param treasuryId - Treasury identifier
   * @returns The treasury or null
   */
  async getTreasury(treasuryId: string): Promise<Treasury | null> {
    const record = await this.call<HolochainRecord | null>('get_treasury', treasuryId);
    if (!record) return null;
    return this.extractEntry<Treasury>(record);
  }

  /**
   * Get treasury by DAO ID
   *
   * @param daoId - DAO identifier
   * @returns The treasury or null
   */
  async getTreasuryByDao(daoId: string): Promise<Treasury | null> {
    const record = await this.call<HolochainRecord | null>('get_treasury_by_dao', daoId);
    if (!record) return null;
    return this.extractEntry<Treasury>(record);
  }

  /**
   * Get all treasuries
   *
   * @returns Array of treasuries
   */
  async getAllTreasuries(): Promise<Treasury[]> {
    const records = await this.call<HolochainRecord[]>('get_all_treasuries', null);
    return records.map(r => this.extractEntry<Treasury>(r));
  }

  /**
   * Get treasury balance
   *
   * @param treasuryId - Treasury identifier
   * @param currency - Currency code
   * @returns Balance amount
   */
  async getTreasuryBalance(treasuryId: string, currency: string): Promise<number> {
    return this.call<number>('get_treasury_balance', { treasury_id: treasuryId, currency });
  }

  /**
   * Deposit to treasury
   *
   * @param treasuryId - Treasury identifier
   * @param amount - Amount to deposit
   * @param currency - Currency code
   * @param depositorDid - Depositor's DID
   * @returns Updated balance
   */
  async deposit(
    treasuryId: string,
    amount: number,
    currency: string,
    depositorDid: string
  ): Promise<number> {
    return this.call<number>('deposit_to_treasury', {
      treasury_id: treasuryId,
      amount,
      currency,
      depositor_did: depositorDid,
    });
  }

  // ============================================================================
  // Allocation Operations
  // ============================================================================

  /**
   * Propose a treasury allocation
   *
   * @param input - Allocation proposal parameters
   * @returns The proposed allocation
   */
  async proposeAllocation(input: ProposeAllocationInput): Promise<TreasuryAllocation> {
    const record = await this.call<HolochainRecord>('propose_allocation', input);
    return this.extractEntry<TreasuryAllocation>(record);
  }

  /**
   * Get an allocation by ID
   *
   * @param allocationId - Allocation identifier
   * @returns The allocation or null
   */
  async getAllocation(allocationId: string): Promise<TreasuryAllocation | null> {
    const record = await this.call<HolochainRecord | null>('get_allocation', allocationId);
    if (!record) return null;
    return this.extractEntry<TreasuryAllocation>(record);
  }

  /**
   * Get allocations for a treasury
   *
   * @param treasuryId - Treasury identifier
   * @returns Array of allocations
   */
  async getAllocationsForTreasury(treasuryId: string): Promise<TreasuryAllocation[]> {
    const records = await this.call<HolochainRecord[]>(
      'get_allocations_for_treasury',
      treasuryId
    );
    return records.map(r => this.extractEntry<TreasuryAllocation>(r));
  }

  /**
   * Approve an allocation
   *
   * @param allocationId - Allocation identifier
   * @param approverDid - Approver's DID
   * @returns Updated allocation
   */
  async approveAllocation(
    allocationId: string,
    approverDid: string
  ): Promise<TreasuryAllocation> {
    const record = await this.call<HolochainRecord>('approve_allocation', {
      allocation_id: allocationId,
      approver_did: approverDid,
    });
    return this.extractEntry<TreasuryAllocation>(record);
  }

  /**
   * Execute an approved allocation
   *
   * @param allocationId - Allocation identifier
   * @returns Updated allocation
   */
  async executeAllocation(allocationId: string): Promise<TreasuryAllocation> {
    const record = await this.call<HolochainRecord>('execute_allocation', allocationId);
    return this.extractEntry<TreasuryAllocation>(record);
  }

  /**
   * Cancel an allocation
   *
   * @param allocationId - Allocation identifier
   * @param reason - Cancellation reason
   * @returns Updated allocation
   */
  async cancelAllocation(
    allocationId: string,
    reason: string
  ): Promise<TreasuryAllocation> {
    const record = await this.call<HolochainRecord>('cancel_allocation', {
      allocation_id: allocationId,
      reason,
    });
    return this.extractEntry<TreasuryAllocation>(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick allocation proposal
   *
   * @param treasuryId - Treasury identifier
   * @param purpose - Allocation purpose
   * @param amount - Amount
   * @param currency - Currency code
   * @param recipientDid - Recipient's DID
   * @returns The proposed allocation
   */
  async quickAllocation(
    treasuryId: string,
    purpose: string,
    amount: number,
    currency: string = 'MYC',
    recipientDid?: string
  ): Promise<TreasuryAllocation> {
    const id = `alloc-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.proposeAllocation({
      id,
      treasury_id: treasuryId,
      purpose,
      amount,
      currency,
      recipient_did: recipientDid,
    });
  }

  /**
   * Get allocation status description
   *
   * @param status - Allocation status
   * @returns Human-readable description
   */
  getStatusDescription(status: AllocationStatus): string {
    const descriptions: Record<AllocationStatus, string> = {
      Proposed: 'Allocation has been proposed and awaits voting',
      Voting: 'Allocation is currently being voted on',
      Approved: 'Allocation has been approved and can be executed',
      Executed: 'Allocation has been executed and funds transferred',
      Rejected: 'Allocation was rejected by governance',
      Cancelled: 'Allocation was cancelled',
    };
    return descriptions[status];
  }

  /**
   * Check if allocation can be executed
   *
   * @param allocation - The allocation
   * @param treasury - The treasury
   * @returns True if allocation can be executed
   */
  canExecute(allocation: TreasuryAllocation, treasury: Treasury): boolean {
    if (allocation.status !== 'Approved') return false;
    const approvalRate = allocation.approved_by.length; // simplified
    return approvalRate >= treasury.governance_threshold;
  }

  /**
   * Calculate total allocated (pending + executed)
   *
   * @param allocations - Array of allocations
   * @param currency - Currency to sum
   * @returns Total allocated amount
   */
  calculateTotalAllocated(
    allocations: TreasuryAllocation[],
    currency: string
  ): number {
    return allocations
      .filter(
        a =>
          a.currency === currency &&
          (a.status === 'Proposed' || a.status === 'Voting' || a.status === 'Approved')
      )
      .reduce((sum, a) => sum + a.amount, 0);
  }

  /**
   * Get pending allocations
   *
   * @param allocations - Array of allocations
   * @returns Allocations that are pending execution
   */
  getPendingAllocations(allocations: TreasuryAllocation[]): TreasuryAllocation[] {
    return allocations.filter(
      a => a.status === 'Proposed' || a.status === 'Voting' || a.status === 'Approved'
    );
  }
}

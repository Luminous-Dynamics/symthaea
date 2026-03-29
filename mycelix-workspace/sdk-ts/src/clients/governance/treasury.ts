// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Treasury Zome Client
 *
 * Handles DAO treasury management, allocations, and multi-sig operations
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/treasury
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Treasury,
  TreasuryBalance,
  TreasuryAllocation,
  CreateTreasuryInput,
  ProposeAllocationInput,
  AllocationStatus,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';
// Note: GovernanceError removed as unused

/**
 * Configuration for the Treasury client
 */
export interface TreasuryClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: TreasuryClientConfig = {
  roleName: 'governance',
};

/**
 * Client for Treasury operations
 *
 * Treasuries are DAO-controlled funds that require governance approval
 * for allocations above certain thresholds. Supports multi-sig operations
 * and discretionary spending limits.
 *
 * @example
 * ```typescript
 * import { TreasuryClient } from '@mycelix/sdk/clients/governance';
 *
 * const treasury = new TreasuryClient(appClient);
 *
 * // Create a DAO treasury
 * const daoTreasury = await treasury.createTreasury({
 *   daoId: 'uhCkkp...',
 *   name: 'Community Development Fund',
 *   description: 'Funds for community projects',
 *   governanceThreshold: 0.66,
 *   discretionaryLimit: 1000,
 *   multiSigThreshold: 3,
 *   multiSigSigners: ['did:mycelix:alice', 'did:mycelix:bob', 'did:mycelix:carol'],
 * });
 *
 * // Propose an allocation
 * const allocation = await treasury.proposeAllocation({
 *   treasuryId: daoTreasury.id,
 *   purpose: 'Equipment purchase for cooperative',
 *   amount: 5000,
 *   currency: 'MYC',
 *   recipientDid: 'did:mycelix:cooperative',
 * });
 *
 * // Check balance
 * const balance = await treasury.getBalance(daoTreasury.id, 'MYC');
 * ```
 */
export class TreasuryClient extends ZomeClient {
  protected readonly zomeName = 'treasury';

  constructor(client: AppClient, config: TreasuryClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Treasury CRUD Operations
  // ============================================================================

  /**
   * Create a DAO treasury
   *
   * @param input - Treasury creation parameters
   * @returns The created treasury
   */
  async createTreasury(input: CreateTreasuryInput): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('create_treasury', {
      dao_id: input.daoId,
      name: input.name,
      description: input.description,
      governance_threshold: input.governanceThreshold ?? 0.51,
      discretionary_limit: input.discretionaryLimit ?? 0,
      discretionary_period_hours: input.discretionaryPeriodHours ?? 168,
      multi_sig_threshold: input.multiSigThreshold ?? 1,
      multi_sig_signers: input.multiSigSigners ?? [],
    });
    return this.mapTreasury(record);
  }

  /**
   * Get a treasury by ID
   *
   * @param treasuryId - Treasury identifier
   * @returns The treasury or null
   */
  async getTreasury(treasuryId: ActionHash): Promise<Treasury | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_treasury', treasuryId);
    if (!record) return null;
    return this.mapTreasury(record);
  }

  /**
   * Get treasury by DAO ID
   *
   * @param daoId - DAO identifier
   * @returns The treasury or null
   */
  async getTreasuryByDAO(daoId: ActionHash): Promise<Treasury | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_treasury_by_dao', daoId);
    if (!record) return null;
    return this.mapTreasury(record);
  }

  /**
   * Update treasury settings
   *
   * Requires governance approval for threshold changes.
   *
   * @param treasuryId - Treasury identifier
   * @param updates - Fields to update
   * @returns Updated treasury
   */
  async updateTreasury(
    treasuryId: ActionHash,
    updates: {
      name?: string;
      description?: string;
      governanceThreshold?: number;
      discretionaryLimit?: number;
      discretionaryPeriodHours?: number;
    }
  ): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('update_treasury', {
      treasury_id: treasuryId,
      name: updates.name,
      description: updates.description,
      governance_threshold: updates.governanceThreshold,
      discretionary_limit: updates.discretionaryLimit,
      discretionary_period_hours: updates.discretionaryPeriodHours,
    });
    return this.mapTreasury(record);
  }

  /**
   * List all treasuries
   *
   * @returns Array of treasuries
   */
  async listTreasuries(): Promise<Treasury[]> {
    const records = await this.callZome<HolochainRecord[]>('list_treasuries', null);
    return records.map(r => this.mapTreasury(r));
  }

  // ============================================================================
  // Multi-Sig Management
  // ============================================================================

  /**
   * Add a multi-sig signer
   *
   * @param treasuryId - Treasury identifier
   * @param signerDid - New signer's DID
   * @returns Updated treasury
   */
  async addSigner(treasuryId: ActionHash, signerDid: string): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('add_multi_sig_signer', {
      treasury_id: treasuryId,
      signer_did: signerDid,
    });
    return this.mapTreasury(record);
  }

  /**
   * Remove a multi-sig signer
   *
   * @param treasuryId - Treasury identifier
   * @param signerDid - Signer to remove
   * @returns Updated treasury
   */
  async removeSigner(treasuryId: ActionHash, signerDid: string): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('remove_multi_sig_signer', {
      treasury_id: treasuryId,
      signer_did: signerDid,
    });
    return this.mapTreasury(record);
  }

  /**
   * Update multi-sig threshold
   *
   * @param treasuryId - Treasury identifier
   * @param threshold - New threshold
   * @returns Updated treasury
   */
  async updateMultiSigThreshold(
    treasuryId: ActionHash,
    threshold: number
  ): Promise<Treasury> {
    const record = await this.callZomeOnce<HolochainRecord>('update_multi_sig_threshold', {
      treasury_id: treasuryId,
      threshold,
    });
    return this.mapTreasury(record);
  }

  // ============================================================================
  // Balance Operations
  // ============================================================================

  /**
   * Get treasury balance
   *
   * @param treasuryId - Treasury identifier
   * @param currency - Currency code
   * @returns Balance details
   */
  async getBalance(treasuryId: ActionHash, currency: string): Promise<TreasuryBalance> {
    const result = await this.callZome<any>('get_treasury_balance', {
      treasury_id: treasuryId,
      currency,
    });
    return {
      treasuryId: result.treasury_id,
      currency: result.currency,
      available: result.available,
      locked: result.locked,
      total: result.total,
      updatedAt: result.updated_at,
    };
  }

  /**
   * Get all balances for a treasury
   *
   * @param treasuryId - Treasury identifier
   * @returns Array of balances by currency
   */
  async getAllBalances(treasuryId: ActionHash): Promise<TreasuryBalance[]> {
    const results = await this.callZome<any[]>('get_all_treasury_balances', treasuryId);
    return results.map(r => ({
      treasuryId: r.treasury_id,
      currency: r.currency,
      available: r.available,
      locked: r.locked,
      total: r.total,
      updatedAt: r.updated_at,
    }));
  }

  /**
   * Deposit funds to treasury
   *
   * @param treasuryId - Treasury identifier
   * @param amount - Amount to deposit
   * @param currency - Currency code
   * @returns New balance
   */
  async deposit(
    treasuryId: ActionHash,
    amount: number,
    currency: string
  ): Promise<TreasuryBalance> {
    const result = await this.callZomeOnce<any>('deposit_to_treasury', {
      treasury_id: treasuryId,
      amount,
      currency,
    });
    return {
      treasuryId: result.treasury_id,
      currency: result.currency,
      available: result.available,
      locked: result.locked,
      total: result.total,
      updatedAt: result.updated_at,
    };
  }

  // ============================================================================
  // Allocation Operations
  // ============================================================================

  /**
   * Propose a treasury allocation
   *
   * For amounts above discretionary limit, requires governance approval.
   *
   * @param input - Allocation proposal parameters
   * @returns The proposed allocation
   */
  async proposeAllocation(input: ProposeAllocationInput): Promise<TreasuryAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('propose_allocation', {
      treasury_id: input.treasuryId,
      purpose: input.purpose,
      amount: input.amount,
      currency: input.currency,
      recipient_did: input.recipientDid,
      recipient_address: input.recipientAddress,
      proposal_id: input.proposalId,
    });
    return this.mapAllocation(record);
  }

  /**
   * Get an allocation by ID
   *
   * @param allocationId - Allocation identifier
   * @returns The allocation or null
   */
  async getAllocation(allocationId: ActionHash): Promise<TreasuryAllocation | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_allocation', allocationId);
    if (!record) return null;
    return this.mapAllocation(record);
  }

  /**
   * Get allocations for a treasury
   *
   * @param treasuryId - Treasury identifier
   * @param statusFilter - Optional status filter
   * @returns Array of allocations
   */
  async getAllocations(
    treasuryId: ActionHash,
    statusFilter?: AllocationStatus
  ): Promise<TreasuryAllocation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_allocations_for_treasury', {
      treasury_id: treasuryId,
      status_filter: statusFilter,
    });
    return records.map(r => this.mapAllocation(r));
  }

  /**
   * Get pending allocations
   *
   * @param treasuryId - Treasury identifier
   * @returns Array of pending allocations
   */
  async getPendingAllocations(treasuryId: ActionHash): Promise<TreasuryAllocation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_pending_allocations', treasuryId);
    return records.map(r => this.mapAllocation(r));
  }

  /**
   * Approve an allocation (multi-sig)
   *
   * @param allocationId - Allocation identifier
   * @returns Updated allocation
   */
  async approveAllocation(allocationId: ActionHash): Promise<TreasuryAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('approve_allocation', allocationId);
    return this.mapAllocation(record);
  }

  /**
   * Execute an approved allocation
   *
   * @param allocationId - Allocation identifier
   * @returns Updated allocation with execution details
   */
  async executeAllocation(allocationId: ActionHash): Promise<TreasuryAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('execute_allocation', allocationId);
    return this.mapAllocation(record);
  }

  /**
   * Cancel an allocation
   *
   * @param allocationId - Allocation identifier
   * @param reason - Cancellation reason
   * @returns Updated allocation
   */
  async cancelAllocation(allocationId: ActionHash, reason: string): Promise<TreasuryAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_allocation', {
      allocation_id: allocationId,
      reason,
    });
    return this.mapAllocation(record);
  }

  /**
   * Reject an allocation (multi-sig rejection)
   *
   * @param allocationId - Allocation identifier
   * @param reason - Rejection reason
   * @returns Updated allocation
   */
  async rejectAllocation(allocationId: ActionHash, reason: string): Promise<TreasuryAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('reject_allocation', {
      allocation_id: allocationId,
      reason,
    });
    return this.mapAllocation(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick allocation (discretionary spending)
   *
   * For amounts within discretionary limit that don't need governance.
   *
   * @param treasuryId - Treasury identifier
   * @param purpose - Purpose description
   * @param amount - Amount
   * @param currency - Currency code
   * @param recipientDid - Recipient's DID
   * @returns The allocation
   */
  async quickAllocation(
    treasuryId: ActionHash,
    purpose: string,
    amount: number,
    currency: string,
    recipientDid: string
  ): Promise<TreasuryAllocation> {
    return this.proposeAllocation({
      treasuryId,
      purpose,
      amount,
      currency,
      recipientDid,
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
      Proposed: 'Allocation has been proposed and awaits approval',
      Voting: 'Allocation is being voted on by governance',
      Approved: 'Allocation has been approved and can be executed',
      Executed: 'Allocation has been executed and funds transferred',
      Rejected: 'Allocation was rejected',
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
    return allocation.approvedBy.length >= treasury.multiSigThreshold;
  }

  /**
   * Check if allocation requires governance
   *
   * @param amount - Allocation amount
   * @param treasury - The treasury
   * @returns True if governance approval required
   */
  requiresGovernance(amount: number, treasury: Treasury): boolean {
    return amount > treasury.discretionaryLimit;
  }

  /**
   * Calculate total allocated (pending + approved)
   *
   * @param allocations - Array of allocations
   * @param currency - Currency to sum
   * @returns Total allocated amount
   */
  calculateTotalAllocated(allocations: TreasuryAllocation[], currency: string): number {
    return allocations
      .filter(
        a =>
          a.currency === currency &&
          (a.status === 'Proposed' || a.status === 'Voting' || a.status === 'Approved')
      )
      .reduce((sum, a) => sum + a.amount, 0);
  }

  /**
   * Get available balance after pending allocations
   *
   * @param treasuryId - Treasury identifier
   * @param currency - Currency code
   * @returns Available balance
   */
  async getAvailableBalance(treasuryId: ActionHash, currency: string): Promise<number> {
    const balance = await this.getBalance(treasuryId, currency);
    return balance.available;
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to Treasury type
   */
  private mapTreasury(record: HolochainRecord): Treasury {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      daoId: entry.dao_id,
      name: entry.name,
      description: entry.description,
      governanceThreshold: entry.governance_threshold,
      discretionaryLimit: entry.discretionary_limit,
      discretionaryPeriodHours: entry.discretionary_period_hours,
      multiSigThreshold: entry.multi_sig_threshold,
      multiSigSigners: entry.multi_sig_signers,
      active: entry.active,
      createdAt: entry.created_at,
    };
  }

  /**
   * Map Holochain record to TreasuryAllocation type
   */
  private mapAllocation(record: HolochainRecord): TreasuryAllocation {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      treasuryId: entry.treasury_id,
      proposalId: entry.proposal_id,
      purpose: entry.purpose,
      amount: entry.amount,
      currency: entry.currency,
      recipientDid: entry.recipient_did,
      recipientAddress: entry.recipient_address,
      status: entry.status,
      approvedBy: entry.approved_by,
      executionTxHash: entry.execution_tx_hash,
      requestedAt: entry.requested_at,
      executedAt: entry.executed_at,
    };
  }
}

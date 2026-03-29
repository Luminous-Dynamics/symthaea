// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Execution Zome Client
 *
 * Handles timelock management, proposal execution, guardian vetoes,
 * and fund allocation for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/execution
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Timelock,
  FundAllocation,
  TimelockStatus,
  CreateTimelockInput,
  ExecuteTimelockInput,
  VetoTimelockInput,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Execution client
 */
export interface ExecutionClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: ExecutionClientConfig = {
  roleName: 'governance',
};

/**
 * Client for Execution operations
 *
 * Manages the timelock-based execution of passed proposals, including:
 * - Timelock creation and lifecycle (Pending → Ready → Executed/Cancelled/Failed)
 * - Guardian vetoes during timelock period
 * - Fund allocation and release
 *
 * @example
 * ```typescript
 * import { ExecutionClient } from '@mycelix/sdk/clients/governance';
 *
 * const execution = new ExecutionClient(appClient);
 *
 * // Create a timelock for a passed proposal
 * const timelock = await execution.createTimelock({
 *   proposalId: 'proposal:123',
 *   actions: JSON.stringify({ action: 'transfer', amount: 5000 }),
 *   delayHours: 48,
 * });
 *
 * // Later, execute the timelock
 * const result = await execution.executeTimelock({ timelockId: timelock.id });
 * ```
 */
export class ExecutionClient extends ZomeClient {
  protected readonly zomeName = 'execution';

  constructor(client: AppClient, config: ExecutionClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Timelock Operations
  // ============================================================================

  /**
   * Create a timelock for a passed proposal
   *
   * @param input - Timelock creation parameters
   * @returns The timelock record
   */
  async createTimelock(input: CreateTimelockInput): Promise<Timelock> {
    const record = await this.callZomeOnce<HolochainRecord>('create_timelock', {
      proposal_id: input.proposalId,
      actions: input.actions,
      delay_hours: input.delayHours,
    });
    return this.mapTimelock(record);
  }

  /**
   * Get timelock for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The timelock or null
   */
  async getProposalTimelock(proposalId: string): Promise<Timelock | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_proposal_timelock', proposalId);
    if (!record) return null;
    return this.mapTimelock(record);
  }

  /**
   * Mark a timelock as ready for execution (delay period expired)
   *
   * @param timelockId - Timelock identifier
   * @returns Updated timelock record
   */
  async markTimelockReady(timelockId: string): Promise<Timelock> {
    const record = await this.callZomeOnce<HolochainRecord>('mark_timelock_ready', {
      timelock_id: timelockId,
    });
    return this.mapTimelock(record);
  }

  /**
   * Execute a ready timelock
   *
   * @param input - Execution parameters
   * @returns The execution result record
   */
  async executeTimelock(input: ExecuteTimelockInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('execute_timelock', {
      timelock_id: input.timelockId,
    });
  }

  /**
   * Get all pending timelocks
   *
   * @returns Array of pending timelocks
   */
  async getPendingTimelocks(): Promise<Timelock[]> {
    const records = await this.callZome<HolochainRecord[]>('get_pending_timelocks', null);
    return records.map(r => this.mapTimelock(r));
  }

  // ============================================================================
  // Guardian Veto
  // ============================================================================

  /**
   * Veto a timelock (guardian action)
   *
   * @param input - Veto parameters
   * @returns The veto and cancelled timelock records
   */
  async vetoTimelock(input: VetoTimelockInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('veto_timelock', {
      timelock_id: input.timelockId,
      reason: input.reason,
    });
  }

  // ============================================================================
  // Fund Allocation
  // ============================================================================

  /**
   * Lock funds for a proposal execution
   *
   * @param proposalId - Proposal identifier
   * @param timelockId - Associated timelock
   * @param sourceAccount - Source account for funds
   * @param amount - Amount to lock
   * @param currency - Currency code
   * @returns The fund allocation record
   */
  async lockFunds(
    proposalId: string,
    timelockId: string,
    sourceAccount: string,
    amount: number,
    currency: string
  ): Promise<FundAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('lock_proposal_funds', {
      proposal_id: proposalId,
      timelock_id: timelockId,
      source_account: sourceAccount,
      amount,
      currency,
    });
    return this.mapFundAllocation(record);
  }

  /**
   * Release locked funds after successful execution
   *
   * @param proposalId - Proposal identifier
   * @returns Updated fund allocation record
   */
  async releaseFunds(proposalId: string): Promise<FundAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('release_locked_funds', {
      proposal_id: proposalId,
    });
    return this.mapFundAllocation(record);
  }

  /**
   * Refund locked funds (execution failed/cancelled/vetoed)
   *
   * @param proposalId - Proposal identifier
   * @returns Updated fund allocation record
   */
  async refundFunds(proposalId: string): Promise<FundAllocation> {
    const record = await this.callZomeOnce<HolochainRecord>('refund_locked_funds', {
      proposal_id: proposalId,
    });
    return this.mapFundAllocation(record);
  }

  /**
   * Get fund allocation for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The fund allocation or null
   */
  async getFundAllocation(proposalId: string): Promise<FundAllocation | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_fund_allocation', proposalId);
    if (!record) return null;
    return this.mapFundAllocation(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get timelock status description
   */
  getStatusDescription(status: TimelockStatus): string {
    const descriptions: Record<TimelockStatus, string> = {
      Pending: 'Timelock delay period is active',
      Ready: 'Timelock delay expired, ready for execution',
      Executed: 'Proposal actions have been executed',
      Cancelled: 'Timelock was cancelled (e.g., via veto)',
      Failed: 'Execution failed',
    };
    return descriptions[status];
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapTimelock(record: HolochainRecord): Timelock {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      proposalId: entry.proposal_id,
      actions: entry.actions,
      started: entry.started,
      expires: entry.expires,
      status: entry.status,
      cancellationReason: entry.cancellation_reason,
    };
  }

  private mapFundAllocation(record: HolochainRecord): FundAllocation {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      proposalId: entry.proposal_id,
      timelockId: entry.timelock_id,
      sourceAccount: entry.source_account,
      amount: entry.amount,
      currency: entry.currency,
      lockedAt: entry.locked_at,
      status: entry.status,
      statusReason: entry.status_reason,
    };
  }
}

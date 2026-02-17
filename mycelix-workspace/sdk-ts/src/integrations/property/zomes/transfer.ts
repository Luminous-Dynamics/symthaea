/**
 * Transfer Zome Client
 *
 * Handles ownership transfers and title management.
 *
 * @module @mycelix/sdk/integrations/property/zomes/transfer
 */

import { PropertySdkError } from '../types';

import type {
  Transfer,
  TransferStatus,
  InitiateTransferInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Transfer client
 */
export interface TransferClientConfig {
  /** Role ID for the property DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: TransferClientConfig = {
  roleId: 'property',
  zomeName: 'property',
};

/**
 * Client for ownership transfer operations
 *
 * @example
 * ```typescript
 * const transfers = new TransferClient(holochainClient);
 *
 * // Initiate a transfer
 * const transfer = await transfers.initiateTransfer({
 *   asset_id: 'property-001',
 *   to_owner: 'did:mycelix:buyer',
 *   percentage: 100,
 *   consideration: 500000,
 *   currency: 'USD',
 *   use_escrow: true,
 * });
 *
 * // Accept the transfer (as buyer)
 * await transfers.acceptTransfer(transfer.id);
 *
 * // Complete the transfer (after escrow)
 * await transfers.completeTransfer(transfer.id);
 * ```
 */
export class TransferClient {
  private readonly config: TransferClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<TransferClientConfig> = {}
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
      throw new PropertySdkError(
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
      throw new PropertySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Transfer Operations
  // ============================================================================

  /**
   * Initiate a transfer
   *
   * @param input - Transfer parameters
   * @returns The created transfer
   */
  async initiateTransfer(input: InitiateTransferInput): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('initiate_transfer', input);
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Get a transfer by ID
   *
   * @param transferId - Transfer identifier
   * @returns The transfer or null
   */
  async getTransfer(transferId: string): Promise<Transfer | null> {
    const record = await this.call<HolochainRecord | null>('get_transfer', transferId);
    if (!record) return null;
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Accept a transfer (as recipient)
   *
   * @param transferId - Transfer identifier
   * @returns Updated transfer
   */
  async acceptTransfer(transferId: string): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('accept_transfer', transferId);
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Approve a transfer (for joint ownership)
   *
   * @param transferId - Transfer identifier
   * @param approverDid - Approver's DID
   * @returns Updated transfer
   */
  async approveTransfer(transferId: string, approverDid: string): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('approve_transfer', {
      transfer_id: transferId,
      approver_did: approverDid,
    });
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Complete a transfer (after all approvals and escrow)
   *
   * @param transferId - Transfer identifier
   * @returns Completed transfer
   */
  async completeTransfer(transferId: string): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('complete_transfer', transferId);
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Cancel a transfer
   *
   * @param transferId - Transfer identifier
   * @param reason - Cancellation reason
   * @returns Cancelled transfer
   */
  async cancelTransfer(transferId: string, reason?: string): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('cancel_transfer', {
      transfer_id: transferId,
      reason,
    });
    return this.extractEntry<Transfer>(record);
  }

  /**
   * Dispute a transfer
   *
   * @param transferId - Transfer identifier
   * @param disputantDid - Disputant's DID
   * @param reason - Dispute reason
   * @returns Disputed transfer
   */
  async disputeTransfer(
    transferId: string,
    disputantDid: string,
    reason: string
  ): Promise<Transfer> {
    const record = await this.call<HolochainRecord>('dispute_transfer', {
      transfer_id: transferId,
      disputant_did: disputantDid,
      reason,
    });
    return this.extractEntry<Transfer>(record);
  }

  // ============================================================================
  // Query Operations
  // ============================================================================

  /**
   * Get transfers for an asset
   *
   * @param assetId - Asset identifier
   * @returns Array of transfers
   */
  async getTransfersForAsset(assetId: string): Promise<Transfer[]> {
    const records = await this.call<HolochainRecord[]>('get_transfers_for_asset', assetId);
    return records.map(r => this.extractEntry<Transfer>(r));
  }

  /**
   * Get pending transfers for a DID
   *
   * @param did - DID to query
   * @returns Array of pending transfers
   */
  async getPendingTransfers(did: string): Promise<Transfer[]> {
    const records = await this.call<HolochainRecord[]>('get_pending_transfers', did);
    return records.map(r => this.extractEntry<Transfer>(r));
  }

  /**
   * Get transfers by status
   *
   * @param status - Transfer status
   * @param limit - Maximum results
   * @returns Array of transfers
   */
  async getTransfersByStatus(status: TransferStatus, limit: number = 50): Promise<Transfer[]> {
    const records = await this.call<HolochainRecord[]>('get_transfers_by_status', {
      status,
      limit,
    });
    return records.map(r => this.extractEntry<Transfer>(r));
  }

  /**
   * Get transfer history for a DID (as sender or recipient)
   *
   * @param did - DID to query
   * @returns Array of transfers
   */
  async getTransferHistory(did: string): Promise<Transfer[]> {
    const records = await this.call<HolochainRecord[]>('get_transfer_history', did);
    return records.map(r => this.extractEntry<Transfer>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Check if transfer can be completed
   *
   * @param transfer - The transfer
   * @returns True if transfer can be completed
   */
  canComplete(transfer: Transfer): boolean {
    return (
      transfer.status === 'Accepted' &&
      transfer.approvals.length >= transfer.required_approvals
    );
  }

  /**
   * Get transfer status description
   *
   * @param status - Transfer status
   * @returns Human-readable description
   */
  getStatusDescription(status: TransferStatus): string {
    const descriptions: Record<TransferStatus, string> = {
      Proposed: 'Transfer has been proposed and awaits acceptance',
      Pending: 'Transfer is pending additional approvals',
      Accepted: 'Transfer has been accepted by recipient',
      Completed: 'Transfer has been completed',
      Cancelled: 'Transfer was cancelled',
      Disputed: 'Transfer is under dispute resolution',
    };
    return descriptions[status];
  }

  /**
   * Check if DID needs to approve a transfer
   *
   * @param transfer - The transfer
   * @param did - DID to check
   * @returns True if DID needs to approve
   */
  needsApproval(transfer: Transfer, did: string): boolean {
    if (transfer.status !== 'Pending') return false;
    return !transfer.approvals.includes(did);
  }

  /**
   * Calculate remaining approvals needed
   *
   * @param transfer - The transfer
   * @returns Number of approvals still needed
   */
  remainingApprovals(transfer: Transfer): number {
    return Math.max(0, transfer.required_approvals - transfer.approvals.length);
  }
}

/**
 * Escrow Zome Client
 *
 * Handles escrow operations for secure transactions.
 *
 * @module @mycelix/sdk/integrations/finance/zomes/escrow
 */

import { FinanceSdkError } from '../types';

import type {
  Escrow,
  EscrowType,
  EscrowStatus,
  CreateEscrowInput,
  ReleaseEscrowInput,
  DisputeEscrowInput,
  DisputeResult,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Escrow client
 */
export interface EscrowClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: EscrowClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Client for escrow operations
 *
 * Escrows provide secure, condition-based fund holding with
 * optional arbiter support for dispute resolution.
 *
 * @example
 * ```typescript
 * const escrow = new EscrowClient(holochainClient);
 *
 * // Create a purchase escrow
 * const purchaseEscrow = await escrow.createEscrow({
 *   id: 'escrow-001',
 *   escrow_type: 'Purchase',
 *   depositor_did: 'did:mycelix:buyer',
 *   beneficiary_did: 'did:mycelix:seller',
 *   arbiter_did: 'did:mycelix:marketplace',
 *   amount: 1000,
 *   currency: 'MYC',
 *   conditions: [
 *     'Product delivered as described',
 *     'Buyer confirms receipt within 7 days',
 *   ],
 * });
 *
 * // Release escrow when conditions are met
 * await escrow.releaseEscrow({
 *   escrow_id: 'escrow-001',
 *   beneficiary_wallet: 'seller-wallet',
 *   amount: 1000,
 *   currency: 'MYC',
 *   released_by: 'did:mycelix:buyer',
 * });
 * ```
 */
export class EscrowClient {
  private readonly config: EscrowClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<EscrowClientConfig> = {}
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
  // Escrow Operations
  // ============================================================================

  /**
   * Create an escrow account
   *
   * @param input - Escrow creation parameters
   * @returns The created escrow
   */
  async createEscrow(input: CreateEscrowInput): Promise<Escrow> {
    const record = await this.call<HolochainRecord>('create_escrow', input);
    return this.extractEntry<Escrow>(record);
  }

  /**
   * Get an escrow by ID
   *
   * @param escrowId - Escrow identifier
   * @returns The escrow or null
   */
  async getEscrow(escrowId: string): Promise<Escrow | null> {
    const record = await this.call<HolochainRecord | null>('get_escrow', escrowId);
    if (!record) return null;
    return this.extractEntry<Escrow>(record);
  }

  /**
   * Get escrows by depositor
   *
   * @param depositorDid - Depositor's DID
   * @returns Array of escrows
   */
  async getEscrowsByDepositor(depositorDid: string): Promise<Escrow[]> {
    const records = await this.call<HolochainRecord[]>(
      'get_escrows_by_depositor',
      depositorDid
    );
    return records.map(r => this.extractEntry<Escrow>(r));
  }

  /**
   * Get escrows by beneficiary
   *
   * @param beneficiaryDid - Beneficiary's DID
   * @returns Array of escrows
   */
  async getEscrowsByBeneficiary(beneficiaryDid: string): Promise<Escrow[]> {
    const records = await this.call<HolochainRecord[]>(
      'get_escrows_by_beneficiary',
      beneficiaryDid
    );
    return records.map(r => this.extractEntry<Escrow>(r));
  }

  /**
   * Get escrows where DID is arbiter
   *
   * @param arbiterDid - Arbiter's DID
   * @returns Array of escrows
   */
  async getEscrowsByArbiter(arbiterDid: string): Promise<Escrow[]> {
    const records = await this.call<HolochainRecord[]>(
      'get_escrows_by_arbiter',
      arbiterDid
    );
    return records.map(r => this.extractEntry<Escrow>(r));
  }

  /**
   * Release escrow funds to beneficiary
   *
   * @param input - Release parameters
   * @returns Transaction record
   */
  async releaseEscrow(input: ReleaseEscrowInput): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>('release_escrow', input);
  }

  /**
   * Refund escrow funds to depositor
   *
   * @param escrowId - Escrow identifier
   * @param reason - Refund reason
   * @param refundedBy - DID of who authorized the refund
   * @returns Transaction record
   */
  async refundEscrow(
    escrowId: string,
    reason: string,
    refundedBy: string
  ): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>('refund_escrow', {
      escrow_id: escrowId,
      reason,
      refunded_by: refundedBy,
    });
  }

  /**
   * Dispute an escrow (routes to Justice hApp)
   *
   * @param input - Dispute parameters
   * @returns Dispute result
   */
  async disputeEscrow(input: DisputeEscrowInput): Promise<DisputeResult> {
    return this.call<DisputeResult>('dispute_escrow', input);
  }

  /**
   * Mark escrow conditions as met
   *
   * @param escrowId - Escrow identifier
   * @param verifiedBy - DID of who verified conditions
   * @returns Updated escrow
   */
  async markConditionsMet(escrowId: string, verifiedBy: string): Promise<Escrow> {
    const record = await this.call<HolochainRecord>('mark_conditions_met', {
      escrow_id: escrowId,
      verified_by: verifiedBy,
    });
    return this.extractEntry<Escrow>(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick escrow creation for purchases
   *
   * @param depositorDid - Buyer's DID
   * @param beneficiaryDid - Seller's DID
   * @param amount - Purchase amount
   * @param currency - Currency code
   * @param conditions - Release conditions
   * @param arbiterDid - Optional arbiter DID
   * @returns The created escrow
   */
  async createPurchaseEscrow(
    depositorDid: string,
    beneficiaryDid: string,
    amount: number,
    currency: string = 'MYC',
    conditions: string[] = ['Product delivered as described'],
    arbiterDid?: string
  ): Promise<Escrow> {
    const id = `escrow-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.createEscrow({
      id,
      escrow_type: 'Purchase',
      depositor_did: depositorDid,
      beneficiary_did: beneficiaryDid,
      arbiter_did: arbiterDid,
      amount,
      currency,
      conditions,
    });
  }

  /**
   * Create a milestone escrow (for project funding)
   *
   * @param depositorDid - Funder's DID
   * @param beneficiaryDid - Project owner's DID
   * @param amount - Milestone amount
   * @param currency - Currency code
   * @param milestoneDescription - Milestone conditions
   * @param arbiterDid - Optional arbiter DID
   * @returns The created escrow
   */
  async createMilestoneEscrow(
    depositorDid: string,
    beneficiaryDid: string,
    amount: number,
    currency: string = 'MYC',
    milestoneDescription: string,
    arbiterDid?: string
  ): Promise<Escrow> {
    const id = `milestone-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.createEscrow({
      id,
      escrow_type: 'Milestone',
      depositor_did: depositorDid,
      beneficiary_did: beneficiaryDid,
      arbiter_did: arbiterDid,
      amount,
      currency,
      conditions: [milestoneDescription],
    });
  }

  /**
   * Get escrow status description
   *
   * @param status - Escrow status
   * @returns Human-readable description
   */
  getStatusDescription(status: EscrowStatus): string {
    const descriptions: Record<EscrowStatus, string> = {
      Funded: 'Escrow is funded and awaiting condition fulfillment',
      ConditionsMet: 'Conditions have been met, awaiting release',
      Released: 'Funds have been released to beneficiary',
      Disputed: 'Escrow is under dispute resolution',
      Refunded: 'Funds have been refunded to depositor',
      Expired: 'Escrow has expired',
    };
    return descriptions[status];
  }

  /**
   * Get escrow type description
   *
   * @param type - Escrow type
   * @returns Human-readable description
   */
  getTypeDescription(type: EscrowType): string {
    const descriptions: Record<EscrowType, string> = {
      Purchase: 'Product or service purchase escrow',
      Service: 'Service delivery escrow',
      Loan: 'Loan collateral escrow',
      Dispute: 'Dispute resolution fund hold',
      Milestone: 'Project milestone payment escrow',
    };
    return descriptions[type];
  }

  /**
   * Check if escrow can be released
   *
   * @param escrow - The escrow
   * @returns True if escrow can be released
   */
  canRelease(escrow: Escrow): boolean {
    return escrow.status === 'Funded' || escrow.status === 'ConditionsMet';
  }

  /**
   * Check if escrow can be disputed
   *
   * @param escrow - The escrow
   * @returns True if escrow can be disputed
   */
  canDispute(escrow: Escrow): boolean {
    return escrow.status === 'Funded' || escrow.status === 'ConditionsMet';
  }

  /**
   * Check if escrow can be refunded
   *
   * @param escrow - The escrow
   * @returns True if escrow can be refunded
   */
  canRefund(escrow: Escrow): boolean {
    return escrow.status === 'Funded';
  }

  /**
   * Get all escrows for a DID (as depositor, beneficiary, or arbiter)
   *
   * @param did - DID to query
   * @returns Array of escrows
   */
  async getAllEscrowsForDid(did: string): Promise<Escrow[]> {
    const [depositor, beneficiary, arbiter] = await Promise.all([
      this.getEscrowsByDepositor(did),
      this.getEscrowsByBeneficiary(did),
      this.getEscrowsByArbiter(did),
    ]);

    // Deduplicate by ID
    const seen = new Set<string>();
    const result: Escrow[] = [];

    for (const escrow of [...depositor, ...beneficiary, ...arbiter]) {
      if (!seen.has(escrow.id)) {
        seen.add(escrow.id);
        result.push(escrow);
      }
    }

    return result;
  }

  /**
   * Get active escrows (not yet finalized)
   *
   * @param escrows - Array of escrows
   * @returns Active escrows
   */
  getActiveEscrows(escrows: Escrow[]): Escrow[] {
    return escrows.filter(
      e => e.status === 'Funded' || e.status === 'ConditionsMet' || e.status === 'Disputed'
    );
  }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finance Bridge Zome Client
 *
 * Cross-hApp communication for credit queries, payment processing,
 * and collateral management across the Mycelix ecosystem.
 *
 * @module @mycelix/sdk/clients/finance/bridge
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  CreditResult,
  QueryCreditInput,
  CrossHappPayment,
  ProcessPaymentInput,
  CollateralRegistration,
  RegisterBridgeCollateralInput,
  FinanceBridgeEvent,
  BroadcastFinanceEventInput,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'bridge';

/**
 * Finance Bridge Client for cross-hApp financial operations
 *
 * Enables other hApps to:
 * - Query credit scores
 * - Process payments
 * - Register collateral
 * - Subscribe to finance events
 *
 * @example
 * ```typescript
 * const bridge = new BridgeClient(client);
 *
 * // Query credit from another hApp
 * const credit = await bridge.queryCredit({
 *   did: 'did:mycelix:alice',
 *   source_happ: 'mycelix-marketplace',
 *   purpose: 'RiskAssessment',
 * });
 *
 * // Process a cross-hApp payment
 * await bridge.processPayment({
 *   source_happ: 'mycelix-marketplace',
 *   from_did: 'did:mycelix:buyer',
 *   to_did: 'did:mycelix:seller',
 *   amount: 100,
 *   currency: 'MYC',
 *   reference: 'order-123',
 * });
 * ```
 */
export class BridgeClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Credit Queries
  // ===========================================================================

  /**
   * Query credit score for a DID (from another hApp)
   */
  async queryCredit(input: QueryCreditInput): Promise<CreditResult> {
    return this.callZome<CreditResult>('query_credit', input);
  }

  // ===========================================================================
  // Cross-hApp Payments
  // ===========================================================================

  /**
   * Process a cross-hApp payment
   */
  async processPayment(input: ProcessPaymentInput): Promise<CrossHappPayment> {
    const record = await this.callZomeOnce<HolochainRecord>('process_payment', input);
    return this.extractEntry<CrossHappPayment>(record);
  }

  /**
   * Get payment history for a DID
   */
  async getPaymentHistory(did: string): Promise<CrossHappPayment[]> {
    const records = await this.callZome<HolochainRecord[]>('get_payment_history', did);
    return records.map((r) => this.extractEntry<CrossHappPayment>(r));
  }

  // ===========================================================================
  // Collateral Registration
  // ===========================================================================

  /**
   * Register collateral from another hApp
   */
  async registerCollateral(input: RegisterBridgeCollateralInput): Promise<CollateralRegistration> {
    const record = await this.callZomeOnce<HolochainRecord>('register_collateral', input);
    return this.extractEntry<CollateralRegistration>(record);
  }

  // ===========================================================================
  // Event Broadcasting
  // ===========================================================================

  /**
   * Broadcast a finance event to subscribers
   */
  async broadcastFinanceEvent(input: BroadcastFinanceEventInput): Promise<FinanceBridgeEvent> {
    const record = await this.callZomeOnce<HolochainRecord>('broadcast_finance_event', input);
    return this.extractEntry<FinanceBridgeEvent>(record);
  }
}

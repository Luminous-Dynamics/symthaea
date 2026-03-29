// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Payments Zome Client
 *
 * Payment processing and channel management for the Finance hApp.
 *
 * @module @mycelix/sdk/clients/finance/payments
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  Payment,
  SendPaymentInput,
  PaymentChannel,
  OpenChannelInput,
  ChannelTransferInput,
  Receipt,
  CreateEscrowPaymentInput,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'payments';

/**
 * Payments Client for direct payments and payment channels
 *
 * @example
 * ```typescript
 * const payments = new PaymentsClient(client);
 *
 * // Send a direct payment
 * const payment = await payments.sendPayment({
 *   from_did: 'did:mycelix:alice',
 *   to_did: 'did:mycelix:bob',
 *   amount: 100,
 *   currency: 'MYC',
 *   payment_type: 'Direct',
 * });
 *
 * // Open a payment channel
 * const channel = await payments.openPaymentChannel({
 *   party_a: 'did:mycelix:alice',
 *   party_b: 'did:mycelix:bob',
 *   currency: 'MYC',
 *   initial_deposit_a: 500,
 *   initial_deposit_b: 500,
 * });
 * ```
 */
export class PaymentsClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Direct Payments
  // ===========================================================================

  /**
   * Send a direct payment
   */
  async sendPayment(input: SendPaymentInput): Promise<Payment> {
    const record = await this.callZomeOnce<HolochainRecord>('send_payment', input);
    return this.extractEntry<Payment>(record);
  }

  /**
   * Get a specific payment by ID
   */
  async getPayment(paymentId: string): Promise<Payment | null> {
    return this.callZomeOrNull<Payment>('get_payment', paymentId);
  }

  /**
   * Get payment history for a DID (sent and received)
   */
  async getPaymentHistory(did: string): Promise<Payment[]> {
    const records = await this.callZome<HolochainRecord[]>('get_payment_history', did);
    return records.map((r) => this.extractEntry<Payment>(r));
  }

  /**
   * Get receipt for a payment
   */
  async getReceipt(paymentId: string): Promise<Receipt | null> {
    return this.callZomeOrNull<Receipt>('get_receipt', paymentId);
  }

  /**
   * Refund a payment (creates reverse payment)
   */
  async refundPayment(paymentId: string): Promise<Payment> {
    const record = await this.callZomeOnce<HolochainRecord>('refund_payment', paymentId);
    return this.extractEntry<Payment>(record);
  }

  // ===========================================================================
  // Payment Channels
  // ===========================================================================

  /**
   * Open a payment channel between two parties
   */
  async openPaymentChannel(input: OpenChannelInput): Promise<PaymentChannel> {
    const record = await this.callZomeOnce<HolochainRecord>('open_payment_channel', input);
    return this.extractEntry<PaymentChannel>(record);
  }

  /**
   * Transfer funds within a payment channel
   */
  async channelTransfer(input: ChannelTransferInput): Promise<PaymentChannel> {
    const record = await this.callZomeOnce<HolochainRecord>('channel_transfer', input);
    return this.extractEntry<PaymentChannel>(record);
  }

  /**
   * Close a payment channel (settle balances)
   */
  async closePaymentChannel(channelId: string): Promise<PaymentChannel> {
    const record = await this.callZomeOnce<HolochainRecord>('close_payment_channel', channelId);
    return this.extractEntry<PaymentChannel>(record);
  }

  /**
   * Get all channels for a party
   */
  async getChannels(did: string): Promise<PaymentChannel[]> {
    const records = await this.callZome<HolochainRecord[]>('get_channels', did);
    return records.map((r) => this.extractEntry<PaymentChannel>(r));
  }

  // ===========================================================================
  // Escrow Payments
  // ===========================================================================

  /**
   * Create an escrow payment (held until release)
   */
  async createEscrow(input: CreateEscrowPaymentInput): Promise<Payment> {
    const record = await this.callZomeOnce<HolochainRecord>('create_escrow', input);
    return this.extractEntry<Payment>(record);
  }

  /**
   * Release escrow to recipient
   */
  async releaseEscrow(paymentId: string): Promise<Payment> {
    const record = await this.callZomeOnce<HolochainRecord>('release_escrow', paymentId);
    return this.extractEntry<Payment>(record);
  }
}

/**
 * TEND (Time Exchange) Zome Client
 *
 * Interest-free mutual credit time banking implementing Commons Charter Article II.
 *
 * Key Principles:
 * - All hours are equal (doctor's hour = gardener's hour)
 * - Mutual credit: total always sums to zero
 * - Balance limits: +/- 40 TEND
 * - Interest-free exchanges
 *
 * @module @mycelix/sdk/clients/finance/tend
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  RecordExchangeInput,
  ExchangeRecord,
  BalanceInfo,
  GetBalanceInput,
  ServiceListing,
  CreateListingInput,
  ServiceRequest,
  CreateRequestInput,
} from './types.js';
import type { AppClient } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'tend';

/**
 * TEND Client for Time Exchange (mutual credit time banking)
 *
 * TEND enables interest-free time exchange between community members.
 * All hours are equal - a fundamental principle of time banking.
 *
 * @example
 * ```typescript
 * const tend = new TendClient(client);
 *
 * // Record a time exchange (as provider)
 * const exchange = await tend.recordExchange({
 *   receiver_did: 'did:mycelix:bob',
 *   hours: 2,
 *   service_description: 'Web development consultation',
 *   service_category: 'Technology',
 *   dao_did: 'did:mycelix:my-dao',
 * });
 *
 * // Receiver confirms the exchange
 * await tend.confirmExchange(exchange.id);
 *
 * // Check balance
 * const balance = await tend.getBalance({
 *   member_did: 'did:mycelix:alice',
 *   dao_did: 'did:mycelix:my-dao',
 * });
 * ```
 */
export class TendClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Exchanges
  // ===========================================================================

  /**
   * Record a time exchange
   *
   * Called by the PROVIDER after providing a service.
   * The exchange starts in "Proposed" status until the receiver confirms.
   *
   * Effect on balances (after confirmation):
   * - Provider: +hours (credit)
   * - Receiver: -hours (debt)
   *
   * @throws If exchange would exceed provider's +40 limit
   * @throws If exchange would exceed receiver's -40 limit
   */
  async recordExchange(input: RecordExchangeInput): Promise<ExchangeRecord> {
    return this.callZomeOnce<ExchangeRecord>('record_exchange', input);
  }

  /**
   * Confirm an exchange (called by receiver)
   *
   * This finalizes the exchange and updates both balances.
   */
  async confirmExchange(exchangeId: string): Promise<ExchangeRecord> {
    return this.callZomeOnce<ExchangeRecord>('confirm_exchange', exchangeId);
  }

  /**
   * Dispute an exchange
   *
   * Either party can dispute. Disputed exchanges are escalated for resolution.
   */
  async disputeExchange(exchangeId: string): Promise<ExchangeRecord> {
    return this.callZomeOnce<ExchangeRecord>('dispute_exchange', exchangeId);
  }

  /**
   * Cancel an exchange (provider only, while still Proposed)
   */
  async cancelExchange(exchangeId: string): Promise<ExchangeRecord> {
    return this.callZomeOnce<ExchangeRecord>('cancel_exchange', exchangeId);
  }

  /**
   * Get all exchanges for the caller in a DAO
   */
  async getMyExchanges(daoDid: string): Promise<ExchangeRecord[]> {
    return this.callZome<ExchangeRecord[]>('get_my_exchanges', daoDid);
  }

  // ===========================================================================
  // Balances
  // ===========================================================================

  /**
   * Get balance info for a member in a DAO
   *
   * Returns balance, whether they can provide (< +40) or receive (> -40),
   * and totals for provided/received hours.
   */
  async getBalance(input: GetBalanceInput): Promise<BalanceInfo> {
    return this.callZome<BalanceInfo>('get_balance', input);
  }

  /**
   * Get TEND activity for reputation calculation
   *
   * Returns normalized value with max 5% weight per Commons Charter.
   */
  async getTendReputationInput(input: GetBalanceInput): Promise<number> {
    return this.callZome<number>('get_tend_reputation_input', input);
  }

  // ===========================================================================
  // Service Marketplace
  // ===========================================================================

  /**
   * Create a service listing (offer to help)
   */
  async createListing(input: CreateListingInput): Promise<ServiceListing> {
    return this.callZomeOnce<ServiceListing>('create_listing', input);
  }

  /**
   * Get all active listings in a DAO
   */
  async getDaoListings(daoDid: string): Promise<ServiceListing[]> {
    return this.callZome<ServiceListing[]>('get_dao_listings', daoDid);
  }

  /**
   * Create a service request (ask for help)
   */
  async createRequest(input: CreateRequestInput): Promise<ServiceRequest> {
    return this.callZomeOnce<ServiceRequest>('create_request', input);
  }

  /**
   * Get all open requests in a DAO
   */
  async getDaoRequests(daoDid: string): Promise<ServiceRequest[]> {
    return this.callZome<ServiceRequest[]>('get_dao_requests', daoDid);
  }
}

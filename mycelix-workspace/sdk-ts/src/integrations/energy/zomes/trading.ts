/**
 * Trading Zome Client
 *
 * Handles P2P energy trading operations.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/trading
 */

import { EnergySdkError } from '../types';

import type {
  EnergyTrade,
  CreateTradeInput,
  TradeSearchParams,
  EnergySettlement,
  RequestPurchaseInput,
  EnergySource,
  TradeStatus,
  AvailableEnergy,
  AvailableEnergyQuery,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Trading client
 */
export interface TradingClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: TradingClientConfig = {
  roleId: 'energy',
  zomeName: 'trading',
};

/**
 * Client for energy trading operations
 *
 * @example
 * ```typescript
 * const trading = new TradingClient(holochainClient);
 *
 * // Create a trade offer
 * const trade = await trading.createTrade({
 *   buyer_did: 'did:mycelix:buyer',
 *   amount_kwh: 50,
 *   source: 'Solar',
 *   price_per_kwh: 0.12,
 *   currency: 'USD',
 * });
 *
 * // Accept the trade
 * await trading.acceptTrade(trade.id);
 *
 * // Confirm delivery
 * await trading.confirmDelivery(trade.id);
 * ```
 */
export class TradingClient {
  private readonly config: TradingClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<TradingClientConfig> = {}
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
      throw new EnergySdkError(
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
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Trade Operations
  // ============================================================================

  /**
   * Create a new trade offer
   *
   * @param input - Trade creation parameters
   * @returns The created trade
   */
  async createTrade(input: CreateTradeInput): Promise<EnergyTrade> {
    const record = await this.call<HolochainRecord>('create_trade', {
      ...input,
      status: 'Pending',
      created_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Get a trade by ID
   *
   * @param tradeId - Trade identifier
   * @returns The trade or null if not found
   */
  async getTrade(tradeId: string): Promise<EnergyTrade | null> {
    const record = await this.call<HolochainRecord | null>('get_trade', tradeId);
    if (!record) return null;
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Accept a trade offer
   *
   * @param tradeId - Trade identifier
   * @returns The updated trade
   */
  async acceptTrade(tradeId: string): Promise<EnergyTrade> {
    const record = await this.call<HolochainRecord>('accept_trade', tradeId);
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Confirm energy delivery
   *
   * @param tradeId - Trade identifier
   * @returns The updated trade
   */
  async confirmDelivery(tradeId: string): Promise<EnergyTrade> {
    const record = await this.call<HolochainRecord>('confirm_delivery', {
      trade_id: tradeId,
      delivered_at: Date.now() * 1000,
    });
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Cancel a pending trade
   *
   * @param tradeId - Trade identifier
   * @param reason - Cancellation reason
   * @returns The updated trade
   */
  async cancelTrade(tradeId: string, reason: string): Promise<EnergyTrade> {
    const record = await this.call<HolochainRecord>('cancel_trade', {
      trade_id: tradeId,
      reason,
    });
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Dispute a trade
   *
   * @param tradeId - Trade identifier
   * @param reason - Dispute reason
   * @returns The updated trade
   */
  async disputeTrade(tradeId: string, reason: string): Promise<EnergyTrade> {
    const record = await this.call<HolochainRecord>('dispute_trade', {
      trade_id: tradeId,
      reason,
    });
    return this.extractEntry<EnergyTrade>(record);
  }

  /**
   * Get trades by participant
   *
   * @param did - Participant's DID
   * @returns Array of trades
   */
  async getTradesByParticipant(did: string): Promise<EnergyTrade[]> {
    const records = await this.call<HolochainRecord[]>('get_trades_by_participant', did);
    return records.map(r => this.extractEntry<EnergyTrade>(r));
  }

  /**
   * Get trades by seller
   *
   * @param sellerDid - Seller's DID
   * @returns Array of trades
   */
  async getTradesBySeller(sellerDid: string): Promise<EnergyTrade[]> {
    const records = await this.call<HolochainRecord[]>('get_trades_by_seller', sellerDid);
    return records.map(r => this.extractEntry<EnergyTrade>(r));
  }

  /**
   * Get trades by buyer
   *
   * @param buyerDid - Buyer's DID
   * @returns Array of trades
   */
  async getTradesByBuyer(buyerDid: string): Promise<EnergyTrade[]> {
    const records = await this.call<HolochainRecord[]>('get_trades_by_buyer', buyerDid);
    return records.map(r => this.extractEntry<EnergyTrade>(r));
  }

  /**
   * Get open trade offers
   *
   * @param source - Optional filter by energy source
   * @returns Array of open trades
   */
  async getOpenTrades(source?: EnergySource): Promise<EnergyTrade[]> {
    const records = await this.call<HolochainRecord[]>('get_open_trades', source ?? null);
    return records.map(r => this.extractEntry<EnergyTrade>(r));
  }

  /**
   * Search trades
   *
   * @param params - Search parameters
   * @returns Array of matching trades
   */
  async searchTrades(params: TradeSearchParams): Promise<EnergyTrade[]> {
    const records = await this.call<HolochainRecord[]>('search_trades', params);
    return records.map(r => this.extractEntry<EnergyTrade>(r));
  }

  // ============================================================================
  // Cross-hApp Trading
  // ============================================================================

  /**
   * Query available energy in the network
   *
   * @param query - Query parameters
   * @returns Array of available energy offerings
   */
  async queryAvailableEnergy(query: AvailableEnergyQuery): Promise<AvailableEnergy[]> {
    return this.call<AvailableEnergy[]>('query_available_energy', query);
  }

  /**
   * Request energy purchase (creates settlement)
   *
   * @param input - Purchase request parameters
   * @returns The created settlement
   */
  async requestPurchase(input: RequestPurchaseInput): Promise<EnergySettlement> {
    return this.call<EnergySettlement>('request_energy_purchase', input);
  }

  /**
   * Confirm a settlement
   *
   * @param settlementId - Settlement identifier
   * @returns The updated settlement
   */
  async confirmSettlement(settlementId: string): Promise<EnergySettlement> {
    return this.call<EnergySettlement>('confirm_settlement', settlementId);
  }

  /**
   * Get pending settlements
   *
   * @param did - Participant's DID
   * @returns Array of pending settlements
   */
  async getPendingSettlements(did: string): Promise<EnergySettlement[]> {
    return this.call<EnergySettlement[]>('get_pending_settlements', did);
  }

  /**
   * Get settlement by ID
   *
   * @param settlementId - Settlement identifier
   * @returns The settlement or null
   */
  async getSettlement(settlementId: string): Promise<EnergySettlement | null> {
    return this.call<EnergySettlement | null>('get_settlement', settlementId);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick trade creation with auto-calculated total
   *
   * @param buyerDid - Buyer's DID
   * @param amountKwh - Amount in kWh
   * @param source - Energy source
   * @param pricePerKwh - Price per kWh
   * @param currency - Currency code
   * @returns The created trade
   */
  async quickTrade(
    buyerDid: string,
    amountKwh: number,
    source: EnergySource,
    pricePerKwh: number,
    currency: string = 'USD'
  ): Promise<EnergyTrade> {
    return this.createTrade({
      buyer_did: buyerDid,
      amount_kwh: amountKwh,
      source,
      price_per_kwh: pricePerKwh,
      currency,
    });
  }

  /**
   * Calculate trade total value
   *
   * @param trade - The trade
   * @returns Total value
   */
  calculateTradeTotal(trade: EnergyTrade): number {
    return trade.amount_kwh * trade.price_per_kwh;
  }

  /**
   * Get trade status description
   *
   * @param status - Trade status
   * @returns Human-readable description
   */
  getStatusDescription(status: TradeStatus): string {
    const descriptions: Record<TradeStatus, string> = {
      Pending: 'Trade offer is pending acceptance',
      Confirmed: 'Trade has been accepted and is awaiting delivery',
      Delivered: 'Energy has been delivered and trade is complete',
      Disputed: 'Trade is under dispute resolution',
      Cancelled: 'Trade has been cancelled',
    };
    return descriptions[status];
  }

  /**
   * Check if trade can be accepted
   *
   * @param trade - The trade
   * @returns True if trade can be accepted
   */
  canAccept(trade: EnergyTrade): boolean {
    return trade.status === 'Pending';
  }

  /**
   * Check if trade can be confirmed as delivered
   *
   * @param trade - The trade
   * @returns True if delivery can be confirmed
   */
  canConfirmDelivery(trade: EnergyTrade): boolean {
    return trade.status === 'Confirmed';
  }

  /**
   * Check if trade can be disputed
   *
   * @param trade - The trade
   * @returns True if trade can be disputed
   */
  canDispute(trade: EnergyTrade): boolean {
    return trade.status === 'Confirmed' || trade.status === 'Delivered';
  }

  /**
   * Check if trade can be cancelled
   *
   * @param trade - The trade
   * @returns True if trade can be cancelled
   */
  canCancel(trade: EnergyTrade): boolean {
    return trade.status === 'Pending';
  }

  /**
   * Get active trades for a participant
   *
   * @param did - Participant's DID
   * @returns Array of active trades
   */
  async getActiveTrades(did: string): Promise<EnergyTrade[]> {
    const trades = await this.getTradesByParticipant(did);
    return trades.filter(t => t.status === 'Pending' || t.status === 'Confirmed');
  }

  /**
   * Calculate total traded volume for a participant
   *
   * @param did - Participant's DID
   * @returns Total traded kWh
   */
  async getTotalTradedVolume(did: string): Promise<number> {
    const trades = await this.getTradesByParticipant(did);
    return trades
      .filter(t => t.status === 'Delivered')
      .reduce((sum, t) => sum + t.amount_kwh, 0);
  }
}

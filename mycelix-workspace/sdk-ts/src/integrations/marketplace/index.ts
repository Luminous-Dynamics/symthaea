// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Marketplace Integration
 *
 * hApp-specific adapter for Mycelix-Marketplace providing:
 * - Transaction reputation tracking for buyers and sellers
 * - Trust score calculation using Bayesian reputation model
 * - Listing verification with scam risk assessment
 * - Cross-hApp reputation aggregation via Bridge
 * - Federated Learning integration for collaborative trust models
 *
 * @packageDocumentation
 * @module integrations/marketplace
 * @see {@link MarketplaceReputationService} - Main service class
 * @see {@link getMarketplaceService} - Singleton accessor
 *
 * @example Basic transaction recording
 * ```typescript
 * import { getMarketplaceService } from '@mycelix/sdk/integrations/marketplace';
 *
 * const marketplace = getMarketplaceService();
 *
 * // Record successful transaction
 * const result = marketplace.recordTransaction({
 *   id: 'tx-001',
 *   type: 'purchase',
 *   buyerId: 'buyer-alice',
 *   sellerId: 'seller-bob',
 *   amount: 99.99,
 *   currency: 'USD',
 *   itemId: 'vintage-lamp',
 *   timestamp: Date.now(),
 *   success: true,
 * });
 *
 * console.log(`Seller score: ${result.sellerScore}`);
 * console.log(`Buyer score: ${result.buyerScore}`);
 * ```
 *
 * @example Verifying a listing before purchase
 * ```typescript
 * const verification = marketplace.verifyListing('listing-123', 'seller-bob');
 *
 * if (verification.verified) {
 *   console.log('Safe to purchase');
 * } else {
 *   console.log('Recommendations:', verification.recommendations);
 *   console.log('Scam risk:', verification.scamRiskScore);
 * }
 * ```
 */

import {
  LocalBridge,
  createReputationQuery,
} from '../../bridge/index.js';
import {
  FLCoordinator,
  AggregationMethod,
} from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';



// ============================================================================
// Marketplace-Specific Types
// ============================================================================

/**
 * Type of marketplace transaction
 *
 * @remarks
 * - `purchase`: Standard buy transaction
 * - `sale`: Direct sale by seller
 * - `exchange`: Barter/trade transaction
 * - `rental`: Temporary use agreement
 */
export type TransactionType = 'purchase' | 'sale' | 'exchange' | 'rental';

/**
 * Complete transaction record for reputation tracking
 *
 * @remarks
 * All transactions are recorded to build trust profiles for both parties.
 * The `success` field determines whether trust scores increase or decrease.
 *
 * @example
 * ```typescript
 * const tx: Transaction = {
 *   id: 'tx-001',
 *   type: 'purchase',
 *   buyerId: 'buyer-123',
 *   sellerId: 'seller-456',
 *   amount: 150.00,
 *   currency: 'USD',
 *   itemId: 'item-789',
 *   timestamp: Date.now(),
 *   success: true,
 * };
 * ```
 */
export interface Transaction {
  id: string;
  type: TransactionType;
  buyerId: string;
  sellerId: string;
  amount: number;
  currency: string;
  itemId: string;
  timestamp: number;
  success: boolean;
  disputeResolution?: 'buyer_favor' | 'seller_favor' | 'split' | 'none';
}

/** Seller profile with trust metrics */
export interface SellerProfile {
  sellerId: string;
  reputation: ReputationScore;
  trustScore: number;
  transactionCount: number;
  successRate: number;
  lastActive: number;
  verified: boolean;
}

/** Buyer profile with trust metrics */
export interface BuyerProfile {
  buyerId: string;
  reputation: ReputationScore;
  trustScore: number;
  transactionCount: number;
  paymentReliability: number;
  disputeRate: number;
  lastActive: number;
}

/** Listing verification result */
export interface ListingVerification {
  listingId: string;
  verified: boolean;
  sellerTrust: number;
  scamRiskScore: number;
  recommendations: string[];
}

// ============================================================================
// Marketplace Reputation Service
// ============================================================================

/**
 * MarketplaceReputationService - Trust and reputation for marketplace transactions
 *
 * @example
 * ```typescript
 * const marketplace = new MarketplaceReputationService();
 *
 * // Record a successful transaction
 * marketplace.recordTransaction({
 *   id: 'tx-123',
 *   type: 'purchase',
 *   buyerId: 'buyer-1',
 *   sellerId: 'seller-1',
 *   amount: 100,
 *   currency: 'USD',
 *   itemId: 'item-1',
 *   timestamp: Date.now(),
 *   success: true,
 * });
 *
 * // Check seller trust
 * const seller = marketplace.getSellerProfile('seller-1');
 * if (seller.trustScore >= 0.8) {
 *   // Safe to proceed
 * }
 * ```
 */
export class MarketplaceReputationService {
  private sellerReputations: Map<string, ReputationScore> = new Map();
  private buyerReputations: Map<string, ReputationScore> = new Map();
  private transactionCounts: Map<string, number> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('marketplace');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34,
    });
  }

  /**
   * Record a transaction outcome and update reputation scores
   *
   * @param transaction - The transaction to record
   * @returns Object containing updated seller and buyer scores
   *
   * @remarks
   * - Successful transactions increase trust scores
   * - Failed transactions decrease trust scores
   * - Scores are automatically stored in the Bridge for cross-hApp queries
   *
   * @example
   * ```typescript
   * const result = service.recordTransaction({
   *   id: 'tx-001',
   *   type: 'purchase',
   *   buyerId: 'buyer-1',
   *   sellerId: 'seller-1',
   *   amount: 100,
   *   currency: 'USD',
   *   itemId: 'item-1',
   *   timestamp: Date.now(),
   *   success: true,
   * });
   *
   * console.log(`Seller: ${result.sellerScore}, Buyer: ${result.buyerScore}`);
   * ```
   */
  recordTransaction(transaction: Transaction): {
    sellerScore: number;
    buyerScore: number;
  } {
    // Update seller reputation
    let sellerRep = this.sellerReputations.get(transaction.sellerId) || createReputation(transaction.sellerId);
    sellerRep = transaction.success ? recordPositive(sellerRep) : recordNegative(sellerRep);
    this.sellerReputations.set(transaction.sellerId, sellerRep);

    // Update buyer reputation
    let buyerRep = this.buyerReputations.get(transaction.buyerId) || createReputation(transaction.buyerId);
    buyerRep = transaction.success ? recordPositive(buyerRep) : recordNegative(buyerRep);
    this.buyerReputations.set(transaction.buyerId, buyerRep);

    // Track transaction counts
    const sellerCount = (this.transactionCounts.get(transaction.sellerId) || 0) + 1;
    const buyerCount = (this.transactionCounts.get(transaction.buyerId) || 0) + 1;
    this.transactionCounts.set(transaction.sellerId, sellerCount);
    this.transactionCounts.set(transaction.buyerId, buyerCount);

    // Store reputations in bridge for cross-hApp queries
    this.bridge.setReputation('marketplace', transaction.sellerId, sellerRep);
    this.bridge.setReputation('marketplace', transaction.buyerId, buyerRep);

    return {
      sellerScore: reputationValue(sellerRep),
      buyerScore: reputationValue(buyerRep),
    };
  }

  /**
   * Get comprehensive seller profile with trust metrics
   *
   * @param sellerId - Unique identifier for the seller
   * @returns Complete seller profile including reputation and verification status
   *
   * @remarks
   * - Sellers are marked as `verified` when they have:
   *   - Trust score >= 0.9
   *   - At least 10 completed transactions
   * - The `successRate` is calculated from all transaction history
   *
   * @example
   * ```typescript
   * const seller = service.getSellerProfile('seller-123');
   *
   * if (seller.verified && seller.successRate > 0.95) {
   *   console.log('Highly trusted seller');
   * }
   * ```
   */
  getSellerProfile(sellerId: string): SellerProfile {
    const reputation = this.sellerReputations.get(sellerId) || createReputation(sellerId);
    const transactionCount = this.transactionCounts.get(sellerId) || 0;
    const total = reputation.positiveCount + reputation.negativeCount;

    return {
      sellerId,
      reputation,
      trustScore: reputationValue(reputation),
      transactionCount,
      successRate: total > 0 ? reputation.positiveCount / total : 0,
      lastActive: Date.now(),
      verified: reputationValue(reputation) >= 0.9 && transactionCount >= 10,
    };
  }

  /**
   * Get buyer profile with trust metrics
   */
  getBuyerProfile(buyerId: string): BuyerProfile {
    const reputation = this.buyerReputations.get(buyerId) || createReputation(buyerId);
    const transactionCount = this.transactionCounts.get(buyerId) || 0;
    const total = reputation.positiveCount + reputation.negativeCount;

    return {
      buyerId,
      reputation,
      trustScore: reputationValue(reputation),
      transactionCount,
      paymentReliability: total > 0 ? reputation.positiveCount / total : 0,
      disputeRate: total > 0 ? reputation.negativeCount / total : 0,
      lastActive: Date.now(),
    };
  }

  /**
   * Verify a listing's trustworthiness before purchase
   *
   * @param listingId - Unique identifier for the listing
   * @param sellerId - Seller offering the listing
   * @returns Verification result with risk assessment and recommendations
   *
   * @remarks
   * The verification process:
   * 1. Retrieves seller's complete trust profile
   * 2. Calculates scam risk based on trust history
   * 3. Generates actionable recommendations
   *
   * A listing is considered `verified` when:
   * - Seller trust score >= 0.7
   * - Scam risk score < 0.3
   *
   * @example
   * ```typescript
   * const verification = service.verifyListing('listing-001', 'seller-123');
   *
   * if (!verification.verified) {
   *   console.log('Warning:', verification.recommendations.join(', '));
   *   console.log(`Scam risk: ${(verification.scamRiskScore * 100).toFixed(1)}%`);
   * }
   * ```
   */
  verifyListing(listingId: string, sellerId: string): ListingVerification {
    const seller = this.getSellerProfile(sellerId);
    const recommendations: string[] = [];

    if (seller.trustScore < 0.5) {
      recommendations.push('Seller has low trust score - proceed with caution');
    }

    const scamRiskScore = Math.max(0, (1 - seller.trustScore) * 0.7);
    if (scamRiskScore > 0.5) {
      recommendations.push('Elevated scam risk - verify seller identity');
    }

    const verified = seller.trustScore >= 0.7 && scamRiskScore < 0.3;
    if (verified) {
      recommendations.push('Listing verified - safe to proceed');
    }

    return {
      listingId,
      verified,
      sellerTrust: seller.trustScore,
      scamRiskScore,
      recommendations,
    };
  }

  /**
   * Check if seller is trustworthy
   */
  isSellerTrustworthy(sellerId: string, threshold = 0.7): boolean {
    const reputation = this.sellerReputations.get(sellerId);
    if (!reputation) return false;
    return isTrustworthy(reputation, threshold);
  }

  /**
   * Query seller reputation from other hApps
   */
  queryExternalReputation(sellerId: string): void {
    const query = createReputationQuery('marketplace', sellerId);
    this.bridge.send('mail', query);
    this.bridge.send('praxis', query);
  }

  /**
   * Get FL coordinator for reputation aggregation
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }

  /**
   * Get FL round statistics
   */
  getFLStats(): {
    totalRounds: number;
    participantCount: number;
    averageParticipation: number;
  } {
    const stats = this.flCoordinator.getRoundStats();
    return {
      totalRounds: stats.totalRounds,
      participantCount: stats.participantCount,
      averageParticipation: stats.averageParticipation,
    };
  }
}

// ============================================================================
// Bridge Client (Holochain Zome Calls)
// ============================================================================

import { type MycelixClient } from '../../client/index.js';

const MARKETPLACE_ROLE = 'marketplace';

/**
 * MarketplaceBridgeClient - Direct Holochain zome calls for marketplace operations
 *
 * Provides the same capabilities as MarketplaceReputationService but backed by the
 * Holochain conductor instead of in-memory Maps.
 */
export class MarketplaceBridgeClient {
  constructor(private client: MycelixClient) {}

  // --- transactions zome ---

  async createTransaction(input: {
    listing_hash: Uint8Array;
    seller: Uint8Array;
    amount: number;
    currency: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'create_transaction',
      payload: input,
    });
  }

  async getTransaction(transactionHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'get_transaction',
      payload: transactionHash,
    });
  }

  async getMyTransactions(): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'get_my_transactions',
      payload: null,
    });
  }

  async confirmTransaction(transactionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'confirm_transaction',
      payload: transactionHash,
    });
  }

  async markShipped(input: {
    transaction_hash: Uint8Array;
    tracking_info?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'mark_shipped',
      payload: input,
    });
  }

  async confirmDelivery(transactionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'confirm_delivery',
      payload: transactionHash,
    });
  }

  async completeTransaction(transactionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'complete_transaction',
      payload: transactionHash,
    });
  }

  async disputeTransaction(input: {
    transaction_hash: Uint8Array;
    reason: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'dispute_transaction',
      payload: input,
    });
  }

  async cancelTransaction(transactionHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'cancel_transaction',
      payload: transactionHash,
    });
  }

  async getListingTransactions(listingHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'transactions',
      fn_name: 'get_listing_transactions',
      payload: listingHash,
    });
  }

  // --- reputation zome ---

  async getAgentMatlScore(agent: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'get_agent_matl_score',
      payload: agent,
    });
  }

  async getAgentMatlScoreFast(agent: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'get_agent_matl_score_fast',
      payload: agent,
    });
  }

  async getCombinedReputation(agent: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'get_combined_reputation',
      payload: agent,
    });
  }

  async getCrossAppReputation(agent: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'get_cross_app_reputation',
      payload: agent,
    });
  }

  async updateMatlScore(input: {
    agent: Uint8Array;
    quality: number;
    consistency: number;
    reputation: number;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'update_matl_score',
      payload: input,
    });
  }

  async isByzantine(agent: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'is_byzantine',
      payload: agent,
    });
  }

  async submitReview(input: {
    seller: Uint8Array;
    transaction_hash: Uint8Array;
    rating: number;
    comment?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'submit_review',
      payload: input,
    });
  }

  async getSellerReviews(seller: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'reputation',
      fn_name: 'get_seller_reviews',
      payload: seller,
    });
  }

  // --- arbitration zome ---

  async fileDispute(input: {
    transaction_hash: Uint8Array;
    reason: string;
    evidence?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'arbitration',
      fn_name: 'file_dispute',
      payload: input,
    });
  }

  async submitArbitrationVote(input: {
    dispute_hash: Uint8Array;
    vote: string;
    reasoning?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'arbitration',
      fn_name: 'submit_arbitration_vote',
      payload: input,
    });
  }

  async finalizeArbitration(disputeHash: Uint8Array): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'arbitration',
      fn_name: 'finalize_arbitration',
      payload: disputeHash,
    });
  }

  async getArbitrationOpportunities(): Promise<Record<string, unknown>> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'arbitration',
      fn_name: 'get_arbitration_opportunities',
      payload: null,
    });
  }

  async getDispute(disputeHash: Uint8Array): Promise<Record<string, unknown> | null> {
    return this.client.callZome({
      role_name: MARKETPLACE_ROLE,
      zome_name: 'arbitration',
      fn_name: 'get_dispute',
      payload: disputeHash,
    });
  }
}

// Bridge client singleton
let marketplaceBridgeInstance: MarketplaceBridgeClient | null = null;

export function getMarketplaceBridgeClient(client: MycelixClient): MarketplaceBridgeClient {
  if (!marketplaceBridgeInstance) marketplaceBridgeInstance = new MarketplaceBridgeClient(client);
  return marketplaceBridgeInstance;
}

export function resetMarketplaceBridgeClient(): void {
  marketplaceBridgeInstance = null;
}

// ============================================================================
// Exports
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
};

// Default service instance
let defaultService: MarketplaceReputationService | null = null;

/**
 * Get the default marketplace reputation service instance
 */
export function getMarketplaceService(): MarketplaceReputationService {
  if (!defaultService) {
    defaultService = new MarketplaceReputationService();
  }
  return defaultService;
}

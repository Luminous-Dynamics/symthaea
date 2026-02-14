/**
 * Wallet Bridge Integration - Cross-hApp Reputation
 *
 * Integrates the wallet with the Cross-hApp Bridge for:
 * - Querying aggregate reputation across the ecosystem
 * - Subscribing to reputation updates
 * - Contributing finance-related reputation data
 *
 * This enables the wallet to display unified trust scores
 * that combine activity across all Mycelix hApps.
 *
 * @example
 * ```typescript
 * const walletBridge = createWalletBridge(financeProvider);
 *
 * // Get aggregate reputation for a contact
 * const rep = await walletBridge.getAggregateReputation('@alice');
 * console.log(`Trust score: ${rep.aggregatedScore}`);
 *
 * // Subscribe to reputation changes
 * walletBridge.onReputationChange('@alice', (newRep) => {
 *   updateUI(newRep);
 * });
 * ```
 */

import {
  type CrossHappBridge,
  getCrossHappBridge,
  type HappId,
  type ReputationQueryResponse,
  type BridgeMessage,
  aggregateReputationWithWeights,
} from '../bridge/cross-happ.js';
import { createReputation, recordPositive, recordNegative } from '../matl/index.js';

import type { HolochainFinanceProvider } from './finance-provider.js';
import type { ReputationScore } from '../matl/index.js';

// =============================================================================
// Types
// =============================================================================

/** Reputation change callback */
export type ReputationChangeHandler = (reputation: ReputationQueryResponse) => void;

/** Wallet bridge configuration */
export interface WalletBridgeConfig {
  /** Default hApps to include in reputation queries */
  defaultContextHapps?: HappId[];
  /** Weights for reputation aggregation */
  reputationWeights?: Partial<Record<HappId, number>>;
  /** Cache duration in ms (default: 60000) */
  cacheDurationMs?: number;
}

/** Cached reputation data */
interface CachedReputation {
  data: ReputationQueryResponse;
  timestamp: number;
}

// =============================================================================
// Wallet Bridge
// =============================================================================

/**
 * Wallet Bridge Integration
 *
 * Provides cross-hApp reputation services for the wallet.
 */
export class WalletBridge {
  private bridge: CrossHappBridge;
  private financeProvider?: HolochainFinanceProvider;
  private config: Required<WalletBridgeConfig>;
  private reputationCache = new Map<string, CachedReputation>();
  private reputationListeners = new Map<string, Set<ReputationChangeHandler>>();
  private subscriptionId?: string;
  private transactionWatcher?: () => void;

  constructor(config?: WalletBridgeConfig, financeProvider?: HolochainFinanceProvider) {
    this.bridge = getCrossHappBridge();
    this.financeProvider = financeProvider;
    this.config = {
      defaultContextHapps: config?.defaultContextHapps ?? [
        'finance',
        'identity',
        'marketplace',
        'governance',
      ],
      reputationWeights: config?.reputationWeights ?? {
        identity: 1.5,
        finance: 1.3,
        marketplace: 1.2,
        governance: 1.1,
        justice: 1.4,
      },
      cacheDurationMs: config?.cacheDurationMs ?? 60000,
    };

    // Register finance hApp with the bridge
    this.bridge.registerHapp('finance');

    // Subscribe to reputation updates
    this.subscriptionId = this.bridge.subscribe('finance', 'reputation_update', (message) =>
      this.handleReputationUpdate(message)
    );
  }

  // ===========================================================================
  // Reputation Queries
  // ===========================================================================

  /**
   * Get aggregate reputation for a DID across all relevant hApps.
   *
   * @param did - The DID to query
   * @param forceRefresh - Bypass cache and fetch fresh data
   * @returns Aggregated reputation response
   */
  async getAggregateReputation(
    did: string,
    forceRefresh = false
  ): Promise<ReputationQueryResponse> {
    // Check cache first
    const cached = this.reputationCache.get(did);
    if (!forceRefresh && cached && Date.now() - cached.timestamp < this.config.cacheDurationMs) {
      return cached.data;
    }

    // Query bridge for cross-hApp reputation
    const response = await this.bridge.queryReputation(did, this.config.defaultContextHapps);

    // Apply weighted aggregation
    response.aggregatedScore = aggregateReputationWithWeights(
      response.scores,
      this.config.reputationWeights
    );

    // Cache the result
    this.reputationCache.set(did, {
      data: response,
      timestamp: Date.now(),
    });

    return response;
  }

  /**
   * Get reputation for a specific hApp context.
   *
   * @param did - The DID to query
   * @param happId - The specific hApp to query
   */
  async getHappReputation(did: string, happId: HappId): Promise<number> {
    const response = await this.bridge.queryReputation(did, [happId]);
    return response.scores[happId] ?? 0.5;
  }

  /**
   * Check if a DID is trustworthy (above threshold).
   *
   * @param did - The DID to check
   * @param threshold - Minimum trust score (default: 0.6)
   */
  async isTrustworthy(did: string, threshold = 0.6): Promise<boolean> {
    const rep = await this.getAggregateReputation(did);
    return rep.aggregatedScore >= threshold && rep.confidence >= 0.5;
  }

  /**
   * Get trust recommendation for a transaction.
   *
   * @param recipientDid - The recipient DID
   * @param amount - Transaction amount
   * @returns Recommendation with risk level
   */
  async getTransactionRecommendation(
    recipientDid: string,
    amount: number
  ): Promise<TransactionRecommendation> {
    const rep = await this.getAggregateReputation(recipientDid);

    // Risk levels based on reputation and amount
    let riskLevel: 'low' | 'medium' | 'high' | 'very_high';
    let recommendation: string;
    let requiresVerification: boolean;

    if (rep.aggregatedScore >= 0.8 && rep.confidence >= 0.7) {
      riskLevel = 'low';
      recommendation = 'This recipient has excellent reputation across the ecosystem.';
      requiresVerification = false;
    } else if (rep.aggregatedScore >= 0.6 && rep.confidence >= 0.5) {
      riskLevel = 'medium';
      recommendation = 'This recipient has moderate reputation. Proceed with normal caution.';
      requiresVerification = amount > 1000;
    } else if (rep.aggregatedScore >= 0.4 || rep.confidence < 0.3) {
      riskLevel = 'high';
      recommendation = 'Limited reputation data available. Consider verifying identity.';
      requiresVerification = amount > 100;
    } else {
      riskLevel = 'very_high';
      recommendation = 'This recipient has poor reputation. Proceed with extreme caution.';
      requiresVerification = true;
    }

    return {
      recipientDid,
      reputation: rep,
      riskLevel,
      recommendation,
      requiresVerification,
    };
  }

  // ===========================================================================
  // Reputation Contributions
  // ===========================================================================

  /**
   * Report a positive interaction (successful payment).
   *
   * @param did - The DID of the counterparty
   * @param transactionId - Reference transaction ID
   */
  reportPositiveInteraction(did: string, transactionId: string): void {
    const reputation = this.getOrCreateLocalReputation(did);
    const updatedRep = recordPositive(reputation);

    this.bridge.updateReputation(did, 'finance', updatedRep);
    this.invalidateCache(did);

    // Broadcast the update
    void this.bridge.send({
      type: 'reputation_update',
      sourceHapp: 'finance',
      targetHapp: 'broadcast',
      payload: {
        did,
        transactionId,
        interactionType: 'positive',
        happContext: 'finance',
      },
    });
  }

  /**
   * Report a negative interaction (failed payment, dispute).
   *
   * @param did - The DID of the counterparty
   * @param reason - Reason for negative report
   */
  reportNegativeInteraction(
    did: string,
    reason: 'payment_failed' | 'dispute' | 'fraud_suspected'
  ): void {
    let reputation = this.getOrCreateLocalReputation(did);

    // Apply multiple negative records for severe cases
    const penaltyCount = reason === 'fraud_suspected' ? 3 : 1;
    for (let i = 0; i < penaltyCount; i++) {
      reputation = recordNegative(reputation);
    }

    this.bridge.updateReputation(did, 'finance', reputation);
    this.invalidateCache(did);

    // Broadcast the update
    void this.bridge.send({
      type: 'reputation_update',
      sourceHapp: 'finance',
      targetHapp: 'broadcast',
      payload: {
        did,
        reason,
        interactionType: 'negative',
        happContext: 'finance',
      },
    });
  }

  // ===========================================================================
  // Subscription Management
  // ===========================================================================

  /**
   * Subscribe to reputation changes for a specific DID.
   *
   * @param did - The DID to watch
   * @param handler - Callback for reputation changes
   * @returns Unsubscribe function
   */
  onReputationChange(did: string, handler: ReputationChangeHandler): () => void {
    let listeners = this.reputationListeners.get(did);
    if (!listeners) {
      listeners = new Set();
      this.reputationListeners.set(did, listeners);
    }
    listeners.add(handler);

    return () => {
      listeners?.delete(handler);
      if (listeners?.size === 0) {
        this.reputationListeners.delete(did);
      }
    };
  }

  /**
   * Watch all transactions and auto-update reputation.
   * Call this after the finance provider is connected.
   */
  watchTransactions(): void {
    if (!this.financeProvider || this.transactionWatcher) return;

    // Subscribe to state changes on the finance provider
    this.transactionWatcher = this.financeProvider.state$.subscribe((state) => {
      // Check for newly confirmed transactions
      for (const tx of state.transactions) {
        if (tx.status === 'confirmed') {
          // Report positive interaction for the counterparty
          const counterpartyDid =
            tx.direction === 'outgoing' ? tx.to.did : tx.from.did;
          if (counterpartyDid) {
            this.reportPositiveInteraction(counterpartyDid, tx.id);
          }
        }
      }
    }).unsubscribe;
  }

  // ===========================================================================
  // Verification Integration
  // ===========================================================================

  /**
   * Request identity verification from the identity hApp.
   *
   * @param did - The DID to verify
   */
  async requestIdentityVerification(did: string): Promise<VerificationStatus> {
    const response = await this.bridge.requestVerification('finance', 'identity', {
      subjectDid: did,
      verificationType: 'identity',
    });

    return {
      did,
      verified: response.verified,
      level: response.level,
      verifiedAt: response.verified ? Date.now() : undefined,
    };
  }

  /**
   * Request credential verification.
   *
   * @param did - The DID to verify
   * @param credentialType - Type of credential to check
   */
  async verifyCredential(did: string, credentialType: string): Promise<VerificationStatus> {
    const response = await this.bridge.requestVerification('finance', 'identity', {
      subjectDid: did,
      verificationType: 'credential',
      resource: credentialType,
    });

    return {
      did,
      verified: response.verified,
      level: response.level,
      credentialType,
      verifiedAt: response.verified ? Date.now() : undefined,
    };
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /**
   * Clean up resources.
   */
  destroy(): void {
    if (this.subscriptionId) {
      this.bridge.unsubscribe(this.subscriptionId);
    }
    if (this.transactionWatcher) {
      this.transactionWatcher();
    }
    this.reputationListeners.clear();
    this.reputationCache.clear();
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private handleReputationUpdate(message: BridgeMessage): Promise<void> {
    const payload = message.payload as { did?: string; score?: number };
    if (!payload.did) return Promise.resolve();

    // Invalidate cache
    this.invalidateCache(payload.did);

    // Notify listeners
    const listeners = this.reputationListeners.get(payload.did);
    if (listeners && listeners.size > 0) {
      // Fetch fresh data and notify
      this.getAggregateReputation(payload.did, true)
        .then((rep) => {
          for (const handler of listeners) {
            try {
              handler(rep);
            } catch (e) {
              console.error('Reputation handler error:', e);
            }
          }
        })
        .catch(console.error);
    }

    return Promise.resolve();
  }

  private invalidateCache(did: string): void {
    this.reputationCache.delete(did);
  }

  private getOrCreateLocalReputation(did: string): ReputationScore {
    // Start with a neutral reputation if none exists (uses Laplace smoothing)
    return createReputation(did);
  }
}

// =============================================================================
// Types
// =============================================================================

/** Transaction recommendation based on reputation */
export interface TransactionRecommendation {
  recipientDid: string;
  reputation: ReputationQueryResponse;
  riskLevel: 'low' | 'medium' | 'high' | 'very_high';
  recommendation: string;
  requiresVerification: boolean;
}

/** Verification status */
export interface VerificationStatus {
  did: string;
  verified: boolean;
  level: number;
  credentialType?: string;
  verifiedAt?: number;
}

// =============================================================================
// Factory
// =============================================================================

/**
 * Create a wallet bridge integration.
 *
 * @param financeProvider - Optional finance provider to watch transactions
 * @param config - Optional configuration
 */
export function createWalletBridge(
  financeProvider?: HolochainFinanceProvider,
  config?: WalletBridgeConfig
): WalletBridge {
  return new WalletBridge(config, financeProvider);
}

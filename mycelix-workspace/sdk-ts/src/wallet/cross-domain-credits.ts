// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-Domain Resource Credits System
 *
 * Manages fungible and non-fungible resource credits across all 5 civilizational domains.
 * Credits can be earned, spent, transferred, and converted between domains with
 * exchange rates reflecting real resource availability and demand.
 *
 * Key features:
 * - Multi-domain wallet with unified balance view
 * - Credit earning through contribution and participation
 * - Domain-specific exchange rates
 * - Transfer and redemption mechanisms
 * - Credit history and audit trail
 * - Community credit pooling
 *
 * @packageDocumentation
 * @module wallet/cross-domain-credits
 */

import type { ResourceDomain } from '../metabolism/cross-domain-dashboard.js';
import type { AgentPubKey } from '@holochain/client';

// ============================================================================
// Credit Types
// ============================================================================

/**
 * Type of resource credit
 */
export type CreditType =
  | 'contribution' // Earned through active contribution
  | 'participation' // Earned through participation
  | 'allocation' // Allocated by community
  | 'transfer' // Received from another agent
  | 'conversion' // Converted from another domain
  | 'subsidy' // Community/government subsidy
  | 'reward'; // Special reward or bonus

/**
 * Credit status
 */
export type CreditStatus =
  | 'active' // Can be used
  | 'pending' // Awaiting confirmation
  | 'reserved' // Reserved for pending transaction
  | 'expired' // Past expiry date
  | 'redeemed' // Already used
  | 'revoked'; // Revoked (fraud, error, etc.)

/**
 * A single credit entry
 */
export interface CreditEntry {
  /** Unique credit ID */
  creditId: string;
  /** Domain this credit applies to */
  domain: ResourceDomain;
  /** Amount of credits */
  amount: number;
  /** Type of credit */
  type: CreditType;
  /** Current status */
  status: CreditStatus;
  /** When the credit was created */
  createdAt: number;
  /** When the credit expires (0 = never) */
  expiresAt: number;
  /** Source of the credit */
  source: CreditSource;
  /** Metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Source of a credit
 */
export interface CreditSource {
  /** Type of source */
  type: 'happ' | 'bridge' | 'community' | 'transfer' | 'system';
  /** Source identifier (hApp name, community ID, etc.) */
  identifier: string;
  /** Agent who issued or transferred */
  agent?: AgentPubKey;
  /** Reference to original action/transaction */
  reference?: string;
}

// ============================================================================
// Wallet Types
// ============================================================================

/**
 * Balance for a single domain
 */
export interface DomainBalance {
  /** Domain */
  domain: ResourceDomain;
  /** Available balance (can be spent) */
  available: number;
  /** Reserved balance (pending transactions) */
  reserved: number;
  /** Total balance (available + reserved) */
  total: number;
  /** Pending incoming credits */
  pending: number;
  /** Credits expiring within 30 days */
  expiringIn30Days: number;
  /** Last activity timestamp */
  lastActivity: number;
}

/**
 * Complete wallet state
 */
export interface Wallet {
  /** Wallet owner */
  owner: AgentPubKey;
  /** Individual domain balances */
  balances: Record<ResourceDomain, DomainBalance>;
  /** Total credits across all domains (in standard units) */
  totalCredits: number;
  /** Credit entries */
  entries: CreditEntry[];
  /** Transaction history */
  transactions: CreditTransaction[];
  /** Wallet created timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
  /** Wallet tier/level */
  tier: WalletTier;
}

/**
 * Wallet tier (affects limits and features)
 */
export type WalletTier = 'basic' | 'standard' | 'premium' | 'community';

/**
 * Wallet summary for quick view
 */
export interface WalletSummary {
  /** Owner */
  owner: AgentPubKey;
  /** Total available credits (all domains) */
  totalAvailable: number;
  /** Domain with highest balance */
  primaryDomain: ResourceDomain;
  /** Domains with low balance */
  lowBalanceDomains: ResourceDomain[];
  /** Number of pending transactions */
  pendingTransactions: number;
  /** Tier */
  tier: WalletTier;
}

// ============================================================================
// Transaction Types
// ============================================================================

/**
 * Credit transaction
 */
export interface CreditTransaction {
  /** Transaction ID */
  transactionId: string;
  /** Transaction type */
  type: TransactionType;
  /** From wallet (null for system/earning) */
  from: AgentPubKey | null;
  /** To wallet (null for system/spending) */
  to: AgentPubKey | null;
  /** Domain */
  domain: ResourceDomain;
  /** Amount */
  amount: number;
  /** Fee (if any) */
  fee: number;
  /** Status */
  status: TransactionStatus;
  /** When initiated */
  initiatedAt: number;
  /** When completed (or null if pending) */
  completedAt: number | null;
  /** Description */
  description: string;
  /** Reference */
  reference?: string;
}

/**
 * Transaction type
 */
export type TransactionType =
  | 'earn' // Earned credits
  | 'spend' // Spent credits
  | 'transfer' // Transferred to another wallet
  | 'receive' // Received from another wallet
  | 'convert' // Converted between domains
  | 'redeem' // Redeemed for goods/services
  | 'pool' // Added to community pool
  | 'withdraw' // Withdrawn from community pool
  | 'expire' // Credits expired
  | 'adjust'; // Manual adjustment

/**
 * Transaction status
 */
export type TransactionStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'reversed';

// ============================================================================
// Exchange Types
// ============================================================================

/**
 * Exchange rate between domains
 */
export interface ExchangeRate {
  /** From domain */
  from: ResourceDomain;
  /** To domain */
  to: ResourceDomain;
  /** Rate (1 from = rate to) */
  rate: number;
  /** When the rate was set */
  timestamp: number;
  /** Rate volatility (0-1) */
  volatility: number;
  /** Minimum exchange amount */
  minAmount: number;
  /** Maximum exchange amount per transaction */
  maxAmount: number;
}

/**
 * Exchange quote
 */
export interface ExchangeQuote {
  /** Quote ID */
  quoteId: string;
  /** From domain */
  fromDomain: ResourceDomain;
  /** To domain */
  toDomain: ResourceDomain;
  /** Input amount */
  inputAmount: number;
  /** Output amount */
  outputAmount: number;
  /** Exchange rate used */
  rate: number;
  /** Fee amount */
  fee: number;
  /** Valid until */
  validUntil: number;
}

// ============================================================================
// Community Pool Types
// ============================================================================

/**
 * Community credit pool
 */
export interface CommunityPool {
  /** Pool ID */
  poolId: string;
  /** Pool name */
  name: string;
  /** Pool description */
  description: string;
  /** Administrators */
  administrators: AgentPubKey[];
  /** Members who can withdraw */
  members: AgentPubKey[];
  /** Domain balances */
  balances: Record<ResourceDomain, number>;
  /** Total contributions */
  totalContributions: number;
  /** Total distributions */
  totalDistributions: number;
  /** Distribution rules */
  distributionRules: DistributionRule[];
  /** Created timestamp */
  createdAt: number;
}

/**
 * Distribution rule for community pool
 */
export interface DistributionRule {
  /** Rule ID */
  ruleId: string;
  /** Rule name */
  name: string;
  /** Domain this rule applies to */
  domain: ResourceDomain;
  /** Trigger condition */
  trigger: 'manual' | 'periodic' | 'threshold' | 'event';
  /** Amount or percentage to distribute */
  amount: number;
  /** Is amount a percentage */
  isPercentage: boolean;
  /** Recipient filter */
  recipientFilter?: string;
  /** Is rule active */
  active: boolean;
}

// ============================================================================
// Resource Credits Service
// ============================================================================

/**
 * Configuration for the credits service
 */
export interface CreditsServiceConfig {
  /** Default credit expiry (days, 0 = never) */
  defaultExpiryDays: number;
  /** Transfer fee percentage */
  transferFeePercent: number;
  /** Exchange fee percentage */
  exchangeFeePercent: number;
  /** Minimum transfer amount */
  minTransferAmount: number;
  /** Maximum daily transfer limit */
  maxDailyTransferLimit: number;
  /** Enable automatic expiry processing */
  enableAutoExpiry: boolean;
  /** Exchange rate update interval (ms) */
  exchangeRateUpdateIntervalMs: number;
}

/**
 * Default configuration
 */
export const DEFAULT_CREDITS_CONFIG: CreditsServiceConfig = {
  defaultExpiryDays: 365,
  transferFeePercent: 0.5,
  exchangeFeePercent: 1.0,
  minTransferAmount: 1,
  maxDailyTransferLimit: 10000,
  enableAutoExpiry: true,
  exchangeRateUpdateIntervalMs: 3600000, // 1 hour
};

/**
 * Cross-Domain Resource Credits Service
 */
export class ResourceCreditsService {
  private config: CreditsServiceConfig;
  private wallets: Map<string, Wallet> = new Map();
  private exchangeRates: Map<string, ExchangeRate> = new Map();
  private communityPools: Map<string, CommunityPool> = new Map();
  private transactionCounter = 0;
  private creditCounter = 0;

  constructor(config: Partial<CreditsServiceConfig> = {}) {
    this.config = { ...DEFAULT_CREDITS_CONFIG, ...config };
    this.initializeExchangeRates();
  }

  // ============================================================================
  // Wallet Operations
  // ============================================================================

  /**
   * Create a new wallet for an agent
   */
  public createWallet(owner: AgentPubKey): Wallet {
    const key = this.agentKey(owner);

    if (this.wallets.has(key)) {
      throw new Error('Wallet already exists for this agent');
    }

    const wallet: Wallet = {
      owner,
      balances: this.createEmptyBalances(),
      totalCredits: 0,
      entries: [],
      transactions: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      tier: 'basic',
    };

    this.wallets.set(key, wallet);
    return wallet;
  }

  /**
   * Get wallet for an agent
   */
  public getWallet(owner: AgentPubKey): Wallet | null {
    return this.wallets.get(this.agentKey(owner)) ?? null;
  }

  /**
   * Get or create wallet
   */
  public getOrCreateWallet(owner: AgentPubKey): Wallet {
    let wallet = this.getWallet(owner);
    if (!wallet) {
      wallet = this.createWallet(owner);
    }
    return wallet;
  }

  /**
   * Get wallet summary
   */
  public getWalletSummary(owner: AgentPubKey): WalletSummary | null {
    const wallet = this.getWallet(owner);
    if (!wallet) return null;

    const lowBalanceDomains = (Object.entries(wallet.balances) as Array<[ResourceDomain, DomainBalance]>)
      .filter(([_, b]) => b.available < 100)
      .map(([d, _]) => d);

    const primaryDomain = (Object.entries(wallet.balances) as Array<[ResourceDomain, DomainBalance]>)
      .reduce((max, [d, b]) => b.available > max[1].available ? [d, b] : max)[0];

    return {
      owner,
      totalAvailable: Object.values(wallet.balances).reduce((sum, b) => sum + b.available, 0),
      primaryDomain: primaryDomain,
      lowBalanceDomains,
      pendingTransactions: wallet.transactions.filter(t => t.status === 'pending').length,
      tier: wallet.tier,
    };
  }

  // ============================================================================
  // Credit Operations
  // ============================================================================

  /**
   * Issue credits to a wallet
   */
  public issueCredits(
    to: AgentPubKey,
    domain: ResourceDomain,
    amount: number,
    type: CreditType,
    source: CreditSource,
    metadata?: Record<string, unknown>
  ): CreditEntry {
    const wallet = this.getOrCreateWallet(to);

    this.creditCounter++;
    const entry: CreditEntry = {
      creditId: `credit-${Date.now()}-${this.creditCounter}`,
      domain,
      amount,
      type,
      status: 'active',
      createdAt: Date.now(),
      expiresAt: this.config.defaultExpiryDays > 0
        ? Date.now() + (this.config.defaultExpiryDays * 24 * 60 * 60 * 1000)
        : 0,
      source,
      metadata,
    };

    wallet.entries.push(entry);
    wallet.balances[domain].available += amount;
    wallet.balances[domain].total += amount;
    wallet.balances[domain].lastActivity = Date.now();
    wallet.totalCredits += amount;
    wallet.updatedAt = Date.now();

    // Record transaction
    this.recordTransaction({
      type: 'earn',
      from: null,
      to,
      domain,
      amount,
      fee: 0,
      description: `Issued ${amount} ${domain} credits (${type})`,
      reference: entry.creditId,
    });

    return entry;
  }

  /**
   * Spend credits from a wallet
   */
  public spendCredits(
    from: AgentPubKey,
    domain: ResourceDomain,
    amount: number,
    description: string,
    reference?: string
  ): CreditTransaction {
    const wallet = this.getWallet(from);
    if (!wallet) {
      throw new Error('Wallet not found');
    }

    if (wallet.balances[domain].available < amount) {
      throw new Error(`Insufficient ${domain} credits: have ${wallet.balances[domain].available}, need ${amount}`);
    }

    // Deduct from oldest active credits first (FIFO)
    let remaining = amount;
    for (const entry of wallet.entries) {
      if (entry.domain === domain && entry.status === 'active' && remaining > 0) {
        const deduct = Math.min(entry.amount, remaining);
        entry.amount -= deduct;
        remaining -= deduct;
        if (entry.amount === 0) {
          entry.status = 'redeemed';
        }
      }
    }

    wallet.balances[domain].available -= amount;
    wallet.balances[domain].total -= amount;
    wallet.balances[domain].lastActivity = Date.now();
    wallet.totalCredits -= amount;
    wallet.updatedAt = Date.now();

    return this.recordTransaction({
      type: 'spend',
      from,
      to: null,
      domain,
      amount,
      fee: 0,
      description,
      reference,
    });
  }

  /**
   * Transfer credits between wallets
   */
  public transfer(
    from: AgentPubKey,
    to: AgentPubKey,
    domain: ResourceDomain,
    amount: number,
    description?: string
  ): CreditTransaction {
    if (amount < this.config.minTransferAmount) {
      throw new Error(`Minimum transfer amount is ${this.config.minTransferAmount}`);
    }

    const fromWallet = this.getWallet(from);
    if (!fromWallet) {
      throw new Error('Source wallet not found');
    }

    if (fromWallet.balances[domain].available < amount) {
      throw new Error(`Insufficient ${domain} credits for transfer`);
    }

    // Calculate fee
    const fee = Math.ceil(amount * this.config.transferFeePercent / 100);
    const netAmount = amount - fee;

    // Deduct from source
    fromWallet.balances[domain].available -= amount;
    fromWallet.balances[domain].total -= amount;
    fromWallet.totalCredits -= amount;
    fromWallet.updatedAt = Date.now();

    // Add to destination (getOrCreateWallet handled inside issueCredits)
    const entry = this.issueCredits(
      to,
      domain,
      netAmount,
      'transfer',
      { type: 'transfer', identifier: this.agentKey(from), agent: from }
    );

    return this.recordTransaction({
      type: 'transfer',
      from,
      to,
      domain,
      amount: netAmount,
      fee,
      description: description ?? `Transfer ${amount} ${domain} credits`,
      reference: entry.creditId,
    });
  }

  // ============================================================================
  // Exchange Operations
  // ============================================================================

  /**
   * Get exchange quote
   */
  public getExchangeQuote(
    fromDomain: ResourceDomain,
    toDomain: ResourceDomain,
    inputAmount: number
  ): ExchangeQuote {
    const rate = this.getExchangeRate(fromDomain, toDomain);
    if (!rate) {
      throw new Error(`No exchange rate available for ${fromDomain} -> ${toDomain}`);
    }

    if (inputAmount < rate.minAmount) {
      throw new Error(`Minimum exchange amount is ${rate.minAmount}`);
    }
    if (inputAmount > rate.maxAmount) {
      throw new Error(`Maximum exchange amount is ${rate.maxAmount}`);
    }

    const fee = Math.ceil(inputAmount * this.config.exchangeFeePercent / 100);
    const netInput = inputAmount - fee;
    const outputAmount = Math.floor(netInput * rate.rate);

    return {
      quoteId: `quote-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      fromDomain,
      toDomain,
      inputAmount,
      outputAmount,
      rate: rate.rate,
      fee,
      validUntil: Date.now() + 300000, // 5 minutes
    };
  }

  /**
   * Execute an exchange
   */
  public executeExchange(
    owner: AgentPubKey,
    quote: ExchangeQuote
  ): CreditTransaction {
    if (Date.now() > quote.validUntil) {
      throw new Error('Quote has expired');
    }

    const wallet = this.getWallet(owner);
    if (!wallet) {
      throw new Error('Wallet not found');
    }

    if (wallet.balances[quote.fromDomain].available < quote.inputAmount) {
      throw new Error(`Insufficient ${quote.fromDomain} credits`);
    }

    // Deduct from source domain
    wallet.balances[quote.fromDomain].available -= quote.inputAmount;
    wallet.balances[quote.fromDomain].total -= quote.inputAmount;

    // Add to destination domain
    wallet.balances[quote.toDomain].available += quote.outputAmount;
    wallet.balances[quote.toDomain].total += quote.outputAmount;

    // Update total (may differ due to exchange rate)
    wallet.totalCredits = Object.values(wallet.balances).reduce((sum, b) => sum + b.total, 0);
    wallet.updatedAt = Date.now();

    // Create credit entry
    this.creditCounter++;
    wallet.entries.push({
      creditId: `credit-${Date.now()}-${this.creditCounter}`,
      domain: quote.toDomain,
      amount: quote.outputAmount,
      type: 'conversion',
      status: 'active',
      createdAt: Date.now(),
      expiresAt: this.config.defaultExpiryDays > 0
        ? Date.now() + (this.config.defaultExpiryDays * 24 * 60 * 60 * 1000)
        : 0,
      source: { type: 'system', identifier: 'exchange' },
      metadata: { quoteId: quote.quoteId, rate: quote.rate },
    });

    return this.recordTransaction({
      type: 'convert',
      from: owner,
      to: owner,
      domain: quote.toDomain,
      amount: quote.outputAmount,
      fee: quote.fee,
      description: `Converted ${quote.inputAmount} ${quote.fromDomain} to ${quote.outputAmount} ${quote.toDomain}`,
      reference: quote.quoteId,
    });
  }

  /**
   * Get exchange rate between domains
   */
  public getExchangeRate(from: ResourceDomain, to: ResourceDomain): ExchangeRate | null {
    return this.exchangeRates.get(`${from}->${to}`) ?? null;
  }

  /**
   * Get all exchange rates
   */
  public getAllExchangeRates(): ExchangeRate[] {
    return Array.from(this.exchangeRates.values());
  }

  // ============================================================================
  // Community Pool Operations
  // ============================================================================

  /**
   * Create a community pool
   */
  public createCommunityPool(
    name: string,
    description: string,
    administrators: AgentPubKey[]
  ): CommunityPool {
    const pool: CommunityPool = {
      poolId: `pool-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      description,
      administrators,
      members: [...administrators],
      balances: {
        food: 0,
        water: 0,
        energy: 0,
        shelter: 0,
        medicine: 0,
      },
      totalContributions: 0,
      totalDistributions: 0,
      distributionRules: [],
      createdAt: Date.now(),
    };

    this.communityPools.set(pool.poolId, pool);
    return pool;
  }

  /**
   * Contribute to a community pool
   */
  public contributeToPool(
    contributor: AgentPubKey,
    poolId: string,
    domain: ResourceDomain,
    amount: number
  ): CreditTransaction {
    const pool = this.communityPools.get(poolId);
    if (!pool) {
      throw new Error('Pool not found');
    }

    // Spend from contributor wallet
    const tx = this.spendCredits(
      contributor,
      domain,
      amount,
      `Contribution to pool ${pool.name}`,
      poolId
    );

    // Add to pool
    pool.balances[domain] += amount;
    pool.totalContributions += amount;

    // Add contributor as member if not already
    const key = this.agentKey(contributor);
    if (!pool.members.some(m => this.agentKey(m) === key)) {
      pool.members.push(contributor);
    }

    return tx;
  }

  /**
   * Distribute from a community pool
   */
  public distributeFromPool(
    admin: AgentPubKey,
    poolId: string,
    recipients: Array<{ agent: AgentPubKey; amount: number }>,
    domain: ResourceDomain,
    description: string
  ): CreditTransaction[] {
    const pool = this.communityPools.get(poolId);
    if (!pool) {
      throw new Error('Pool not found');
    }

    // Check admin
    if (!pool.administrators.some(a => this.agentKey(a) === this.agentKey(admin))) {
      throw new Error('Only administrators can distribute from pool');
    }

    const totalAmount = recipients.reduce((sum, r) => sum + r.amount, 0);
    if (pool.balances[domain] < totalAmount) {
      throw new Error(`Insufficient pool balance: have ${pool.balances[domain]}, need ${totalAmount}`);
    }

    const transactions: CreditTransaction[] = [];

    for (const recipient of recipients) {
      this.issueCredits(
        recipient.agent,
        domain,
        recipient.amount,
        'allocation',
        { type: 'community', identifier: poolId }
      );

      transactions.push(this.recordTransaction({
        type: 'withdraw',
        from: null,
        to: recipient.agent,
        domain,
        amount: recipient.amount,
        fee: 0,
        description,
        reference: poolId,
      }));
    }

    pool.balances[domain] -= totalAmount;
    pool.totalDistributions += totalAmount;

    return transactions;
  }

  /**
   * Get community pool
   */
  public getPool(poolId: string): CommunityPool | null {
    return this.communityPools.get(poolId) ?? null;
  }

  // ============================================================================
  // History and Reporting
  // ============================================================================

  /**
   * Get transaction history for a wallet
   */
  public getTransactionHistory(
    owner: AgentPubKey,
    options?: {
      domain?: ResourceDomain;
      type?: TransactionType;
      limit?: number;
      offset?: number;
    }
  ): CreditTransaction[] {
    const wallet = this.getWallet(owner);
    if (!wallet) return [];

    let transactions = wallet.transactions;

    if (options?.domain) {
      transactions = transactions.filter(t => t.domain === options.domain);
    }
    if (options?.type) {
      transactions = transactions.filter(t => t.type === options.type);
    }

    // Sort by date descending
    transactions = transactions.sort((a, b) => b.initiatedAt - a.initiatedAt);

    // Apply pagination
    const offset = options?.offset ?? 0;
    const limit = options?.limit ?? 50;
    return transactions.slice(offset, offset + limit);
  }

  /**
   * Get credit balance report
   */
  public getBalanceReport(owner: AgentPubKey): {
    balances: Record<ResourceDomain, DomainBalance>;
    byType: Record<CreditType, number>;
    bySource: Record<string, number>;
    expiringSoon: CreditEntry[];
  } | null {
    const wallet = this.getWallet(owner);
    if (!wallet) return null;

    const byType: Record<CreditType, number> = {
      contribution: 0,
      participation: 0,
      allocation: 0,
      transfer: 0,
      conversion: 0,
      subsidy: 0,
      reward: 0,
    };

    const bySource: Record<string, number> = {};
    const expiringSoon: CreditEntry[] = [];
    const thirtyDaysMs = 30 * 24 * 60 * 60 * 1000;

    for (const entry of wallet.entries) {
      if (entry.status === 'active') {
        byType[entry.type] += entry.amount;
        const sourceKey = entry.source.identifier;
        bySource[sourceKey] = (bySource[sourceKey] ?? 0) + entry.amount;

        if (entry.expiresAt > 0 && entry.expiresAt < Date.now() + thirtyDaysMs) {
          expiringSoon.push(entry);
        }
      }
    }

    return {
      balances: wallet.balances,
      byType,
      bySource,
      expiringSoon,
    };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private agentKey(agent: AgentPubKey): string {
    return typeof agent === 'string' ? agent : Buffer.from(agent).toString('hex');
  }

  private createEmptyBalances(): Record<ResourceDomain, DomainBalance> {
    const domains: ResourceDomain[] = ['food', 'water', 'energy', 'shelter', 'medicine'];
    const balances: Record<ResourceDomain, DomainBalance> = {} as Record<ResourceDomain, DomainBalance>;

    for (const domain of domains) {
      balances[domain] = {
        domain,
        available: 0,
        reserved: 0,
        total: 0,
        pending: 0,
        expiringIn30Days: 0,
        lastActivity: 0,
      };
    }

    return balances;
  }

  private initializeExchangeRates(): void {
    const domains: ResourceDomain[] = ['food', 'water', 'energy', 'shelter', 'medicine'];

    // Base rates relative to a standard unit
    const baseRates: Record<ResourceDomain, number> = {
      food: 1.0,
      water: 0.8, // Water slightly cheaper
      energy: 1.2, // Energy premium
      shelter: 1.5, // Shelter premium
      medicine: 1.3, // Medicine premium
    };

    // Create exchange rates for all pairs
    for (const from of domains) {
      for (const to of domains) {
        if (from !== to) {
          const rate = baseRates[to] / baseRates[from];
          this.exchangeRates.set(`${from}->${to}`, {
            from,
            to,
            rate,
            timestamp: Date.now(),
            volatility: 0.1,
            minAmount: 10,
            maxAmount: 10000,
          });
        }
      }
    }
  }

  private recordTransaction(input: Omit<CreditTransaction, 'transactionId' | 'status' | 'initiatedAt' | 'completedAt'>): CreditTransaction {
    this.transactionCounter++;
    const tx: CreditTransaction = {
      ...input,
      transactionId: `tx-${Date.now()}-${this.transactionCounter}`,
      status: 'completed',
      initiatedAt: Date.now(),
      completedAt: Date.now(),
    };

    // Add to wallets
    if (input.from) {
      const fromWallet = this.getWallet(input.from);
      fromWallet?.transactions.push(tx);
    }
    if (input.to && (!input.from || this.agentKey(input.from) !== this.agentKey(input.to))) {
      const toWallet = this.getWallet(input.to);
      toWallet?.transactions.push(tx);
    }

    return tx;
  }
}

// ============================================================================
// Factory and Singleton
// ============================================================================

let creditsServiceInstance: ResourceCreditsService | null = null;

/**
 * Get the singleton credits service instance
 */
export function getCreditsService(config?: Partial<CreditsServiceConfig>): ResourceCreditsService {
  if (!creditsServiceInstance) {
    creditsServiceInstance = new ResourceCreditsService(config);
  }
  return creditsServiceInstance;
}

/**
 * Reset the credits service instance (for testing)
 */
export function resetCreditsService(): void {
  creditsServiceInstance = null;
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Finance Integration
 *
 * hApp-specific adapter for Mycelix-Finance providing:
 * - Multi-currency wallet management with trust verification
 * - Transaction tracking with reputation impact
 * - Credit union style lending with community backing
 * - Mutual credit systems with MATL risk assessment
 * - Cross-hApp financial coordination via Bridge
 * - Treasury management for DAOs
 * - Escrow services for secure transactions
 *
 * @packageDocumentation
 * @module integrations/finance
 */

// ============================================================================
// New Holochain Zome Clients (Recommended)
// ============================================================================

export { MycelixFinanceClient } from './client';
export type {
  FinanceClientConfig,
  FinanceConnectionOptions,
} from './client';

// Zome clients
export {
  WalletClient,
  CreditClient,
  LendingClient,
  TreasuryClient,
  EscrowClient,
} from './zomes';
export type {
  WalletClientConfig,
  CreditClientConfig,
  LendingClientConfig,
  TreasuryClientConfig,
  EscrowClientConfig,
} from './zomes';

// Types
export type {
  // Wallet types
  WalletType,
  CreateWalletInput,
  Balance,
  BalanceQuery,
  // Transaction types
  TransferInput,
  DepositInput,
  WithdrawInput,
  // Credit types
  CreditScore,
  CreditTier,
  CreditFactors,
  CreditCalculationInput,
  CreditScoreResult,
  // Loan types
  Loan,
  LoanStatus,
  CollateralType,
  ApplicationStatus,
  RequestLoanInput,
  LoanPayment,
  MakePaymentInput,
  LoanApplication,
  SubmitApplicationInput,
  ApproveLoanInput,
  PaymentSchedule,
  PaymentScheduleInput,
  ScheduledPayment,
  LoanDefaultInput,
  DefaultResult,
  // Treasury types
  Treasury,
  CreateTreasuryInput,
  TreasuryAllocation,
  AllocationStatus,
  ProposeAllocationInput,
  // Escrow types
  Escrow,
  EscrowType,
  EscrowStatus,
  CreateEscrowInput,
  ReleaseEscrowInput,
  DisputeEscrowInput,
  DisputeResult,
  // Cross-hApp types
  CollateralLockRequest,
  CollateralLockResult,
  JusticeEnforcementRequest,
  EnforcementResult,
  FinancialSummary,
  // Error types
  FinanceSdkError,
  FinanceSdkErrorCode,
} from './types';

// ============================================================================
// Legacy Mock Services (for testing/development)
// ============================================================================


import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  createReputation,
  recordPositive,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';
import { FinanceValidators } from '../../utils/validation.js';

// ============================================================================
// Bridge Zome Types (matching Rust finance_bridge zome)
// ============================================================================

/** Credit query purpose */
export type CreditPurpose =
  | 'LoanApplication'
  | 'TrustVerification'
  | 'MarketplaceTransaction'
  | 'PropertyPurchase'
  | 'EnergyInvestment';

/** Credit query */
export interface CreditQuery {
  id: string;
  did: string;
  source_happ: string;
  purpose: CreditPurpose;
  amount_requested?: number;
  queried_at: number;
}

/** Credit result */
export interface CreditResult {
  id: string;
  query_id: string;
  did: string;
  credit_score: number;
  available_credit: number;
  payment_history_score: number;
  matl_score: number;
  active_loans: number;
  total_debt: number;
  calculated_at: number;
}

/** Cross-hApp payment */
export interface CrossHappPayment {
  id: string;
  payer_did: string;
  payee_did: string;
  amount: number;
  currency: string;
  source_happ: string;
  target_happ: string;
  reference_id: string;
  status: 'Pending' | 'Confirmed' | 'Failed' | 'Reversed';
  created_at: number;
  confirmed_at?: number;
}

/** Collateral registration */
export interface CollateralRegistration {
  id: string;
  owner_did: string;
  asset_type: 'RealEstate' | 'Vehicle' | 'Cryptocurrency' | 'EnergyAsset' | 'Equipment';
  asset_id: string;
  source_happ: string;
  valuation: number;
  pledged_amount?: number;
  loan_id?: string;
  registered_at: number;
}

/** Finance bridge event types */
export type FinanceBridgeEventType =
  | 'PaymentProcessed'
  | 'LoanApproved'
  | 'LoanDefaulted'
  | 'CollateralPledged'
  | 'CollateralReleased'
  | 'CreditScoreUpdated';

/** Finance bridge event */
export interface FinanceBridgeEvent {
  id: string;
  event_type: FinanceBridgeEventType;
  did?: string;
  payment_id?: string;
  loan_id?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Query credit input */
export interface QueryCreditInput {
  did: string;
  purpose: CreditPurpose;
  amount_requested?: number;
}

/** Process payment input */
export interface ProcessPaymentInput {
  payee_did: string;
  amount: number;
  currency: string;
  target_happ: string;
  reference_id: string;
}

/** Register collateral input */
export interface RegisterCollateralInput {
  asset_type: CollateralRegistration['asset_type'];
  asset_id: string;
  valuation: number;
}

// ============================================================================
// Finance-Specific Types
// ============================================================================

/** Supported currency types */
export type CurrencyType = 'mutual_credit' | 'token' | 'fiat_backed' | 'time_bank';

/** Transaction status */
export type TransactionStatus = 'pending' | 'confirmed' | 'failed' | 'disputed';

/** Account type */
export type AccountType = 'personal' | 'business' | 'cooperative' | 'escrow';

/** Wallet representation */
export interface Wallet {
  id: string;
  ownerId: string;
  balances: Map<string, number>;
  creditLimit: number;
  reputation: ReputationScore;
  accountType: AccountType;
  createdAt: number;
}

/** Financial transaction */
export interface Transaction {
  id: string;
  fromWallet: string;
  toWallet: string;
  amount: number;
  currency: string;
  currencyType: CurrencyType;
  status: TransactionStatus;
  memo?: string;
  timestamp: number;
  confirmedAt?: number;
}

/** Loan request */
export interface LoanRequest {
  id: string;
  borrowerId: string;
  amount: number;
  currency: string;
  purpose: string;
  termMonths: number;
  interestRate: number;
  collateralType?: string;
  collateralValue?: number;
  backers: string[];
  status: 'pending' | 'approved' | 'funded' | 'repaying' | 'completed' | 'defaulted';
  createdAt: number;
}

/** Credit limit calculation input */
export interface CreditLimitInput {
  reputation: ReputationScore;
  transactionHistory: number;
  communityBacking: number;
  collateralValue: number;
}

// ============================================================================
// Finance Service
// ============================================================================

/**
 * Finance service for wallet and transaction management
 */
export class FinanceService {
  private wallets = new Map<string, Wallet>();
  private transactions: Transaction[] = [];
  private loans = new Map<string, LoanRequest>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('finance');
  }

  /**
   * Create a new wallet
   */
  createWallet(ownerId: string, accountType: AccountType = 'personal'): Wallet {
    const id = `wallet-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const wallet: Wallet = {
      id,
      ownerId,
      balances: new Map([['MCX', 0]]),
      creditLimit: 0,
      reputation: createReputation(ownerId),
      accountType,
      createdAt: Date.now(),
    };
    this.wallets.set(id, wallet);
    return wallet;
  }

  /**
   * Transfer funds between wallets
   * @throws {MycelixError} If amount is not positive
   */
  transfer(
    fromWalletId: string,
    toWalletId: string,
    amount: number,
    currency: string = 'MCX',
    memo?: string
  ): Transaction {
    // Validate transfer amount (must be positive)
    FinanceValidators.transfer(amount, currency);

    const fromWallet = this.wallets.get(fromWalletId);
    const toWallet = this.wallets.get(toWalletId);

    if (!fromWallet || !toWallet) {
      throw new Error('Wallet not found');
    }

    const balance = fromWallet.balances.get(currency) || 0;
    const effectiveBalance = balance + fromWallet.creditLimit;

    if (effectiveBalance < amount) {
      throw new Error('Insufficient funds');
    }

    // Execute transfer
    fromWallet.balances.set(currency, balance - amount);
    const toBalance = toWallet.balances.get(currency) || 0;
    toWallet.balances.set(currency, toBalance + amount);

    const tx: Transaction = {
      id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      fromWallet: fromWalletId,
      toWallet: toWalletId,
      amount,
      currency,
      currencyType: 'mutual_credit',
      status: 'confirmed',
      memo,
      timestamp: Date.now(),
      confirmedAt: Date.now(),
    };

    this.transactions.push(tx);

    // Update reputations
    fromWallet.reputation = recordPositive(fromWallet.reputation);
    toWallet.reputation = recordPositive(toWallet.reputation);

    return tx;
  }

  /**
   * Calculate credit limit based on trust factors
   */
  calculateCreditLimit(input: CreditLimitInput): number {
    const reputationFactor = reputationValue(input.reputation);
    const historyFactor = Math.min(input.transactionHistory / 100, 1);
    const backingFactor = input.communityBacking * 0.5;
    const collateralFactor = input.collateralValue * 0.8;

    const baseLimit = 1000;
    const trustMultiplier = reputationFactor * historyFactor;

    return baseLimit * trustMultiplier + backingFactor + collateralFactor;
  }

  /**
   * Request a community-backed loan
   * @throws {MycelixError} If loan parameters are invalid
   */
  requestLoan(input: Omit<LoanRequest, 'id' | 'status' | 'createdAt' | 'backers'>): LoanRequest {
    // Validate loan parameters
    FinanceValidators.loan(input.amount, input.termMonths, input.interestRate);

    const loan: LoanRequest = {
      ...input,
      id: `loan-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      backers: [],
      status: 'pending',
      createdAt: Date.now(),
    };
    this.loans.set(loan.id, loan);
    return loan;
  }

  /**
   * Back a loan request
   */
  backLoan(loanId: string, backerId: string, _amount: number): LoanRequest {
    const loan = this.loans.get(loanId);
    if (!loan) throw new Error('Loan not found');

    loan.backers.push(backerId);

    // Check if fully backed
    const totalBacking = loan.backers.length * (loan.amount / 5); // Simplified
    if (totalBacking >= loan.amount) {
      loan.status = 'funded';
    }

    return loan;
  }

  /**
   * Get wallet by ID
   */
  getWallet(walletId: string): Wallet | undefined {
    return this.wallets.get(walletId);
  }

  /**
   * Get transaction history for a wallet
   */
  getTransactionHistory(walletId: string): Transaction[] {
    return this.transactions.filter(
      (tx) => tx.fromWallet === walletId || tx.toWallet === walletId
    );
  }

  /**
   * Get wallet balance
   */
  getBalance(walletId: string, currency: string = 'MCX'): number {
    const wallet = this.wallets.get(walletId);
    return wallet?.balances.get(currency) || 0;
  }

  /**
   * Cross-hApp credit check
   */
  async getCrossHappCreditScore(did: string): Promise<number> {
    const scores = this.bridge.getCrossHappReputation(did);
    const total = scores.reduce((sum, s) => sum + s.score, 0);
    return scores.length > 0 ? total / scores.length : 0.5;
  }
}

// Singleton
let instance: FinanceService | null = null;

export function getFinanceService(): FinanceService {
  if (!instance) instance = new FinanceService();
  return instance;
}

export function resetFinanceService(): void {
  instance = null;
}

// ============================================================================
// Finance Bridge Client (Holochain Zome Calls)
// ============================================================================

const FINANCE_ROLE = 'finance';
const BRIDGE_ZOME = 'finance_bridge';

/**
 * Finance Bridge Client - Direct Holochain zome calls for cross-hApp finance
 */
export class FinanceBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Query credit score for a DID
   */
  async queryCredit(input: QueryCreditInput): Promise<CreditResult> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_credit',
      payload: input,
    });
  }

  /**
   * Process a cross-hApp payment
   */
  async processPayment(input: ProcessPaymentInput): Promise<CrossHappPayment> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'process_payment',
      payload: input,
    });
  }

  /**
   * Register collateral from another hApp
   */
  async registerCollateral(input: RegisterCollateralInput): Promise<CollateralRegistration> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'register_collateral',
      payload: input,
    });
  }

  /**
   * Get payment history for a DID
   */
  async getPaymentHistory(did: string, limit?: number): Promise<CrossHappPayment[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_payment_history',
      payload: { did, limit: limit ?? 50 },
    });
  }

  /**
   * Get collateral by owner
   */
  async getCollateralByOwner(ownerDid: string): Promise<CollateralRegistration[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_collateral_by_owner',
      payload: ownerDid,
    });
  }

  /**
   * Broadcast a finance event
   */
  async broadcastFinanceEvent(eventType: FinanceBridgeEventType, did?: string, paymentId?: string, payload: string = '{}'): Promise<FinanceBridgeEvent> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_finance_event',
      payload: { event_type: eventType, did, payment_id: paymentId, payload },
    });
  }

  /**
   * Get recent finance events
   */
  async getRecentEvents(limit?: number): Promise<FinanceBridgeEvent[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }
}

// Bridge client singleton
let bridgeInstance: FinanceBridgeClient | null = null;

export function getFinanceBridgeClient(client: MycelixClient): FinanceBridgeClient {
  if (!bridgeInstance) bridgeInstance = new FinanceBridgeClient(client);
  return bridgeInstance;
}

export function resetFinanceBridgeClient(): void {
  bridgeInstance = null;
}

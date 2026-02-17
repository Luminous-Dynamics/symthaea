/**
 * Mycelix Finance Client
 *
 * Unified client for the Finance hApp SDK, providing access to
 * wallets, credit scoring, lending, treasury, and escrow operations.
 *
 * @module @mycelix/sdk/integrations/finance
 */

import {
  type AppClient,
  AppWebsocket,

} from '@holochain/client';

import { FinanceSdkError } from './types';
import { CreditClient } from './zomes/credit';
import { EscrowClient } from './zomes/escrow';
import { LendingClient } from './zomes/lending';
import { TreasuryClient } from './zomes/treasury';
import { WalletClient } from './zomes/wallet';

import type {
  Wallet,
  Transaction,
  CreditScore,
  Loan,
  LoanPayment,
  Escrow,
  FinancialSummary,
} from './types';


/**
 * Configuration for the Finance client
 */
export interface FinanceClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: FinanceClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Connection options for creating a new client
 */
export interface FinanceConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Optional app client constructor args */
  appClientArgs?: Record<string, unknown>;
}

/**
 * Unified Mycelix Finance Client
 *
 * Provides access to all finance functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixFinanceClient } from '@mycelix/sdk/integrations/finance';
 *
 * // Connect to Holochain
 * const finance = await MycelixFinanceClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Create a wallet
 * const wallet = await finance.wallets.createWallet({
 *   id: 'my-wallet',
 *   owner_did: 'did:mycelix:alice',
 *   wallet_type: 'Personal',
 * });
 *
 * // Check credit score
 * const creditScore = await finance.credit.calculateCreditScore('did:mycelix:alice');
 * console.log(`Credit score: ${creditScore.score}`);
 *
 * // Request a loan
 * const loan = await finance.lending.requestLoan({
 *   id: 'loan-001',
 *   borrower_did: 'did:mycelix:alice',
 *   lender_did: 'did:mycelix:community-fund',
 *   principal: 5000,
 *   currency: 'MYC',
 *   interest_rate: 5.0,
 *   term_days: 365,
 * });
 *
 * // Create an escrow for a purchase
 * const escrow = await finance.escrow.createPurchaseEscrow(
 *   'did:mycelix:buyer',
 *   'did:mycelix:seller',
 *   1000,
 *   'MYC',
 *   ['Product delivered as described'],
 *   'did:mycelix:marketplace'
 * );
 * ```
 */
export class MycelixFinanceClient {
  /** Wallet and transaction operations */
  public readonly wallets: WalletClient;

  /** Credit scoring operations */
  public readonly credit: CreditClient;

  /** P2P lending operations */
  public readonly lending: LendingClient;

  /** Treasury management operations */
  public readonly treasury: TreasuryClient;

  /** Escrow operations */
  public readonly escrow: EscrowClient;

  private readonly config: FinanceClientConfig;

  /**
   * Create a finance client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<FinanceClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize sub-clients
    this.wallets = new WalletClient(client, this.config);
    this.credit = new CreditClient(client, this.config);
    this.lending = new LendingClient(client, this.config);
    this.treasury = new TreasuryClient(client, this.config);
    this.escrow = new EscrowClient(client, this.config);
  }

  /**
   * Connect to Holochain and create a finance client
   *
   * @param options - Connection options
   * @returns Connected finance client
   *
   * @example
   * ```typescript
   * const finance = await MycelixFinanceClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(
    options: FinanceConnectionOptions
  ): Promise<MycelixFinanceClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
        ...options.appClientArgs,
      });

      return new MycelixFinanceClient(client);
    } catch (error) {
      throw new FinanceSdkError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Create a finance client from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Finance client
   */
  static fromClient(
    client: AppClient,
    config: Partial<FinanceClientConfig> = {}
  ): MycelixFinanceClient {
    return new MycelixFinanceClient(client, config);
  }

  // ============================================================================
  // Cross-hApp Integration
  // ============================================================================

  /**
   * Get financial summary for a DID
   *
   * Used by other hApps for credit/trust verification.
   *
   * @param did - DID to query
   * @returns Financial summary
   */
  async getFinancialSummary(did: string): Promise<FinancialSummary> {
    return this.callZome<FinancialSummary>('get_financial_summary', did);
  }

  /**
   * Query finance data for cross-hApp requests
   *
   * @param did - DID to query
   * @returns Finance data
   */
  async queryFinance(did: string): Promise<Record<string, unknown>> {
    return this.callZome<Record<string, unknown>>('query_finance', did);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get comprehensive account status for a DID
   *
   * Returns wallets, loans, credit score, and escrows.
   *
   * @param did - DID to query
   * @returns Account status summary
   */
  async getAccountStatus(did: string): Promise<{
    wallets: Wallet[];
    creditScore: CreditScore | null;
    activeLoans: Loan[];
    activeEscrows: Escrow[];
  }> {
    const [wallets, creditScore, loans, escrows] = await Promise.all([
      this.wallets.getWalletsByOwner(did),
      this.credit.getCreditScore(did),
      this.lending.getActiveLoans(did),
      this.escrow.getAllEscrowsForDid(did),
    ]);

    return {
      wallets,
      creditScore,
      activeLoans: loans,
      activeEscrows: this.escrow.getActiveEscrows(escrows),
    };
  }

  /**
   * Get loan details with payments
   *
   * @param loanId - Loan identifier
   * @returns Loan with payment history
   */
  async getLoanDetails(loanId: string): Promise<{
    loan: Loan | null;
    payments: LoanPayment[];
    remainingBalance: number;
    isOverdue: boolean;
  }> {
    const loan = await this.lending.getLoan(loanId);
    if (!loan) {
      return {
        loan: null,
        payments: [],
        remainingBalance: 0,
        isOverdue: false,
      };
    }

    const payments = await this.lending.getPaymentsForLoan(loanId);
    const remainingBalance = this.lending.calculateRemainingBalance(loan, payments);
    const isOverdue = this.lending.isOverdue(loan);

    return {
      loan,
      payments,
      remainingBalance,
      isOverdue,
    };
  }

  /**
   * Quick transfer between wallets
   *
   * @param fromWallet - Source wallet ID
   * @param toWallet - Destination wallet ID
   * @param amount - Transfer amount
   * @param currency - Currency code
   * @param memo - Optional memo
   * @returns The transaction
   */
  async transfer(
    fromWallet: string,
    toWallet: string,
    amount: number,
    currency: string = 'MYC',
    memo?: string
  ): Promise<Transaction> {
    return this.wallets.quickTransfer(fromWallet, toWallet, amount, currency, memo);
  }

  /**
   * Get total net worth for a DID
   *
   * @param did - DID to query
   * @param currency - Currency to calculate in
   * @returns Net worth (assets - liabilities)
   */
  async getNetWorth(did: string, currency: string = 'MYC'): Promise<number> {
    const wallets = await this.wallets.getWalletsByOwner(did);
    const loans = await this.lending.getLoansByBorrower(did);

    // Sum wallet balances
    let assets = 0;
    for (const wallet of wallets) {
      const balance = await this.wallets.getBalance(wallet.id, currency);
      assets += balance;
    }

    // Sum loan liabilities
    const liabilities = loans
      .filter(l => l.status === 'Active' && l.currency === currency)
      .reduce((sum, l) => sum + l.principal, 0);

    return assets - liabilities;
  }

  /**
   * Check if DID is eligible for a loan
   *
   * @param did - DID to check
   * @param amount - Requested loan amount
   * @returns Eligibility result
   */
  async checkLoanEligibility(
    did: string,
    amount: number
  ): Promise<{
    eligible: boolean;
    maxLoanAmount: number;
    recommendedRate: number;
    reason?: string;
  }> {
    const creditScore = await this.credit.getCreditScore(did);

    if (!creditScore) {
      return {
        eligible: false,
        maxLoanAmount: 0,
        recommendedRate: 25.0,
        reason: 'No credit history found',
      };
    }

    const tier = this.credit.getCreditTier(creditScore.score);
    const recommendedRate = this.credit.getRecommendedRate(creditScore.score);

    // Get active loans to check utilization
    const activeLoans = await this.lending.getActiveLoans(did);
    const currentDebt = activeLoans.reduce((sum, l) => sum + l.principal, 0);

    // Simplified max loan calculation
    const baseMax = this.credit.calculateMaxLoan(creditScore.score, 50000); // Assume 50k income
    const maxLoanAmount = Math.max(0, baseMax - currentDebt);

    const eligible = amount <= maxLoanAmount && tier !== 'VeryPoor';

    return {
      eligible,
      maxLoanAmount,
      recommendedRate,
      reason: eligible
        ? undefined
        : amount > maxLoanAmount
          ? `Requested amount exceeds maximum (${maxLoanAmount})`
          : 'Credit score too low',
    };
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Helper to call zome functions
   */
  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
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
}

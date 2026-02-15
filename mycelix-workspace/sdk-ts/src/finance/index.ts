/**
 * Mycelix Finance Module
 *
 * TypeScript client for the mycelix-finance hApp.
 * Provides credit scoring, lending, payments, and treasury management.
 *
 * @module @mycelix/sdk/finance
 */

// ============================================================================
// Types
// ============================================================================

/** Holochain record wrapper */
export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
  entry: { Present: T };
}

/** Generic zome call interface */
export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Wallet Types
// ============================================================================

/** Wallet type */
export type WalletType = 'Personal' | 'Business' | 'DAO' | 'Escrow';

/** Transaction type */
export type TransactionType =
  | 'Transfer'
  | 'Deposit'
  | 'Withdrawal'
  | 'LoanDisbursement'
  | 'LoanRepayment'
  | 'Fee';

/** Transaction status */
export type TransactionStatus = 'Pending' | 'Confirmed' | 'Failed' | 'Reversed';

/** Currency supported */
export type Currency = 'MYC' | 'USD' | 'EUR' | 'BTC' | 'ETH' | 'ENERGY';

/** A wallet */
export interface Wallet {
  /** Wallet ID */
  id: string;
  /** Owner DID */
  owner: string;
  /** Wallet type */
  type_: WalletType;
  /** Wallet name */
  name: string;
  /** Balances by currency */
  balances: Record<string, number>;
  /** Created timestamp */
  created_at: number;
  /** Last update timestamp */
  updated_at: number;
  /** Last activity timestamp */
  last_activity?: number;
  /** Frozen status */
  frozen?: boolean;
}

/** Input for creating a wallet */
export interface CreateWalletInput {
  type_: WalletType;
  name: string;
}

/** A transaction */
export interface Transaction {
  /** Transaction ID */
  id: string;
  /** From wallet (null for deposits) */
  from_wallet?: string;
  /** To wallet (null for withdrawals) */
  to_wallet?: string;
  /** Amount */
  amount: number;
  /** Currency */
  currency: Currency;
  /** Transaction type */
  type_: TransactionType;
  /** Status */
  status: TransactionStatus;
  /** Memo */
  memo?: string;
  /** Reference (e.g., loan ID) */
  reference?: string;
  /** Created timestamp */
  created_at: number;
  /** Confirmed timestamp */
  confirmed_at?: number;
}

/** Input for transfer */
export interface TransferInput {
  from_wallet: string;
  to_wallet: string;
  amount: number;
  currency: Currency;
  memo?: string;
}

// ============================================================================
// Credit Types
// ============================================================================

/** Credit tier */
export type CreditTier = 'Unrated' | 'Bronze' | 'Silver' | 'Gold' | 'Platinum';

/** Credit score */
export interface CreditScore {
  /** DID */
  did: string;
  /** Numeric score (0-1000) */
  score: number;
  /** Credit tier */
  tier: CreditTier;
  /** MATL contribution (0-1) */
  matl_factor: number;
  /** Payment history factor (0-1) */
  payment_factor: number;
  /** Utilization factor (0-1) */
  utilization_factor: number;
  /** Length of history factor (0-1) */
  history_factor: number;
  /** Total credit limit */
  total_limit: number;
  /** Current utilization */
  current_utilization: number;
  /** Last updated timestamp */
  updated_at: number;
}

// ============================================================================
// Loan Types
// ============================================================================

/** Loan status */
export type LoanStatus = 'Pending' | 'Active' | 'Repaid' | 'Defaulted' | 'Cancelled';

/** Loan type */
export type LoanType = 'Personal' | 'Business' | 'Microfinance' | 'Collateralized';

/** A loan */
export interface Loan {
  /** Loan ID */
  id: string;
  /** Borrower DID */
  borrower: string;
  /** Lender DID */
  lender: string;
  /** Principal amount */
  principal: number;
  /** Currency */
  currency: Currency;
  /** Interest rate (annual %) */
  interest_rate: number;
  /** Term in days */
  term_days: number;
  /** Loan type */
  type_: LoanType;
  /** Status */
  status: LoanStatus;
  /** Collateral reference (if any) */
  collateral_ref?: string;
  /** Amount repaid */
  amount_repaid: number;
  /** Created timestamp */
  created_at: number;
  /** Disbursed timestamp */
  disbursed_at?: number;
  /** Due timestamp */
  due_at?: number;
}

/** Input for loan request */
export interface LoanRequestInput {
  amount: number;
  currency: Currency;
  term_days: number;
  type_: LoanType;
  collateral_ref?: string;
  purpose: string;
}

/** Loan offer */
export interface LoanOffer {
  /** Offer ID */
  id: string;
  /** Lender DID */
  lender: string;
  /** Loan request ID */
  request_id: string;
  /** Interest rate offered */
  interest_rate: number;
  /** Conditions */
  conditions?: string;
  /** Valid until */
  valid_until: number;
  /** Created timestamp */
  created_at: number;
}

// ============================================================================
// Treasury Types
// ============================================================================

/** Treasury operation type */
export type TreasuryOperation = 'Deposit' | 'Withdrawal' | 'Grant' | 'Investment';

/** A treasury */
export interface Treasury {
  /** Treasury ID */
  id: string;
  /** DAO ID that owns this treasury */
  dao_id: string;
  /** Balances by currency */
  balances: Record<Currency, number>;
  /** Created timestamp */
  created_at: number;
  /** Last operation timestamp */
  last_operation: number;
}

/** Treasury transaction */
export interface TreasuryTransaction {
  /** Transaction ID */
  id: string;
  /** Treasury ID */
  treasury_id: string;
  /** Operation type */
  operation: TreasuryOperation;
  /** Amount */
  amount: number;
  /** Currency */
  currency: Currency;
  /** Proposal ID that authorized this */
  proposal_id?: string;
  /** Recipient (for grants/investments) */
  recipient?: string;
  /** Description */
  description: string;
  /** Created timestamp */
  created_at: number;
}

// ============================================================================
// Wallet Client
// ============================================================================

const FINANCE_ROLE = 'finance';
const WALLET_ZOME = 'wallets';

/**
 * Wallet Client - Manage wallets and transactions
 *
 * @example
 * ```typescript
 * const wallets = new WalletClient(conductor);
 *
 * const myWallet = await wallets.createWallet({ type_: 'Personal' });
 *
 * await wallets.transfer({
 *   from_wallet: myWallet.id,
 *   to_wallet: 'recipient-wallet-id',
 *   amount: 100,
 *   currency: 'MYC',
 * });
 * ```
 */
export class WalletClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a wallet */
  async createWallet(input: CreateWalletInput): Promise<HolochainRecord<Wallet>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'create_wallet',
      payload: input,
    }) as Promise<HolochainRecord<Wallet>>;
  }

  /** Get wallet by ID */
  async getWallet(walletId: string): Promise<HolochainRecord<Wallet> | null> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'get_wallet',
      payload: walletId,
    }) as Promise<HolochainRecord<Wallet> | null>;
  }

  /** Get wallets by owner */
  async getWalletsByOwner(ownerDid: string): Promise<HolochainRecord<Wallet>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'get_wallets_by_owner',
      payload: ownerDid,
    }) as Promise<HolochainRecord<Wallet>[]>;
  }

  /** Transfer funds - can accept either TransferInput object or individual params */
  async transfer(
    inputOrFromWallet: TransferInput | string,
    toWallet?: string,
    amount?: number,
    currency?: Currency,
    memo?: string
  ): Promise<HolochainRecord<Transaction>> {
    const input: TransferInput =
      typeof inputOrFromWallet === 'string'
        ? {
            from_wallet: inputOrFromWallet,
            to_wallet: toWallet!,
            amount: amount!,
            currency: currency as Currency,
            memo,
          }
        : inputOrFromWallet;

    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'transfer',
      payload: input,
    }) as Promise<HolochainRecord<Transaction>>;
  }

  /** Get transaction history */
  async getTransactionHistory(
    walletId: string,
    limit?: number
  ): Promise<HolochainRecord<Transaction>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'get_transaction_history',
      payload: { wallet_id: walletId, limit: limit ?? 50 },
    }) as Promise<HolochainRecord<Transaction>[]>;
  }

  /** Get balance */
  async getBalance(walletId: string, currency: Currency): Promise<number> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'get_balance',
      payload: { wallet_id: walletId, currency },
    }) as Promise<number>;
  }

  /** Deposit funds to wallet */
  async deposit(
    walletId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<Transaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'deposit',
      payload: { wallet_id: walletId, amount, currency },
    }) as Promise<HolochainRecord<Transaction>>;
  }

  /** Withdraw funds from wallet */
  async withdraw(
    walletId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<Transaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: WALLET_ZOME,
      fn_name: 'withdraw',
      payload: { wallet_id: walletId, amount, currency },
    }) as Promise<HolochainRecord<Transaction>>;
  }
}

// ============================================================================
// Credit Client
// ============================================================================

const CREDIT_ZOME = 'credit_scoring';

/**
 * Credit Client - MATL-based credit scoring
 */
export class CreditClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Get credit score for a DID */
  async getCreditScore(did: string): Promise<HolochainRecord<CreditScore> | null> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: CREDIT_ZOME,
      fn_name: 'get_credit_score',
      payload: did,
    }) as Promise<HolochainRecord<CreditScore> | null>;
  }

  /** Calculate credit score (recalculate) */
  async calculateCreditScore(did: string): Promise<HolochainRecord<CreditScore>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: CREDIT_ZOME,
      fn_name: 'calculate_credit_score',
      payload: did,
    }) as Promise<HolochainRecord<CreditScore>>;
  }

  /** Get my credit score */
  async getMyCreditScore(): Promise<HolochainRecord<CreditScore> | null> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: CREDIT_ZOME,
      fn_name: 'get_my_credit_score',
      payload: null,
    }) as Promise<HolochainRecord<CreditScore> | null>;
  }

  /** Get credit limit */
  async getCreditLimit(did: string): Promise<number> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: CREDIT_ZOME,
      fn_name: 'get_credit_limit',
      payload: did,
    }) as Promise<number>;
  }

  /** Update credit score based on payment outcome */
  async updateCreditScore(
    did: string,
    paymentOutcome: 'repaid' | 'defaulted'
  ): Promise<HolochainRecord<CreditScore>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: CREDIT_ZOME,
      fn_name: 'update_credit_score',
      payload: { did, payment_outcome: paymentOutcome },
    }) as Promise<HolochainRecord<CreditScore>>;
  }
}

// ============================================================================
// Lending Client
// ============================================================================

const LENDING_ZOME = 'lending';

/**
 * Lending Client - P2P lending
 */
export class LendingClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Request a loan */
  async requestLoan(input: LoanRequestInput): Promise<HolochainRecord<Loan>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'request_loan',
      payload: input,
    }) as Promise<HolochainRecord<Loan>>;
  }

  /** Get loan by ID */
  async getLoan(loanId: string): Promise<HolochainRecord<Loan> | null> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'get_loan',
      payload: loanId,
    }) as Promise<HolochainRecord<Loan> | null>;
  }

  /** Get loans by borrower */
  async getLoansByBorrower(borrowerDid: string): Promise<HolochainRecord<Loan>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'get_loans_by_borrower',
      payload: borrowerDid,
    }) as Promise<HolochainRecord<Loan>[]>;
  }

  /** Get loans by lender */
  async getLoansByLender(lenderDid: string): Promise<HolochainRecord<Loan>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'get_loans_by_lender',
      payload: lenderDid,
    }) as Promise<HolochainRecord<Loan>[]>;
  }

  /** Make a loan offer */
  async makeOffer(
    requestId: string,
    interestRate: number,
    conditions?: string
  ): Promise<HolochainRecord<LoanOffer>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'make_offer',
      payload: { request_id: requestId, interest_rate: interestRate, conditions },
    }) as Promise<HolochainRecord<LoanOffer>>;
  }

  /** Offer a loan (alias for makeOffer with full input) */
  async offerLoan(input: {
    amount: number;
    currency: Currency;
    interest_rate: number;
    term_days: number;
    borrower_did?: string;
  }): Promise<HolochainRecord<LoanOffer>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'offer_loan',
      payload: input,
    }) as Promise<HolochainRecord<LoanOffer>>;
  }

  /** Accept an offer */
  async acceptOffer(offerId: string): Promise<HolochainRecord<Loan>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'accept_offer',
      payload: offerId,
    }) as Promise<HolochainRecord<Loan>>;
  }

  /** Make a repayment */
  async makeRepayment(loanId: string, amount: number): Promise<HolochainRecord<Transaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'make_repayment',
      payload: { loan_id: loanId, amount },
    }) as Promise<HolochainRecord<Transaction>>;
  }

  /** Get open loan requests */
  async getOpenRequests(limit?: number): Promise<HolochainRecord<Loan>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: LENDING_ZOME,
      fn_name: 'get_open_requests',
      payload: limit ?? 50,
    }) as Promise<HolochainRecord<Loan>[]>;
  }
}

// ============================================================================
// Treasury Client
// ============================================================================

const TREASURY_ZOME = 'treasury';

/**
 * Treasury Client - DAO treasury management
 */
export class TreasuryClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a treasury for a DAO */
  async createTreasury(daoId: string): Promise<HolochainRecord<Treasury>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'create_treasury',
      payload: daoId,
    }) as Promise<HolochainRecord<Treasury>>;
  }

  /** Get treasury by DAO ID */
  async getTreasury(daoId: string): Promise<HolochainRecord<Treasury> | null> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'get_treasury',
      payload: daoId,
    }) as Promise<HolochainRecord<Treasury> | null>;
  }

  /** Deposit to treasury */
  async deposit(
    daoId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<TreasuryTransaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'deposit',
      payload: { dao_id: daoId, amount, currency },
    }) as Promise<HolochainRecord<TreasuryTransaction>>;
  }

  /** Deposit to treasury (alias) */
  async depositToTreasury(
    treasuryId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<TreasuryTransaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'deposit_to_treasury',
      payload: { treasury_id: treasuryId, amount, currency },
    }) as Promise<HolochainRecord<TreasuryTransaction>>;
  }

  /** Execute grant (requires governance approval) */
  async executeGrant(
    daoId: string,
    proposalId: string,
    recipient: string,
    amount: number,
    currency: Currency,
    description: string
  ): Promise<HolochainRecord<TreasuryTransaction>> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'execute_grant',
      payload: { dao_id: daoId, proposal_id: proposalId, recipient, amount, currency, description },
    }) as Promise<HolochainRecord<TreasuryTransaction>>;
  }

  /** Get treasury transaction history */
  async getTransactionHistory(
    daoId: string,
    limit?: number
  ): Promise<HolochainRecord<TreasuryTransaction>[]> {
    return this.client.callZome({
      role_name: FINANCE_ROLE,
      zome_name: TREASURY_ZOME,
      fn_name: 'get_transaction_history',
      payload: { dao_id: daoId, limit: limit ?? 50 },
    }) as Promise<HolochainRecord<TreasuryTransaction>[]>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Finance hApp clients
 */
export function createFinanceClients(client: ZomeCallable) {
  const walletClient = new WalletClient(client);
  return {
    wallets: walletClient,
    wallet: walletClient, // alias for compatibility
    credit: new CreditClient(client),
    lending: new LendingClient(client),
    treasury: new TreasuryClient(client),
  };
}

export default {
  WalletClient,
  CreditClient,
  LendingClient,
  TreasuryClient,
  createFinanceClients,
};

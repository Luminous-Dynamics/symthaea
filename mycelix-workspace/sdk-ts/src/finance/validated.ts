/**
 * @mycelix/sdk Finance Validated Clients
 *
 * Provides input-validated versions of all Finance clients.
 *
 * @packageDocumentation
 * @module finance/validated
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  WalletClient,
  CreditClient,
  LendingClient,
  TreasuryClient,
  type ZomeCallable,
  type HolochainRecord,
  type Wallet,
  type CreateWalletInput,
  type Transaction,
  type TransferInput,
  type CreditScore,
  type Loan,
  type LoanRequestInput,
  type LoanOffer,
  type Treasury,
  type TreasuryTransaction,
  type Currency,
} from './index.js';

// ============================================================================
// Validation Schemas
// ============================================================================

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID (start with "did:")' });

const walletTypeSchema = z.enum(['Personal', 'Business', 'DAO', 'Escrow']);

const currencySchema = z.enum(['MYC', 'USD', 'EUR', 'BTC', 'ETH', 'ENERGY']);

const loanTypeSchema = z.enum(['Personal', 'Business', 'Microfinance', 'Collateralized']);

const createWalletInputSchema = z.object({
  type_: walletTypeSchema,
  name: z.string().min(1, 'Wallet name is required'),
});

const transferInputSchema = z.object({
  from_wallet: z.string().min(1, 'From wallet is required'),
  to_wallet: z.string().min(1, 'To wallet is required'),
  amount: z.number().positive('Amount must be positive'),
  currency: currencySchema,
  memo: z.string().optional(),
});

const loanRequestInputSchema = z.object({
  amount: z.number().positive('Amount must be positive'),
  currency: currencySchema,
  term_days: z
    .number()
    .min(1, 'Term must be at least 1 day')
    .max(3650, 'Term cannot exceed 10 years'),
  type_: loanTypeSchema,
  collateral_ref: z.string().optional(),
  purpose: z.string().min(10, 'Purpose must be at least 10 characters'),
});

// ============================================================================
// Validation Utility
// ============================================================================

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

// ============================================================================
// Validated Wallet Client
// ============================================================================

export class ValidatedWalletClient {
  private client: WalletClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new WalletClient(zomeClient);
  }

  async createWallet(input: CreateWalletInput): Promise<HolochainRecord<Wallet>> {
    validateOrThrow(createWalletInputSchema, input, 'createWallet input');
    return this.client.createWallet(input);
  }

  async getWallet(walletId: string): Promise<HolochainRecord<Wallet> | null> {
    validateOrThrow(z.string().min(1), walletId, 'walletId');
    return this.client.getWallet(walletId);
  }

  async getWalletsByOwner(ownerDid: string): Promise<HolochainRecord<Wallet>[]> {
    validateOrThrow(didSchema, ownerDid, 'ownerDid');
    return this.client.getWalletsByOwner(ownerDid);
  }

  async transfer(input: TransferInput): Promise<HolochainRecord<Transaction>> {
    validateOrThrow(transferInputSchema, input, 'transfer input');
    return this.client.transfer(input);
  }

  async getTransactionHistory(
    walletId: string,
    limit?: number
  ): Promise<HolochainRecord<Transaction>[]> {
    validateOrThrow(z.string().min(1), walletId, 'walletId');
    if (limit !== undefined) {
      validateOrThrow(z.number().min(1).max(1000), limit, 'limit');
    }
    return this.client.getTransactionHistory(walletId, limit);
  }

  async getBalance(walletId: string, currency: Currency): Promise<number> {
    validateOrThrow(z.string().min(1), walletId, 'walletId');
    validateOrThrow(currencySchema, currency, 'currency');
    return this.client.getBalance(walletId, currency);
  }

  async deposit(
    walletId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<Transaction>> {
    validateOrThrow(z.string().min(1), walletId, 'walletId');
    validateOrThrow(z.number().positive(), amount, 'amount');
    return this.client.deposit(walletId, amount, currency);
  }
}

// ============================================================================
// Validated Credit Client
// ============================================================================

export class ValidatedCreditClient {
  private client: CreditClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new CreditClient(zomeClient);
  }

  async getCreditScore(did: string): Promise<HolochainRecord<CreditScore> | null> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getCreditScore(did);
  }

  async calculateCreditScore(did: string): Promise<HolochainRecord<CreditScore>> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.calculateCreditScore(did);
  }

  async getMyCreditScore(): Promise<HolochainRecord<CreditScore> | null> {
    return this.client.getMyCreditScore();
  }

  async getCreditLimit(did: string): Promise<number> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getCreditLimit(did);
  }
}

// ============================================================================
// Validated Lending Client
// ============================================================================

export class ValidatedLendingClient {
  private client: LendingClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new LendingClient(zomeClient);
  }

  async requestLoan(input: LoanRequestInput): Promise<HolochainRecord<Loan>> {
    validateOrThrow(loanRequestInputSchema, input, 'requestLoan input');
    return this.client.requestLoan(input);
  }

  async getLoan(loanId: string): Promise<HolochainRecord<Loan> | null> {
    validateOrThrow(z.string().min(1), loanId, 'loanId');
    return this.client.getLoan(loanId);
  }

  async getLoansByBorrower(borrowerDid: string): Promise<HolochainRecord<Loan>[]> {
    validateOrThrow(didSchema, borrowerDid, 'borrowerDid');
    return this.client.getLoansByBorrower(borrowerDid);
  }

  async getLoansByLender(lenderDid: string): Promise<HolochainRecord<Loan>[]> {
    validateOrThrow(didSchema, lenderDid, 'lenderDid');
    return this.client.getLoansByLender(lenderDid);
  }

  async makeOffer(
    requestId: string,
    interestRate: number,
    conditions?: string
  ): Promise<HolochainRecord<LoanOffer>> {
    validateOrThrow(z.string().min(1), requestId, 'requestId');
    validateOrThrow(z.number().min(0).max(100), interestRate, 'interestRate');
    if (conditions !== undefined) {
      validateOrThrow(z.string(), conditions, 'conditions');
    }
    return this.client.makeOffer(requestId, interestRate, conditions);
  }

  async acceptOffer(offerId: string): Promise<HolochainRecord<Loan>> {
    validateOrThrow(z.string().min(1), offerId, 'offerId');
    return this.client.acceptOffer(offerId);
  }

  async makeRepayment(loanId: string, amount: number): Promise<HolochainRecord<Transaction>> {
    validateOrThrow(z.string().min(1), loanId, 'loanId');
    validateOrThrow(z.number().positive(), amount, 'amount');
    return this.client.makeRepayment(loanId, amount);
  }

  async getOpenRequests(limit?: number): Promise<HolochainRecord<Loan>[]> {
    if (limit !== undefined) {
      validateOrThrow(z.number().min(1).max(1000), limit, 'limit');
    }
    return this.client.getOpenRequests(limit);
  }
}

// ============================================================================
// Validated Treasury Client
// ============================================================================

export class ValidatedTreasuryClient {
  private client: TreasuryClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new TreasuryClient(zomeClient);
  }

  async createTreasury(daoId: string): Promise<HolochainRecord<Treasury>> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.createTreasury(daoId);
  }

  async getTreasury(daoId: string): Promise<HolochainRecord<Treasury> | null> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    return this.client.getTreasury(daoId);
  }

  async deposit(
    daoId: string,
    amount: number,
    currency: Currency
  ): Promise<HolochainRecord<TreasuryTransaction>> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    validateOrThrow(z.number().positive(), amount, 'amount');
    validateOrThrow(currencySchema, currency, 'currency');
    return this.client.deposit(daoId, amount, currency);
  }

  async executeGrant(
    daoId: string,
    proposalId: string,
    recipient: string,
    amount: number,
    currency: Currency,
    description: string
  ): Promise<HolochainRecord<TreasuryTransaction>> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    validateOrThrow(z.string().min(1), proposalId, 'proposalId');
    validateOrThrow(didSchema, recipient, 'recipient');
    validateOrThrow(z.number().positive(), amount, 'amount');
    validateOrThrow(currencySchema, currency, 'currency');
    validateOrThrow(z.string().min(1), description, 'description');
    return this.client.executeGrant(daoId, proposalId, recipient, amount, currency, description);
  }

  async getTransactionHistory(
    daoId: string,
    limit?: number
  ): Promise<HolochainRecord<TreasuryTransaction>[]> {
    validateOrThrow(z.string().min(1), daoId, 'daoId');
    if (limit !== undefined) {
      validateOrThrow(z.number().min(1).max(1000), limit, 'limit');
    }
    return this.client.getTransactionHistory(daoId, limit);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createValidatedFinanceClients(client: ZomeCallable) {
  const walletClient = new ValidatedWalletClient(client);
  return {
    wallets: walletClient,
    wallet: walletClient, // alias for compatibility
    credit: new ValidatedCreditClient(client),
    lending: new ValidatedLendingClient(client),
    treasury: new ValidatedTreasuryClient(client),
  };
}

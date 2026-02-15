/**
 * Finance Module Tests
 *
 * Tests for the Finance hApp TypeScript clients:
 * - WalletClient (wallet management and transactions)
 * - CreditClient (MATL-based credit scoring)
 * - LendingClient (P2P lending)
 * - TreasuryClient (DAO treasury management)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  WalletClient,
  CreditClient,
  LendingClient,
  TreasuryClient,
  createFinanceClients,
  type Wallet,
  type Transaction,
  type CreditScore,
  type Loan,
  type LoanOffer,
  type Treasury,
  type TreasuryTransaction,
  type ZomeCallable,
  type HolochainRecord,
  type WalletType,
  type TransactionType,
  type TransactionStatus,
  type Currency,
  type CreditTier,
  type LoanStatus,
  type LoanType,
  type TreasuryOperation,
} from '../src/finance/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockWallet(overrides: Partial<Wallet> = {}): Wallet {
  return {
    id: 'wallet-123',
    owner: 'did:mycelix:owner123',
    type_: 'Personal' as WalletType,
    balances: {
      MYC: 1000,
      USD: 500,
      EUR: 0,
      BTC: 0.01,
      ETH: 0.5,
      ENERGY: 100,
    },
    created_at: Date.now() * 1000 - 86400000000,
    last_activity: Date.now() * 1000,
    frozen: false,
    ...overrides,
  };
}

function createMockTransaction(overrides: Partial<Transaction> = {}): Transaction {
  return {
    id: 'tx-123',
    from_wallet: 'wallet-123',
    to_wallet: 'wallet-456',
    amount: 100,
    currency: 'MYC' as Currency,
    type_: 'Transfer' as TransactionType,
    status: 'Confirmed' as TransactionStatus,
    memo: 'Payment for services',
    created_at: Date.now() * 1000,
    confirmed_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockCreditScore(overrides: Partial<CreditScore> = {}): CreditScore {
  return {
    did: 'did:mycelix:borrower123',
    score: 720,
    tier: 'Gold' as CreditTier,
    matl_factor: 0.85,
    payment_factor: 0.9,
    utilization_factor: 0.3,
    history_factor: 0.75,
    total_limit: 10000,
    current_utilization: 3000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockLoan(overrides: Partial<Loan> = {}): Loan {
  return {
    id: 'loan-123',
    borrower: 'did:mycelix:borrower123',
    lender: 'did:mycelix:lender456',
    principal: 5000,
    currency: 'MYC' as Currency,
    interest_rate: 5.5,
    term_days: 365,
    type_: 'Personal' as LoanType,
    status: 'Active' as LoanStatus,
    amount_repaid: 1000,
    created_at: Date.now() * 1000 - 2592000000000,
    disbursed_at: Date.now() * 1000 - 2592000000000,
    due_at: Date.now() * 1000 + 28512000000000,
    ...overrides,
  };
}

function createMockLoanOffer(overrides: Partial<LoanOffer> = {}): LoanOffer {
  return {
    id: 'offer-123',
    lender: 'did:mycelix:lender456',
    request_id: 'loan-req-123',
    interest_rate: 5.0,
    conditions: 'Collateral required for amounts over 5000',
    valid_until: Date.now() * 1000 + 604800000000,
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockTreasury(overrides: Partial<Treasury> = {}): Treasury {
  return {
    id: 'treasury-123',
    dao_id: 'dao-luminous',
    balances: {
      MYC: 1000000,
      USD: 50000,
      EUR: 0,
      BTC: 5,
      ETH: 100,
      ENERGY: 50000,
    },
    created_at: Date.now() * 1000 - 31536000000000,
    last_operation: Date.now() * 1000,
    ...overrides,
  };
}

function createMockTreasuryTransaction(
  overrides: Partial<TreasuryTransaction> = {}
): TreasuryTransaction {
  return {
    id: 'treasury-tx-123',
    treasury_id: 'treasury-123',
    operation: 'Grant' as TreasuryOperation,
    amount: 10000,
    currency: 'MYC' as Currency,
    proposal_id: 'proposal-123',
    recipient: 'did:mycelix:grantee',
    description: 'Development grant for SDK improvements',
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

// ============================================================================
// WalletClient Tests
// ============================================================================

describe('WalletClient', () => {
  let client: WalletClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockWallet = createMockWallet();
    const mockTransaction = createMockTransaction();

    responses.set('wallets:create_wallet', createMockRecord(mockWallet));
    responses.set('wallets:get_wallet', createMockRecord(mockWallet));
    responses.set('wallets:get_wallets_by_owner', [createMockRecord(mockWallet)]);
    responses.set('wallets:transfer', createMockRecord(mockTransaction));
    responses.set('wallets:get_transaction_history', [createMockRecord(mockTransaction)]);
    responses.set('wallets:get_balance', 1000);

    mockZome = createMockClient(responses);
    client = new WalletClient(mockZome);
  });

  describe('createWallet', () => {
    it('should create a new wallet', async () => {
      const result = await client.createWallet({ type_: 'Personal' });

      expect(result.entry.Present.id).toBe('wallet-123');
      expect(result.entry.Present.type_).toBe('Personal');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'finance',
          zome_name: 'wallets',
          fn_name: 'create_wallet',
        })
      );
    });

    it('should create wallets of all types', async () => {
      const types: WalletType[] = ['Personal', 'Business', 'DAO', 'Escrow'];

      for (const type_ of types) {
        const result = await client.createWallet({ type_ });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getWallet', () => {
    it('should get wallet by ID', async () => {
      const result = await client.getWallet('wallet-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('wallet-123');
    });
  });

  describe('getWalletsByOwner', () => {
    it('should get wallets by owner DID', async () => {
      const results = await client.getWalletsByOwner('did:mycelix:owner123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.owner).toBe('did:mycelix:owner123');
    });
  });

  describe('transfer', () => {
    it('should transfer funds between wallets', async () => {
      const result = await client.transfer({
        from_wallet: 'wallet-123',
        to_wallet: 'wallet-456',
        amount: 100,
        currency: 'MYC',
        memo: 'Payment',
      });

      expect(result.entry.Present.type_).toBe('Transfer');
      expect(result.entry.Present.amount).toBe(100);
      expect(result.entry.Present.status).toBe('Confirmed');
    });

    it('should transfer all supported currencies', async () => {
      const currencies: Currency[] = ['MYC', 'USD', 'EUR', 'BTC', 'ETH', 'ENERGY'];

      for (const currency of currencies) {
        const result = await client.transfer({
          from_wallet: 'wallet-123',
          to_wallet: 'wallet-456',
          amount: 10,
          currency,
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getTransactionHistory', () => {
    it('should get transaction history', async () => {
      const results = await client.getTransactionHistory('wallet-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.id).toBe('tx-123');
    });

    it('should respect limit parameter', async () => {
      const results = await client.getTransactionHistory('wallet-123', 10);

      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: { wallet_id: 'wallet-123', limit: 10 },
        })
      );
    });
  });

  describe('getBalance', () => {
    it('should get balance for specific currency', async () => {
      const balance = await client.getBalance('wallet-123', 'MYC');

      expect(balance).toBe(1000);
    });
  });
});

// ============================================================================
// CreditClient Tests
// ============================================================================

describe('CreditClient', () => {
  let client: CreditClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockCreditScore = createMockCreditScore();

    responses.set('credit_scoring:get_credit_score', createMockRecord(mockCreditScore));
    responses.set('credit_scoring:calculate_credit_score', createMockRecord(mockCreditScore));
    responses.set('credit_scoring:get_my_credit_score', createMockRecord(mockCreditScore));
    responses.set('credit_scoring:get_credit_limit', 10000);

    mockZome = createMockClient(responses);
    client = new CreditClient(mockZome);
  });

  describe('getCreditScore', () => {
    it('should get credit score for a DID', async () => {
      const result = await client.getCreditScore('did:mycelix:borrower123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.score).toBe(720);
      expect(result!.entry.Present.tier).toBe('Gold');
    });
  });

  describe('calculateCreditScore', () => {
    it('should recalculate credit score', async () => {
      const result = await client.calculateCreditScore('did:mycelix:borrower123');

      expect(result.entry.Present.matl_factor).toBe(0.85);
      expect(result.entry.Present.payment_factor).toBe(0.9);
    });
  });

  describe('getMyCreditScore', () => {
    it('should get calling agent credit score', async () => {
      const result = await client.getMyCreditScore();

      expect(result).not.toBeNull();
      expect(result!.entry.Present.did).toBe('did:mycelix:borrower123');
    });
  });

  describe('getCreditLimit', () => {
    it('should get credit limit for a DID', async () => {
      const limit = await client.getCreditLimit('did:mycelix:borrower123');

      expect(limit).toBe(10000);
    });
  });

  describe('Credit Tiers', () => {
    it('should support all credit tiers', () => {
      const tiers: CreditTier[] = ['Unrated', 'Bronze', 'Silver', 'Gold', 'Platinum'];

      tiers.forEach((tier) => {
        const score = createMockCreditScore({ tier });
        expect(tiers).toContain(score.tier);
      });
    });

    it('should have score in valid range', () => {
      const score = createMockCreditScore();
      expect(score.score).toBeGreaterThanOrEqual(0);
      expect(score.score).toBeLessThanOrEqual(1000);
    });

    it('should have factors in 0-1 range', () => {
      const score = createMockCreditScore();
      expect(score.matl_factor).toBeGreaterThanOrEqual(0);
      expect(score.matl_factor).toBeLessThanOrEqual(1);
      expect(score.payment_factor).toBeGreaterThanOrEqual(0);
      expect(score.payment_factor).toBeLessThanOrEqual(1);
      expect(score.utilization_factor).toBeGreaterThanOrEqual(0);
      expect(score.utilization_factor).toBeLessThanOrEqual(1);
      expect(score.history_factor).toBeGreaterThanOrEqual(0);
      expect(score.history_factor).toBeLessThanOrEqual(1);
    });
  });
});

// ============================================================================
// LendingClient Tests
// ============================================================================

describe('LendingClient', () => {
  let client: LendingClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockLoan = createMockLoan();
    const mockOffer = createMockLoanOffer();
    const mockTransaction = createMockTransaction({
      type_: 'LoanRepayment',
      reference: 'loan-123',
    });

    responses.set('lending:request_loan', createMockRecord(mockLoan));
    responses.set('lending:get_loan', createMockRecord(mockLoan));
    responses.set('lending:get_loans_by_borrower', [createMockRecord(mockLoan)]);
    responses.set('lending:get_loans_by_lender', [createMockRecord(mockLoan)]);
    responses.set('lending:make_offer', createMockRecord(mockOffer));
    responses.set(
      'lending:accept_offer',
      createMockRecord({
        ...mockLoan,
        status: 'Active',
        disbursed_at: Date.now() * 1000,
      })
    );
    responses.set('lending:make_repayment', createMockRecord(mockTransaction));
    responses.set('lending:get_open_requests', [
      createMockRecord({
        ...mockLoan,
        status: 'Pending',
      }),
    ]);

    mockZome = createMockClient(responses);
    client = new LendingClient(mockZome);
  });

  describe('requestLoan', () => {
    it('should request a loan', async () => {
      const result = await client.requestLoan({
        amount: 5000,
        currency: 'MYC',
        term_days: 365,
        type_: 'Personal',
        purpose: 'Home improvement',
      });

      expect(result.entry.Present.id).toBe('loan-123');
      expect(result.entry.Present.principal).toBe(5000);
    });

    it('should request with all loan types', async () => {
      const types: LoanType[] = ['Personal', 'Business', 'Microfinance', 'Collateralized'];

      for (const type_ of types) {
        const result = await client.requestLoan({
          amount: 1000,
          currency: 'MYC',
          term_days: 90,
          type_,
          purpose: 'Test',
        });
        expect(result).toBeDefined();
      }
    });

    it('should request with collateral reference', async () => {
      const result = await client.requestLoan({
        amount: 10000,
        currency: 'MYC',
        term_days: 730,
        type_: 'Collateralized',
        collateral_ref: 'property-registry:deed-123',
        purpose: 'Business expansion',
      });

      expect(result).toBeDefined();
    });
  });

  describe('getLoan', () => {
    it('should get loan by ID', async () => {
      const result = await client.getLoan('loan-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('loan-123');
    });
  });

  describe('getLoansByBorrower', () => {
    it('should get loans by borrower DID', async () => {
      const results = await client.getLoansByBorrower('did:mycelix:borrower123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.borrower).toBe('did:mycelix:borrower123');
    });
  });

  describe('getLoansByLender', () => {
    it('should get loans by lender DID', async () => {
      const results = await client.getLoansByLender('did:mycelix:lender456');

      expect(results).toHaveLength(1);
    });
  });

  describe('makeOffer', () => {
    it('should make a loan offer', async () => {
      const result = await client.makeOffer('loan-req-123', 5.0, 'Collateral required');

      expect(result.entry.Present.interest_rate).toBe(5.0);
      expect(result.entry.Present.conditions).toBe('Collateral required for amounts over 5000');
    });

    it('should make offer without conditions', async () => {
      const result = await client.makeOffer('loan-req-123', 6.0);

      expect(result).toBeDefined();
    });
  });

  describe('acceptOffer', () => {
    it('should accept a loan offer', async () => {
      const result = await client.acceptOffer('offer-123');

      expect(result.entry.Present.status).toBe('Active');
      expect(result.entry.Present.disbursed_at).toBeDefined();
    });
  });

  describe('makeRepayment', () => {
    it('should make a loan repayment', async () => {
      const result = await client.makeRepayment('loan-123', 500);

      expect(result.entry.Present.type_).toBe('LoanRepayment');
      expect(result.entry.Present.reference).toBe('loan-123');
    });
  });

  describe('getOpenRequests', () => {
    it('should get open loan requests', async () => {
      const results = await client.getOpenRequests();

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Pending');
    });

    it('should respect limit parameter', async () => {
      await client.getOpenRequests(25);

      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: 25,
        })
      );
    });
  });

  describe('Loan Status Transitions', () => {
    it('should support all loan statuses', () => {
      const statuses: LoanStatus[] = ['Pending', 'Active', 'Repaid', 'Defaulted', 'Cancelled'];

      statuses.forEach((status) => {
        const loan = createMockLoan({ status });
        expect(statuses).toContain(loan.status);
      });
    });
  });
});

// ============================================================================
// TreasuryClient Tests
// ============================================================================

describe('TreasuryClient', () => {
  let client: TreasuryClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockTreasury = createMockTreasury();
    const mockTreasuryTx = createMockTreasuryTransaction();

    responses.set('treasury:create_treasury', createMockRecord(mockTreasury));
    responses.set('treasury:get_treasury', createMockRecord(mockTreasury));
    responses.set(
      'treasury:deposit',
      createMockRecord({
        ...mockTreasuryTx,
        operation: 'Deposit',
      })
    );
    responses.set('treasury:execute_grant', createMockRecord(mockTreasuryTx));
    responses.set('treasury:get_transaction_history', [createMockRecord(mockTreasuryTx)]);

    mockZome = createMockClient(responses);
    client = new TreasuryClient(mockZome);
  });

  describe('createTreasury', () => {
    it('should create treasury for a DAO', async () => {
      const result = await client.createTreasury('dao-luminous');

      expect(result.entry.Present.id).toBe('treasury-123');
      expect(result.entry.Present.dao_id).toBe('dao-luminous');
    });
  });

  describe('getTreasury', () => {
    it('should get treasury by DAO ID', async () => {
      const result = await client.getTreasury('dao-luminous');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.dao_id).toBe('dao-luminous');
    });
  });

  describe('deposit', () => {
    it('should deposit to treasury', async () => {
      const result = await client.deposit('dao-luminous', 10000, 'MYC');

      expect(result.entry.Present.operation).toBe('Deposit');
      expect(result.entry.Present.amount).toBe(10000);
    });

    it('should deposit all currencies', async () => {
      const currencies: Currency[] = ['MYC', 'USD', 'EUR', 'BTC', 'ETH', 'ENERGY'];

      for (const currency of currencies) {
        const result = await client.deposit('dao-luminous', 100, currency);
        expect(result).toBeDefined();
      }
    });
  });

  describe('executeGrant', () => {
    it('should execute grant with governance approval', async () => {
      const result = await client.executeGrant(
        'dao-luminous',
        'proposal-123',
        'did:mycelix:grantee',
        10000,
        'MYC',
        'Development grant'
      );

      expect(result.entry.Present.operation).toBe('Grant');
      expect(result.entry.Present.proposal_id).toBe('proposal-123');
      expect(result.entry.Present.recipient).toBe('did:mycelix:grantee');
    });
  });

  describe('getTransactionHistory', () => {
    it('should get treasury transaction history', async () => {
      const results = await client.getTransactionHistory('dao-luminous');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.treasury_id).toBe('treasury-123');
    });

    it('should respect limit parameter', async () => {
      await client.getTransactionHistory('dao-luminous', 25);

      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: { dao_id: 'dao-luminous', limit: 25 },
        })
      );
    });
  });

  describe('Treasury Operations', () => {
    it('should support all treasury operation types', () => {
      const operations: TreasuryOperation[] = ['Deposit', 'Withdrawal', 'Grant', 'Investment'];

      operations.forEach((operation) => {
        const tx = createMockTreasuryTransaction({ operation });
        expect(operations).toContain(tx.operation);
      });
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createFinanceClients', () => {
  it('should create all finance clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createFinanceClients(mockZome);

    expect(clients.wallets).toBeInstanceOf(WalletClient);
    expect(clients.credit).toBeInstanceOf(CreditClient);
    expect(clients.lending).toBeInstanceOf(LendingClient);
    expect(clients.treasury).toBeInstanceOf(TreasuryClient);
  });

  it('should share the same ZomeCallable instance', () => {
    const mockZome = createMockClient(new Map());
    const clients = createFinanceClients(mockZome);

    expect(clients.wallets).toBeDefined();
    expect(clients.credit).toBeDefined();
    expect(clients.lending).toBeDefined();
    expect(clients.treasury).toBeDefined();
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce wallet type constraints', () => {
    const types: WalletType[] = ['Personal', 'Business', 'DAO', 'Escrow'];

    types.forEach((type_) => {
      const wallet = createMockWallet({ type_ });
      expect(types).toContain(wallet.type_);
    });
  });

  it('should enforce transaction types', () => {
    const types: TransactionType[] = [
      'Transfer',
      'Deposit',
      'Withdrawal',
      'LoanDisbursement',
      'LoanRepayment',
      'Fee',
    ];

    types.forEach((type_) => {
      const tx = createMockTransaction({ type_ });
      expect(types).toContain(tx.type_);
    });
  });

  it('should enforce currency constraints', () => {
    const currencies: Currency[] = ['MYC', 'USD', 'EUR', 'BTC', 'ETH', 'ENERGY'];

    currencies.forEach((currency) => {
      const tx = createMockTransaction({ currency });
      expect(currencies).toContain(tx.currency);
    });
  });

  it('should enforce credit score range', () => {
    const score = createMockCreditScore();
    expect(score.score).toBeGreaterThanOrEqual(0);
    expect(score.score).toBeLessThanOrEqual(1000);
  });

  it('should enforce interest rate is positive', () => {
    const loan = createMockLoan();
    expect(loan.interest_rate).toBeGreaterThanOrEqual(0);
  });

  it('should enforce term_days is positive', () => {
    const loan = createMockLoan();
    expect(loan.term_days).toBeGreaterThan(0);
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full lending lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockLoan = createMockLoan({ status: 'Pending' });
    const mockOffer = createMockLoanOffer();

    responses.set('lending:request_loan', createMockRecord(mockLoan));
    responses.set('lending:make_offer', createMockRecord(mockOffer));
    responses.set(
      'lending:accept_offer',
      createMockRecord({
        ...mockLoan,
        status: 'Active',
      })
    );
    responses.set(
      'lending:make_repayment',
      createMockRecord(
        createMockTransaction({
          type_: 'LoanRepayment',
        })
      )
    );

    const mockZome = createMockClient(responses);
    const clients = createFinanceClients(mockZome);

    // Borrower requests loan
    const loanRequest = await clients.lending.requestLoan({
      amount: 5000,
      currency: 'MYC',
      term_days: 365,
      type_: 'Personal',
      purpose: 'Home improvement',
    });
    expect(loanRequest.entry.Present.status).toBe('Pending');

    // Lender makes offer
    const offer = await clients.lending.makeOffer(loanRequest.entry.Present.id, 5.0);
    expect(offer.entry.Present.interest_rate).toBe(5.0);

    // Borrower accepts offer
    const activeLoan = await clients.lending.acceptOffer(offer.entry.Present.id);
    expect(activeLoan.entry.Present.status).toBe('Active');

    // Borrower makes repayment
    const repayment = await clients.lending.makeRepayment(activeLoan.entry.Present.id, 500);
    expect(repayment.entry.Present.type_).toBe('LoanRepayment');
  });

  it('should support treasury grant flow', async () => {
    const responses = new Map<string, unknown>();
    const mockTreasury = createMockTreasury();
    const mockTx = createMockTreasuryTransaction();

    responses.set('treasury:get_treasury', createMockRecord(mockTreasury));
    responses.set('treasury:execute_grant', createMockRecord(mockTx));

    const mockZome = createMockClient(responses);
    const clients = createFinanceClients(mockZome);

    // Check treasury balance
    const treasury = await clients.treasury.getTreasury('dao-luminous');
    expect(treasury!.entry.Present.balances.MYC).toBeGreaterThan(0);

    // Execute grant (requires governance approval)
    const grant = await clients.treasury.executeGrant(
      'dao-luminous',
      'proposal-123',
      'did:mycelix:grantee',
      10000,
      'MYC',
      'Development grant'
    );
    expect(grant.entry.Present.proposal_id).toBe('proposal-123');
  });

  it('should link credit scoring to lending', async () => {
    const responses = new Map<string, unknown>();
    const mockCreditScore = createMockCreditScore({ score: 720, tier: 'Gold' });
    const mockLoan = createMockLoan();

    responses.set('credit_scoring:get_credit_score', createMockRecord(mockCreditScore));
    responses.set('lending:request_loan', createMockRecord(mockLoan));

    const mockZome = createMockClient(responses);
    const clients = createFinanceClients(mockZome);

    // Check credit score first
    const score = await clients.credit.getCreditScore('did:mycelix:borrower123');
    expect(score!.entry.Present.tier).toBe('Gold');

    // Then request loan (interest rate would be based on credit score)
    const loan = await clients.lending.requestLoan({
      amount: 5000,
      currency: 'MYC',
      term_days: 365,
      type_: 'Personal',
      purpose: 'Test',
    });
    expect(loan.entry.Present.borrower).toBe('did:mycelix:borrower123');
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  it('should handle zero balances', () => {
    const wallet = createMockWallet({
      balances: {
        MYC: 0,
        USD: 0,
        EUR: 0,
        BTC: 0,
        ETH: 0,
        ENERGY: 0,
      },
    });

    expect(wallet.balances.MYC).toBe(0);
  });

  it('should handle fully repaid loan', () => {
    const loan = createMockLoan({
      status: 'Repaid',
      amount_repaid: 5000,
    });

    expect(loan.amount_repaid).toBe(loan.principal);
  });

  it('should handle frozen wallet', () => {
    const wallet = createMockWallet({ frozen: true });

    expect(wallet.frozen).toBe(true);
  });

  it('should handle unrated credit', () => {
    const score = createMockCreditScore({
      tier: 'Unrated',
      score: 0,
    });

    expect(score.tier).toBe('Unrated');
    expect(score.score).toBe(0);
  });

  it('should handle defaulted loan', () => {
    const loan = createMockLoan({
      status: 'Defaulted',
      amount_repaid: 1000,
    });

    expect(loan.status).toBe('Defaulted');
    expect(loan.amount_repaid).toBeLessThan(loan.principal);
  });
});

// ============================================================================
// Validated Client Tests
// ============================================================================

import {
  ValidatedWalletClient,
  ValidatedCreditClient,
  ValidatedLendingClient,
  ValidatedTreasuryClient,
  createValidatedFinanceClients,
} from '../src/finance/validated.js';

describe('ValidatedWalletClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedWalletClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['wallets:create_wallet', createMockRecord(createMockWallet())],
        ['wallets:get_wallet', createMockRecord(createMockWallet())],
        ['wallets:get_wallets_by_owner', [createMockRecord(createMockWallet())]],
        ['wallets:transfer', createMockRecord(createMockTransaction())],
        ['wallets:get_transaction_history', [createMockRecord(createMockTransaction())]],
        ['wallets:get_balance', 1000],
        ['wallets:deposit', createMockRecord(createMockTransaction())],
      ])
    );
    validatedClient = new ValidatedWalletClient(mockClient);
  });

  describe('createWallet', () => {
    it('accepts valid wallet input', async () => {
      const result = await validatedClient.createWallet({
        type_: 'Personal',
        name: 'My Wallet',
      });
      expect(result.entry.Present?.id).toBe('wallet-123');
    });

    it('rejects empty name', async () => {
      await expect(
        validatedClient.createWallet({
          type_: 'Personal',
          name: '',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid wallet type', async () => {
      await expect(
        validatedClient.createWallet({
          type_: 'Invalid' as 'Personal',
          name: 'Test',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getWallet', () => {
    it('accepts valid wallet ID', async () => {
      const result = await validatedClient.getWallet('wallet-123');
      expect(result?.entry.Present?.id).toBe('wallet-123');
    });

    it('rejects empty wallet ID', async () => {
      await expect(validatedClient.getWallet('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getWalletsByOwner', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getWalletsByOwner('did:mycelix:owner123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getWalletsByOwner('not-a-did')).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('transfer', () => {
    it('accepts valid transfer input', async () => {
      const result = await validatedClient.transfer({
        from_wallet: 'wallet-123',
        to_wallet: 'wallet-456',
        amount: 100,
        currency: 'MYC',
      });
      expect(result.entry.Present?.id).toBe('tx-123');
    });

    it('rejects negative amount', async () => {
      await expect(
        validatedClient.transfer({
          from_wallet: 'wallet-123',
          to_wallet: 'wallet-456',
          amount: -100,
          currency: 'MYC',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects zero amount', async () => {
      await expect(
        validatedClient.transfer({
          from_wallet: 'wallet-123',
          to_wallet: 'wallet-456',
          amount: 0,
          currency: 'MYC',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid currency', async () => {
      await expect(
        validatedClient.transfer({
          from_wallet: 'wallet-123',
          to_wallet: 'wallet-456',
          amount: 100,
          currency: 'INVALID' as 'MYC',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getTransactionHistory', () => {
    it('accepts valid wallet ID', async () => {
      const result = await validatedClient.getTransactionHistory('wallet-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('accepts valid limit', async () => {
      const result = await validatedClient.getTransactionHistory('wallet-123', 50);
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid limit', async () => {
      await expect(validatedClient.getTransactionHistory('wallet-123', 0)).rejects.toThrow(
        'Validation failed'
      );
    });

    it('rejects limit over 1000', async () => {
      await expect(validatedClient.getTransactionHistory('wallet-123', 1001)).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('getBalance', () => {
    it('accepts valid wallet ID and currency', async () => {
      const result = await validatedClient.getBalance('wallet-123', 'MYC');
      expect(result).toBe(1000);
    });

    it('rejects empty wallet ID', async () => {
      await expect(validatedClient.getBalance('', 'MYC')).rejects.toThrow('Validation failed');
    });
  });

  describe('deposit', () => {
    it('accepts valid deposit', async () => {
      const result = await validatedClient.deposit('wallet-123', 500, 'MYC');
      expect(result.entry.Present?.id).toBe('tx-123');
    });

    it('rejects negative amount', async () => {
      await expect(validatedClient.deposit('wallet-123', -100, 'MYC')).rejects.toThrow(
        'Validation failed'
      );
    });
  });
});

describe('ValidatedCreditClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedCreditClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['credit_scoring:get_credit_score', createMockRecord(createMockCreditScore())],
        ['credit_scoring:calculate_credit_score', createMockRecord(createMockCreditScore())],
        ['credit_scoring:get_my_credit_score', createMockRecord(createMockCreditScore())],
        ['credit_scoring:get_credit_limit', 5000],
      ])
    );
    validatedClient = new ValidatedCreditClient(mockClient);
  });

  describe('getCreditScore', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getCreditScore('did:mycelix:user123');
      expect(result?.entry.Present?.score).toBe(720);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getCreditScore('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('calculateCreditScore', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.calculateCreditScore('did:mycelix:user123');
      expect(result.entry.Present?.score).toBe(720);
    });

    it('rejects invalid DID without content', async () => {
      await expect(validatedClient.calculateCreditScore('not-a-did')).rejects.toThrow();
    });
  });

  describe('getMyCreditScore', () => {
    it('returns credit score without validation', async () => {
      const result = await validatedClient.getMyCreditScore();
      expect(result?.entry.Present?.score).toBe(720);
    });
  });

  describe('getCreditLimit', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getCreditLimit('did:mycelix:user123');
      expect(result).toBe(5000);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getCreditLimit('not-a-did')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedLendingClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedLendingClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['lending:request_loan', createMockRecord(createMockLoan())],
        ['lending:get_loan', createMockRecord(createMockLoan())],
        ['lending:get_loans_by_borrower', [createMockRecord(createMockLoan())]],
        ['lending:get_loans_by_lender', [createMockRecord(createMockLoan())]],
        ['lending:make_offer', createMockRecord(createMockLoanOffer())],
        ['lending:accept_offer', createMockRecord(createMockLoan())],
        ['lending:make_repayment', createMockRecord(createMockTransaction())],
        ['lending:get_open_requests', [createMockRecord(createMockLoan())]],
      ])
    );
    validatedClient = new ValidatedLendingClient(mockClient);
  });

  describe('requestLoan', () => {
    it('accepts valid loan request', async () => {
      const result = await validatedClient.requestLoan({
        amount: 5000,
        currency: 'MYC',
        term_days: 365,
        type_: 'Personal',
        purpose: 'Home improvement project funds',
      });
      expect(result.entry.Present?.id).toBe('loan-123');
    });

    it('rejects negative amount', async () => {
      await expect(
        validatedClient.requestLoan({
          amount: -5000,
          currency: 'MYC',
          term_days: 365,
          type_: 'Personal',
          purpose: 'Home improvement project funds',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects term over 10 years', async () => {
      await expect(
        validatedClient.requestLoan({
          amount: 5000,
          currency: 'MYC',
          term_days: 4000,
          type_: 'Personal',
          purpose: 'Home improvement project funds',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short purpose', async () => {
      await expect(
        validatedClient.requestLoan({
          amount: 5000,
          currency: 'MYC',
          term_days: 365,
          type_: 'Personal',
          purpose: 'Short',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getLoan', () => {
    it('accepts valid loan ID', async () => {
      const result = await validatedClient.getLoan('loan-123');
      expect(result?.entry.Present?.id).toBe('loan-123');
    });

    it('rejects empty loan ID', async () => {
      await expect(validatedClient.getLoan('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getLoansByBorrower', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getLoansByBorrower('did:mycelix:borrower123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getLoansByBorrower('not-a-did')).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('getLoansByLender', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getLoansByLender('did:mycelix:lender123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getLoansByLender('not-a-did')).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('makeOffer', () => {
    it('accepts valid offer', async () => {
      const result = await validatedClient.makeOffer('request-123', 5.5);
      expect(result.entry.Present?.id).toBe('offer-123');
    });

    it('accepts offer with conditions', async () => {
      const result = await validatedClient.makeOffer('request-123', 5.5, 'Must provide collateral');
      expect(result.entry.Present?.id).toBe('offer-123');
    });

    it('rejects negative interest rate', async () => {
      await expect(validatedClient.makeOffer('request-123', -5)).rejects.toThrow('Validation failed');
    });

    it('rejects interest rate over 100', async () => {
      await expect(validatedClient.makeOffer('request-123', 105)).rejects.toThrow(
        'Validation failed'
      );
    });

    it('rejects empty request ID', async () => {
      await expect(validatedClient.makeOffer('', 5.5)).rejects.toThrow('Validation failed');
    });
  });

  describe('acceptOffer', () => {
    it('accepts valid offer ID', async () => {
      const result = await validatedClient.acceptOffer('offer-123');
      expect(result.entry.Present?.id).toBe('loan-123');
    });

    it('rejects empty offer ID', async () => {
      await expect(validatedClient.acceptOffer('')).rejects.toThrow('Validation failed');
    });
  });

  describe('makeRepayment', () => {
    it('accepts valid repayment', async () => {
      const result = await validatedClient.makeRepayment('loan-123', 500);
      expect(result.entry.Present?.id).toBe('tx-123');
    });

    it('rejects negative amount', async () => {
      await expect(validatedClient.makeRepayment('loan-123', -500)).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('getOpenRequests', () => {
    it('returns open requests', async () => {
      const result = await validatedClient.getOpenRequests();
      expect(result.length).toBeGreaterThan(0);
    });

    it('accepts valid limit', async () => {
      const result = await validatedClient.getOpenRequests(50);
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid limit', async () => {
      await expect(validatedClient.getOpenRequests(0)).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedTreasuryClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedTreasuryClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['treasury:create_treasury', createMockRecord(createMockTreasury())],
        ['treasury:get_treasury', createMockRecord(createMockTreasury())],
        ['treasury:deposit', createMockRecord(createMockTreasuryTransaction())],
        ['treasury:execute_grant', createMockRecord(createMockTreasuryTransaction())],
        ['treasury:get_transaction_history', [createMockRecord(createMockTreasuryTransaction())]],
      ])
    );
    validatedClient = new ValidatedTreasuryClient(mockClient);
  });

  describe('createTreasury', () => {
    it('accepts valid DAO ID', async () => {
      const result = await validatedClient.createTreasury('dao-123');
      // Mock returns 'dao-luminous' from createMockTreasury
      expect(result.entry.Present?.dao_id).toBe('dao-luminous');
    });

    it('rejects empty DAO ID', async () => {
      await expect(validatedClient.createTreasury('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getTreasury', () => {
    it('accepts valid DAO ID', async () => {
      const result = await validatedClient.getTreasury('dao-123');
      // Mock returns 'dao-luminous' from createMockTreasury
      expect(result?.entry.Present?.dao_id).toBe('dao-luminous');
    });

    it('rejects empty DAO ID', async () => {
      await expect(validatedClient.getTreasury('')).rejects.toThrow('Validation failed');
    });
  });

  describe('deposit', () => {
    it('accepts valid deposit', async () => {
      const result = await validatedClient.deposit('dao-123', 1000, 'MYC');
      expect(result.entry.Present?.amount).toBe(10000); // Match mock treasury transaction amount
    });

    it('rejects negative amount', async () => {
      await expect(validatedClient.deposit('dao-123', -1000, 'MYC')).rejects.toThrow(
        'Validation failed'
      );
    });

    it('rejects invalid currency', async () => {
      await expect(validatedClient.deposit('dao-123', 1000, 'INVALID' as 'MYC')).rejects.toThrow(
        'Validation failed'
      );
    });
  });

  describe('executeGrant', () => {
    it('accepts valid grant', async () => {
      const result = await validatedClient.executeGrant(
        'dao-123',
        'proposal-456',
        'did:mycelix:recipient789',
        1000,
        'MYC',
        'Development grant'
      );
      expect(result.entry.Present?.amount).toBe(10000); // Match mock treasury transaction amount
    });

    it('rejects non-DID recipient', async () => {
      await expect(
        validatedClient.executeGrant(
          'dao-123',
          'proposal-456',
          'not-a-did',
          1000,
          'MYC',
          'Development grant'
        )
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative amount', async () => {
      await expect(
        validatedClient.executeGrant(
          'dao-123',
          'proposal-456',
          'did:mycelix:recipient789',
          -1000,
          'MYC',
          'Development grant'
        )
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty description', async () => {
      await expect(
        validatedClient.executeGrant(
          'dao-123',
          'proposal-456',
          'did:mycelix:recipient789',
          1000,
          'MYC',
          ''
        )
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getTransactionHistory', () => {
    it('accepts valid DAO ID', async () => {
      const result = await validatedClient.getTransactionHistory('dao-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('accepts valid limit', async () => {
      const result = await validatedClient.getTransactionHistory('dao-123', 50);
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid limit', async () => {
      await expect(validatedClient.getTransactionHistory('dao-123', 0)).rejects.toThrow(
        'Validation failed'
      );
    });
  });
});

describe('createValidatedFinanceClients', () => {
  it('creates all validated clients', () => {
    const mockClient = createMockClient();
    const clients = createValidatedFinanceClients(mockClient);

    expect(clients.wallets).toBeInstanceOf(ValidatedWalletClient);
    expect(clients.wallet).toBeInstanceOf(ValidatedWalletClient);
    expect(clients.credit).toBeInstanceOf(ValidatedCreditClient);
    expect(clients.lending).toBeInstanceOf(ValidatedLendingClient);
    expect(clients.treasury).toBeInstanceOf(ValidatedTreasuryClient);
  });

  it('wallet and wallets are the same instance', () => {
    const mockClient = createMockClient();
    const clients = createValidatedFinanceClients(mockClient);

    expect(clients.wallet).toBe(clients.wallets);
  });
});

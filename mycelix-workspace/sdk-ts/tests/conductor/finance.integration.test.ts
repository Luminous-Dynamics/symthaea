/**
 * Finance hApp Conductor Integration Tests
 *
 * These tests verify the Finance clients work correctly with a real
 * Holochain conductor. They require the conductor harness to be available.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  WalletClient,
  CreditClient,
  LendingClient,
  TreasuryClient,
  createFinanceClients,
  type ZomeCallable,
  type Wallet,
  type Transaction,
  type CreditScore,
  type Loan,
  type LoanOffer,
  type Treasury,
} from '../../src/finance/index.js';
import { createValidatedFinanceClients } from '../../src/finance/validated.js';
import { MycelixError } from '../../src/errors.js';
import { CONDUCTOR_ENABLED, generateTestAgentId } from './conductor-harness.js';

function createMockClient(): ZomeCallable {
  const wallets = new Map<string, Wallet>();
  const transactions = new Map<string, Transaction[]>();
  const creditScores = new Map<string, CreditScore>();
  const loans = new Map<string, Loan>();
  const loanOffers = new Map<string, LoanOffer[]>();
  const treasuries = new Map<string, Treasury>();

  return {
    async callZome({
      fn_name,
      payload,
    }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) {
      const agentId = `did:mycelix:${generateTestAgentId()}`;

      switch (fn_name) {
        case 'create_wallet': {
          const input = payload as { type_: string; name: string };
          const wallet: Wallet = {
            id: `wallet-${Date.now()}`,
            owner: agentId,
            type_: input.type_ as Wallet['type_'],
            name: input.name,
            balances: {},
            created_at: Date.now(),
            updated_at: Date.now(),
          };
          wallets.set(wallet.id, wallet);
          return {
            signed_action: { hashed: { hash: wallet.id, content: {} }, signature: 'sig' },
            entry: { Present: wallet },
          };
        }

        case 'get_wallet': {
          const id = payload as string;
          const w = wallets.get(id);
          return w
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: w },
              }
            : null;
        }

        case 'get_wallets_by_owner': {
          const owner = payload as string;
          return Array.from(wallets.values())
            .filter((w) => w.owner === owner)
            .map((w) => ({
              signed_action: { hashed: { hash: w.id, content: {} }, signature: 'sig' },
              entry: { Present: w },
            }));
        }

        case 'deposit': {
          const { wallet_id, amount, currency } = payload as {
            wallet_id: string;
            amount: number;
            currency: string;
          };
          const w = wallets.get(wallet_id);
          if (w) {
            w.balances[currency] = (w.balances[currency] || 0) + amount;
            w.updated_at = Date.now();
            const tx: Transaction = {
              id: `tx-${Date.now()}`,
              from_wallet: 'external',
              to_wallet: wallet_id,
              amount,
              currency: currency as Transaction['currency'],
              type_: 'Deposit',
              status: 'Completed',
              memo: 'Deposit',
              created_at: Date.now(),
              completed_at: Date.now(),
            };
            const existing = transactions.get(wallet_id) || [];
            existing.push(tx);
            transactions.set(wallet_id, existing);
            return {
              signed_action: { hashed: { hash: tx.id, content: {} }, signature: 'sig' },
              entry: { Present: tx },
            };
          }
          throw new Error('Wallet not found');
        }

        case 'withdraw': {
          const { wallet_id, amount, currency } = payload as {
            wallet_id: string;
            amount: number;
            currency: string;
          };
          const w = wallets.get(wallet_id);
          if (w) {
            if ((w.balances[currency] || 0) < amount) throw new Error('Insufficient funds');
            w.balances[currency] -= amount;
            w.updated_at = Date.now();
            const tx: Transaction = {
              id: `tx-${Date.now()}`,
              from_wallet: wallet_id,
              to_wallet: 'external',
              amount,
              currency: currency as Transaction['currency'],
              type_: 'Withdrawal',
              status: 'Completed',
              created_at: Date.now(),
              completed_at: Date.now(),
            };
            return {
              signed_action: { hashed: { hash: tx.id, content: {} }, signature: 'sig' },
              entry: { Present: tx },
            };
          }
          throw new Error('Wallet not found');
        }

        case 'transfer': {
          const { from_wallet, to_wallet, amount, currency, memo } = payload as {
            from_wallet: string;
            to_wallet: string;
            amount: number;
            currency: string;
            memo?: string;
          };
          const fromW = wallets.get(from_wallet);
          const toW = wallets.get(to_wallet);
          if (!fromW || !toW) throw new Error('Wallet not found');
          if ((fromW.balances[currency] || 0) < amount) throw new Error('Insufficient funds');
          fromW.balances[currency] -= amount;
          toW.balances[currency] = (toW.balances[currency] || 0) + amount;
          fromW.updated_at = Date.now();
          toW.updated_at = Date.now();
          const tx: Transaction = {
            id: `tx-${Date.now()}`,
            from_wallet,
            to_wallet,
            amount,
            currency: currency as Transaction['currency'],
            type_: 'Transfer',
            status: 'Completed',
            memo,
            created_at: Date.now(),
            completed_at: Date.now(),
          };
          return {
            signed_action: { hashed: { hash: tx.id, content: {} }, signature: 'sig' },
            entry: { Present: tx },
          };
        }

        case 'get_transactions': {
          const walletId = payload as string;
          const txs = transactions.get(walletId) || [];
          return txs.map((t) => ({
            signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
            entry: { Present: t },
          }));
        }

        case 'get_credit_score': {
          const did = payload as string;
          let score = creditScores.get(did);
          if (!score) {
            score = {
              did,
              matl_score: 0.5,
              payment_history_score: 0.5,
              collateral_ratio: 1.0,
              composite_score: 0.5,
              loans_repaid: 0,
              loans_defaulted: 0,
              total_borrowed: 0,
              total_repaid: 0,
              last_updated: Date.now(),
            };
            creditScores.set(did, score);
          }
          return {
            signed_action: { hashed: { hash: did, content: {} }, signature: 'sig' },
            entry: { Present: score },
          };
        }

        case 'update_credit_score': {
          const { did, payment_outcome } = payload as { did: string; payment_outcome: string };
          let score = creditScores.get(did);
          if (!score) {
            score = {
              did,
              matl_score: 0.5,
              payment_history_score: 0.5,
              collateral_ratio: 1.0,
              composite_score: 0.5,
              loans_repaid: 0,
              loans_defaulted: 0,
              total_borrowed: 0,
              total_repaid: 0,
              last_updated: Date.now(),
            };
          }
          if (payment_outcome === 'repaid') {
            score.loans_repaid++;
            score.payment_history_score = Math.min(1, score.payment_history_score + 0.05);
          } else {
            score.loans_defaulted++;
            score.payment_history_score = Math.max(0, score.payment_history_score - 0.2);
          }
          score.composite_score =
            (score.matl_score + score.payment_history_score + score.collateral_ratio) / 3;
          score.last_updated = Date.now();
          creditScores.set(did, score);
          return {
            signed_action: { hashed: { hash: did, content: {} }, signature: 'sig' },
            entry: { Present: score },
          };
        }

        case 'request_loan': {
          const input = payload as {
            amount: number;
            currency: string;
            term_days: number;
            collateral_asset_id?: string;
          };
          const loan: Loan = {
            id: `loan-${Date.now()}`,
            borrower: agentId,
            lender: undefined,
            amount: input.amount,
            currency: input.currency as Loan['currency'],
            interest_rate: 0.05,
            term_days: input.term_days,
            collateral_asset_id: input.collateral_asset_id,
            status: 'Requested',
            disbursed_at: undefined,
            due_at: undefined,
            repaid_at: undefined,
            created_at: Date.now(),
          };
          loans.set(loan.id, loan);
          return {
            signed_action: { hashed: { hash: loan.id, content: {} }, signature: 'sig' },
            entry: { Present: loan },
          };
        }

        case 'offer_loan': {
          const input = payload as {
            amount: number;
            currency: string;
            interest_rate: number;
            term_days: number;
          };
          const offer: LoanOffer = {
            id: `offer-${Date.now()}`,
            lender: agentId,
            amount: input.amount,
            currency: input.currency as LoanOffer['currency'],
            interest_rate: input.interest_rate,
            term_days: input.term_days,
            min_credit_score: 0.3,
            available: true,
            created_at: Date.now(),
          };
          const existing = loanOffers.get(agentId) || [];
          existing.push(offer);
          loanOffers.set(agentId, existing);
          return {
            signed_action: { hashed: { hash: offer.id, content: {} }, signature: 'sig' },
            entry: { Present: offer },
          };
        }

        case 'accept_offer': {
          const offerId = payload as string;
          for (const offers of loanOffers.values()) {
            const offer = offers.find((o) => o.id === offerId);
            if (offer) {
              offer.available = false;
              const loan: Loan = {
                id: `loan-${Date.now()}`,
                borrower: agentId,
                lender: offer.lender,
                amount: offer.amount,
                currency: offer.currency,
                interest_rate: offer.interest_rate,
                term_days: offer.term_days,
                status: 'Active',
                disbursed_at: Date.now(),
                due_at: Date.now() + offer.term_days * 24 * 60 * 60 * 1000,
                created_at: Date.now(),
              };
              loans.set(loan.id, loan);
              return {
                signed_action: { hashed: { hash: loan.id, content: {} }, signature: 'sig' },
                entry: { Present: loan },
              };
            }
          }
          throw new Error('Offer not found');
        }

        case 'repay_loan': {
          const { loan_id, amount } = payload as { loan_id: string; amount: number };
          const loan = loans.get(loan_id);
          if (loan) {
            if (amount >= loan.amount * (1 + loan.interest_rate)) {
              loan.status = 'Repaid';
              loan.repaid_at = Date.now();
            }
            return {
              signed_action: { hashed: { hash: loan_id, content: {} }, signature: 'sig' },
              entry: { Present: loan },
            };
          }
          throw new Error('Loan not found');
        }

        case 'get_loan': {
          const id = payload as string;
          const loan = loans.get(id);
          return loan
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: loan },
              }
            : null;
        }

        case 'get_loans_by_borrower': {
          const did = payload as string;
          return Array.from(loans.values())
            .filter((l) => l.borrower === did)
            .map((l) => ({
              signed_action: { hashed: { hash: l.id, content: {} }, signature: 'sig' },
              entry: { Present: l },
            }));
        }

        case 'get_available_offers': {
          const currency = payload as string | undefined;
          const allOffers: LoanOffer[] = [];
          for (const offers of loanOffers.values()) {
            for (const o of offers) {
              if (o.available && (!currency || o.currency === currency)) {
                allOffers.push(o);
              }
            }
          }
          return allOffers.map((o) => ({
            signed_action: { hashed: { hash: o.id, content: {} }, signature: 'sig' },
            entry: { Present: o },
          }));
        }

        case 'create_treasury': {
          const input = payload as { dao_id: string; name: string };
          const treasury: Treasury = {
            id: `treasury-${Date.now()}`,
            dao_id: input.dao_id,
            name: input.name,
            balances: {},
            signers: [agentId],
            required_signatures: 1,
            created_at: Date.now(),
            updated_at: Date.now(),
          };
          treasuries.set(treasury.id, treasury);
          return {
            signed_action: { hashed: { hash: treasury.id, content: {} }, signature: 'sig' },
            entry: { Present: treasury },
          };
        }

        case 'get_treasury': {
          const id = payload as string;
          const t = treasuries.get(id);
          return t
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: t },
              }
            : null;
        }

        case 'get_treasury_by_dao': {
          const daoId = payload as string;
          for (const t of treasuries.values()) {
            if (t.dao_id === daoId) {
              return {
                signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
                entry: { Present: t },
              };
            }
          }
          return null;
        }

        case 'deposit_to_treasury': {
          const { treasury_id, amount, currency } = payload as {
            treasury_id: string;
            amount: number;
            currency: string;
          };
          const t = treasuries.get(treasury_id);
          if (t) {
            t.balances[currency] = (t.balances[currency] || 0) + amount;
            t.updated_at = Date.now();
            return {
              signed_action: { hashed: { hash: treasury_id, content: {} }, signature: 'sig' },
              entry: { Present: t },
            };
          }
          throw new Error('Treasury not found');
        }

        default:
          throw new Error(`Unknown function: ${fn_name}`);
      }
    },
  };
}

const describeConductor = CONDUCTOR_ENABLED ? describe : describe.skip;
const describeUnit = describe;

describeUnit('Finance Clients (Mock)', () => {
  let mockClient: ZomeCallable;
  let walletClient: WalletClient;
  let creditClient: CreditClient;
  let lendingClient: LendingClient;
  let treasuryClient: TreasuryClient;

  beforeAll(() => {
    mockClient = createMockClient();
    const clients = createFinanceClients(mockClient);
    walletClient = clients.wallet;
    creditClient = clients.credit;
    lendingClient = clients.lending;
    treasuryClient = clients.treasury;
  });

  describe('WalletClient', () => {
    it('should create a wallet', async () => {
      const result = await walletClient.createWallet({
        type_: 'Personal',
        name: 'My Main Wallet',
      });

      expect(result).toBeDefined();
      const wallet = result.entry.Present as Wallet;
      expect(wallet.name).toBe('My Main Wallet');
      expect(wallet.type_).toBe('Personal');
    });

    it('should deposit funds', async () => {
      const created = await walletClient.createWallet({ type_: 'Personal', name: 'Deposit Test' });
      const walletId = (created.entry.Present as Wallet).id;

      const result = await walletClient.deposit(walletId, 1000, 'MYC');

      expect(result).toBeDefined();
      const tx = result.entry.Present as Transaction;
      expect(tx.amount).toBe(1000);
      expect(tx.type_).toBe('Deposit');
    });

    it('should transfer between wallets', async () => {
      const from = await walletClient.createWallet({ type_: 'Personal', name: 'From Wallet' });
      const to = await walletClient.createWallet({ type_: 'Personal', name: 'To Wallet' });
      const fromId = (from.entry.Present as Wallet).id;
      const toId = (to.entry.Present as Wallet).id;

      await walletClient.deposit(fromId, 500, 'MYC');
      const result = await walletClient.transfer(fromId, toId, 200, 'MYC', 'Test transfer');

      expect(result).toBeDefined();
      const tx = result.entry.Present as Transaction;
      expect(tx.amount).toBe(200);
      expect(tx.type_).toBe('Transfer');
    });

    it('should reject insufficient funds', async () => {
      const created = await walletClient.createWallet({ type_: 'Personal', name: 'Empty Wallet' });
      const walletId = (created.entry.Present as Wallet).id;

      await expect(walletClient.withdraw(walletId, 100, 'MYC')).rejects.toThrow(
        'Insufficient funds'
      );
    });
  });

  describe('CreditClient', () => {
    it('should get credit score', async () => {
      const result = await creditClient.getCreditScore('did:mycelix:borrower1');

      expect(result).toBeDefined();
      const score = result.entry.Present as CreditScore;
      expect(score.composite_score).toBeGreaterThanOrEqual(0);
      expect(score.composite_score).toBeLessThanOrEqual(1);
    });

    it('should update credit score on repayment', async () => {
      await creditClient.getCreditScore('did:mycelix:borrower2');
      const result = await creditClient.updateCreditScore('did:mycelix:borrower2', 'repaid');

      expect(result).toBeDefined();
      const score = result.entry.Present as CreditScore;
      expect(score.loans_repaid).toBe(1);
    });
  });

  describe('LendingClient', () => {
    it('should request a loan', async () => {
      const result = await lendingClient.requestLoan({
        amount: 10000,
        currency: 'MYC',
        term_days: 30,
      });

      expect(result).toBeDefined();
      const loan = result.entry.Present as Loan;
      expect(loan.amount).toBe(10000);
      expect(loan.status).toBe('Requested');
    });

    it('should offer a loan', async () => {
      const result = await lendingClient.offerLoan({
        amount: 5000,
        currency: 'MYC',
        interest_rate: 0.08,
        term_days: 60,
      });

      expect(result).toBeDefined();
      const offer = result.entry.Present as LoanOffer;
      expect(offer.amount).toBe(5000);
      expect(offer.available).toBe(true);
    });

    it('should accept a loan offer', async () => {
      const offered = await lendingClient.offerLoan({
        amount: 2000,
        currency: 'MYC',
        interest_rate: 0.05,
        term_days: 14,
      });
      const offerId = (offered.entry.Present as LoanOffer).id;

      const result = await lendingClient.acceptOffer(offerId);

      expect(result).toBeDefined();
      const loan = result.entry.Present as Loan;
      expect(loan.status).toBe('Active');
    });
  });

  describe('TreasuryClient', () => {
    it('should create a treasury', async () => {
      const result = await treasuryClient.createTreasury({
        dao_id: 'dao-123',
        name: 'Community Treasury',
      });

      expect(result).toBeDefined();
      const treasury = result.entry.Present as Treasury;
      expect(treasury.name).toBe('Community Treasury');
    });

    it('should deposit to treasury', async () => {
      const created = await treasuryClient.createTreasury({
        dao_id: 'dao-456',
        name: 'Test Treasury',
      });
      const treasuryId = (created.entry.Present as Treasury).id;

      const result = await treasuryClient.depositToTreasury(treasuryId, 10000, 'MYC');

      expect(result).toBeDefined();
      const treasury = result.entry.Present as Treasury;
      expect(treasury.balances['MYC']).toBe(10000);
    });
  });
});

describe('Validated Finance Clients', () => {
  let mockClient: ZomeCallable;
  let validatedClients: ReturnType<typeof createValidatedFinanceClients>;

  beforeAll(() => {
    mockClient = createMockClient();
    validatedClients = createValidatedFinanceClients(mockClient);
  });

  it('should reject invalid wallet creation', async () => {
    await expect(
      validatedClients.wallet.createWallet({
        type_: 'Personal',
        name: '', // Empty name
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject negative deposit', async () => {
    await expect(validatedClients.wallet.deposit('wallet-1', -100, 'MYC')).rejects.toThrow(
      MycelixError
    );
  });

  it('should reject invalid loan request', async () => {
    await expect(
      validatedClients.lending.requestLoan({
        amount: 0, // Zero amount
        currency: 'MYC',
        term_days: 0, // Zero term
      })
    ).rejects.toThrow(MycelixError);
  });
});

describeConductor('Finance Conductor Integration Tests', () => {
  it.todo('should create wallets on the real conductor');
  it.todo('should execute atomic transfers');
  it.todo('should calculate real credit scores from on-chain data');
  it.todo('should manage P2P loans with smart contract enforcement');
});

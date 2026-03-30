// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finance Integration Tests
 *
 * Tests for FinanceService - wallets, transactions, and lending
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  FinanceService,
  getFinanceService,
  resetFinanceService,
  type Wallet,
  type Transaction,
  type LoanRequest,
} from '../../src/integrations/finance/index.js';

describe('Finance Integration', () => {
  let service: FinanceService;

  beforeEach(() => {
    resetFinanceService();
    service = new FinanceService();
  });

  describe('FinanceService', () => {
    describe('createWallet', () => {
      it('should create a new wallet', () => {
        const wallet = service.createWallet('did:mycelix:owner1');

        expect(wallet).toBeDefined();
        expect(wallet.id).toMatch(/^wallet-/);
        expect(wallet.ownerId).toBe('did:mycelix:owner1');
        expect(wallet.accountType).toBe('personal');
        expect(wallet.balances.get('MCX')).toBe(0);
      });

      it('should support different account types', () => {
        const business = service.createWallet('did:mycelix:corp', 'business');
        const coop = service.createWallet('did:mycelix:coop', 'cooperative');
        const escrow = service.createWallet('did:mycelix:escrow', 'escrow');

        expect(business.accountType).toBe('business');
        expect(coop.accountType).toBe('cooperative');
        expect(escrow.accountType).toBe('escrow');
      });

      it('should initialize with reputation', () => {
        const wallet = service.createWallet('did:mycelix:newuser');

        expect(wallet.reputation).toBeDefined();
        expect(wallet.reputation.agentId).toBe('did:mycelix:newuser');
      });

      it('should initialize with zero credit limit', () => {
        const wallet = service.createWallet('did:mycelix:user');

        expect(wallet.creditLimit).toBe(0);
      });
    });

    describe('transfer', () => {
      it('should transfer funds between wallets', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');

        // Give sender some funds (via credit limit for testing)
        sender.creditLimit = 1000;

        const tx = service.transfer(sender.id, receiver.id, 100, 'MCX', 'Payment for services');

        expect(tx).toBeDefined();
        expect(tx.id).toMatch(/^tx-/);
        expect(tx.amount).toBe(100);
        expect(tx.status).toBe('confirmed');
        expect(tx.memo).toBe('Payment for services');
      });

      it('should update balances correctly', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 500;

        service.transfer(sender.id, receiver.id, 200, 'MCX');

        expect(service.getBalance(sender.id)).toBe(-200);
        expect(service.getBalance(receiver.id)).toBe(200);
      });

      it('should throw for insufficient funds', () => {
        const sender = service.createWallet('did:mycelix:broke');
        const receiver = service.createWallet('did:mycelix:receiver');

        expect(() => {
          service.transfer(sender.id, receiver.id, 100, 'MCX');
        }).toThrow('Insufficient funds');
      });

      it('should throw for non-existent wallet', () => {
        const sender = service.createWallet('did:mycelix:sender');

        expect(() => {
          service.transfer(sender.id, 'wallet-fake', 100, 'MCX');
        }).toThrow('Wallet not found');
      });

      it('should update reputations after successful transfer', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 1000;

        const senderRepBefore = sender.reputation.positiveCount;
        const receiverRepBefore = receiver.reputation.positiveCount;

        service.transfer(sender.id, receiver.id, 100, 'MCX');

        expect(sender.reputation.positiveCount).toBeGreaterThan(senderRepBefore);
        expect(receiver.reputation.positiveCount).toBeGreaterThan(receiverRepBefore);
      });
    });

    describe('transfer error paths', () => {
      it('should throw for negative transfer amount', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 1000;

        expect(() => {
          service.transfer(sender.id, receiver.id, -100, 'MCX');
        }).toThrow('Transfer amount must be positive');
      });

      it('should throw for zero transfer amount', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 1000;

        expect(() => {
          service.transfer(sender.id, receiver.id, 0, 'MCX');
        }).toThrow('Transfer amount must be positive');
      });

      it('should throw for NaN transfer amount', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 1000;

        expect(() => {
          service.transfer(sender.id, receiver.id, NaN, 'MCX');
        }).toThrow('Transfer amount must be a valid number');
      });

      it('should throw when sender wallet does not exist', () => {
        const receiver = service.createWallet('did:mycelix:receiver');

        expect(() => {
          service.transfer('wallet-nonexistent', receiver.id, 100, 'MCX');
        }).toThrow('Wallet not found');
      });

      it('should throw for overdraft beyond credit limit', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        sender.creditLimit = 100; // Only 100 credit

        expect(() => {
          service.transfer(sender.id, receiver.id, 200, 'MCX');
        }).toThrow('Insufficient funds');
      });

      it('should throw for transfer in unsupported currency when balance is zero', () => {
        const sender = service.createWallet('did:mycelix:sender');
        const receiver = service.createWallet('did:mycelix:receiver');
        // No credit limit, no balance in ETH

        expect(() => {
          service.transfer(sender.id, receiver.id, 50, 'ETH');
        }).toThrow('Insufficient funds');
      });
    });

    describe('requestLoan error paths', () => {
      it('should throw for negative loan amount', () => {
        expect(() => {
          service.requestLoan({
            borrowerId: 'did:mycelix:borrower',
            amount: -5000,
            currency: 'MCX',
            purpose: 'Test',
            termMonths: 12,
            interestRate: 0.05,
          });
        }).toThrow('Loan amount must be positive');
      });

      it('should throw for zero loan amount', () => {
        expect(() => {
          service.requestLoan({
            borrowerId: 'did:mycelix:borrower',
            amount: 0,
            currency: 'MCX',
            purpose: 'Test',
            termMonths: 12,
            interestRate: 0.05,
          });
        }).toThrow('Loan amount must be positive');
      });

      it('should throw for negative loan term', () => {
        expect(() => {
          service.requestLoan({
            borrowerId: 'did:mycelix:borrower',
            amount: 5000,
            currency: 'MCX',
            purpose: 'Test',
            termMonths: -6,
            interestRate: 0.05,
          });
        }).toThrow('Loan term must be positive');
      });

      it('should throw for non-integer loan term', () => {
        expect(() => {
          service.requestLoan({
            borrowerId: 'did:mycelix:borrower',
            amount: 5000,
            currency: 'MCX',
            purpose: 'Test',
            termMonths: 6.5,
            interestRate: 0.05,
          });
        }).toThrow('Loan term must be a whole number of months');
      });

      it('should throw for negative interest rate', () => {
        expect(() => {
          service.requestLoan({
            borrowerId: 'did:mycelix:borrower',
            amount: 5000,
            currency: 'MCX',
            purpose: 'Test',
            termMonths: 12,
            interestRate: -0.05,
          });
        }).toThrow('Interest rate cannot be negative');
      });

      it('should allow zero interest rate (interest-free loan)', () => {
        const loan = service.requestLoan({
          borrowerId: 'did:mycelix:borrower',
          amount: 1000,
          currency: 'MCX',
          purpose: 'Community support',
          termMonths: 6,
          interestRate: 0,
        });

        expect(loan).toBeDefined();
        expect(loan.interestRate).toBe(0);
        expect(loan.status).toBe('pending');
      });
    });

    describe('calculateCreditLimit', () => {
      it('should calculate credit limit based on trust factors', () => {
        const wallet = service.createWallet('did:mycelix:user');

        const limit = service.calculateCreditLimit({
          reputation: wallet.reputation,
          transactionHistory: 50,
          communityBacking: 500,
          collateralValue: 2000,
        });

        expect(limit).toBeGreaterThan(0);
        // Base 1000 * 0.5 * 0.5 + 500 * 0.5 + 2000 * 0.8 = 250 + 250 + 1600 = 2100
        expect(limit).toBeGreaterThanOrEqual(1600); // At minimum collateral factor
      });

      it('should increase limit with better reputation', () => {
        const lowRepWallet = service.createWallet('did:mycelix:low');
        const highRepWallet = service.createWallet('did:mycelix:high');

        // Simulate high reputation
        for (let i = 0; i < 10; i++) {
          highRepWallet.reputation.positiveCount++;
        }

        const lowLimit = service.calculateCreditLimit({
          reputation: lowRepWallet.reputation,
          transactionHistory: 10,
          communityBacking: 0,
          collateralValue: 0,
        });

        const highLimit = service.calculateCreditLimit({
          reputation: highRepWallet.reputation,
          transactionHistory: 10,
          communityBacking: 0,
          collateralValue: 0,
        });

        expect(highLimit).toBeGreaterThan(lowLimit);
      });

      it('should factor in collateral value', () => {
        const wallet = service.createWallet('did:mycelix:user');

        const noCollateral = service.calculateCreditLimit({
          reputation: wallet.reputation,
          transactionHistory: 50,
          communityBacking: 0,
          collateralValue: 0,
        });

        const withCollateral = service.calculateCreditLimit({
          reputation: wallet.reputation,
          transactionHistory: 50,
          communityBacking: 0,
          collateralValue: 5000,
        });

        expect(withCollateral).toBeGreaterThan(noCollateral);
        expect(withCollateral - noCollateral).toBe(5000 * 0.8); // 80% of collateral
      });
    });

    describe('requestLoan', () => {
      it('should create a loan request', () => {
        const loan = service.requestLoan({
          borrowerId: 'did:mycelix:borrower',
          amount: 5000,
          currency: 'MCX',
          purpose: 'Business expansion',
          termMonths: 12,
          interestRate: 0.05,
        });

        expect(loan).toBeDefined();
        expect(loan.id).toMatch(/^loan-/);
        expect(loan.amount).toBe(5000);
        expect(loan.status).toBe('pending');
        expect(loan.backers).toEqual([]);
      });

      it('should support collateral-backed loans', () => {
        const loan = service.requestLoan({
          borrowerId: 'did:mycelix:borrower',
          amount: 10000,
          currency: 'MCX',
          purpose: 'Property purchase',
          termMonths: 24,
          interestRate: 0.04,
          collateralType: 'real_estate',
          collateralValue: 15000,
        });

        expect(loan.collateralType).toBe('real_estate');
        expect(loan.collateralValue).toBe(15000);
      });
    });

    describe('backLoan', () => {
      it('should add backer to loan', () => {
        const loan = service.requestLoan({
          borrowerId: 'did:mycelix:borrower',
          amount: 1000,
          currency: 'MCX',
          purpose: 'Education',
          termMonths: 6,
          interestRate: 0.03,
        });

        const backed = service.backLoan(loan.id, 'did:mycelix:backer1', 200);

        expect(backed.backers).toContain('did:mycelix:backer1');
      });

      it('should mark loan as funded when fully backed', () => {
        const loan = service.requestLoan({
          borrowerId: 'did:mycelix:borrower',
          amount: 1000,
          currency: 'MCX',
          purpose: 'Test',
          termMonths: 6,
          interestRate: 0.03,
        });

        // Back with 5 backers (each providing 200 = 1/5 of amount)
        for (let i = 0; i < 5; i++) {
          service.backLoan(loan.id, `did:mycelix:backer${i}`, 200);
        }

        const updated = service.backLoan(loan.id, 'did:mycelix:finalBacker', 200);

        expect(updated.status).toBe('funded');
      });

      it('should throw for non-existent loan', () => {
        expect(() => {
          service.backLoan('loan-fake', 'did:mycelix:backer', 100);
        }).toThrow('Loan not found');
      });
    });

    describe('getWallet', () => {
      it('should retrieve an existing wallet', () => {
        const created = service.createWallet('did:mycelix:test');
        const retrieved = service.getWallet(created.id);

        expect(retrieved).toBeDefined();
        expect(retrieved!.id).toBe(created.id);
      });

      it('should return undefined for non-existent wallet', () => {
        const result = service.getWallet('wallet-fake');
        expect(result).toBeUndefined();
      });
    });

    describe('getTransactionHistory', () => {
      it('should return transactions for a wallet', () => {
        const w1 = service.createWallet('did:mycelix:user1');
        const w2 = service.createWallet('did:mycelix:user2');
        const w3 = service.createWallet('did:mycelix:user3');

        w1.creditLimit = 1000;
        w2.creditLimit = 1000;

        service.transfer(w1.id, w2.id, 100, 'MCX');
        service.transfer(w2.id, w3.id, 50, 'MCX');
        service.transfer(w1.id, w3.id, 75, 'MCX');

        const w1History = service.getTransactionHistory(w1.id);
        const w2History = service.getTransactionHistory(w2.id);
        const w3History = service.getTransactionHistory(w3.id);

        expect(w1History.length).toBe(2);
        expect(w2History.length).toBe(2);
        expect(w3History.length).toBe(2);
      });

      it('should return empty array for wallet with no transactions', () => {
        const wallet = service.createWallet('did:mycelix:noactivity');
        const history = service.getTransactionHistory(wallet.id);

        expect(history).toEqual([]);
      });
    });

    describe('getBalance', () => {
      it('should return current balance for currency', () => {
        const wallet = service.createWallet('did:mycelix:user');
        const sender = service.createWallet('did:mycelix:sender');
        sender.creditLimit = 1000;

        service.transfer(sender.id, wallet.id, 250, 'MCX');

        expect(service.getBalance(wallet.id, 'MCX')).toBe(250);
      });

      it('should return 0 for unknown currency', () => {
        const wallet = service.createWallet('did:mycelix:user');

        expect(service.getBalance(wallet.id, 'UNKNOWN')).toBe(0);
      });
    });

    describe('getCrossHappCreditScore', () => {
      it('should return a score between 0 and 1', async () => {
        const score = await service.getCrossHappCreditScore('did:mycelix:user');

        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('getFinanceService', () => {
    it('should return singleton instance', () => {
      const service1 = getFinanceService();
      const service2 = getFinanceService();

      expect(service1).toBe(service2);
    });
  });
});

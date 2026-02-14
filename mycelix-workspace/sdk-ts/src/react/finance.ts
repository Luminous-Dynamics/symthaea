/**
 * React Hooks for Mycelix Finance Module
 *
 * Provides React hooks for wallets, credit, lending, and treasury.
 *
 * @module @mycelix/sdk/react/finance
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Wallet,
  CreateWalletInput,
  Transaction,
  TransferInput,
  CreditScore,
  Loan,
  LoanRequestInput,
  LoanOffer,
  Treasury,
  TreasuryTransaction,
  Currency,
  HolochainRecord,
} from '../finance/index.js';

// ============================================================================
// Wallet Hooks
// ============================================================================

/**
 * Hook to fetch a wallet by ID
 */
export function useWalletById(_walletId: string): QueryState<HolochainRecord<Wallet> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch wallets by owner
 */
export function useWalletsByOwnerDid(_ownerDid: string): QueryState<HolochainRecord<Wallet>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get balance
 */
export function useBalance(_walletId: string, _currency: Currency): QueryState<number> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get transaction history
 */
export function useTransactionHistory(
  _walletId: string,
  _limit?: number
): QueryState<HolochainRecord<Transaction>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to create a wallet
 */
export function useCreateWalletMutation(): MutationState<
  HolochainRecord<Wallet>,
  CreateWalletInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to transfer funds
 */
export function useTransferMutation(): MutationState<HolochainRecord<Transaction>, TransferInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Credit Hooks
// ============================================================================

/**
 * Hook to get credit score
 */
export function useCreditScoreByDid(_did: string): QueryState<HolochainRecord<CreditScore> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get my credit score
 */
export function useMyCreditScore(): QueryState<HolochainRecord<CreditScore> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get credit limit
 */
export function useCreditLimitByDid(_did: string): QueryState<number> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to recalculate credit score
 */
export function useCalculateCreditScore(): MutationState<HolochainRecord<CreditScore>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Lending Hooks
// ============================================================================

/**
 * Hook to get a loan by ID
 */
export function useLoanById(_loanId: string): QueryState<HolochainRecord<Loan> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get loans by borrower
 */
export function useLoansByBorrower(_borrowerDid: string): QueryState<HolochainRecord<Loan>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get loans by lender
 */
export function useLoansByLender(_lenderDid: string): QueryState<HolochainRecord<Loan>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get open loan requests
 */
export function useOpenLoanRequests(_limit?: number): QueryState<HolochainRecord<Loan>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to request a loan
 */
export function useRequestLoan(): MutationState<HolochainRecord<Loan>, LoanRequestInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to make a loan offer
 */
export function useMakeLoanOffer(): MutationState<
  HolochainRecord<LoanOffer>,
  { requestId: string; interestRate: number; conditions?: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to accept a loan offer
 */
export function useAcceptLoanOffer(): MutationState<HolochainRecord<Loan>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to make a loan repayment
 */
export function useMakeRepayment(): MutationState<
  HolochainRecord<Transaction>,
  { loanId: string; amount: number }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Treasury Hooks
// ============================================================================

/**
 * Hook to get treasury by DAO ID
 */
export function useTreasuryByDao(_daoId: string): QueryState<HolochainRecord<Treasury> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get treasury transaction history
 */
export function useTreasuryHistory(
  _daoId: string,
  _limit?: number
): QueryState<HolochainRecord<TreasuryTransaction>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to create a treasury
 */
export function useCreateTreasury(): MutationState<HolochainRecord<Treasury>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to deposit to treasury
 */
export function useTreasuryDeposit(): MutationState<
  HolochainRecord<TreasuryTransaction>,
  { daoId: string; amount: number; currency: Currency }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to execute a treasury grant
 */
export function useExecuteGrant(): MutationState<
  HolochainRecord<TreasuryTransaction>,
  {
    daoId: string;
    proposalId: string;
    recipient: string;
    amount: number;
    currency: Currency;
    description: string;
  }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

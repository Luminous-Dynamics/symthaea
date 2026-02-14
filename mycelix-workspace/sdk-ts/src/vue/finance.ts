/**
 * @mycelix/sdk Vue 3 Composables for Finance Module
 *
 * Provides Vue 3 Composition API composables for the Finance hApp integration.
 *
 * @packageDocumentation
 * @module vue/finance
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions, Ref } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface Wallet {
  id: string;
  owner_did: string;
  type: 'personal' | 'business' | 'escrow' | 'treasury';
  balances: Map<string, number>;
  created_at: number;
  frozen: boolean;
}

export interface Transaction {
  id: string;
  from_wallet: string;
  to_wallet: string;
  amount: number;
  currency: string;
  memo?: string;
  status: 'pending' | 'completed' | 'failed' | 'reversed';
  created_at: number;
  completed_at?: number;
}

export interface CreditScore {
  did: string;
  score: number; // 0-1000
  factors: {
    payment_history: number;
    credit_utilization: number;
    credit_age: number;
    matl_score: number;
    collateral_ratio: number;
  };
  updated_at: number;
}

export interface Loan {
  id: string;
  borrower_did: string;
  lender_did: string;
  principal: number;
  currency: string;
  interest_rate: number;
  term_days: number;
  collateral_asset_id?: string;
  status: 'pending' | 'active' | 'repaid' | 'defaulted';
  payments: LoanPayment[];
  created_at: number;
  maturity_date: number;
}

export interface LoanPayment {
  id: string;
  loan_id: string;
  amount: number;
  principal_portion: number;
  interest_portion: number;
  paid_at: number;
}

// ============================================================================
// Wallet Composables
// ============================================================================

export interface UseWalletReturn extends UseQueryReturn<HolochainRecord<Wallet> | null> {
  balances: Ref<Map<string, number>>;
}

/**
 * Composable to create a wallet
 */
export function useCreateWallet(): UseMutationReturn<
  HolochainRecord<Wallet>,
  {
    type: Wallet['type'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a wallet
 */
export function useWallet(_walletId: string, _options?: UseQueryOptions): UseWalletReturn {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get wallets by owner
 */
export function useWalletsByOwner(_ownerDid: string): UseQueryReturn<HolochainRecord<Wallet>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get wallet balance for a currency
 */
export function useBalance(_walletId: string, _currency: string): UseQueryReturn<number> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to freeze wallet (admin/enforcement only)
 */
export function useFreezeWallet(): UseMutationReturn<
  HolochainRecord<Wallet>,
  {
    wallet_id: string;
    reason: string;
    case_id?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to unfreeze wallet
 */
export function useUnfreezeWallet(): UseMutationReturn<
  HolochainRecord<Wallet>,
  {
    wallet_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Transaction Composables
// ============================================================================

/**
 * Composable to transfer funds
 */
export function useTransfer(): UseMutationReturn<
  HolochainRecord<Transaction>,
  {
    from_wallet: string;
    to_wallet: string;
    amount: number;
    currency: string;
    memo?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get transaction history
 */
export function useTransactionHistory(
  _walletId: string,
  _options?: UseQueryOptions & { limit?: number; offset?: number }
): UseQueryReturn<HolochainRecord<Transaction>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a transaction
 */
export function useTransaction(
  _transactionId: string
): UseQueryReturn<HolochainRecord<Transaction> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get transactions between two wallets
 */
export function useTransactionsBetween(
  _walletA: string,
  _walletB: string
): UseQueryReturn<HolochainRecord<Transaction>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Credit Composables
// ============================================================================

/**
 * Composable to get credit score
 */
export function useCreditScore(_did: string): UseQueryReturn<CreditScore> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get credit score history
 */
export function useCreditScoreHistory(
  _did: string
): UseQueryReturn<Array<{ score: number; date: number }>> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to calculate credit score (does not save)
 */
export function useCalculateCreditScore(): UseMutationReturn<CreditScore, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Lending Composables
// ============================================================================

/**
 * Composable to request a loan
 */
export function useRequestLoan(): UseMutationReturn<
  HolochainRecord<Loan>,
  {
    amount: number;
    currency: string;
    term_days: number;
    collateral_asset_id?: string;
    purpose?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to offer a loan
 */
export function useOfferLoan(): UseMutationReturn<
  HolochainRecord<Loan>,
  {
    borrower_did: string;
    amount: number;
    currency: string;
    interest_rate: number;
    term_days: number;
    require_collateral?: boolean;
    collateral_percentage?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to accept a loan offer
 */
export function useAcceptLoan(): UseMutationReturn<
  HolochainRecord<Loan>,
  {
    loan_id: string;
    collateral_asset_id?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a loan
 */
export function useLoan(_loanId: string): UseQueryReturn<HolochainRecord<Loan> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get loans by borrower
 */
export function useLoansByBorrower(_borrowerDid: string): UseQueryReturn<HolochainRecord<Loan>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get loans by lender
 */
export function useLoansByLender(_lenderDid: string): UseQueryReturn<HolochainRecord<Loan>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to make loan payment
 */
export function useMakeLoanPayment(): UseMutationReturn<
  HolochainRecord<LoanPayment>,
  {
    loan_id: string;
    amount: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get loan payment schedule
 */
export function useLoanPaymentSchedule(_loanId: string): UseQueryReturn<
  Array<{
    due_date: number;
    amount: number;
    principal: number;
    interest: number;
    paid: boolean;
  }>
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Treasury Composables
// ============================================================================

/**
 * Composable to get treasury balance
 */
export function useTreasuryBalance(_treasuryId: string): UseQueryReturn<Map<string, number>> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to propose treasury allocation
 */
export function useProposeAllocation(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    treasury_id: string;
    recipient_wallet: string;
    amount: number;
    currency: string;
    purpose: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get treasury transactions
 */
export function useTreasuryTransactions(
  _treasuryId: string
): UseQueryReturn<HolochainRecord<Transaction>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Escrow Composables
// ============================================================================

/**
 * Composable to create escrow
 */
export function useCreateEscrow(): UseMutationReturn<
  HolochainRecord<Wallet>,
  {
    parties: string[];
    release_conditions: string;
    expiry_date?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to release escrow
 */
export function useReleaseEscrow(): UseMutationReturn<
  HolochainRecord<Transaction>,
  {
    escrow_wallet_id: string;
    recipient_wallet: string;
    amount: number;
    currency: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to refund escrow
 */
export function useRefundEscrow(): UseMutationReturn<
  HolochainRecord<Transaction[]>,
  {
    escrow_wallet_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

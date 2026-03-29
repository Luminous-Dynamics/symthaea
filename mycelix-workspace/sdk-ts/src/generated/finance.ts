// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finance types (Loan, Payment, Wallet, etc.)
 *
 * Auto-generated TypeScript types from Rust SDK.
 * DO NOT EDIT MANUALLY - regenerate with: pnpm generate:types
 *
 * @module @mycelix/sdk/generated/finance
 * @generated
 */

/**
 * A financial wallet
 * @generated
 */
export interface Wallet {
  /** Wallet ID */
  id: string;
  /** Owner DID */
  owner: string;
  /** Type of wallet */
  walletType: WalletType;
  /** Wallet name */
  name: string;
  /** Balances by currency */
  balances: Record<string, number>;
  /** Creation timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
}

/**
 * Type of wallet
 * @generated
 */
export type WalletType = 'Personal' | 'Business' | 'DAO' | 'Escrow';

/**
 * A financial transaction
 * @generated
 */
export interface Transaction {
  /** Transaction ID */
  id: string;
  /** Source wallet */
  fromWallet?: string;
  /** Destination wallet */
  toWallet?: string;
  /** Transaction amount */
  amount: number;
  /** Currency */
  currency: Currency;
  /** Type of transaction */
  transactionType: TransactionType;
  /** Transaction status */
  status: TransactionStatus;
  /** Transaction memo */
  memo?: string;
  /** Creation timestamp */
  createdAt: number;
}

/**
 * Type of transaction
 * @generated
 */
export type TransactionType = 'Transfer' | 'Deposit' | 'Withdrawal' | 'LoanDisbursement' | 'LoanRepayment' | 'Fee';

/**
 * Status of a transaction
 * @generated
 */
export type TransactionStatus = 'Pending' | 'Confirmed' | 'Failed' | 'Reversed';

/**
 * Supported currencies
 * @generated
 */
export type Currency = 'MYC' | 'USD' | 'EUR' | 'BTC' | 'ETH' | 'ENERGY';

/**
 * A loan record
 * @generated
 */
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
  interestRate: number;
  /** Term in days */
  termDays: number;
  /** Type of loan */
  loanType: LoanType;
  /** Loan status */
  status: LoanStatus;
  /** Amount repaid */
  amountRepaid: number;
  /** Creation timestamp */
  createdAt: number;
  /** Due date timestamp */
  dueAt?: number;
}

/**
 * Type of loan
 * @generated
 */
export type LoanType = 'Personal' | 'Business' | 'Microfinance' | 'Collateralized';

/**
 * Status of a loan
 * @generated
 */
export type LoanStatus = 'Pending' | 'Active' | 'Repaid' | 'Defaulted' | 'Cancelled';

/**
 * MATL-based credit score
 * @generated
 */
export interface CreditScore {
  /** Subject DID */
  did: string;
  /** Numeric score (0-1000) */
  score: number;
  /** Credit tier */
  tier: CreditTier;
  /** MATL contribution (0-1) */
  matlFactor: number;
  /** Payment history factor */
  paymentFactor: number;
  /** Utilization factor */
  utilizationFactor: number;
  /** History length factor */
  historyFactor: number;
  /** Total credit limit */
  totalLimit: number;
  /** Current utilization */
  currentUtilization: number;
  /** Last update timestamp */
  updatedAt: number;
}

/**
 * Credit score tier
 * @generated
 */
export type CreditTier = 'Unrated' | 'Bronze' | 'Silver' | 'Gold' | 'Platinum';

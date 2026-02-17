/**
 * Transactions Zome Functions
 *
 * Type-safe wrappers for all transaction-related zome calls
 */

import type { AppClient } from '@holochain/client';
import { callZome } from './client';
import type {
  Transaction,
  CreateTransactionInput,
  UpdateTransactionStatusInput,
} from '$types';

/**
 * Get a single transaction by hash
 *
 * @param client - Holochain client
 * @param transactionHash - Transaction action hash
 * @returns Transaction details
 */
export async function getTransaction(
  client: AppClient,
  transactionHash: string
): Promise<Transaction> {
  return callZome<Transaction>(client, 'transactions', 'get_transaction', {
    transaction_hash: transactionHash,
  });
}

/**
 * Get all my transactions (purchases + sales)
 *
 * @param client - Holochain client
 * @returns Array of transactions
 */
export async function getMyTransactions(client: AppClient): Promise<Transaction[]> {
  return callZome<Transaction[]>(client, 'transactions', 'get_my_transactions', {});
}

/**
 * Get my purchases
 *
 * @param client - Holochain client
 * @returns Array of purchase transactions
 */
export async function getMyPurchases(client: AppClient): Promise<Transaction[]> {
  return callZome<Transaction[]>(client, 'transactions', 'get_my_purchases', {});
}

/**
 * Get my sales
 *
 * @param client - Holochain client
 * @returns Array of sale transactions
 */
export async function getMySales(client: AppClient): Promise<Transaction[]> {
  return callZome<Transaction[]>(client, 'transactions', 'get_my_sales', {});
}

/**
 * Create a new transaction (purchase)
 *
 * @param client - Holochain client
 * @param input - Transaction data
 * @returns Created transaction
 */
export async function createTransaction(
  client: AppClient,
  input: CreateTransactionInput
): Promise<Transaction> {
  return callZome<Transaction>(client, 'transactions', 'create_transaction', input);
}

/**
 * Update transaction status
 *
 * @param client - Holochain client
 * @param input - Status update data
 * @returns Updated transaction
 */
export async function updateTransactionStatus(
  client: AppClient,
  input: UpdateTransactionStatusInput
): Promise<Transaction> {
  return callZome<Transaction>(client, 'transactions', 'update_transaction_status', input);
}

/**
 * Confirm delivery (buyer action)
 *
 * @param client - Holochain client
 * @param transactionHash - Transaction action hash
 * @returns Updated transaction
 */
export async function confirmDelivery(
  client: AppClient,
  transactionHash: string
): Promise<Transaction> {
  return callZome<Transaction>(client, 'transactions', 'confirm_delivery', {
    transaction_hash: transactionHash,
  });
}

/**
 * Mark as shipped (seller action)
 *
 * @param client - Holochain client
 * @param transactionHash - Transaction action hash
 * @param trackingNumber - Shipping tracking number
 * @returns Updated transaction
 */
export async function markAsShipped(
  client: AppClient,
  transactionHash: string,
  trackingNumber: string
): Promise<Transaction> {
  return callZome<Transaction>(client, 'transactions', 'mark_as_shipped', {
    transaction_hash: transactionHash,
    tracking_number: trackingNumber,
  });
}

/**
 * Get transactions by listing
 *
 * @param client - Holochain client
 * @param listingHash - Listing action hash
 * @returns Array of transactions for this listing
 */
export async function getTransactionsByListing(
  client: AppClient,
  listingHash: string
): Promise<Transaction[]> {
  return callZome<Transaction[]>(client, 'transactions', 'get_transactions_by_listing', {
    listing_hash: listingHash,
  });
}

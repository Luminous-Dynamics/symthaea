// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Transaction Type Definitions
 *
 * Core data structures for marketplace transactions
 */

/**
 * Transaction status in lifecycle
 */
export type TransactionStatus =
  | 'pending' // Order placed, awaiting seller action
  | 'shipped' // Seller marked as shipped
  | 'delivered' // Package delivered (awaiting confirmation)
  | 'completed' // Buyer confirmed delivery
  | 'disputed'; // Dispute filed

/**
 * Payment method options
 */
export type PaymentMethod = 'crypto' | 'credit_card' | 'paypal' | 'bank_transfer';

/**
 * Shipping address
 */
export interface ShippingAddress {
  name: string;
  address_line_1: string;
  address_line_2?: string;
  city: string;
  state: string;
  postal_code: string;
  country: string;
}

/**
 * Complete transaction data structure
 */
export interface Transaction {
  /** Transaction ID (Holochain action hash) */
  id: string;

  /** Associated listing hash */
  listing_hash: string;

  /** Listing title (denormalized for display) */
  listing_title: string;

  /** Listing photo CID (first photo) */
  listing_photo_cid?: string;

  /** Buyer agent ID */
  buyer_agent_id: string;

  /** Buyer username */
  buyer_name: string;

  /** Seller agent ID */
  seller_agent_id: string;

  /** Seller username */
  seller_name: string;

  /** Seller trust score */
  seller_trust_score: number;

  /** Buyer trust score */
  buyer_trust_score?: number;

  /** Quantity purchased */
  quantity: number;

  /** Price per unit */
  unit_price: number;

  /** Total price (unit_price * quantity) */
  total_price: number;

  /** Current status */
  status: TransactionStatus;

  /** Shipping address */
  shipping_address: ShippingAddress;

  /** Payment method */
  payment_method: PaymentMethod;

  /** Cryptocurrency wallet address (if crypto payment) */
  wallet_address?: string;

  /** Tracking number (when shipped) */
  tracking_number?: string;

  /** Order placed timestamp */
  created_at: number;

  /** Shipped timestamp */
  shipped_at?: number;

  /** Delivered timestamp */
  delivered_at?: number;

  /** Completed timestamp */
  completed_at?: number;

  /** Transaction hash (for Holochain DHT) */
  transaction_hash?: string;

  /** Whether buyer can confirm delivery */
  can_confirm_delivery?: boolean;

  /** Whether buyer can leave review */
  can_leave_review?: boolean;

  /** Whether buyer can file dispute */
  can_file_dispute?: boolean;

  /** Whether seller can mark as shipped */
  can_mark_shipped?: boolean;

  /** Risk snapshot hash at checkout time (optional) */
  risk_snapshot_hash?: string;
}

/**
 * Transaction creation input
 */
export interface CreateTransactionInput {
  listing_hash: string;
  quantity: number;
  shipping_address: ShippingAddress;
  payment_method: PaymentMethod;
  wallet_address?: string;
  trust_override_note?: string;
  trust_override_flags?: Record<string, string | undefined>;
  risk_snapshot_hash?: string;
}

/**
 * Transaction status update input
 */
export interface UpdateTransactionStatusInput {
  transaction_hash: string;
  status: TransactionStatus;
  tracking_number?: string;
}

/**
 * Transaction filter criteria
 */
export interface TransactionFilters {
  /** Filter by type (all, purchases, sales) */
  type?: 'all' | 'purchases' | 'sales';

  /** Filter by status */
  status?: TransactionStatus | 'all';
}

/**
 * Transaction statistics for user dashboard
 */
export interface TransactionStats {
  /** Total purchases made */
  total_purchases: number;

  /** Total sales made */
  total_sales: number;

  /** Total amount spent */
  total_spent: number;

  /** Total amount earned */
  total_earned: number;

  /** Average transaction value */
  average_transaction_value: number;
}

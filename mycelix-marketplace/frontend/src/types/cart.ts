/**
 * Shopping Cart Type Definitions
 *
 * Data structures for shopping cart management
 */

import type { Listing } from './listing';

/**
 * Cart item (listing + quantity)
 */
export interface CartItem {
  /** Listing hash */
  listing_hash: string;

  /** Listing title */
  title: string;

  /** Price per unit */
  price: number;

  /** Quantity in cart */
  quantity: number;

  /** Listing photo CID (first photo) */
  photo_cid?: string;

  /** Seller agent ID */
  seller_agent_id: string;

  /** Seller username */
  seller_name: string;

  /** Maximum available quantity */
  max_quantity?: number;

  /** Added to cart timestamp */
  added_at: number;
}

/**
 * Cart state
 */
export interface CartState {
  /** All items in cart */
  items: CartItem[];

  /** Total number of items */
  itemCount: number;

  /** Subtotal (sum of price * quantity) */
  subtotal: number;

  /** Tax amount (8%) */
  tax: number;

  /** Shipping cost */
  shipping: number;

  /** Total cost */
  total: number;

  /** Last updated timestamp */
  lastUpdated: number;
}

/**
 * Add to cart input
 */
export interface AddToCartInput {
  listing: Listing;
  seller_name: string;
  quantity?: number;
}

/**
 * Update cart item quantity input
 */
export interface UpdateCartItemInput {
  listing_hash: string;
  quantity: number;
}

/**
 * Shopping Cart Store
 *
 * Centralized state management for shopping cart with localStorage persistence
 */

import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';
import type { CartItem, CartState } from '$types';

/**
 * Tax rate (8%)
 */
const TAX_RATE = 0.08;

/**
 * Flat shipping rate
 */
const SHIPPING_COST = 5.99;

/**
 * localStorage key
 */
const STORAGE_KEY = 'mycelix_cart';

/**
 * Load cart from localStorage
 */
function loadCart(): CartItem[] {
  if (!browser) return [];

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (e) {
    console.error('Failed to load cart from localStorage:', e);
    return [];
  }
}

/**
 * Save cart to localStorage
 */
function saveCart(items: CartItem[]): void {
  if (!browser) return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch (e) {
    console.error('Failed to save cart to localStorage:', e);
  }
}

/**
 * Cart items store
 */
function createCartStore() {
  const { subscribe, set, update } = writable<CartItem[]>(loadCart());

  // Subscribe to changes and persist to localStorage
  if (browser) {
    subscribe((items) => {
      saveCart(items);
    });
  }

  return {
    subscribe,

    /**
     * Add item to cart or increase quantity if already exists
     */
    addItem: (item: Omit<CartItem, 'added_at'>) => {
      update((items) => {
        const existingIndex = items.findIndex((i) => i.listing_hash === item.listing_hash);

        if (existingIndex >= 0) {
          // Item exists, increase quantity
          items[existingIndex].quantity += item.quantity;
        } else {
          // New item
          items.push({
            ...item,
            added_at: Date.now(),
          });
        }

        return items;
      });
    },

    /**
     * Remove item from cart
     */
    removeItem: (listingHash: string) => {
      update((items) => items.filter((item) => item.listing_hash !== listingHash));
    },

    /**
     * Update item quantity
     */
    updateQuantity: (listingHash: string, quantity: number) => {
      update((items) => {
        const item = items.find((i) => i.listing_hash === listingHash);
        if (item) {
          if (quantity <= 0) {
            // Remove if quantity is 0 or negative
            return items.filter((i) => i.listing_hash !== listingHash);
          }
          item.quantity = quantity;
        }
        return items;
      });
    },

    /**
     * Clear entire cart
     */
    clear: () => {
      set([]);
    },

    /**
     * Get item by listing hash
     */
    getItem: (listingHash: string, items: CartItem[]) => {
      return items.find((item) => item.listing_hash === listingHash);
    },
  };
}

/**
 * Cart items store
 */
export const cartItems = createCartStore();

/**
 * Derived store: item count
 */
export const itemCount = derived(cartItems, ($items) =>
  $items.reduce((sum, item) => sum + item.quantity, 0)
);

/**
 * Derived store: subtotal
 */
export const subtotal = derived(cartItems, ($items) =>
  $items.reduce((sum, item) => sum + item.price * item.quantity, 0)
);

/**
 * Derived store: tax
 */
export const tax = derived(subtotal, ($subtotal) => $subtotal * TAX_RATE);

/**
 * Derived store: shipping
 */
export const shipping = derived(cartItems, ($items) => ($items.length > 0 ? SHIPPING_COST : 0));

/**
 * Derived store: total
 */
export const total = derived([subtotal, tax, shipping], ([$subtotal, $tax, $shipping]) =>
  $subtotal + $tax + $shipping
);

/**
 * Derived store: complete cart state
 */
export const cartState = derived(
  [cartItems, itemCount, subtotal, tax, shipping, total],
  ([$items, $itemCount, $subtotal, $tax, $shipping, $total]): CartState => ({
    items: $items,
    itemCount: $itemCount,
    subtotal: $subtotal,
    tax: $tax,
    shipping: $shipping,
    total: $total,
    lastUpdated: Date.now(),
  })
);

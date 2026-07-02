// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Transaction } from '$types';

const now = Date.now();

export function getMockTransactions(): Transaction[] {
  return [
    {
      id: 'tx-mock-1',
      transaction_hash: 'tx-mock-1',
      listing_hash: 'mock-listing-3',
      listing_title: 'Retro Handheld Console',
      listing_photo_cid: undefined,
      buyer_agent_id: 'buyer-mock-1',
      buyer_name: 'Retro Fan',
      seller_agent_id: 'agent_beta',
      seller_name: 'Beta Bazaar',
      seller_trust_score: 88,
      buyer_trust_score: 82,
      quantity: 1,
      unit_price: 79.99,
      total_price: 79.99,
      status: 'shipped',
      shipping_address: {
        name: 'Retro Fan',
        address_line_1: '123 Memory Ln',
        address_line_2: '',
        city: 'Arcadia',
        state: 'CA',
        postal_code: '90001',
        country: 'USA',
      },
      payment_method: 'crypto',
      wallet_address: '0xabc123',
      tracking_number: 'TRACK12345',
      created_at: now - 1000 * 60 * 60 * 24 * 5,
      shipped_at: now - 1000 * 60 * 60 * 24 * 3,
      can_confirm_delivery: true,
    },
    {
      id: 'tx-mock-2',
      transaction_hash: 'tx-mock-2',
      listing_hash: 'mock-listing-1',
      listing_title: 'Hand-painted Ceramic Planter',
      listing_photo_cid: undefined,
      buyer_agent_id: 'buyer-mock-2',
      buyer_name: 'Plant Lover',
      seller_agent_id: 'agent_alpha',
      seller_name: 'Alice Artisans',
      seller_trust_score: 92,
      buyer_trust_score: 80,
      quantity: 2,
      unit_price: 24.99,
      total_price: 49.98,
      status: 'pending',
      shipping_address: {
        name: 'Plant Lover',
        address_line_1: '456 Green St',
        address_line_2: 'Apt 2',
        city: 'Verdant',
        state: 'OR',
        postal_code: '97035',
        country: 'USA',
      },
      payment_method: 'credit_card',
      created_at: now - 1000 * 60 * 60 * 24 * 2,
      can_mark_shipped: true,
    },
  ];
}

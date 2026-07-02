// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Central Type Definitions Export
 *
 * Import all types from this single location:
 * import type { Listing, Transaction, UserProfile } from '$types';
 */

// Listing types
export type {
  Listing,
  ListingCategory,
  ListingStatus,
  SellerInfo,
  Review,
  ListingWithContext,
  CreateListingInput,
  UpdateListingInput,
  ListingFilters,
  ListingSortFn,
} from './listing';

// Transaction types
export type {
  Transaction,
  TransactionStatus,
  PaymentMethod,
  ShippingAddress,
  CreateTransactionInput,
  UpdateTransactionStatusInput,
  TransactionFilters,
  TransactionStats,
} from './transaction';

// User types
export type {
  UserProfile,
  UserRole,
  UserStats,
  TrustBreakdown,
  AuthState,
  UserPreferences,
} from './user';

// Dispute types
export type {
  Dispute,
  DisputeStatus,
  DisputeReason,
  ArbitratorVote,
  Arbitrator,
  ArbitratorVoteRecord,
  ArbitratorProfile,
  CreateDisputeInput,
  CastVoteInput,
  DisputeFilters,
  MRCStats,
} from './dispute';

// Cart types
export type {
  CartItem,
  CartState,
  AddToCartInput,
  UpdateCartItemInput,
} from './cart';

// Reputation and trust graph types
export type {
  ReputationClaim,
  TrustGraphEdge,
  TrustGraphNode,
  TrustGraphSnapshot,
  ProofTrailItem,
} from './reputation';

// Proof request tracking
export type { ProofRequest, ProofRequestStatus } from './proof';

// Risk signals
export type { RiskSignal } from './risk';

// Intent-based buying
export type { IntentRequest, IntentBundleSuggestion, IntentBundleItem } from './intent';

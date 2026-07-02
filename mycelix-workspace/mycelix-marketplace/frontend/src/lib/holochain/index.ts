// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Barrel export for Holochain client functionality
 */

// Client initialization and management
export { initHolochainClient, disconnectHolochainClient } from './client';

// Identity integration (mycelix-identity)
export {
  // Connection
  checkIdentityConnection,
  isIdentityAvailable,
  // DID Resolution
  resolveDID,
  resolveDIDFromAgent,
  getMyDID,
  // Credential Verification
  verifyCredential,
  checkRevocationStatus,
  // Assurance Levels
  getAssuranceLevel,
  meetsAssuranceLevel,
  getRequiredAssuranceLevel,
  // Transaction Identity
  verifyTransactionIdentity,
  verifyHighValueTransaction,
  // Profile Enhancement
  getEnhancedSellerProfile,
  getEnhancedBuyerProfile,
  // Cache
  clearDIDCache,
  getCacheStats,
  // Utilities
  formatAssuranceLevel,
  getAssuranceLevelColor,
  isValidDID,
  // Constants
  ASSURANCE_LEVEL_VALUE,
  ASSURANCE_LEVEL_DESCRIPTION,
  // Types
  type AssuranceLevel,
  type DidDocument,
  type DidResolutionResult,
  type CredentialVerificationResult,
  type TransactionIdentityVerification,
  type EnhancedProfile,
  type HighValueTransactionConfig,
} from './identity';

// Listings management
export {
  createListing,
  getListing,
  getListingsByCategory,
  updateListing,
  searchListings
} from './listings';

// Transactions management
export {
  createTransaction,
  getTransaction,
  getMyTransactions,
  getMyPurchases,
  getMySales,
  updateTransactionStatus,
  confirmDelivery,
  markAsShipped,
  getTransactionsByListing
} from './transactions';

// User profile management and reviews
export {
  getUserProfile,
  getMyProfile,
  updateMyProfile,
  createReview,
  getReviewsForListing,
  getReviewsForSeller,
  getMyReviews,
  getReviewsIWrote
} from './users';

// Disputes and arbitration
export {
  createDispute,
  getDispute,
  getDisputesByStatus,
  getMyDisputes,
  getMyArbitrationCases,
  castArbitratorVote,
  getArbitratorProfile,
  isArbitrator,
  getAllDisputes
} from './disputes';

// Reputation + trust graph
export { getTrustGraph, getReputationClaims, getProofTrail } from './reputation';

// Types
export type {
  CreateListingInput,
  Listing,
  CreateTransactionInput,
  Transaction,
  TransactionStatus,
  UserProfile,
  CreateDisputeInput,
  Dispute,
  DisputeStatus,
  Review
} from '$types';

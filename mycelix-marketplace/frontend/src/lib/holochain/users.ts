/**
 * Users Zome Functions
 *
 * Type-safe wrappers for all user profile and review zome calls
 */

import type { AppClient } from '@holochain/client';
import { callZome } from './client';
import type { UserProfile, Review } from '$types';

/**
 * Get my profile (current user)
 *
 * @param client - Holochain client
 * @returns User profile
 */
export async function getMyProfile(client: AppClient): Promise<UserProfile> {
  return callZome<UserProfile>(client, 'users', 'get_my_profile', {});
}

/**
 * Get user profile by agent ID
 *
 * @param client - Holochain client
 * @param agentId - User agent ID
 * @returns User profile
 */
export async function getUserProfile(client: AppClient, agentId: string): Promise<UserProfile> {
  return callZome<UserProfile>(client, 'users', 'get_user_profile', {
    agent_id: agentId,
  });
}

/**
 * Update my profile
 *
 * @param client - Holochain client
 * @param profile - Updated profile data
 * @returns Updated profile
 */
export async function updateMyProfile(
  client: AppClient,
  profile: Partial<UserProfile>
): Promise<UserProfile> {
  return callZome<UserProfile>(client, 'users', 'update_my_profile', profile);
}

/**
 * Create a review for a seller
 *
 * @param client - Holochain client
 * @param input - Review data
 * @returns Created review
 */
export async function createReview(
  client: AppClient,
  input: {
    listing_hash: string;
    transaction_hash: string;
    rating: number;
    comment: string;
  }
): Promise<Review> {
  return callZome<Review>(client, 'reviews', 'create_review', input);
}

/**
 * Get reviews for a listing
 *
 * @param client - Holochain client
 * @param listingHash - Listing action hash
 * @returns Array of reviews
 */
export async function getReviewsForListing(
  client: AppClient,
  listingHash: string
): Promise<Review[]> {
  return callZome<Review[]>(client, 'reviews', 'get_reviews_for_listing', {
    listing_hash: listingHash,
  });
}

/**
 * Get reviews for a seller
 *
 * @param client - Holochain client
 * @param agentId - Seller agent ID (optional, defaults to current user)
 * @returns Array of reviews
 */
export async function getReviewsForSeller(
  client: AppClient,
  agentId?: string
): Promise<Review[]> {
  return callZome<Review[]>(client, 'reviews', 'get_reviews_for_seller', {
    agent_id: agentId,
  });
}

/**
 * Get my reviews (reviews I received)
 *
 * @param client - Holochain client
 * @returns Array of reviews
 */
export async function getMyReviews(client: AppClient): Promise<Review[]> {
  return callZome<Review[]>(client, 'reviews', 'get_my_reviews', {});
}

/**
 * Get reviews I've written
 *
 * @param client - Holochain client
 * @returns Array of reviews I authored
 */
export async function getReviewsIWrote(client: AppClient): Promise<Review[]> {
  return callZome<Review[]>(client, 'reviews', 'get_reviews_i_wrote', {});
}

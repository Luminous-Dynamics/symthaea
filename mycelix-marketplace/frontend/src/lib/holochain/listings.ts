/**
 * Listings Zome Functions
 *
 * Type-safe wrappers for all listing-related zome calls
 */

import type { AppClient } from '@holochain/client';
import { callZome } from './client';
import type {
  Listing,
  ListingWithContext,
  CreateListingInput,
  UpdateListingInput,
} from '$types';

/**
 * Get a single listing by hash
 *
 * @param client - Holochain client
 * @param listingHash - Listing action hash
 * @returns Listing with context (seller, reviews)
 */
export async function getListing(
  client: AppClient,
  listingHash: string
): Promise<ListingWithContext> {
  return callZome<ListingWithContext>(client, 'listings', 'get_listing', {
    listing_hash: listingHash,
  });
}

/**
 * Get all listings in the marketplace
 *
 * @param client - Holochain client
 * @returns Array of listings
 */
export async function getAllListings(client: AppClient): Promise<Listing[]> {
  const result = await callZome<{ listings: Listing[] }>(
    client,
    'listings',
    'get_all_listings',
    {}
  );
  return result.listings;
}

/**
 * Get listings by seller agent ID
 *
 * @param client - Holochain client
 * @param agentId - Seller agent ID
 * @returns Array of listings
 */
export async function getListingsBySeller(
  client: AppClient,
  agentId: string
): Promise<Listing[]> {
  return callZome<Listing[]>(client, 'listings', 'get_listings_by_seller', {
    agent_id: agentId,
  });
}

/**
 * Get my listings (current user)
 *
 * @param client - Holochain client
 * @returns Array of listings
 */
export async function getMyListings(client: AppClient): Promise<Listing[]> {
  return callZome<Listing[]>(client, 'listings', 'get_my_listings', {});
}

/**
 * Create a new listing
 *
 * @param client - Holochain client
 * @param input - Listing data
 * @returns Created listing with hash
 */
export async function createListing(
  client: AppClient,
  input: CreateListingInput
): Promise<Listing> {
  return callZome<Listing>(client, 'listings', 'create_listing', input);
}

/**
 * Update an existing listing
 *
 * @param client - Holochain client
 * @param input - Update data
 * @returns Updated listing
 */
export async function updateListing(
  client: AppClient,
  input: UpdateListingInput
): Promise<Listing> {
  return callZome<Listing>(client, 'listings', 'update_listing', input);
}

/**
 * Delete a listing
 *
 * @param client - Holochain client
 * @param listingHash - Listing action hash
 * @returns Success boolean
 */
export async function deleteListing(client: AppClient, listingHash: string): Promise<boolean> {
  return callZome<boolean>(client, 'listings', 'delete_listing', {
    listing_hash: listingHash,
  });
}

/**
 * Search listings by query
 *
 * @param client - Holochain client
 * @param query - Search query
 * @returns Array of matching listings
 */
export async function searchListings(client: AppClient, query: string): Promise<Listing[]> {
  return callZome<Listing[]>(client, 'listings', 'search_listings', {
    query,
  });
}

/**
 * Get listings by category
 *
 * @param client - Holochain client
 * @param category - Category name
 * @returns Array of listings in category
 */
export async function getListingsByCategory(
  client: AppClient,
  category: string
): Promise<Listing[]> {
  return callZome<Listing[]>(client, 'listings', 'get_listings_by_category', {
    category,
  });
}

/**
 * Increment listing view count
 *
 * @param client - Holochain client
 * @param listingHash - Listing action hash
 */
export async function incrementListingViews(
  client: AppClient,
  listingHash: string
): Promise<void> {
  await callZome(client, 'listings', 'increment_views', {
    listing_hash: listingHash,
  });
}

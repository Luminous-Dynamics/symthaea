// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Listing, ListingCategory, SellerInfo, Review, ListingWithContext } from '$types';
import { getMockReviews } from './reviews';
import { getMockSeller } from './sellers';

const sampleCategories: ListingCategory[] = [
  'Electronics',
  'Home & Garden',
  'Art & Collectibles',
  'Toys & Games',
  'Fashion',
];

const sellers: SellerInfo[] = [
  {
    agent_id: 'agent_alpha',
    username: 'Alice Artisans',
    trust_score: 92,
    total_listings: 12,
    total_sales: 58,
    average_rating: 4.8,
    member_since: Date.now() - 1000 * 60 * 60 * 24 * 365 * 2,
  },
  {
    agent_id: 'agent_beta',
    username: 'Beta Bazaar',
    trust_score: 88,
    total_listings: 20,
    total_sales: 134,
    average_rating: 4.6,
    member_since: Date.now() - 1000 * 60 * 60 * 24 * 365 * 3,
  },
];

const baseListings: Listing[] = [
  {
    id: 'mock-listing-1',
    listing_hash: 'mock-listing-1',
    title: 'Hand-painted Ceramic Planter',
    description: 'A vibrant, hand-painted ceramic planter perfect for succulents and herbs.',
    price: 24.99,
    category: sampleCategories[1],
    photos_ipfs_cids: [],
    seller_agent_id: sellers[0].agent_id,
    created_at: Date.now() - 1000 * 60 * 60 * 24 * 5,
    status: 'active',
    quantity_available: 8,
    views: 42,
  },
  {
    id: 'mock-listing-2',
    listing_hash: 'mock-listing-2',
    title: 'Upcycled Denim Tote Bag',
    description: 'Durable tote made from upcycled denim with reinforced stitching and inner pocket.',
    price: 39.5,
    category: sampleCategories[4],
    photos_ipfs_cids: [],
    seller_agent_id: sellers[1].agent_id,
    created_at: Date.now() - 1000 * 60 * 60 * 24 * 10,
    status: 'active',
    quantity_available: 15,
    views: 73,
  },
  {
    id: 'mock-listing-3',
    listing_hash: 'mock-listing-3',
    title: 'Retro Handheld Console',
    description: 'Play 500 classic games on a crisp IPS screen with USB-C charging.',
    price: 79.99,
    category: sampleCategories[0],
    photos_ipfs_cids: [],
    seller_agent_id: sellers[1].agent_id,
    created_at: Date.now() - 1000 * 60 * 60 * 24 * 2,
    status: 'active',
    quantity_available: 25,
    views: 112,
  },
  {
    id: 'mock-listing-4',
    listing_hash: 'mock-listing-4',
    title: 'Ocean Resin Coasters (Set of 4)',
    description: 'Epoxy resin coasters with swirling ocean patterns and cork backing.',
    price: 29.0,
    category: sampleCategories[2],
    photos_ipfs_cids: [],
    seller_agent_id: sellers[0].agent_id,
    created_at: Date.now() - 1000 * 60 * 60 * 24 * 8,
    status: 'active',
    quantity_available: 12,
    views: 65,
  },
];

const listingMap = new Map<string, ListingWithContext>();
baseListings.forEach((listing) => {
  const seller =
    getMockSeller(listing.seller_agent_id) ||
    sellers.find((s) => s.agent_id === listing.seller_agent_id) ||
    sellers[0];
  listingMap.set(listing.listing_hash || listing.id, {
    listing,
    seller,
    reviews: getMockReviews(listing.listing_hash || listing.id),
  });
});

export function getMockListings(): Listing[] {
  return baseListings;
}

export function getMockListingWithContext(id: string): ListingWithContext | null {
  return listingMap.get(id) || null;
}

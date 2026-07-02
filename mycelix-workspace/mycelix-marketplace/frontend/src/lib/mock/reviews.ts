import type { Review } from '$types';

const now = Date.now();

const baseReviews: Review[] = [
  {
    id: 'rev-1',
    listing_hash: 'mock-listing-1',
    reviewer_agent_id: 'buyer-mock-2',
    reviewer_name: 'Plant Lover',
    rating: 5,
    comment: 'Beautiful planter, well packaged and quick ship.',
    created_at: now - 1000 * 60 * 60 * 24 * 2,
    transaction_hash: 'tx-mock-2',
  },
  {
    id: 'rev-2',
    listing_hash: 'mock-listing-3',
    reviewer_agent_id: 'buyer-mock-1',
    reviewer_name: 'Retro Fan',
    rating: 4,
    comment: 'Console works great, display is crisp. Minor scuffs on box.',
    created_at: now - 1000 * 60 * 60 * 24 * 5,
    transaction_hash: 'tx-mock-1',
  },
];

export function getMockReviews(listingHash: string): Review[] {
  return baseReviews.filter((review) => review.listing_hash === listingHash);
}

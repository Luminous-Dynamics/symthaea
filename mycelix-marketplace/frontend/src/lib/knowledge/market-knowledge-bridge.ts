import type { AppClient } from '@holochain/client';
import {
  KnowledgeService,
  type CreateClaimInput,
  type EpistemicPosition,
} from '@mycelix/knowledge-client';

import type { Listing, ListingCategory } from '$types';

export interface ListingClaimContext {
  listing: Listing;
  sellerAgentId?: string | null;
}

export async function submitListingClaim(
  appClient: AppClient,
  context: ListingClaimContext
): Promise<void> {
  const { listing, sellerAgentId } = context;

  const classification: EpistemicPosition = {
    empirical: 0.6,
    normative: 0.4,
    mythic: 0.2,
  };

  const listingId = listing.listing_hash ?? listing.id;
  const topics = buildTopics(listing.category);

  const claimInput: CreateClaimInput = {
    content: listing.title,
    classification,
    domain: 'marketplace',
    topics,
    sources: [
      {
        uri: `happ://marketplace/listing/${listingId}`,
        title: listing.title,
        author: sellerAgentId ?? undefined,
        publishedAt: listing.created_at,
        reliability: 0.5,
      },
    ],
    evidence: [],
  };

  const service = new KnowledgeService(appClient, 'knowledge');

  try {
    await service.submitAndAnalyzeClaim(claimInput);
  } catch (error) {
    // Knowledge integration is best-effort and should never block listing creation.
    console.warn('Failed to submit listing claim to Mycelix Knowledge:', error);
  }
}

function buildTopics(category: ListingCategory): string[] {
  const topics = ['marketplace', 'listing'];

  if (category) {
    topics.push(category);
  }

  return topics;
}


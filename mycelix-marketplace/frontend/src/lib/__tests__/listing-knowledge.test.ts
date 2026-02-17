import { describe, expect, it, vi } from 'vitest';

import { initKnowledgeClient } from '$lib/knowledge/listing-knowledge';
import type { Listing } from '$types';

// Mock Holochain client init
vi.mock('$lib/holochain', () => ({
  initHolochainClient: vi.fn(async () => ({})),
}));

// Prepare KnowledgeClient mock shape
const searchMock = vi.fn();
const getDependencyTreeMock = vi.fn();
const calculateCredMock = vi.fn();
const requestVerificationMarketMock = vi.fn();

vi.mock('@mycelix/knowledge-client', () => {
  return {
    KnowledgeClient: vi.fn().mockImplementation(() => ({
      query: {
        search: searchMock,
      },
      graph: {
        getDependencyTree: getDependencyTreeMock,
      },
      inference: {
        calculateEnhancedCredibility: calculateCredMock,
      },
      marketsIntegration: {
        requestVerificationMarket: requestVerificationMarketMock,
      },
    })),
    calculateInformationValue: vi.fn((uncertainty: number, dependentCount: number, averageWeight: number) => {
      // Simple deterministic function for testing
      return uncertainty * dependentCount * averageWeight;
    }),
    recommendVerification: vi.fn(() => ({
      recommend: true,
      reason: 'High information value with low verifiability - verification would be valuable',
      suggestedTargetE: 0.7,
    })),
  };
});

const baseListing: Listing = {
  id: 'listing-1',
  title: 'Solar capacity increased 15% in Q3',
  description: 'Test listing',
  price: 10,
  category: 'Books & Media',
  photos_ipfs_cids: [],
  seller_agent_id: 'agent_alpha',
  created_at: 1_700_000_000_000,
  status: 'active',
  quantity_available: 1,
  views: 0,
};

describe('ListingKnowledgeClient', () => {
  it('returns null snapshot when no claims are found', async () => {
    searchMock.mockResolvedValueOnce([]);

    const client = await initKnowledgeClient();
    const snapshot = await client.getListingKnowledgeSnapshot(baseListing);

    expect(snapshot.claim).toBeNull();
    expect(snapshot.credibility).toBeNull();
    expect(snapshot.verificationRecommendation).toBeNull();
  });

  it('returns claim, credibility, and verification recommendation when claim exists', async () => {
    searchMock.mockResolvedValueOnce([
      {
        id: 'claim-1',
        content: baseListing.title,
        classification: { empirical: 0.4, normative: 0.3, mythic: 0.2 },
      },
    ]);

    getDependencyTreeMock.mockResolvedValueOnce({
      totalDependencies: 5,
      aggregateWeight: 2.5,
    });

    calculateCredMock.mockResolvedValueOnce({
      id: 'cred-1',
      subject: 'claim-1',
      subjectType: 'Claim',
      overallScore: 0.85,
      components: {},
      matl: {},
      evidenceStrength: {},
      factors: [],
      assessedAt: Date.now(),
    });

    const client = await initKnowledgeClient();
    const snapshot = await client.getListingKnowledgeSnapshot(baseListing);

    expect(snapshot.claim).not.toBeNull();
    expect(snapshot?.claim?.id).toBe('claim-1');
    expect(snapshot.credibility?.overallScore).toBe(0.85);
    expect(snapshot.verificationRecommendation).not.toBeNull();
    expect(snapshot.verificationRecommendation?.recommend).toBe(true);
  });

  it('requests a verification market for a claim', async () => {
    requestVerificationMarketMock.mockResolvedValueOnce('market-hash-1');
    const client = await initKnowledgeClient();

    const result = await client.requestVerificationMarketForClaim('claim-xyz', 0.7, 0.8, Date.now() + 3600_000, [
      'marketplace',
    ]);

    expect(requestVerificationMarketMock).toHaveBeenCalledWith(
      'claim-xyz',
      0.7,
      0.8,
      expect.any(Number),
      ['marketplace']
    );
    expect(result).toBe('market-hash-1');
  });
});

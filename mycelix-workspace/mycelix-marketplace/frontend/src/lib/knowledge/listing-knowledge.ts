import type { AppClient } from '@holochain/client';
import {
  KnowledgeClient,
  type Claim,
  type EnhancedCredibilityScore,
  type InformationValue,
  calculateInformationValue,
  recommendVerification,
} from '@mycelix/knowledge-client';

import type { Listing } from '$types';
import { initHolochainClient } from '$lib/holochain';

export interface VerificationRecommendation {
  recommend: boolean;
  reason: string;
  suggestedTargetE: number;
}

export interface ListingKnowledgeSnapshot {
  claim: Claim | null;
  credibility: EnhancedCredibilityScore | null;
  verificationRecommendation: VerificationRecommendation | null;
}

export interface ListingKnowledgeClient {
  getListingKnowledgeSnapshot(listing: Listing): Promise<ListingKnowledgeSnapshot>;
  requestVerificationMarketForClaim(
    claimId: string,
    targetE: number,
    minConfidence: number,
    closesAt: number,
    tags?: string[]
  ): Promise<string>;
}

class ListingKnowledgeClientImpl implements ListingKnowledgeClient {
  private client: KnowledgeClient;

  constructor(appClient: AppClient, roleName: string = 'knowledge') {
    this.client = new KnowledgeClient(appClient, roleName);
  }

  async getListingKnowledgeSnapshot(listing: Listing): Promise<ListingKnowledgeSnapshot> {
    const listingId = listing.listing_hash ?? listing.id;
    const uri = `happ://marketplace/listing/${listingId}`;

    try {
      const claims = await this.client.query.search(uri, {
        minE: 0.0,
        limit: 1,
      });

      if (!claims.length) {
        return { claim: null, credibility: null, verificationRecommendation: null };
      }

      const claim = claims[0];
      const credibility = await this.client.inference.calculateEnhancedCredibility(
        claim.id,
        'Claim'
      );

      // Derive information value using the same approach as KnowledgeService.
      const dependencyTree = await this.client.graph.getDependencyTree(claim.id, 3);
      const uncertainty = 1 - claim.classification.empirical;
      const expectedValue = calculateInformationValue(
        uncertainty,
        dependencyTree.totalDependencies,
        dependencyTree.aggregateWeight / Math.max(dependencyTree.totalDependencies, 1)
      );

      const informationValue: InformationValue = {
        id: `iv-${claim.id}`,
        claimId: claim.id,
        expectedValue,
        dependentCount: dependencyTree.totalDependencies,
        averageDependencyWeight:
          dependencyTree.aggregateWeight / Math.max(dependencyTree.totalDependencies, 1),
        uncertainty,
        impactScore: dependencyTree.aggregateWeight,
        recommendedForVerification: uncertainty > 0.3,
        assessedAt: Date.now(),
        reasoning: 'Automatic marketplace listing assessment',
      };

      const verificationRecommendation = recommendVerification(claim, informationValue);

      return { claim, credibility, verificationRecommendation };
    } catch {
      return { claim: null, credibility: null, verificationRecommendation: null };
    }
  }

  async requestVerificationMarketForClaim(
    claimId: string,
    targetE: number,
    minConfidence: number,
    closesAt: number,
    tags?: string[]
  ): Promise<string> {
    const hash = await this.client.marketsIntegration.requestVerificationMarket(
      claimId,
      targetE,
      minConfidence,
      closesAt,
      tags
    );
    // Return a short string representation that can be surfaced in UI.
    return typeof hash === 'string' ? hash : JSON.stringify(hash);
  }
}

export async function initKnowledgeClient(): Promise<ListingKnowledgeClient> {
  const appClient = (await initHolochainClient()) as AppClient;
  return new ListingKnowledgeClientImpl(appClient);
}

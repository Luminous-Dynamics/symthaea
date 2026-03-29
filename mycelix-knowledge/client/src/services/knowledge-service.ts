// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge Service - High-level operations for the Knowledge Graph
 *
 * Provides composable operations that combine multiple zome calls
 * for common workflows.
 */

import { AppClient, AgentPubKey, ActionHash } from "@holochain/client";
import {
  KnowledgeClient,
  Claim,
  CreateClaimInput,
  EpistemicPosition,
  FactCheckResult,
  EnhancedCredibilityScore,
  InformationValue,
  CascadeResult,
  MarketSummary,
  calculateInformationValue,
  recommendVerification,
  toDiscreteEpistemic,
} from "../index";

/**
 * Result of submitting and verifying a claim
 */
export interface ClaimSubmissionResult {
  claimHash: ActionHash;
  credibility: EnhancedCredibilityScore;
  informationValue: InformationValue;
  verificationRecommendation: {
    recommend: boolean;
    reason: string;
    suggestedTargetE: number;
  };
}

/**
 * Result of a comprehensive fact check
 */
export interface ComprehensiveFactCheckResult {
  factCheck: FactCheckResult;
  relatedClaims: Claim[];
  contradictions: Claim[];
  credibilityAnalysis: EnhancedCredibilityScore[];
}

/**
 * Claim with full context
 */
export interface EnrichedClaim extends Claim {
  credibility?: EnhancedCredibilityScore;
  informationValue?: InformationValue;
  markets?: MarketSummary[];
  relatedClaims?: Claim[];
}

/**
 * High-level knowledge graph service
 */
export class KnowledgeService {
  private client: KnowledgeClient;

  constructor(appClient: AppClient, roleName: string = "knowledge") {
    this.client = new KnowledgeClient(appClient, roleName);
  }

  /**
   * Submit a claim with automatic credibility assessment and verification recommendation
   */
  async submitAndAnalyzeClaim(
    input: CreateClaimInput
  ): Promise<ClaimSubmissionResult> {
    // Create the claim
    const claimHash = await this.client.claims.createClaim(input);

    // Get credibility assessment
    const credibility = await this.client.inference.calculateEnhancedCredibility(
      claimHash.toString(),
      "Claim"
    );

    // Calculate information value
    const dependencyTree = await this.client.graph.getDependencyTree(
      claimHash.toString(),
      3
    );
    const uncertainty = 1 - input.classification.empirical;
    const informationValue: InformationValue = {
      id: `iv-${claimHash}`,
      claimId: claimHash.toString(),
      expectedValue: calculateInformationValue(
        uncertainty,
        dependencyTree.totalDependencies,
        dependencyTree.aggregateWeight / Math.max(dependencyTree.totalDependencies, 1)
      ),
      dependentCount: dependencyTree.totalDependencies,
      averageDependencyWeight:
        dependencyTree.aggregateWeight / Math.max(dependencyTree.totalDependencies, 1),
      uncertainty,
      impactScore: dependencyTree.aggregateWeight,
      recommendedForVerification: uncertainty > 0.3,
      assessedAt: Date.now(),
      reasoning: "Automatic assessment on submission",
    };

    // Get verification recommendation
    const claim = await this.client.claims.getClaim(claimHash);
    const verificationRecommendation = claim
      ? recommendVerification(claim, informationValue)
      : { recommend: false, reason: "Claim not found", suggestedTargetE: 0 };

    return {
      claimHash,
      credibility,
      informationValue,
      verificationRecommendation,
    };
  }

  /**
   * Perform comprehensive fact check with context
   */
  async comprehensiveFactCheck(
    statement: string,
    context?: string,
    minE: number = 0.5
  ): Promise<ComprehensiveFactCheckResult> {
    // Basic fact check
    const factCheck = await this.client.factcheck.factCheck({
      statement,
      context,
      minE,
    });

    // Get related claims
    const relatedClaims = await this.client.query.search(statement, {
      minE: minE * 0.8, // Slightly lower threshold for related
      limit: 20,
    });

    // Find contradictions among supporting claims
    const contradictions: Claim[] = [];
    for (const supporting of factCheck.supportingClaims) {
      const claimContradictions = await this.client.query.findContradictions(
        supporting.claimId
      );
      contradictions.push(...claimContradictions);
    }

    // Get credibility for key claims
    const credibilityAnalysis: EnhancedCredibilityScore[] = [];
    const claimIds = [
      ...factCheck.supportingClaims.map((c) => c.claimId),
      ...factCheck.contradictingClaims.map((c) => c.claimId),
    ].slice(0, 10); // Limit to top 10

    for (const claimId of claimIds) {
      try {
        const cred = await this.client.inference.calculateEnhancedCredibility(
          claimId,
          "Claim"
        );
        credibilityAnalysis.push(cred);
      } catch {
        // Skip claims that fail credibility assessment
      }
    }

    return {
      factCheck,
      relatedClaims,
      contradictions,
      credibilityAnalysis,
    };
  }

  /**
   * Get an enriched claim with all context
   */
  async getEnrichedClaim(claimHash: ActionHash): Promise<EnrichedClaim | null> {
    const claim = await this.client.claims.getClaim(claimHash);
    if (!claim) return null;

    const claimId = claimHash.toString();

    // Parallel fetch of additional data
    const [credibility, informationValue, markets, relatedClaims] =
      await Promise.allSettled([
        this.client.inference.calculateEnhancedCredibility(claimId, "Claim"),
        this.client.graph.rankByInformationValue(1).then((ivs) =>
          ivs.find((iv) => iv.claimId === claimId)
        ),
        this.client.claims.getClaimMarkets(claimId),
        this.client.query.findRelated(claimId, 2, 10),
      ]);

    return {
      ...claim,
      credibility:
        credibility.status === "fulfilled" ? credibility.value : undefined,
      informationValue:
        informationValue.status === "fulfilled" ? informationValue.value : undefined,
      markets: markets.status === "fulfilled" ? markets.value : undefined,
      relatedClaims:
        relatedClaims.status === "fulfilled" ? relatedClaims.value : undefined,
    };
  }

  /**
   * Update a claim and cascade changes through the belief graph
   */
  async updateAndCascade(
    claimHash: ActionHash,
    newClassification: EpistemicPosition
  ): Promise<CascadeResult> {
    // Update the claim's classification
    await this.client.claims.updateClaim({
      claimHash,
      classification: newClassification,
    });

    // Propagate belief changes
    await this.client.graph.propagateBelief(claimHash.toString());

    // Trigger cascade update to notify dependents and markets
    return this.client.claims.cascadeUpdate(claimHash.toString());
  }

  /**
   * Get claims that would have the highest value if verified
   */
  async getHighValueVerificationTargets(
    limit: number = 10
  ): Promise<Array<{
    claim: Claim;
    informationValue: InformationValue;
    recommendation: ReturnType<typeof recommendVerification>;
  }>> {
    // Get claims ranked by information value
    const ivRecords = await this.client.graph.rankByInformationValue(limit * 2);
    const results: Array<{
      claim: Claim;
      informationValue: InformationValue;
      recommendation: ReturnType<typeof recommendVerification>;
    }> = [];

    for (const iv of ivRecords) {
      // This is a simplification - in practice we'd need the claim hash
      const claims = await this.client.query.search(iv.claimId, { limit: 1 });
      if (claims.length > 0) {
        const claim = claims[0];
        const recommendation = recommendVerification(claim, iv);

        if (recommendation.recommend) {
          results.push({
            claim,
            informationValue: iv,
            recommendation,
          });
        }
      }

      if (results.length >= limit) break;
    }

    return results;
  }

  /**
   * Request verification market for a claim if recommended
   */
  async autoVerifyIfRecommended(claimHash: ActionHash): Promise<{
    requested: boolean;
    marketHash?: ActionHash;
    reason: string;
  }> {
    const claim = await this.client.claims.getClaim(claimHash);
    if (!claim) {
      return { requested: false, reason: "Claim not found" };
    }

    const claimId = claimHash.toString();

    // Get information value
    const ivRecords = await this.client.graph.rankByInformationValue(100);
    const iv = ivRecords.find((r) => r.claimId === claimId);

    if (!iv) {
      return { requested: false, reason: "No information value assessment" };
    }

    const recommendation = recommendVerification(claim, iv);

    if (!recommendation.recommend) {
      return { requested: false, reason: recommendation.reason };
    }

    // Request verification market
    const marketHash = await this.client.claims.spawnVerificationMarket({
      claimId,
      targetE: recommendation.suggestedTargetE,
      minConfidence: 0.7,
      closesAt: Date.now() + 7 * 24 * 60 * 60 * 1000, // 1 week
      tags: claim.topics,
    });

    return {
      requested: true,
      marketHash,
      reason: recommendation.reason,
    };
  }

  /**
   * Get author's credibility profile
   */
  async getAuthorProfile(authorDid: string): Promise<{
    reputation: Awaited<ReturnType<typeof this.client.inference.getAuthorReputation>>;
    recentClaims: Claim[];
    topDomains: string[];
    verificationRate: number;
  }> {
    const reputation = await this.client.inference.getAuthorReputation(authorDid);

    // Get author's recent claims
    // Note: This would need author's AgentPubKey in practice
    const recentClaims: Claim[] = []; // Placeholder

    // Calculate top domains from reputation
    const topDomains = reputation.domainScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map((d) => d.domain);

    // Calculate verification rate
    const totalVerified =
      reputation.claimsVerifiedTrue + reputation.claimsVerifiedFalse;
    const verificationRate =
      reputation.claimsAuthored > 0
        ? totalVerified / reputation.claimsAuthored
        : 0;

    return {
      reputation,
      recentClaims,
      topDomains,
      verificationRate,
    };
  }

  /**
   * Detect potential echo chambers in a topic area
   */
  async detectEchoChambers(
    topic: string,
    minClaims: number = 10
  ): Promise<{
    detected: boolean;
    claimCount: number;
    diversityScore: number;
    dominantPosition?: EpistemicPosition;
    warningLevel: "none" | "low" | "medium" | "high";
  }> {
    // Get claims for topic
    const claims = await this.client.claims.searchClaimsByTopic(topic);

    if (claims.length < minClaims) {
      return {
        detected: false,
        claimCount: claims.length,
        diversityScore: 1,
        warningLevel: "none",
      };
    }

    // Calculate variance in epistemic positions
    const positions = claims.map((c) => c.classification);
    const avgE =
      positions.reduce((acc, p) => acc + p.empirical, 0) / positions.length;
    const avgN =
      positions.reduce((acc, p) => acc + p.normative, 0) / positions.length;
    const avgM = positions.reduce((acc, p) => acc + p.mythic, 0) / positions.length;

    // Calculate standard deviation
    const stdE = Math.sqrt(
      positions.reduce((acc, p) => acc + Math.pow(p.empirical - avgE, 2), 0) /
        positions.length
    );
    const stdN = Math.sqrt(
      positions.reduce((acc, p) => acc + Math.pow(p.normative - avgN, 2), 0) /
        positions.length
    );
    const stdM = Math.sqrt(
      positions.reduce((acc, p) => acc + Math.pow(p.mythic - avgM, 2), 0) /
        positions.length
    );

    // Diversity score based on standard deviations (higher = more diverse)
    const diversityScore = (stdE + stdN + stdM) / 3;

    // Low diversity in normative dimension is most concerning for echo chambers
    const isEchoChamber = stdN < 0.15 && claims.length >= minClaims;

    let warningLevel: "none" | "low" | "medium" | "high" = "none";
    if (isEchoChamber) {
      if (stdN < 0.05) warningLevel = "high";
      else if (stdN < 0.1) warningLevel = "medium";
      else warningLevel = "low";
    }

    return {
      detected: isEchoChamber,
      claimCount: claims.length,
      diversityScore,
      dominantPosition: { empirical: avgE, normative: avgN, mythic: avgM },
      warningLevel,
    };
  }
}

export default KnowledgeService;

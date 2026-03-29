// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Evidence Bridge
 *
 * Integrates the Knowledge module's epistemic classification system with
 * the Justice module's evidence verification. Enables claims-based
 * verification of evidence submitted in dispute cases.
 *
 * @packageDocumentation
 * @module bridge/epistemic-evidence
 */

import { getCrossHappBridge } from './cross-happ.js';
import {
  type Evidence as JusticeEvidence,
  type EvidenceType as JusticeEvidenceType,
  type CaseCategory,
} from '../justice/index.js';
import {
  type EpistemicPosition,
  type FactCheckResult,
  type FactCheckVerdict,
  type CredibilityScore,
  type ClaimType,
} from '../knowledge/index.js';



// ============================================================================
// Epistemic Evidence Types
// ============================================================================

/**
 * Extended evidence with epistemic classification
 * Bridges Justice evidence with Knowledge module's epistemic framework
 */
export interface EpistemicEvidence extends JusticeEvidence {
  /** Epistemic classification from Knowledge module */
  epistemic_position?: EpistemicPosition;
  /** Credibility score (0.0 to 1.0) */
  credibility_score?: number;
  /** Linked claim IDs from Knowledge graph */
  linked_claim_ids?: string[];
  /** Fact-check verdict if applicable */
  fact_check_verdict?: FactCheckVerdict;
  /** Evidence strength adjusted by epistemic factors */
  adjusted_strength?: number;
  /** Verification timestamp */
  epistemically_verified_at?: number;
}

/**
 * Result of epistemic evidence verification
 */
export interface EpistemicVerificationResult {
  /** Original evidence ID */
  evidence_id: string;
  /** Whether verification was successful */
  verified: boolean;
  /** Epistemic classification */
  classification: EpistemicPosition;
  /** Overall credibility (0.0 to 1.0) */
  credibility: number;
  /** Confidence in the assessment (0.0 to 1.0) */
  confidence: number;
  /** Supporting claims from Knowledge graph */
  supporting_claims: string[];
  /** Contradicting claims from Knowledge graph */
  contradicting_claims: string[];
  /** Detailed factors affecting verification */
  factors: EpistemicVerificationFactor[];
  /** Recommendation for arbitrators */
  recommendation: 'Strong' | 'Moderate' | 'Weak' | 'Unreliable' | 'Insufficient';
  /** Verification timestamp */
  verified_at: number;
}

/**
 * Factor affecting epistemic verification
 */
export interface EpistemicVerificationFactor {
  /** Factor name */
  name: string;
  /** Factor weight (-1.0 to 1.0) */
  weight: number;
  /** Description */
  description: string;
  /** Source (which analysis produced this) */
  source: 'knowledge_graph' | 'credibility_model' | 'fact_check' | 'cross_reference';
}

/**
 * Input for batch evidence verification
 */
export interface BatchVerificationInput {
  /** Case ID */
  case_id: string;
  /** Evidence IDs to verify */
  evidence_ids: string[];
  /** Minimum credibility threshold */
  min_credibility?: number;
  /** Whether to include fact-checking */
  include_fact_check?: boolean;
  /** Whether to search for related claims */
  include_graph_search?: boolean;
}

/**
 * Result of batch evidence verification
 */
export interface BatchVerificationResult {
  /** Case ID */
  case_id: string;
  /** Individual verification results */
  results: EpistemicVerificationResult[];
  /** Summary statistics */
  summary: BatchVerificationSummary;
  /** Overall case evidence quality */
  case_evidence_quality: 'Excellent' | 'Good' | 'Mixed' | 'Poor' | 'Insufficient';
  /** Total verification time in ms */
  duration_ms: number;
}

/**
 * Summary of batch verification
 */
export interface BatchVerificationSummary {
  /** Total evidence items */
  total: number;
  /** Successfully verified */
  verified: number;
  /** Failed verification */
  failed: number;
  /** Average credibility score */
  avg_credibility: number;
  /** Average empirical score */
  avg_empirical: number;
  /** Average normative score */
  avg_normative: number;
  /** Count by recommendation */
  by_recommendation: Record<string, number>;
}

// ============================================================================
// Mapping Functions
// ============================================================================

/**
 * Map Justice evidence type to Knowledge claim type
 */
export function mapEvidenceTypeToClaimType(evidenceType: JusticeEvidenceType): ClaimType {
  const mapping: Record<JusticeEvidenceType, ClaimType> = {
    Document: 'Fact',
    Testimony: 'Opinion',
    Transaction: 'Historical',
    Screenshot: 'Fact',
    Witness: 'Opinion',
    Other: 'Fact',
  };
  return mapping[evidenceType] ?? 'Fact';
}

/**
 * Calculate adjusted evidence strength based on epistemic factors
 *
 * @param baseStrength - Original evidence strength (0-1)
 * @param epistemicPosition - E/N/M classification
 * @param caseCategory - Type of dispute case
 * @returns Adjusted strength (0-1)
 */
export function calculateAdjustedStrength(
  baseStrength: number,
  epistemicPosition: EpistemicPosition,
  caseCategory: CaseCategory
): number {
  // Weight epistemic dimensions differently by case category
  const weights = getEpistemicWeightsForCategory(caseCategory);

  const epistemicMultiplier =
    epistemicPosition.empirical * weights.empirical +
    epistemicPosition.normative * weights.normative +
    epistemicPosition.mythic * weights.mythic;

  // Normalize to 0-1 range with some boost for high epistemic scores
  const adjustedStrength = baseStrength * (0.5 + epistemicMultiplier * 0.5);

  return Math.min(Math.max(adjustedStrength, 0), 1);
}

/**
 * Get epistemic weight factors for a case category
 */
export function getEpistemicWeightsForCategory(category: CaseCategory): {
  empirical: number;
  normative: number;
  mythic: number;
} {
  switch (category) {
    case 'ContractDispute':
      // Contracts rely heavily on empirical facts
      return { empirical: 0.7, normative: 0.2, mythic: 0.1 };
    case 'ConductViolation':
      // Conduct violations (fraud, defamation, etc.) need strong empirical evidence
      return { empirical: 0.6, normative: 0.3, mythic: 0.1 };
    case 'PropertyDispute':
      // Property disputes are mainly empirical
      return { empirical: 0.65, normative: 0.25, mythic: 0.1 };
    case 'FinancialDispute':
      // Financial disputes rely heavily on empirical evidence
      return { empirical: 0.75, normative: 0.2, mythic: 0.05 };
    case 'GovernanceDispute':
      // Governance is normative-heavy
      return { empirical: 0.4, normative: 0.5, mythic: 0.1 };
    case 'IdentityDispute':
      // Identity disputes balance empirical evidence with normative questions
      return { empirical: 0.6, normative: 0.3, mythic: 0.1 };
    case 'IPDispute':
      // IP is empirical (did it happen) + normative (is it fair use)
      return { empirical: 0.55, normative: 0.35, mythic: 0.1 };
    case 'Other':
    default:
      // Balanced weights for unknown categories
      return { empirical: 0.5, normative: 0.35, mythic: 0.15 };
  }
}

/**
 * Convert fact-check verdict to recommendation
 */
export function verdictToRecommendation(
  verdict: FactCheckVerdict,
  credibility: number
): 'Strong' | 'Moderate' | 'Weak' | 'Unreliable' | 'Insufficient' {
  if (verdict === 'True' && credibility >= 0.8) return 'Strong';
  if (verdict === 'True' && credibility >= 0.5) return 'Moderate';
  if (verdict === 'PartiallyTrue' && credibility >= 0.6) return 'Moderate';
  if (verdict === 'PartiallyTrue' || verdict === 'Misleading') return 'Weak';
  if (verdict === 'False') return 'Unreliable';
  return 'Insufficient';
}

/**
 * Calculate overall credibility from multiple factors
 */
export function calculateOverallCredibility(
  epistemicPosition: EpistemicPosition,
  factCheckResult?: FactCheckResult,
  credibilityScore?: CredibilityScore
): { credibility: number; confidence: number } {
  let credibility = 0.5;
  let confidence = 0.3;

  // Base epistemic contribution (empirical dominates for evidence)
  const epistemicBase =
    epistemicPosition.empirical * 0.6 +
    epistemicPosition.normative * 0.3 +
    epistemicPosition.mythic * 0.1;

  credibility = epistemicBase;
  confidence = 0.4;

  // Boost from fact-check if available
  if (factCheckResult) {
    const verdictBoost = {
      True: 0.3,
      PartiallyTrue: 0.1,
      Misleading: -0.2,
      False: -0.4,
      Unverified: 0,
    }[factCheckResult.verdict] ?? 0;

    credibility = credibility * 0.6 + (0.5 + verdictBoost) * 0.4;
    confidence = Math.min(confidence + factCheckResult.confidence * 0.3, 1);
  }

  // Boost from credibility score if available
  if (credibilityScore) {
    credibility = credibility * 0.7 + credibilityScore.score * 0.3;
    confidence = Math.min(confidence + 0.2, 1);
  }

  return {
    credibility: Math.min(Math.max(credibility, 0), 1),
    confidence: Math.min(Math.max(confidence, 0), 1),
  };
}

// ============================================================================
// Epistemic Evidence Bridge Class
// ============================================================================

/**
 * Bridge for epistemic evidence verification
 *
 * Coordinates between Justice evidence submission and Knowledge graph
 * to provide epistemically-grounded evidence assessment.
 *
 * @example
 * ```typescript
 * const bridge = new EpistemicEvidenceBridge();
 *
 * // Verify a single piece of evidence
 * const result = await bridge.verifyEvidence({
 *   id: 'evidence-123',
 *   case_id: 'case-456',
 *   submitter: 'did:mycelix:alice',
 *   type_: 'Document',
 *   title: 'Contract Agreement',
 *   description: 'The signed contract between parties',
 *   content_hash: 'Qm...',
 *   submitted_at: Date.now(),
 *   verified: false,
 * }, 'ContractDispute');
 *
 * // Batch verify all evidence for a case
 * const batchResult = await bridge.verifyEvidenceBatch({
 *   case_id: 'case-456',
 *   evidence_ids: ['evidence-1', 'evidence-2'],
 *   include_fact_check: true,
 * });
 * ```
 */
export class EpistemicEvidenceBridge {
  /**
   * Verify a single piece of evidence using epistemic classification
   */
  async verifyEvidence(
    evidence: JusticeEvidence,
    caseCategory: CaseCategory
  ): Promise<EpistemicVerificationResult> {
    const bridge = getCrossHappBridge();
    const factors: EpistemicVerificationFactor[] = [];
    const supportingClaims: string[] = [];
    const contradictingClaims: string[] = [];

    // Step 1: Perform initial epistemic classification based on evidence type
    const baseClassification = this.classifyEvidence(evidence);
    factors.push({
      name: 'Base Type Classification',
      weight: 0.3,
      description: `Evidence type '${evidence.type_}' suggests ${this.describeClassification(baseClassification)}`,
      source: 'credibility_model',
    });

    // Step 2: Query Knowledge graph for related claims
    try {
      // Search for claims related to the evidence content
      const knowledgeQuery = await bridge.queryReputation(evidence.submitter, [
        'knowledge',
        'media',
      ]);

      if (knowledgeQuery.aggregatedScore > 0.7) {
        factors.push({
          name: 'Submitter Knowledge Reputation',
          weight: 0.2,
          description: `Submitter has high knowledge reputation (${knowledgeQuery.aggregatedScore.toFixed(2)})`,
          source: 'knowledge_graph',
        });
      } else if (knowledgeQuery.aggregatedScore < 0.4) {
        factors.push({
          name: 'Submitter Knowledge Reputation',
          weight: -0.1,
          description: `Submitter has low knowledge reputation (${knowledgeQuery.aggregatedScore.toFixed(2)})`,
          source: 'knowledge_graph',
        });
      }
    } catch {
      // Knowledge graph query failed - continue with base classification
      factors.push({
        name: 'Knowledge Graph Unavailable',
        weight: 0,
        description: 'Could not query knowledge graph for additional context',
        source: 'knowledge_graph',
      });
    }

    // Step 3: Calculate adjusted classification based on case category
    const adjustedClassification = this.adjustClassificationForCase(
      baseClassification,
      caseCategory
    );

    // Step 4: Calculate overall credibility
    const { credibility, confidence } = calculateOverallCredibility(adjustedClassification);

    // Step 5: Determine recommendation
    const recommendation = this.determineRecommendation(credibility, confidence, factors);

    return {
      evidence_id: evidence.id,
      verified: credibility >= 0.5 && confidence >= 0.4,
      classification: adjustedClassification,
      credibility,
      confidence,
      supporting_claims: supportingClaims,
      contradicting_claims: contradictingClaims,
      factors,
      recommendation,
      verified_at: Date.now(),
    };
  }

  /**
   * Verify a batch of evidence for a case
   */
  async verifyEvidenceBatch(
    input: BatchVerificationInput,
    evidenceItems: JusticeEvidence[],
    caseCategory: CaseCategory
  ): Promise<BatchVerificationResult> {
    const batchStartTime = Date.now();
    const results: EpistemicVerificationResult[] = [];

    // Verify each piece of evidence
    for (const evidence of evidenceItems) {
      if (input.evidence_ids.includes(evidence.id)) {
        const result = await this.verifyEvidence(evidence, caseCategory);

        // Apply minimum credibility filter if specified
        if (!input.min_credibility || result.credibility >= input.min_credibility) {
          results.push(result);
        }
      }
    }

    // Calculate summary statistics
    const summary = this.calculateSummary(results);

    // Determine overall case evidence quality
    const caseEvidenceQuality = this.determineCaseQuality(summary);

    return {
      case_id: input.case_id,
      results,
      summary,
      case_evidence_quality: caseEvidenceQuality,
      duration_ms: Date.now() - batchStartTime,
    };
  }

  /**
   * Enhance evidence with epistemic classification
   * Returns the evidence object augmented with epistemic data
   */
  async enhanceEvidence(
    evidence: JusticeEvidence,
    caseCategory: CaseCategory
  ): Promise<EpistemicEvidence> {
    const verification = await this.verifyEvidence(evidence, caseCategory);

    return {
      ...evidence,
      epistemic_position: verification.classification,
      credibility_score: verification.credibility,
      linked_claim_ids: verification.supporting_claims,
      adjusted_strength: calculateAdjustedStrength(
        0.5, // Base strength - would come from actual evidence assessment
        verification.classification,
        caseCategory
      ),
      epistemically_verified_at: verification.verified_at,
    };
  }

  /**
   * Perform fact-check on evidence claim content
   */
  async factCheckEvidence(
    evidence: JusticeEvidence,
    _claimText: string // Used in production to call Knowledge graph factCheck API
  ): Promise<{
    verdict: FactCheckVerdict;
    confidence: number;
    supporting: string[];
    contradicting: string[];
  }> {
    // In production, this would call the Knowledge module's factCheck API
    // For now, return a simulated result based on evidence type
    const typeReliability: Record<JusticeEvidenceType, number> = {
      Document: 0.8,
      Transaction: 0.9,
      Screenshot: 0.6,
      Testimony: 0.5,
      Witness: 0.5,
      Other: 0.4,
    };

    const reliability = typeReliability[evidence.type_] ?? 0.5;

    let verdict: FactCheckVerdict;
    if (reliability >= 0.8) verdict = 'True';
    else if (reliability >= 0.6) verdict = 'PartiallyTrue';
    else if (reliability >= 0.4) verdict = 'Unverified';
    else verdict = 'Misleading';

    return {
      verdict,
      confidence: reliability,
      supporting: [],
      contradicting: [],
    };
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private classifyEvidence(evidence: JusticeEvidence): EpistemicPosition {
    // Base classification by evidence type
    const typeClassifications: Record<JusticeEvidenceType, EpistemicPosition> = {
      Document: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
      Transaction: { empirical: 0.95, normative: 0.03, mythic: 0.02 },
      Screenshot: { empirical: 0.7, normative: 0.15, mythic: 0.15 },
      Testimony: { empirical: 0.4, normative: 0.4, mythic: 0.2 },
      Witness: { empirical: 0.5, normative: 0.35, mythic: 0.15 },
      Other: { empirical: 0.5, normative: 0.3, mythic: 0.2 },
    };

    return typeClassifications[evidence.type_] ?? { empirical: 0.5, normative: 0.3, mythic: 0.2 };
  }

  private adjustClassificationForCase(
    classification: EpistemicPosition,
    caseCategory: CaseCategory
  ): EpistemicPosition {
    const weights = getEpistemicWeightsForCategory(caseCategory);

    // Emphasize dimensions that matter more for this case type
    const total = weights.empirical + weights.normative + weights.mythic;

    return {
      empirical: classification.empirical * (weights.empirical / total) * 3,
      normative: classification.normative * (weights.normative / total) * 3,
      mythic: classification.mythic * (weights.mythic / total) * 3,
    };
  }

  private describeClassification(position: EpistemicPosition): string {
    const dominant =
      position.empirical >= position.normative && position.empirical >= position.mythic
        ? 'empirically-grounded'
        : position.normative >= position.mythic
          ? 'normatively-oriented'
          : 'narrative-contextual';

    return dominant;
  }

  private determineRecommendation(
    credibility: number,
    confidence: number,
    factors: EpistemicVerificationFactor[]
  ): 'Strong' | 'Moderate' | 'Weak' | 'Unreliable' | 'Insufficient' {
    const totalWeight = factors.reduce((sum, f) => sum + f.weight, 0);

    if (credibility >= 0.8 && confidence >= 0.7 && totalWeight > 0.3) return 'Strong';
    if (credibility >= 0.6 && confidence >= 0.5 && totalWeight > 0.1) return 'Moderate';
    if (credibility >= 0.4 && confidence >= 0.3) return 'Weak';
    if (credibility < 0.3 || totalWeight < -0.2) return 'Unreliable';
    return 'Insufficient';
  }

  private calculateSummary(results: EpistemicVerificationResult[]): BatchVerificationSummary {
    const verified = results.filter((r) => r.verified).length;
    const failed = results.length - verified;

    const avgCredibility =
      results.length > 0
        ? results.reduce((sum, r) => sum + r.credibility, 0) / results.length
        : 0;

    const avgEmpirical =
      results.length > 0
        ? results.reduce((sum, r) => sum + r.classification.empirical, 0) / results.length
        : 0;

    const avgNormative =
      results.length > 0
        ? results.reduce((sum, r) => sum + r.classification.normative, 0) / results.length
        : 0;

    const byRecommendation: Record<string, number> = {};
    for (const result of results) {
      byRecommendation[result.recommendation] = (byRecommendation[result.recommendation] || 0) + 1;
    }

    return {
      total: results.length,
      verified,
      failed,
      avg_credibility: avgCredibility,
      avg_empirical: avgEmpirical,
      avg_normative: avgNormative,
      by_recommendation: byRecommendation,
    };
  }

  private determineCaseQuality(
    summary: BatchVerificationSummary
  ): 'Excellent' | 'Good' | 'Mixed' | 'Poor' | 'Insufficient' {
    if (summary.total === 0) return 'Insufficient';

    const verifiedRatio = summary.verified / summary.total;
    const strongModerateCount =
      (summary.by_recommendation['Strong'] || 0) + (summary.by_recommendation['Moderate'] || 0);
    const qualityRatio = strongModerateCount / summary.total;

    if (verifiedRatio >= 0.9 && summary.avg_credibility >= 0.8 && qualityRatio >= 0.8)
      return 'Excellent';
    if (verifiedRatio >= 0.7 && summary.avg_credibility >= 0.6 && qualityRatio >= 0.6) return 'Good';
    if (verifiedRatio >= 0.5 && summary.avg_credibility >= 0.4) return 'Mixed';
    if (verifiedRatio < 0.3 || summary.avg_credibility < 0.3) return 'Poor';
    return 'Insufficient';
  }
}

// ============================================================================
// Factory and Singleton
// ============================================================================

/** Shared bridge instance */
let sharedBridge: EpistemicEvidenceBridge | null = null;

/**
 * Get the shared epistemic evidence bridge instance
 */
export function getEpistemicEvidenceBridge(): EpistemicEvidenceBridge {
  if (!sharedBridge) {
    sharedBridge = new EpistemicEvidenceBridge();
  }
  return sharedBridge;
}

/**
 * Reset the bridge (for testing)
 */
export function resetEpistemicEvidenceBridge(): void {
  sharedBridge = null;
}

// ============================================================================
// Utility Exports
// ============================================================================

/**
 * Quick check if evidence meets minimum epistemic standards for a case
 */
export function meetsEpistemicStandards(
  result: EpistemicVerificationResult,
  caseCategory: CaseCategory
): boolean {
  const weights = getEpistemicWeightsForCategory(caseCategory);

  // Calculate weighted epistemic score
  const weightedScore =
    result.classification.empirical * weights.empirical +
    result.classification.normative * weights.normative +
    result.classification.mythic * weights.mythic;

  // Evidence meets standards if:
  // - Weighted score >= 0.5
  // - Credibility >= 0.4
  // - Recommendation is not Unreliable or Insufficient
  return (
    weightedScore >= 0.5 &&
    result.credibility >= 0.4 &&
    result.recommendation !== 'Unreliable' &&
    result.recommendation !== 'Insufficient'
  );
}

/**
 * Get a human-readable explanation of epistemic classification
 */
export function explainEpistemicClassification(position: EpistemicPosition): string {
  const explanations: string[] = [];

  if (position.empirical >= 0.7) {
    explanations.push('strongly empirically verifiable');
  } else if (position.empirical >= 0.4) {
    explanations.push('partially empirically verifiable');
  } else {
    explanations.push('limited empirical verifiability');
  }

  if (position.normative >= 0.5) {
    explanations.push('significant normative/ethical dimensions');
  }

  if (position.mythic >= 0.4) {
    explanations.push('notable narrative/meaning context');
  }

  return `Evidence is ${explanations.join(', ')}.`;
}

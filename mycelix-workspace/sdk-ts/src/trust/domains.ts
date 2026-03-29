// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Trust Domains
 *
 * Cross-domain trust translation - translate K-Vector trust profiles
 * between domains with different trust semantics.
 *
 * @packageDocumentation
 * @module trust/domains
 */

import {
  type KVector,
  KVectorDimension,
  KVECTOR_WEIGHTS,
  kvectorToArray,
  kvectorFromArray,
  isVerified,
  type TrustAttestation,
  type TranslatedAttestation,
} from './types.js';
// Note: computeTrustScore removed as unused

// ============================================================================
// Domain Relevance
// ============================================================================

/**
 * How relevant each K-Vector dimension is to a domain
 */
export interface DomainRelevance {
  /** Reputation relevance (0.0-1.0) */
  reputation: number;
  /** Activity relevance */
  activity: number;
  /** Integrity relevance */
  integrity: number;
  /** Performance relevance */
  performance: number;
  /** Membership relevance */
  membership: number;
  /** Stake relevance */
  stake: number;
  /** Historical relevance */
  historical: number;
  /** Topology relevance */
  topology: number;
  /** Verification relevance */
  verification: number;
}

/**
 * Create a balanced relevance (all dimensions equal)
 */
export function balancedRelevance(): DomainRelevance {
  return {
    reputation: 1.0,
    activity: 1.0,
    integrity: 1.0,
    performance: 1.0,
    membership: 1.0,
    stake: 1.0,
    historical: 1.0,
    topology: 1.0,
    verification: 1.0,
  };
}

/**
 * Convert relevance to array
 */
export function relevanceToArray(r: DomainRelevance): number[] {
  return [
    r.reputation,
    r.activity,
    r.integrity,
    r.performance,
    r.membership,
    r.stake,
    r.historical,
    r.topology,
    r.verification,
  ];
}

/**
 * Compute cosine similarity between two relevance profiles
 */
export function relevanceSimilarity(a: DomainRelevance, b: DomainRelevance): number {
  const arrA = relevanceToArray(a);
  const arrB = relevanceToArray(b);

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < 9; i++) {
    dot += arrA[i] * arrB[i];
    magA += arrA[i] * arrA[i];
    magB += arrB[i] * arrB[i];
  }

  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);

  return magA > 0 && magB > 0 ? dot / (magA * magB) : 0;
}

// ============================================================================
// Trust Domain
// ============================================================================

/**
 * A trust domain with specific requirements
 */
export interface TrustDomain {
  /** Unique domain identifier */
  domainId: string;
  /** Human-readable name */
  name: string;
  /** Description */
  description: string;
  /** Dimension relevance */
  dimensionRelevance: DomainRelevance;
  /** Minimum trust threshold */
  minTrustThreshold: number;
  /** Whether verification is required */
  requiresVerification: boolean;
  /** Optional weight overrides */
  weightOverrides?: Partial<Record<KVectorDimension, number>>;
}

// ============================================================================
// Predefined Domains
// ============================================================================

/**
 * Financial operations domain
 */
export const DOMAIN_FINANCIAL: TrustDomain = {
  domainId: 'financial',
  name: 'Financial Operations',
  description: 'High-stakes financial transactions requiring verified identity and stake',
  dimensionRelevance: {
    reputation: 0.8,
    activity: 0.3,
    integrity: 1.0,
    performance: 0.5,
    membership: 0.7,
    stake: 1.0,
    historical: 0.9,
    topology: 0.2,
    verification: 1.0,
  },
  minTrustThreshold: 0.6,
  requiresVerification: true,
};

/**
 * Code review domain
 */
export const DOMAIN_CODE_REVIEW: TrustDomain = {
  domainId: 'code_review',
  name: 'Code Review',
  description: 'Technical review of code changes',
  dimensionRelevance: {
    reputation: 0.8,
    activity: 0.6,
    integrity: 0.9,
    performance: 1.0,
    membership: 0.4,
    stake: 0.2,
    historical: 0.7,
    topology: 0.5,
    verification: 0.5,
  },
  minTrustThreshold: 0.4,
  requiresVerification: false,
};

/**
 * Social community domain
 */
export const DOMAIN_SOCIAL: TrustDomain = {
  domainId: 'social',
  name: 'Social Community',
  description: 'Community interactions and social activities',
  dimensionRelevance: {
    reputation: 1.0,
    activity: 0.9,
    integrity: 0.6,
    performance: 0.5,
    membership: 0.6,
    stake: 0.1,
    historical: 0.5,
    topology: 1.0,
    verification: 0.3,
  },
  minTrustThreshold: 0.3,
  requiresVerification: false,
};

/**
 * Governance domain
 */
export const DOMAIN_GOVERNANCE: TrustDomain = {
  domainId: 'governance',
  name: 'Governance',
  description: 'Participation in governance decisions',
  dimensionRelevance: {
    reputation: 1.0,
    activity: 0.5,
    integrity: 0.9,
    performance: 0.6,
    membership: 0.8,
    stake: 0.9,
    historical: 0.8,
    topology: 0.6,
    verification: 0.7,
  },
  minTrustThreshold: 0.5,
  requiresVerification: false,
};

/**
 * Research domain
 */
export const DOMAIN_RESEARCH: TrustDomain = {
  domainId: 'research',
  name: 'Research',
  description: 'Academic and research contributions',
  dimensionRelevance: {
    reputation: 0.9,
    activity: 0.4,
    integrity: 0.8,
    performance: 1.0,
    membership: 0.3,
    stake: 0.2,
    historical: 0.7,
    topology: 0.6,
    verification: 0.6,
  },
  minTrustThreshold: 0.4,
  requiresVerification: false,
};

/**
 * Infrastructure domain
 */
export const DOMAIN_INFRASTRUCTURE: TrustDomain = {
  domainId: 'infrastructure',
  name: 'Infrastructure',
  description: 'System operations and infrastructure management',
  dimensionRelevance: {
    reputation: 0.7,
    activity: 0.5,
    integrity: 1.0,
    performance: 0.9,
    membership: 0.6,
    stake: 0.5,
    historical: 0.9,
    topology: 0.4,
    verification: 1.0,
  },
  minTrustThreshold: 0.6,
  requiresVerification: true,
};

/**
 * All predefined domains
 */
export const ALL_DOMAINS: TrustDomain[] = [
  DOMAIN_FINANCIAL,
  DOMAIN_CODE_REVIEW,
  DOMAIN_SOCIAL,
  DOMAIN_GOVERNANCE,
  DOMAIN_RESEARCH,
  DOMAIN_INFRASTRUCTURE,
];

// ============================================================================
// Translation Result
// ============================================================================

/**
 * Dimension translation details
 */
export interface DimensionTranslation {
  /** Dimension */
  dimension: KVectorDimension;
  /** Source value */
  sourceValue: number;
  /** Target value */
  targetValue: number;
  /** Source relevance */
  sourceRelevance: number;
  /** Target relevance */
  targetRelevance: number;
  /** Translation factor */
  translationFactor: number;
}

/**
 * Result of trust translation
 */
export interface TranslationResult {
  /** Source K-Vector */
  sourceKVector: KVector;
  /** Target K-Vector */
  targetKVector: KVector;
  /** Source domain ID */
  sourceDomainId: string;
  /** Target domain ID */
  targetDomainId: string;
  /** Trust in source domain */
  sourceTrust: number;
  /** Trust in target domain */
  targetTrust: number;
  /** Translation confidence (0.0-1.0) */
  translationConfidence: number;
  /** Dimension translations */
  dimensionTranslations: DimensionTranslation[];
  /** Whether target requirements are met */
  meetsTargetRequirements: boolean;
  /** Warnings */
  warnings: string[];
}

// ============================================================================
// Trust Translation
// ============================================================================

/**
 * Compute domain-weighted trust score
 */
export function computeDomainTrust(kv: KVector, domain: TrustDomain): number {
  const arr = kvectorToArray(kv);
  const rel = relevanceToArray(domain.dimensionRelevance);
  const weights = Object.values(KVECTOR_WEIGHTS);

  let weightedSum = 0;
  let weightSum = 0;

  for (let i = 0; i < 9; i++) {
    const domainWeight = weights[i] * rel[i];
    weightedSum += arr[i] * domainWeight;
    weightSum += domainWeight;
  }

  return weightSum > 0 ? weightedSum / weightSum : 0;
}

/**
 * Translate trust between domains
 */
export function translateTrust(
  kvector: KVector,
  sourceDomain: TrustDomain,
  targetDomain: TrustDomain
): TranslationResult {
  const sourceArr = kvectorToArray(kvector);
  const sourceRel = relevanceToArray(sourceDomain.dimensionRelevance);
  const targetRel = relevanceToArray(targetDomain.dimensionRelevance);
  const weights = Object.values(KVECTOR_WEIGHTS);

  // Compute domain similarity
  const domainSimilarity = relevanceSimilarity(
    sourceDomain.dimensionRelevance,
    targetDomain.dimensionRelevance
  );

  // Translate each dimension
  const targetArr: number[] = new Array(9);
  const dimensionTranslations: DimensionTranslation[] = [];
  let totalTransfer = 0;
  let totalWeight = 0;

  const dimensions = [
    KVectorDimension.Reputation,
    KVectorDimension.Activity,
    KVectorDimension.Integrity,
    KVectorDimension.Performance,
    KVectorDimension.Membership,
    KVectorDimension.Stake,
    KVectorDimension.Historical,
    KVectorDimension.Topology,
    KVectorDimension.Verification,
  ];

  for (let i = 0; i < 9; i++) {
    const sourceVal = sourceArr[i];
    const sourceRelevance = sourceRel[i];
    const targetRelevance = targetRel[i];

    // Translation factor based on relevance ratio
    const translationFactor = sourceRelevance > 0
      ? Math.min(1.0, targetRelevance / sourceRelevance) * Math.sqrt(domainSimilarity)
      : 0;

    const targetVal = Math.max(0, Math.min(1, sourceVal * translationFactor));
    targetArr[i] = targetVal;

    const weight = targetRelevance * weights[i];
    totalTransfer += translationFactor * weight;
    totalWeight += weight;

    dimensionTranslations.push({
      dimension: dimensions[i],
      sourceValue: sourceVal,
      targetValue: targetVal,
      sourceRelevance,
      targetRelevance,
      translationFactor,
    });
  }

  const targetKVector = kvectorFromArray(targetArr as [number, number, number, number, number, number, number, number, number]);
  const sourceTrust = computeDomainTrust(kvector, sourceDomain);
  const targetTrust = computeDomainTrust(targetKVector, targetDomain);

  // Translation confidence
  const transferQuality = totalWeight > 0 ? totalTransfer / totalWeight : 0;
  const translationConfidence = Math.max(0, Math.min(1,
    domainSimilarity * 0.4 + transferQuality * 0.6
  ));

  // Check requirements
  const meetsThreshold = targetTrust >= targetDomain.minTrustThreshold;
  const meetsVerification = !targetDomain.requiresVerification || isVerified(kvector);
  const meetsTargetRequirements = meetsThreshold && meetsVerification;

  // Generate warnings
  const warnings: string[] = [];
  if (translationConfidence < 0.5) {
    warnings.push(
      `Low translation confidence (${Math.round(translationConfidence * 100)}%): domains have different trust priorities`
    );
  }
  if (targetTrust < sourceTrust * 0.5) {
    warnings.push(
      `Significant trust reduction: ${Math.round((targetTrust / sourceTrust) * 100)}% of source trust preserved`
    );
  }
  if (targetDomain.requiresVerification && !isVerified(kvector)) {
    warnings.push('Target domain requires verification but agent is not verified');
  }

  return {
    sourceKVector: kvector,
    targetKVector,
    sourceDomainId: sourceDomain.domainId,
    targetDomainId: targetDomain.domainId,
    sourceTrust,
    targetTrust,
    translationConfidence,
    dimensionTranslations,
    meetsTargetRequirements,
    warnings,
  };
}

/**
 * Translate a trust attestation to a target domain
 */
export function translateAttestation(
  attestation: TrustAttestation,
  sourceDomain: TrustDomain,
  targetDomain: TrustDomain
): TranslatedAttestation {
  const result = translateTrust(
    attestation.update.updatedKVector,
    sourceDomain,
    targetDomain
  );

  return {
    original: attestation,
    targetDomainId: targetDomain.domainId,
    translatedKVector: result.targetKVector,
    translationConfidence: result.translationConfidence,
    meetsRequirements: result.meetsTargetRequirements,
    warnings: result.warnings,
  };
}

// ============================================================================
// Domain Registry
// ============================================================================

/**
 * Registry of trust domains
 */
export class DomainRegistry {
  private domains: Map<string, TrustDomain> = new Map();

  constructor() {
    // Register default domains
    for (const domain of ALL_DOMAINS) {
      this.register(domain);
    }
  }

  /**
   * Register a domain
   */
  register(domain: TrustDomain): void {
    this.domains.set(domain.domainId, domain);
  }

  /**
   * Get a domain by ID
   */
  get(domainId: string): TrustDomain | undefined {
    return this.domains.get(domainId);
  }

  /**
   * List all domain IDs
   */
  listDomains(): string[] {
    return Array.from(this.domains.keys());
  }

  /**
   * Translate between domains by ID
   */
  translate(
    kvector: KVector,
    sourceDomainId: string,
    targetDomainId: string
  ): TranslationResult | undefined {
    const source = this.domains.get(sourceDomainId);
    const target = this.domains.get(targetDomainId);

    if (!source || !target) {
      return undefined;
    }

    return translateTrust(kvector, source, target);
  }

  /**
   * Compute similarity matrix for all domains
   */
  similarityMatrix(): Map<string, Map<string, number>> {
    const matrix = new Map<string, Map<string, number>>();
    const entries1 = Array.from(this.domains.entries());

    for (const [id1, domain1] of entries1) {
      const row = new Map<string, number>();
      const entries2 = Array.from(this.domains.entries());
      for (const [id2, domain2] of entries2) {
        const similarity = relevanceSimilarity(
          domain1.dimensionRelevance,
          domain2.dimensionRelevance
        );
        row.set(id2, similarity);
      }
      matrix.set(id1, row);
    }

    return matrix;
  }
}

// ============================================================================
// Domain Compatibility
// ============================================================================

/**
 * Domain compatibility analysis
 */
export interface DomainCompatibility {
  /** Source domain */
  source: string;
  /** Target domain */
  target: string;
  /** Compatibility score (0.0-1.0) */
  compatibility: number;
  /** Dimensions that transfer well */
  strongTransfers: KVectorDimension[];
  /** Dimensions that transfer poorly */
  weakTransfers: KVectorDimension[];
  /** Recommendations */
  recommendations: string[];
}

/**
 * Analyze compatibility between domains
 */
export function analyzeDomainCompatibility(
  source: TrustDomain,
  target: TrustDomain
): DomainCompatibility {
  const sourceRel = relevanceToArray(source.dimensionRelevance);
  const targetRel = relevanceToArray(target.dimensionRelevance);

  const strongTransfers: KVectorDimension[] = [];
  const weakTransfers: KVectorDimension[] = [];
  const recommendations: string[] = [];

  const dimensions = [
    KVectorDimension.Reputation,
    KVectorDimension.Activity,
    KVectorDimension.Integrity,
    KVectorDimension.Performance,
    KVectorDimension.Membership,
    KVectorDimension.Stake,
    KVectorDimension.Historical,
    KVectorDimension.Topology,
    KVectorDimension.Verification,
  ];

  for (let i = 0; i < 9; i++) {
    const transferQuality = sourceRel[i] > 0
      ? Math.min(1, targetRel[i] / sourceRel[i])
      : 0;

    if (transferQuality >= 0.7 && targetRel[i] >= 0.5) {
      strongTransfers.push(dimensions[i]);
    } else if (transferQuality < 0.3 && targetRel[i] >= 0.5) {
      weakTransfers.push(dimensions[i]);
    }
  }

  // Generate recommendations
  if (target.requiresVerification && !source.requiresVerification) {
    recommendations.push('Target domain requires verification - ensure agent is verified');
  }

  for (const dim of weakTransfers) {
    const dimName = ['Reputation', 'Activity', 'Integrity', 'Performance', 'Membership', 'Stake', 'Historical', 'Topology', 'Verification'][dim];
    recommendations.push(`Build ${dimName} in target domain context for better translation`);
  }

  if (target.minTrustThreshold > source.minTrustThreshold) {
    recommendations.push(
      `Target has higher trust threshold (${Math.round(target.minTrustThreshold * 100)}% vs ${Math.round(source.minTrustThreshold * 100)}%) - may need trust building`
    );
  }

  const compatibility = relevanceSimilarity(
    source.dimensionRelevance,
    target.dimensionRelevance
  );

  return {
    source: source.domainId,
    target: target.domainId,
    compatibility,
    strongTransfers,
    weakTransfers,
    recommendations,
  };
}

/**
 * Default domain registry instance
 */
export const defaultDomainRegistry = new DomainRegistry();

/**
 * @mycelix/sdk Epistemic Algebra
 *
 * Mathematical operations for combining claim confidences with
 * intellectual honesty. Core principle: claims can only be as
 * strong as their weakest component.
 *
 * Key operations:
 * - Weakest-link (conjunction): Combined = min(components)
 * - Chain degradation: Confidence degrades through inference
 * - E-level ceiling: Verification method caps max confidence
 * - Corroboration boost: Independent sources (diminishing returns)
 * - Contradiction penalty: Conflicting sources reduce confidence
 *
 * @packageDocumentation
 * @module epistemic/algebra
 */

import {
  type EpistemicClaim,
  EmpiricalLevel,
} from './types.js';

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Maximum confidence for each empirical level.
 * No claim can exceed this ceiling regardless of evidence.
 */
export const E_LEVEL_CONFIDENCE_CEILING: Record<EmpiricalLevel, number> = {
  [EmpiricalLevel.E0_Unverified]: 0.40,      // Unverified can't exceed 40%
  [EmpiricalLevel.E1_Testimonial]: 0.60,      // Testimony caps at 60%
  [EmpiricalLevel.E2_PrivateVerify]: 0.80,    // Private verification caps at 80%
  [EmpiricalLevel.E3_Cryptographic]: 0.95,    // Crypto proof caps at 95%
  [EmpiricalLevel.E4_Consensus]: 0.99,        // Network consensus caps at 99%
};

/**
 * Degradation factor per inference step in a reasoning chain.
 * Longer chains = less confidence.
 */
export const DEFAULT_CHAIN_DEGRADATION = 0.95;

/**
 * Maximum corroboration boost (asymptotic limit).
 * Even infinite sources can't make uncertain things certain.
 */
export const MAX_CORROBORATION_BOOST = 1.3;

/**
 * Diminishing returns factor for corroboration.
 * Each additional source contributes less than the previous.
 */
export const CORROBORATION_DIMINISHING_FACTOR = 0.7;

/**
 * Penalty per contradicting source.
 */
export const CONTRADICTION_PENALTY = 0.15;

// ============================================================================
// TYPES
// ============================================================================

/**
 * Result of combining claims
 */
export interface CombinedConfidence {
  /** Final combined confidence */
  confidence: number;

  /** How the confidence was computed */
  method: 'conjunction' | 'disjunction' | 'corroboration' | 'chain' | 'weighted';

  /** The limiting factor (if any) */
  limitingFactor?: string;

  /** Components that contributed */
  components: number[];

  /** Whether E-level ceiling was applied */
  ceilingApplied: boolean;

  /** Original value before ceiling */
  preCeilingConfidence?: number;

  /** Explanation of the calculation */
  explanation: string;
}

/**
 * Options for corroboration calculation
 */
export interface CorroborationOptions {
  /** Are the sources truly independent? (affects boost) */
  independent: boolean;

  /** Number of sources */
  sourceCount: number;

  /** Individual source confidences */
  confidences: number[];

  /** Any contradicting sources? */
  contradictionCount?: number;
}

/**
 * Result of chain degradation
 */
export interface ChainDegradation {
  /** Final confidence after chain */
  finalConfidence: number;

  /** Original starting confidence */
  startingConfidence: number;

  /** Number of inference steps */
  chainLength: number;

  /** Degradation per step used */
  degradationFactor: number;

  /** Warning if chain is too long */
  warning?: string;
}

/**
 * Conflict resolution result
 */
export interface ConflictResolution {
  /** Resolved confidence */
  confidence: number;

  /** Which sources were in conflict */
  conflicts: Array<{ sourceA: string; sourceB: string; severity: number }>;

  /** Total penalty applied */
  totalPenalty: number;

  /** Recommendation */
  recommendation: string;
}

// ============================================================================
// CORE ALGEBRA OPERATIONS
// ============================================================================

/**
 * Conjunction (AND) - Weakest Link Principle
 *
 * When combining claims via logical AND, the result cannot be
 * more confident than the least confident component.
 *
 * @example
 * ```typescript
 * // "Alice verified AND Bob verified" = min(Alice, Bob)
 * const combined = conjunction([0.9, 0.6, 0.8]);
 * // combined.confidence === 0.6
 * ```
 */
export function conjunction(confidences: number[]): CombinedConfidence {
  if (confidences.length === 0) {
    return {
      confidence: 0,
      method: 'conjunction',
      components: [],
      ceilingApplied: false,
      explanation: 'No components to combine',
    };
  }

  const minConfidence = Math.min(...confidences);
  const limitingIndex = confidences.indexOf(minConfidence);

  return {
    confidence: minConfidence,
    method: 'conjunction',
    limitingFactor: `Component ${limitingIndex + 1} at ${(minConfidence * 100).toFixed(1)}%`,
    components: confidences,
    ceilingApplied: false,
    explanation: `Weakest-link principle: combined confidence = min(${confidences.map((c) => (c * 100).toFixed(1) + '%').join(', ')}) = ${(minConfidence * 100).toFixed(1)}%`,
  };
}

/**
 * Disjunction (OR) - Maximum Principle
 *
 * When combining claims via logical OR, the result is
 * the maximum of the components (if any source is reliable,
 * we have some reason to believe).
 *
 * @example
 * ```typescript
 * // "Alice verified OR Bob verified" = max(Alice, Bob)
 * const combined = disjunction([0.4, 0.8, 0.6]);
 * // combined.confidence === 0.8
 * ```
 */
export function disjunction(confidences: number[]): CombinedConfidence {
  if (confidences.length === 0) {
    return {
      confidence: 0,
      method: 'weighted',
      components: [],
      ceilingApplied: false,
      explanation: 'No components to combine',
    };
  }

  const maxConfidence = Math.max(...confidences);
  const leadingIndex = confidences.indexOf(maxConfidence);

  return {
    confidence: maxConfidence,
    method: 'weighted',
    limitingFactor: `Best source: Component ${leadingIndex + 1} at ${(maxConfidence * 100).toFixed(1)}%`,
    components: confidences,
    ceilingApplied: false,
    explanation: `Disjunction: combined confidence = max(${confidences.map((c) => (c * 100).toFixed(1) + '%').join(', ')}) = ${(maxConfidence * 100).toFixed(1)}%`,
  };
}

/**
 * Apply E-level ceiling to a confidence value
 *
 * Claims cannot be more confident than their verification
 * method allows. Hearsay cannot be 99% confident.
 *
 * @example
 * ```typescript
 * // Testimonial evidence can't exceed 60%
 * const capped = applyELevelCeiling(0.95, EmpiricalLevel.E1_Testimonial);
 * // capped === 0.60
 * ```
 */
export function applyELevelCeiling(
  confidence: number,
  eLevel: EmpiricalLevel
): { confidence: number; ceilingApplied: boolean; ceiling: number } {
  const ceiling = E_LEVEL_CONFIDENCE_CEILING[eLevel];

  if (confidence > ceiling) {
    return {
      confidence: ceiling,
      ceilingApplied: true,
      ceiling,
    };
  }

  return {
    confidence,
    ceilingApplied: false,
    ceiling,
  };
}

/**
 * Calculate confidence degradation through an inference chain
 *
 * Each step of inference degrades confidence. A→B→C→D is less
 * certain than A→B.
 *
 * @example
 * ```typescript
 * // 3-step inference chain starting at 90%
 * const result = chainDegradation(0.9, 3);
 * // result.finalConfidence ≈ 0.77 (0.9 * 0.95^3)
 * ```
 */
export function chainDegradation(
  startingConfidence: number,
  chainLength: number,
  degradationFactor: number = DEFAULT_CHAIN_DEGRADATION
): ChainDegradation {
  // Input validation
  if (typeof startingConfidence !== 'number' || !Number.isFinite(startingConfidence)) {
    throw new Error(`startingConfidence must be a finite number, got: ${startingConfidence}`);
  }
  if (startingConfidence < 0 || startingConfidence > 1) {
    throw new Error(`startingConfidence must be between 0 and 1, got: ${startingConfidence}`);
  }

  if (typeof chainLength !== 'number' || !Number.isInteger(chainLength)) {
    throw new Error(`chainLength must be an integer, got: ${chainLength}`);
  }
  if (chainLength < 0) {
    throw new Error('Chain length cannot be negative');
  }

  if (typeof degradationFactor !== 'number' || !Number.isFinite(degradationFactor)) {
    throw new Error(`degradationFactor must be a finite number, got: ${degradationFactor}`);
  }
  if (degradationFactor <= 0 || degradationFactor >= 1) {
    throw new Error(`degradationFactor must be strictly between 0 and 1 (exclusive), got: ${degradationFactor}`);
  }

  if (chainLength === 0) {
    return {
      finalConfidence: startingConfidence,
      startingConfidence,
      chainLength,
      degradationFactor,
    };
  }

  const finalConfidence = Math.max(0, Math.min(1,
    startingConfidence * Math.pow(degradationFactor, chainLength)
  ));

  const result: ChainDegradation = {
    finalConfidence,
    startingConfidence,
    chainLength,
    degradationFactor,
  };

  // Warn about long chains
  if (chainLength > 5) {
    result.warning = `Long inference chain (${chainLength} steps) - confidence has degraded significantly. Consider finding more direct evidence.`;
  }

  if (finalConfidence < 0.3 && startingConfidence > 0.7) {
    result.warning = `Chain degradation has reduced high confidence to low confidence. The reasoning chain may be too long.`;
  }

  return result;
}

/**
 * Calculate corroboration boost from multiple independent sources
 *
 * Multiple sources increase confidence, but with diminishing returns.
 * The boost is asymptotic - infinite sources can't make uncertain things certain.
 *
 * @example
 * ```typescript
 * // 3 independent sources at 70%
 * const boost = corroborate({
 *   independent: true,
 *   sourceCount: 3,
 *   confidences: [0.7, 0.7, 0.7]
 * });
 * // boost.confidence ≈ 0.84 (boosted but still <100%)
 * ```
 */
export function corroborate(options: CorroborationOptions): CombinedConfidence {
  const { independent, sourceCount, confidences, contradictionCount = 0 } = options;

  // Input validation
  if (typeof independent !== 'boolean') {
    throw new Error(`independent must be a boolean, got: ${typeof independent}`);
  }

  if (typeof sourceCount !== 'number' || !Number.isInteger(sourceCount) || sourceCount < 0) {
    throw new Error(`sourceCount must be a non-negative integer, got: ${sourceCount}`);
  }

  if (!Array.isArray(confidences)) {
    throw new Error(`confidences must be an array, got: ${typeof confidences}`);
  }

  // Validate all confidence values
  for (let i = 0; i < confidences.length; i++) {
    const c = confidences[i];
    if (typeof c !== 'number' || !Number.isFinite(c)) {
      throw new Error(`confidences[${i}] must be a finite number, got: ${c}`);
    }
    if (c < 0 || c > 1) {
      throw new Error(`confidences[${i}] must be between 0 and 1, got: ${c}`);
    }
  }

  if (typeof contradictionCount !== 'number' || contradictionCount < 0) {
    throw new Error(`contradictionCount must be a non-negative number, got: ${contradictionCount}`);
  }

  if (sourceCount === 0 || confidences.length === 0) {
    return {
      confidence: 0,
      method: 'corroboration',
      components: [],
      ceilingApplied: false,
      explanation: 'No sources to corroborate',
    };
  }

  if (sourceCount === 1) {
    return {
      confidence: confidences[0],
      method: 'corroboration',
      components: confidences,
      ceilingApplied: false,
      explanation: 'Single source - no corroboration possible',
    };
  }

  // Start with average confidence
  const avgConfidence =
    confidences.reduce((a, b) => a + b, 0) / confidences.length;

  // Calculate boost based on number of sources (diminishing returns)
  let boost = 1.0;
  if (independent) {
    // Independent sources get full diminishing returns boost
    for (let i = 1; i < sourceCount; i++) {
      boost += (MAX_CORROBORATION_BOOST - 1) * Math.pow(CORROBORATION_DIMINISHING_FACTOR, i);
    }
    boost = Math.min(boost, MAX_CORROBORATION_BOOST);
  } else {
    // Non-independent sources get reduced boost (potential echo chamber)
    for (let i = 1; i < sourceCount; i++) {
      boost +=
        (MAX_CORROBORATION_BOOST - 1) *
        0.3 *
        Math.pow(CORROBORATION_DIMINISHING_FACTOR, i);
    }
    boost = Math.min(boost, 1.1); // Much lower ceiling for non-independent
  }

  // Apply boost
  let boostedConfidence = avgConfidence * boost;

  // Apply contradiction penalty
  if (contradictionCount > 0) {
    const penalty = contradictionCount * CONTRADICTION_PENALTY;
    boostedConfidence = boostedConfidence * (1 - penalty);
  }

  // Cap at 0.99
  boostedConfidence = Math.min(0.99, Math.max(0, boostedConfidence));

  const explanation = independent
    ? `${sourceCount} independent sources with average ${(avgConfidence * 100).toFixed(1)}% confidence. Boost factor: ${boost.toFixed(2)}x`
    : `${sourceCount} non-independent sources (potential echo chamber). Limited boost: ${boost.toFixed(2)}x`;

  return {
    confidence: boostedConfidence,
    method: 'corroboration',
    components: confidences,
    ceilingApplied: boostedConfidence === 0.99,
    explanation:
      contradictionCount > 0
        ? `${explanation}. Penalty for ${contradictionCount} contradicting source(s).`
        : explanation,
  };
}

/**
 * Resolve conflicts between contradicting claims
 *
 * When sources disagree, confidence must be reduced and
 * the conflict flagged for resolution.
 */
export function resolveConflicts(
  claims: Array<{ id: string; confidence: number; content: string }>
): ConflictResolution {
  if (claims.length < 2) {
    return {
      confidence: claims[0]?.confidence ?? 0,
      conflicts: [],
      totalPenalty: 0,
      recommendation: 'No conflicts to resolve',
    };
  }

  // Identify conflicts (simplified - real implementation would use semantic similarity)
  const conflicts: Array<{ sourceA: string; sourceB: string; severity: number }> = [];

  // For now, assume all claims are about the same topic and might conflict
  // In practice, this would use NLP/semantic analysis
  for (let i = 0; i < claims.length; i++) {
    for (let j = i + 1; j < claims.length; j++) {
      const confidenceDiff = Math.abs(claims[i].confidence - claims[j].confidence);
      if (confidenceDiff > 0.3) {
        conflicts.push({
          sourceA: claims[i].id,
          sourceB: claims[j].id,
          severity: confidenceDiff,
        });
      }
    }
  }

  // Calculate penalty
  const totalPenalty = Math.min(
    0.5,
    conflicts.reduce((sum, c) => sum + c.severity * 0.2, 0)
  );

  // Average confidence with penalty
  const avgConfidence =
    claims.reduce((sum, c) => sum + c.confidence, 0) / claims.length;
  const resolvedConfidence = Math.max(0.1, avgConfidence * (1 - totalPenalty));

  let recommendation: string;
  if (conflicts.length === 0) {
    recommendation = 'Claims are consistent';
  } else if (totalPenalty > 0.3) {
    recommendation =
      'Significant disagreement between sources. Seek additional evidence or human adjudication.';
  } else {
    recommendation = 'Minor disagreements. Combined estimate accounts for uncertainty.';
  }

  return {
    confidence: resolvedConfidence,
    conflicts,
    totalPenalty,
    recommendation,
  };
}

// ============================================================================
// CLAIM COMBINATION
// ============================================================================

/**
 * Combine claims with full epistemic algebra
 *
 * This is the main entry point for combining multiple claims
 * while respecting all epistemic constraints.
 *
 * @example
 * ```typescript
 * const combined = combineClaims(
 *   [claim1, claim2, claim3],
 *   { mode: 'conjunction', applyECeiling: true }
 * );
 * ```
 */
export function combineClaims(
  claims: EpistemicClaim[],
  options: {
    mode: 'conjunction' | 'disjunction' | 'corroboration';
    applyECeiling?: boolean;
    areIndependent?: boolean;
  }
): CombinedConfidence {
  const { mode, applyECeiling = true, areIndependent = false } = options;

  if (claims.length === 0) {
    return {
      confidence: 0,
      method: mode,
      components: [],
      ceilingApplied: false,
      explanation: 'No claims to combine',
    };
  }

  // Extract confidences (would need claim.confidence in real implementation)
  // For now, derive from E-level ceiling as max possible
  const confidences = claims.map(
    (c) => E_LEVEL_CONFIDENCE_CEILING[c.classification.empirical]
  );

  // Find lowest E-level for ceiling
  const lowestELevel = Math.min(
    ...claims.map((c) => c.classification.empirical)
  );

  let result: CombinedConfidence;

  switch (mode) {
    case 'conjunction':
      result = conjunction(confidences);
      break;
    case 'disjunction':
      result = disjunction(confidences);
      break;
    case 'corroboration':
      result = corroborate({
        independent: areIndependent,
        sourceCount: claims.length,
        confidences,
      });
      break;
    default: {
      const _exhaustive: never = mode;
      throw new Error(`Unknown combination mode: ${_exhaustive as string}`);
    }
  }

  // Apply E-level ceiling if requested
  if (applyECeiling) {
    const ceiling = E_LEVEL_CONFIDENCE_CEILING[lowestELevel as EmpiricalLevel];
    if (result.confidence > ceiling) {
      result.preCeilingConfidence = result.confidence;
      result.confidence = ceiling;
      result.ceilingApplied = true;
      result.explanation += ` E-level ceiling (E${lowestELevel}) applied: capped at ${(ceiling * 100).toFixed(1)}%.`;
    }
  }

  return result;
}

/**
 * Combine claim along an inference chain
 *
 * @example
 * ```typescript
 * // A implies B implies C, each step degrades
 * const chained = combineChain([claimA, claimB, claimC]);
 * ```
 */
export function combineChain(
  claims: EpistemicClaim[],
  degradationFactor: number = DEFAULT_CHAIN_DEGRADATION
): CombinedConfidence & { chainWarning?: string } {
  if (claims.length === 0) {
    return {
      confidence: 0,
      method: 'chain',
      components: [],
      ceilingApplied: false,
      explanation: 'No claims in chain',
    };
  }

  // Apply conjunction across all claims first
  const conjunctionResult = conjunction(
    claims.map((c) => E_LEVEL_CONFIDENCE_CEILING[c.classification.empirical])
  );

  // Then apply chain degradation
  const degradation = chainDegradation(
    conjunctionResult.confidence,
    claims.length - 1, // n claims = n-1 inference steps
    degradationFactor
  );

  return {
    confidence: degradation.finalConfidence,
    method: 'chain',
    components: claims.map(
      (c) => E_LEVEL_CONFIDENCE_CEILING[c.classification.empirical]
    ),
    ceilingApplied: false,
    explanation: `Inference chain of ${claims.length} claims. Conjunction: ${(conjunctionResult.confidence * 100).toFixed(1)}%, then ${claims.length - 1} degradation steps: ${(degradation.finalConfidence * 100).toFixed(1)}%`,
    chainWarning: degradation.warning,
  };
}

// ============================================================================
// UNCERTAINTY PROPAGATION
// ============================================================================

/**
 * Propagate uncertainty through a dependency graph
 *
 * When claim A depends on claims B and C, A's uncertainty
 * must account for B and C's uncertainties.
 */
export function propagateUncertainty(
  dependencies: Array<{
    claimId: string;
    confidence: number;
    dependencyType: 'requires' | 'supports' | 'weakens';
  }>
): {
  effectiveConfidence: number;
  uncertaintyContributions: Map<string, number>;
  totalUncertainty: number;
} {
  if (dependencies.length === 0) {
    return {
      effectiveConfidence: 1,
      uncertaintyContributions: new Map(),
      totalUncertainty: 0,
    };
  }

  const contributions = new Map<string, number>();
  let effectiveConfidence = 1;

  for (const dep of dependencies) {
    const uncertainty = 1 - dep.confidence;

    switch (dep.dependencyType) {
      case 'requires':
        // Required dependencies multiply confidence
        effectiveConfidence *= dep.confidence;
        contributions.set(dep.claimId, uncertainty);
        break;

      case 'supports':
        // Supporting dependencies boost slightly
        effectiveConfidence *= 1 + uncertainty * 0.1;
        contributions.set(dep.claimId, -uncertainty * 0.1); // Negative = helps
        break;

      case 'weakens':
        // Weakening dependencies reduce confidence
        effectiveConfidence *= 1 - uncertainty * 0.3;
        contributions.set(dep.claimId, uncertainty * 0.3);
        break;
    }
  }

  // Cap at reasonable bounds
  effectiveConfidence = Math.min(0.99, Math.max(0.01, effectiveConfidence));

  const totalUncertainty = 1 - effectiveConfidence;

  return {
    effectiveConfidence,
    uncertaintyContributions: contributions,
    totalUncertainty,
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if a confidence claim is epistemically valid
 * given the E-level of the evidence
 */
export function isValidConfidence(
  confidence: number,
  eLevel: EmpiricalLevel
): { valid: boolean; reason?: string; maxAllowed: number } {
  const ceiling = E_LEVEL_CONFIDENCE_CEILING[eLevel];

  if (confidence > ceiling) {
    return {
      valid: false,
      reason: `Confidence ${(confidence * 100).toFixed(1)}% exceeds E${eLevel} ceiling of ${(ceiling * 100).toFixed(1)}%`,
      maxAllowed: ceiling,
    };
  }

  if (confidence < 0 || confidence > 1) {
    return {
      valid: false,
      reason: 'Confidence must be between 0 and 1',
      maxAllowed: ceiling,
    };
  }

  return {
    valid: true,
    maxAllowed: ceiling,
  };
}

/**
 * Suggest appropriate confidence based on E-level and evidence strength
 */
export function suggestConfidence(
  eLevel: EmpiricalLevel,
  evidenceStrength: 'weak' | 'moderate' | 'strong'
): number {
  const ceiling = E_LEVEL_CONFIDENCE_CEILING[eLevel];

  switch (evidenceStrength) {
    case 'weak':
      return ceiling * 0.4;
    case 'moderate':
      return ceiling * 0.7;
    case 'strong':
      return ceiling * 0.9;
    default:
      return ceiling * 0.5;
  }
}

/**
 * Format a confidence value with appropriate uncertainty language
 */
export function formatConfidence(confidence: number): string {
  if (confidence >= 0.95) return 'extremely high confidence (>95%)';
  if (confidence >= 0.85) return 'high confidence (85-95%)';
  if (confidence >= 0.70) return 'moderate confidence (70-85%)';
  if (confidence >= 0.50) return 'low-moderate confidence (50-70%)';
  if (confidence >= 0.30) return 'low confidence (30-50%)';
  return 'very low confidence (<30%)';
}

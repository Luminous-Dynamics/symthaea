/**
 * Branded Types Module
 *
 * Provides compile-time type safety through branded types.
 * This module has no dependencies to avoid circular imports.
 *
 * @packageDocumentation
 * @module utils/branded
 */

// Branded type symbols
declare const AgentIdBrand: unique symbol;
declare const HappIdBrand: unique symbol;
declare const TrustScoreBrand: unique symbol;
declare const TimestampMsBrand: unique symbol;

/**
 * Branded AgentId type for compile-time safety
 */
export type BrandedAgentId = string & { readonly [AgentIdBrand]: typeof AgentIdBrand };

/**
 * Branded HappId type for compile-time safety
 */
export type BrandedHappId = string & { readonly [HappIdBrand]: typeof HappIdBrand };

/**
 * Branded TrustScore type (0-1) for compile-time safety
 */
export type BrandedTrustScore = number & { readonly [TrustScoreBrand]: typeof TrustScoreBrand };

/**
 * Branded Timestamp (ms) type for compile-time safety
 */
export type BrandedTimestampMs = number & { readonly [TimestampMsBrand]: typeof TimestampMsBrand };

/**
 * Create a branded AgentId
 */
export function brandAgentId(id: string): BrandedAgentId {
  if (!id || typeof id !== 'string') {
    throw new Error('Invalid agent ID');
  }
  return id as BrandedAgentId;
}

/**
 * Create a branded HappId
 */
export function brandHappId(id: string): BrandedHappId {
  if (!id || typeof id !== 'string') {
    throw new Error('Invalid hApp ID');
  }
  return id as BrandedHappId;
}

/**
 * Create a branded TrustScore
 */
export function brandTrustScore(score: number): BrandedTrustScore {
  if (typeof score !== 'number' || score < 0 || score > 1) {
    throw new Error('Trust score must be between 0 and 1');
  }
  return score as BrandedTrustScore;
}

/**
 * Create a branded Timestamp
 */
export function brandTimestamp(ts: number): BrandedTimestampMs {
  if (typeof ts !== 'number' || ts < 0) {
    throw new Error('Invalid timestamp');
  }
  return ts as BrandedTimestampMs;
}

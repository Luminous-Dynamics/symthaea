// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Charter v2.0
 *
 * 3D truth classification system for claims and credentials.
 */

import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClassification,
  type EpistemicClaim,
  type Evidence,
} from './types.js';
import { EpistemicError, ErrorCode, validate, assertDefined } from '../errors.js';
import { secureUUID } from '../security/index.js';

// Re-export all types from types module (no circular dependency)
export * from './types.js';

// Re-export algebra module
export * from './algebra.js';

// Re-export GIS v4.0 module (Graceful Ignorance System with Harmonic extensions)
export * from './gis.js';

/**
 * Get classification code (e.g., "E3-N2-M2")
 */
export function classificationCode(c: EpistemicClassification): string {
  return `E${c.empirical}-N${c.normative}-M${c.materiality}`;
}

/**
 * Check if classification meets minimum requirements
 */
export function meetsMinimum(
  c: EpistemicClassification,
  minE: EmpiricalLevel,
  minN: NormativeLevel,
  minM: MaterialityLevel = MaterialityLevel.M0_Ephemeral
): boolean {
  return c.empirical >= minE && c.normative >= minN && c.materiality >= minM;
}

/**
 * Parse classification code back to classification
 * Returns null if the code format is invalid
 */
export function parseClassificationCode(code: string): EpistemicClassification | null {
  if (typeof code !== 'string') return null;

  const match = code.match(/^E(\d)-N(\d)-M(\d)$/);
  if (!match) return null;

  const [, e, n, m] = match;
  const empirical = parseInt(e) as EmpiricalLevel;
  const normative = parseInt(n) as NormativeLevel;
  const materiality = parseInt(m) as MaterialityLevel;

  // Validate parsed values are in valid ranges
  if (
    (empirical as number) < 0 ||
    (empirical as number) > 4 ||
    (normative as number) < 0 ||
    (normative as number) > 3 ||
    (materiality as number) < 0 ||
    (materiality as number) > 3
  ) {
    return null;
  }

  return { empirical, normative, materiality };
}

/**
 * Parse classification code, throwing if invalid
 * @throws {EpistemicError} If the code format is invalid
 */
export function parseClassificationCodeStrict(code: string): EpistemicClassification {
  const result = parseClassificationCode(code);
  if (result === null) {
    throw new EpistemicError(
      `Invalid classification code: ${code}`,
      ErrorCode.EPISTEMIC_INVALID_CODE,
      { code, expectedFormat: 'E[0-4]-N[0-3]-M[0-3]' }
    );
  }
  return result;
}

// Evidence and EpistemicClaim interfaces are exported from ./types.js

/**
 * Validate enum level is within valid range
 */
function validateEmpiricalLevel(level: EmpiricalLevel): void {
  if (level < EmpiricalLevel.E0_Unverified || level > EmpiricalLevel.E4_Consensus) {
    throw new EpistemicError(
      `Invalid empirical level: ${level}`,
      ErrorCode.EPISTEMIC_INVALID_LEVEL,
      { level, validRange: [0, 4] }
    );
  }
}

function validateNormativeLevel(level: NormativeLevel): void {
  if (level < NormativeLevel.N0_Personal || level > NormativeLevel.N3_Universal) {
    throw new EpistemicError(
      `Invalid normative level: ${level}`,
      ErrorCode.EPISTEMIC_INVALID_LEVEL,
      { level, validRange: [0, 3] }
    );
  }
}

function validateMaterialityLevel(level: MaterialityLevel): void {
  if (level < MaterialityLevel.M0_Ephemeral || level > MaterialityLevel.M3_Immutable) {
    throw new EpistemicError(
      `Invalid materiality level: ${level}`,
      ErrorCode.EPISTEMIC_INVALID_LEVEL,
      { level, validRange: [0, 3] }
    );
  }
}

/**
 * Create a new epistemic claim
 * @throws {EpistemicError} If content or issuer is empty, or levels are invalid
 */
export function createClaim(
  content: string,
  empirical: EmpiricalLevel,
  normative: NormativeLevel,
  materiality: MaterialityLevel,
  issuer: string
): EpistemicClaim {
  validate().notEmpty('content', content).notEmpty('issuer', issuer).throwIfInvalid();

  validateEmpiricalLevel(empirical);
  validateNormativeLevel(normative);
  validateMaterialityLevel(materiality);

  return {
    id: generateId(),
    content,
    classification: { empirical, normative, materiality },
    evidence: [],
    issuer,
    issuedAt: Date.now(),
  };
}

/**
 * Validate evidence object
 */
function validateEvidence(evidence: Evidence): void {
  validate()
    .notEmpty('evidence.type', evidence.type)
    .notEmpty('evidence.source', evidence.source)
    .throwIfInvalid();
}

/**
 * Add evidence to a claim
 * @throws {EpistemicError} If claim is not defined or evidence is invalid
 */
export function addEvidence(claim: EpistemicClaim, evidence: Evidence): EpistemicClaim {
  assertDefined(claim, 'claim');
  assertDefined(evidence, 'evidence');
  validateEvidence(evidence);

  return {
    ...claim,
    evidence: [...claim.evidence, evidence],
  };
}

/**
 * Check if claim meets epistemic standard
 * @throws {EpistemicError} If claim is not defined
 */
export function meetsStandard(
  claim: EpistemicClaim,
  minE: EmpiricalLevel,
  minN: NormativeLevel = NormativeLevel.N0_Personal
): boolean {
  assertDefined(claim, 'claim');
  return meetsMinimum(claim.classification, minE, minN);
}

/**
 * Check if claim is expired
 * @throws {EpistemicError} If claim is not defined
 */
export function isExpired(claim: EpistemicClaim): boolean {
  assertDefined(claim, 'claim');
  if (!claim.expiresAt) return false;
  return Date.now() > claim.expiresAt;
}

/**
 * Claim builder for fluent API
 */
export class ClaimBuilder {
  private content: string;
  private classification: EpistemicClassification;
  private evidence: Evidence[] = [];
  private issuer = '';
  private expiresAt?: number;

  /**
   * @throws {EpistemicError} If content is empty
   */
  constructor(content: string) {
    if (typeof content !== 'string' || content.trim() === '') {
      throw new EpistemicError('Claim content must not be empty', ErrorCode.INVALID_ARGUMENT, {
        content,
      });
    }
    this.content = content;
    this.classification = {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    };
  }

  /**
   * @throws {EpistemicError} If level is invalid
   */
  withEmpirical(level: EmpiricalLevel): this {
    validateEmpiricalLevel(level);
    this.classification.empirical = level;
    return this;
  }

  /**
   * @throws {EpistemicError} If level is invalid
   */
  withNormative(level: NormativeLevel): this {
    validateNormativeLevel(level);
    this.classification.normative = level;
    return this;
  }

  /**
   * @throws {EpistemicError} If level is invalid
   */
  withMateriality(level: MaterialityLevel): this {
    validateMaterialityLevel(level);
    this.classification.materiality = level;
    return this;
  }

  /**
   * @throws {EpistemicError} If any level is invalid
   */
  withClassification(e: EmpiricalLevel, n: NormativeLevel, m: MaterialityLevel): this {
    validateEmpiricalLevel(e);
    validateNormativeLevel(n);
    validateMaterialityLevel(m);
    this.classification = { empirical: e, normative: n, materiality: m };
    return this;
  }

  /**
   * @throws {EpistemicError} If evidence is invalid
   */
  withEvidence(evidence: Evidence): this {
    assertDefined(evidence, 'evidence');
    validateEvidence(evidence);
    this.evidence.push(evidence);
    return this;
  }

  withIssuer(issuer: string): this {
    validate().notEmpty('issuer', issuer).throwIfInvalid();
    this.issuer = issuer;
    return this;
  }

  /**
   * @throws {EpistemicError} If expiresAt is in the past
   */
  withExpiration(expiresAt: number): this {
    if (expiresAt <= Date.now()) {
      throw new EpistemicError(
        'Expiration date must be in the future',
        ErrorCode.INVALID_ARGUMENT,
        { expiresAt, now: Date.now() }
      );
    }
    this.expiresAt = expiresAt;
    return this;
  }

  build(): EpistemicClaim {
    return {
      id: generateId(),
      content: this.content,
      classification: this.classification,
      evidence: this.evidence,
      issuer: this.issuer,
      issuedAt: Date.now(),
      expiresAt: this.expiresAt,
    };
  }
}

/**
 * Create a claim builder
 */
export function claim(content: string): ClaimBuilder {
  return new ClaimBuilder(content);
}

// Utility functions
/**
 * Generate a cryptographically secure unique ID for claims
 */
function generateId(): string {
  return 'claim_' + secureUUID();
}

// Standard requirements for common use cases
export const Standards = {
  // Requires cryptographic proof, network-wide, persistent
  HighTrust: {
    minE: EmpiricalLevel.E3_Cryptographic,
    minN: NormativeLevel.N2_Network,
    minM: MaterialityLevel.M2_Persistent,
  },
  // Requires private verification, communal, temporal
  MediumTrust: {
    minE: EmpiricalLevel.E2_PrivateVerify,
    minN: NormativeLevel.N1_Communal,
    minM: MaterialityLevel.M1_Temporal,
  },
  // Requires testimony, personal, ephemeral
  LowTrust: {
    minE: EmpiricalLevel.E1_Testimonial,
    minN: NormativeLevel.N0_Personal,
    minM: MaterialityLevel.M0_Ephemeral,
  },
} as const;

/**
 * Summary statistics for an epistemic batch
 */
export interface EpistemicBatchSummary {
  total: number;
  expired: number;
  valid: number;
  byEmpiricalLevel: Record<string, number>;
  byNormativeLevel: Record<string, number>;
  byMaterialityLevel: Record<string, number>;
}

/**
 * Batch operations for epistemic claims.
 * Provides ergonomic APIs for creating, filtering, and analyzing multiple claims.
 *
 * @example
 * ```typescript
 * const batch = new EpistemicBatch();
 * batch.addClaim("User verified identity", b => b
 *   .withEmpirical(EmpiricalLevel.E3_Cryptographic)
 *   .withNormative(NormativeLevel.N2_Network));
 * batch.addClaim("Location check passed");
 *
 * const highTrust = batch.filterByStandard(
 *   EmpiricalLevel.E2_PrivateVerify,
 *   NormativeLevel.N1_Communal
 * );
 * console.log(batch.summary());
 * ```
 */
export class EpistemicBatch {
  private claims: EpistemicClaim[] = [];

  /**
   * Get the number of claims in the batch
   */
  get size(): number {
    return this.claims.length;
  }

  /**
   * Add a claim to the batch.
   *
   * @param content - The claim content
   * @param builder - Optional builder callback to configure the claim
   * @returns This batch for chaining
   */
  addClaim(content: string, builder?: (b: ClaimBuilder) => ClaimBuilder): this {
    let claimBuilder = claim(content);
    if (builder) {
      claimBuilder = builder(claimBuilder);
    }
    this.claims.push(claimBuilder.build());
    return this;
  }

  /**
   * Add an existing claim to the batch.
   *
   * @param existingClaim - An already-built EpistemicClaim
   * @returns This batch for chaining
   */
  addExistingClaim(existingClaim: EpistemicClaim): this {
    this.claims.push(existingClaim);
    return this;
  }

  /**
   * Get all claims in the batch.
   *
   * @returns Array of all claims
   */
  getClaims(): EpistemicClaim[] {
    return [...this.claims];
  }

  /**
   * Filter claims by minimum epistemic standard.
   * Returns claims that meet or exceed the specified levels.
   *
   * @param minE - Minimum empirical level
   * @param minN - Minimum normative level
   * @param minM - Optional minimum materiality level
   * @returns Array of claims meeting the standard
   */
  filterByStandard(
    minE: EmpiricalLevel,
    minN: NormativeLevel,
    minM?: MaterialityLevel
  ): EpistemicClaim[] {
    return this.claims.filter((c) => meetsMinimum(c.classification, minE, minN, minM));
  }

  /**
   * Filter out expired claims, returning only valid ones.
   *
   * @returns Array of non-expired claims
   */
  filterByExpiry(): EpistemicClaim[] {
    return this.claims.filter((c) => !isExpired(c));
  }

  /**
   * Get claims that match a specific standard preset.
   *
   * @param standard - One of Standards.HighTrust, MediumTrust, or LowTrust
   * @returns Array of claims meeting the standard
   */
  filterByPreset(standard: (typeof Standards)[keyof typeof Standards]): EpistemicClaim[] {
    return this.filterByStandard(standard.minE, standard.minN, standard.minM);
  }

  /**
   * Get summary statistics for the batch.
   *
   * @returns Summary object with counts by level and expiry status
   */
  summary(): EpistemicBatchSummary {
    const now = Date.now();
    const byEmpiricalLevel: Record<string, number> = {};
    const byNormativeLevel: Record<string, number> = {};
    const byMaterialityLevel: Record<string, number> = {};
    let expired = 0;

    for (const c of this.claims) {
      // Count empirical levels
      const eName = EmpiricalLevel[c.classification.empirical];
      byEmpiricalLevel[eName] = (byEmpiricalLevel[eName] || 0) + 1;

      // Count normative levels
      const nName = NormativeLevel[c.classification.normative];
      byNormativeLevel[nName] = (byNormativeLevel[nName] || 0) + 1;

      // Count materiality levels
      const mName = MaterialityLevel[c.classification.materiality];
      byMaterialityLevel[mName] = (byMaterialityLevel[mName] || 0) + 1;

      // Count expired
      if (c.expiresAt !== undefined && c.expiresAt <= now) {
        expired++;
      }
    }

    return {
      total: this.claims.length,
      expired,
      valid: this.claims.length - expired,
      byEmpiricalLevel,
      byNormativeLevel,
      byMaterialityLevel,
    };
  }

  /**
   * Clear all claims from the batch.
   */
  clear(): void {
    this.claims = [];
  }

  /**
   * Create a new batch containing only claims that pass a predicate.
   *
   * @param predicate - Function to test each claim
   * @returns New EpistemicBatch with filtered claims
   */
  filter(predicate: (claim: EpistemicClaim) => boolean): EpistemicBatch {
    const newBatch = new EpistemicBatch();
    for (const c of this.claims) {
      if (predicate(c)) {
        newBatch.addExistingClaim(c);
      }
    }
    return newBatch;
  }

  /**
   * Map claims to a new array of values.
   *
   * @param mapper - Function to transform each claim
   * @returns Array of transformed values
   */
  map<T>(mapper: (claim: EpistemicClaim) => T): T[] {
    return this.claims.map(mapper);
  }
}

/**
 * Configuration for claim pool TTL strategies
 */
export interface ClaimPoolConfig {
  /** Default TTL in milliseconds for claims without explicit expiration */
  defaultTtlMs?: number;
  /** TTL multipliers by empirical level (higher levels get longer TTL) */
  ttlByEmpiricalLevel?: Partial<Record<EmpiricalLevel, number>>;
  /** Auto-cleanup interval in milliseconds (0 = disabled) */
  cleanupIntervalMs?: number;
  /** Maximum number of claims to hold */
  maxClaims?: number;
}

/**
 * Statistics for the claim pool
 */
export interface ClaimPoolStats {
  total: number;
  active: number;
  expired: number;
  expiringSoon: number; // Within 10% of TTL remaining
  oldestClaimAge: number;
  newestClaimAge: number;
  cleanupCount: number;
}

/**
 * Default TTL multipliers by empirical level.
 * Higher trust claims live longer by default.
 */
const DEFAULT_TTL_MULTIPLIERS: Record<EmpiricalLevel, number> = {
  [EmpiricalLevel.E0_Unverified]: 0.25, // 25% of base TTL
  [EmpiricalLevel.E1_Testimonial]: 0.5, // 50% of base TTL
  [EmpiricalLevel.E2_PrivateVerify]: 1.0, // 100% of base TTL
  [EmpiricalLevel.E3_Cryptographic]: 2.0, // 200% of base TTL
  [EmpiricalLevel.E4_Consensus]: 4.0, // 400% of base TTL (highest trust)
};

/**
 * Managed pool for epistemic claims with automatic lifecycle management.
 * Handles claim expiration, TTL strategies, and cleanup.
 *
 * @example
 * ```typescript
 * const pool = new EpistemicClaimPool({
 *   defaultTtlMs: 3600000, // 1 hour base TTL
 *   cleanupIntervalMs: 60000, // Cleanup every minute
 * });
 *
 * // Add claims - TTL automatically assigned based on empirical level
 * pool.add(claim("User verified").withEmpirical(EmpiricalLevel.E3_Cryptographic).build());
 * pool.add(claim("Temporary check").build()); // Gets shorter TTL
 *
 * // Get only active claims
 * const active = pool.getActive();
 *
 * // Manual cleanup
 * const removed = pool.cleanup();
 * console.log(`Removed ${removed} expired claims`);
 *
 * // Don't forget to stop when done
 * pool.stop();
 * ```
 */
export class EpistemicClaimPool {
  private claims: Map<string, EpistemicClaim> = new Map();
  private effectiveExpiry: Map<string, number> = new Map();
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;
  private cleanupCount = 0;

  private readonly defaultTtlMs: number;
  private readonly ttlMultipliers: Record<EmpiricalLevel, number>;
  private readonly maxClaims: number;

  constructor(config: ClaimPoolConfig = {}) {
    this.defaultTtlMs = config.defaultTtlMs ?? 3600000; // 1 hour default
    this.maxClaims = config.maxClaims ?? 10000;
    this.ttlMultipliers = {
      ...DEFAULT_TTL_MULTIPLIERS,
      ...config.ttlByEmpiricalLevel,
    };

    if (config.cleanupIntervalMs && config.cleanupIntervalMs > 0) {
      this.cleanupTimer = setInterval(() => {
        this.cleanup();
      }, config.cleanupIntervalMs);
    }
  }

  /**
   * Add a claim to the pool.
   * If the claim has no expiration, one is assigned based on its empirical level.
   *
   * @param claimToAdd - The claim to add
   * @returns This pool for chaining
   */
  add(claimToAdd: EpistemicClaim): this {
    // Enforce max claims limit
    if (this.claims.size >= this.maxClaims) {
      this.cleanup(); // Try cleanup first
      if (this.claims.size >= this.maxClaims) {
        // Remove oldest claim if still at limit
        const oldest = this.getOldestClaim();
        if (oldest) {
          this.remove(oldest.id);
        }
      }
    }

    this.claims.set(claimToAdd.id, claimToAdd);

    // Calculate effective expiry
    if (claimToAdd.expiresAt !== undefined) {
      this.effectiveExpiry.set(claimToAdd.id, claimToAdd.expiresAt);
    } else {
      const multiplier = this.ttlMultipliers[claimToAdd.classification.empirical];
      const ttl = this.defaultTtlMs * multiplier;
      this.effectiveExpiry.set(claimToAdd.id, Date.now() + ttl);
    }

    return this;
  }

  /**
   * Add multiple claims at once.
   *
   * @param claimsToAdd - Array of claims to add
   * @returns This pool for chaining
   */
  addMany(claimsToAdd: EpistemicClaim[]): this {
    for (const c of claimsToAdd) {
      this.add(c);
    }
    return this;
  }

  /**
   * Remove a claim by ID.
   *
   * @param id - The claim ID to remove
   * @returns True if the claim was removed
   */
  remove(id: string): boolean {
    this.effectiveExpiry.delete(id);
    return this.claims.delete(id);
  }

  /**
   * Get a claim by ID if it exists and is not expired.
   *
   * @param id - The claim ID
   * @returns The claim or undefined
   */
  get(id: string): EpistemicClaim | undefined {
    const claim = this.claims.get(id);
    if (!claim) return undefined;

    const expiry = this.effectiveExpiry.get(id);
    if (expiry !== undefined && expiry <= Date.now()) {
      return undefined; // Expired
    }

    return claim;
  }

  /**
   * Check if a claim exists and is active.
   *
   * @param id - The claim ID
   * @returns True if claim exists and is not expired
   */
  has(id: string): boolean {
    return this.get(id) !== undefined;
  }

  /**
   * Get all active (non-expired) claims.
   *
   * @returns Array of active claims
   */
  getActive(): EpistemicClaim[] {
    const now = Date.now();
    const active: EpistemicClaim[] = [];

    for (const [id, claim] of this.claims) {
      const expiry = this.effectiveExpiry.get(id);
      if (expiry === undefined || expiry > now) {
        active.push(claim);
      }
    }

    return active;
  }

  /**
   * Get all expired claims (for inspection before cleanup).
   *
   * @returns Array of expired claims
   */
  getExpired(): EpistemicClaim[] {
    const now = Date.now();
    const expired: EpistemicClaim[] = [];

    for (const [id, claim] of this.claims) {
      const expiry = this.effectiveExpiry.get(id);
      if (expiry !== undefined && expiry <= now) {
        expired.push(claim);
      }
    }

    return expired;
  }

  /**
   * Get claims expiring soon (within percentage of their TTL).
   *
   * @param withinMs - Time window in milliseconds (default: 10% of default TTL)
   * @returns Array of claims expiring soon
   */
  getExpiringSoon(withinMs?: number): EpistemicClaim[] {
    const now = Date.now();
    const threshold = withinMs ?? this.defaultTtlMs * 0.1;
    const cutoff = now + threshold;
    const expiring: EpistemicClaim[] = [];

    for (const [id, claim] of this.claims) {
      const expiry = this.effectiveExpiry.get(id);
      if (expiry !== undefined && expiry > now && expiry <= cutoff) {
        expiring.push(claim);
      }
    }

    return expiring;
  }

  /**
   * Remove all expired claims.
   *
   * @returns Number of claims removed
   */
  cleanup(): number {
    const now = Date.now();
    let removed = 0;

    for (const [id] of this.claims) {
      const expiry = this.effectiveExpiry.get(id);
      if (expiry !== undefined && expiry <= now) {
        this.claims.delete(id);
        this.effectiveExpiry.delete(id);
        removed++;
      }
    }

    this.cleanupCount += removed;
    return removed;
  }

  /**
   * Extend the expiration of a claim.
   *
   * @param id - The claim ID
   * @param additionalMs - Additional time in milliseconds
   * @returns True if the claim was extended
   */
  extend(id: string, additionalMs: number): boolean {
    const currentExpiry = this.effectiveExpiry.get(id);
    if (currentExpiry === undefined) return false;

    this.effectiveExpiry.set(id, currentExpiry + additionalMs);
    return true;
  }

  /**
   * Get the remaining TTL for a claim.
   *
   * @param id - The claim ID
   * @returns Remaining time in ms, or -1 if not found, or Infinity if no expiry
   */
  getRemainingTtl(id: string): number {
    if (!this.claims.has(id)) return -1;

    const expiry = this.effectiveExpiry.get(id);
    if (expiry === undefined) return Infinity;

    return Math.max(0, expiry - Date.now());
  }

  /**
   * Get pool statistics.
   *
   * @returns Pool statistics
   */
  getStats(): ClaimPoolStats {
    const now = Date.now();
    let active = 0;
    let expired = 0;
    let expiringSoon = 0;
    let oldestAge = 0;
    let newestAge = Infinity;
    const threshold = this.defaultTtlMs * 0.1;

    for (const [id, claim] of this.claims) {
      const age = now - claim.issuedAt;
      oldestAge = Math.max(oldestAge, age);
      newestAge = Math.min(newestAge, age);

      const expiry = this.effectiveExpiry.get(id);
      if (expiry !== undefined && expiry <= now) {
        expired++;
      } else {
        active++;
        if (expiry !== undefined && expiry <= now + threshold) {
          expiringSoon++;
        }
      }
    }

    return {
      total: this.claims.size,
      active,
      expired,
      expiringSoon,
      oldestClaimAge: this.claims.size > 0 ? oldestAge : 0,
      newestClaimAge: this.claims.size > 0 && newestAge !== Infinity ? newestAge : 0,
      cleanupCount: this.cleanupCount,
    };
  }

  /**
   * Get the oldest claim in the pool.
   *
   * @returns The oldest claim or undefined
   */
  private getOldestClaim(): EpistemicClaim | undefined {
    let oldest: EpistemicClaim | undefined;
    let oldestTime = Infinity;

    for (const claim of this.claims.values()) {
      if (claim.issuedAt < oldestTime) {
        oldestTime = claim.issuedAt;
        oldest = claim;
      }
    }

    return oldest;
  }

  /**
   * Clear all claims from the pool.
   */
  clear(): void {
    this.claims.clear();
    this.effectiveExpiry.clear();
  }

  /**
   * Stop automatic cleanup and release resources.
   */
  stop(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
  }

  /**
   * Get total number of claims (including expired).
   */
  get size(): number {
    return this.claims.size;
  }

  /**
   * Get number of active claims.
   */
  get activeSize(): number {
    return this.getActive().length;
  }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Unified Identity Verification
 *
 * Cross-hApp identity verification for the Mycelix ecosystem.
 * Provides a simple, unified interface for any hApp to verify identities,
 * credentials, and reputation without needing direct access to the identity hApp.
 *
 * @packageDocumentation
 * @module bridge/unified-identity
 */

import { validateDID, validateTrustScore, validateNonEmpty, assertValid, combineResults } from '../utils/validation.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Verification level for identity checks
 */
export enum VerificationLevel {
  /** Basic - DID exists and is not deactivated */
  BASIC = 'basic',
  /** Standard - DID valid + minimum trust score */
  STANDARD = 'standard',
  /** Enhanced - DID valid + high trust + verified credentials */
  ENHANCED = 'enhanced',
  /** Maximum - Full verification with credential chain */
  MAXIMUM = 'maximum',
}

/**
 * Identity verification request
 */
export interface VerifyIdentityRequest {
  /** DID to verify */
  did: string;
  /** Verification level required */
  level?: VerificationLevel;
  /** Minimum trust score required (0-1) */
  minTrustScore?: number;
  /** Required credential types */
  requiredCredentials?: string[];
  /** Source hApp requesting verification */
  sourceHapp: string;
  /** Purpose of verification (for audit) */
  purpose?: string;
}

/**
 * Identity verification result
 */
export interface VerifyIdentityResult {
  /** Verification passed */
  verified: boolean;
  /** DID that was verified */
  did: string;
  /** Trust score (0-1) */
  trustScore: number;
  /** Verification level achieved */
  level: VerificationLevel;
  /** Whether DID is active */
  isActive: boolean;
  /** Credentials that were verified */
  verifiedCredentials: VerifiedCredential[];
  /** Reasons for failure (if not verified) */
  failureReasons: string[];
  /** Verification timestamp */
  timestamp: number;
  /** Time-to-live for caching (ms) */
  ttlMs: number;
}

/**
 * Verified credential info
 */
export interface VerifiedCredential {
  /** Credential type */
  type: string;
  /** Issuer DID */
  issuer: string;
  /** Is credential valid */
  isValid: boolean;
  /** Is credential revoked */
  isRevoked: boolean;
  /** Expiration timestamp */
  expiresAt?: number;
}

/**
 * Credential verification request
 */
export interface VerifyCredentialRequest {
  /** Subject DID */
  subjectDid: string;
  /** Credential type to verify */
  credentialType: string;
  /** Issuer DID (optional - any issuer if not specified) */
  issuerDid?: string;
  /** Source hApp requesting verification */
  sourceHapp: string;
}

/**
 * Credential verification result
 */
export interface VerifyCredentialResult {
  /** Verification passed */
  verified: boolean;
  /** Credential details */
  credential?: VerifiedCredential;
  /** Failure reason */
  failureReason?: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Cross-hApp reputation query
 */
export interface ReputationQuery {
  /** DID to query */
  did: string;
  /** Filter by hApp (optional) */
  happFilter?: string[];
  /** Include breakdown by hApp */
  includeBreakdown?: boolean;
}

/**
 * Reputation result
 */
export interface ReputationResult {
  /** DID */
  did: string;
  /** Aggregate trust score (0-1) */
  aggregateScore: number;
  /** Number of hApps contributing */
  contributingHapps: number;
  /** Breakdown by hApp (if requested) */
  breakdown?: HappReputation[];
  /** Last update timestamp */
  lastUpdate: number;
}

/**
 * Per-hApp reputation
 */
export interface HappReputation {
  /** hApp ID */
  happId: string;
  /** Trust score from this hApp */
  score: number;
  /** Weight of this score */
  weight: number;
  /** Last interaction timestamp */
  lastInteraction: number;
}

/**
 * Identity provider configuration
 */
export interface IdentityProviderConfig {
  /** Cache verification results */
  enableCache?: boolean;
  /** Cache TTL in ms */
  cacheTtlMs?: number;
  /** Default verification level */
  defaultLevel?: VerificationLevel;
  /** Minimum trust score for standard verification */
  standardMinTrust?: number;
  /** Minimum trust score for enhanced verification */
  enhancedMinTrust?: number;
  /** Callback for verification events */
  onVerification?: (result: VerifyIdentityResult) => void;
}

// ============================================================================
// Implementation
// ============================================================================

/**
 * Verification cache entry
 */
interface CacheEntry {
  result: VerifyIdentityResult;
  expiresAt: number;
}

/**
 * Unified Identity Provider
 *
 * Provides a unified interface for cross-hApp identity verification.
 * Any hApp can use this to verify identities without direct access
 * to the identity hApp.
 */
export class UnifiedIdentityProvider {
  private config: Required<IdentityProviderConfig>;
  private cache: Map<string, CacheEntry> = new Map();
  private verificationCount = 0;
  private cacheHits = 0;

  // Simulated identity store (in production, this would call the identity hApp)
  private identities: Map<string, {
    active: boolean;
    trustScore: number;
    credentials: VerifiedCredential[];
  }> = new Map();

  constructor(config: IdentityProviderConfig = {}) {
    this.config = {
      enableCache: config.enableCache ?? true,
      cacheTtlMs: config.cacheTtlMs ?? 60000, // 1 minute default
      defaultLevel: config.defaultLevel ?? VerificationLevel.STANDARD,
      standardMinTrust: config.standardMinTrust ?? 0.5,
      enhancedMinTrust: config.enhancedMinTrust ?? 0.75,
      onVerification: config.onVerification ?? (() => {}),
    };
  }

  /**
   * Verify an identity
   */
  async verifyIdentity(request: VerifyIdentityRequest): Promise<VerifyIdentityResult> {
    // Validate input
    assertValid(combineResults(
      validateDID(request.did),
      validateNonEmpty(request.sourceHapp, 'Source hApp')
    ));

    this.verificationCount++;
    const level = request.level ?? this.config.defaultLevel;

    // Check cache
    if (this.config.enableCache) {
      const cached = this.getCached(request.did, level);
      if (cached) {
        this.cacheHits++;
        return cached;
      }
    }

    // Perform verification
    const result = await this.performVerification(request, level);

    // Cache result
    if (this.config.enableCache && result.verified) {
      this.setCache(request.did, level, result);
    }

    // Callback
    this.config.onVerification(result);

    return result;
  }

  /**
   * Verify a specific credential
   */
  async verifyCredential(request: VerifyCredentialRequest): Promise<VerifyCredentialResult> {
    assertValid(combineResults(
      validateDID(request.subjectDid, 'Subject DID'),
      validateNonEmpty(request.credentialType, 'Credential type'),
      validateNonEmpty(request.sourceHapp, 'Source hApp')
    ));

    // Get identity
    const identity = this.identities.get(request.subjectDid);
    if (!identity) {
      return {
        verified: false,
        failureReason: 'Identity not found',
        timestamp: Date.now(),
      };
    }

    // Find matching credential
    const credential = identity.credentials.find(c => {
      const typeMatch = c.type === request.credentialType;
      const issuerMatch = !request.issuerDid || c.issuer === request.issuerDid;
      return typeMatch && issuerMatch;
    });

    if (!credential) {
      return {
        verified: false,
        failureReason: 'Credential not found',
        timestamp: Date.now(),
      };
    }

    // Check credential validity
    const isExpired = credential.expiresAt && credential.expiresAt < Date.now();
    const isValid = credential.isValid && !credential.isRevoked && !isExpired;

    return {
      verified: isValid,
      credential,
      failureReason: isValid ? undefined :
        credential.isRevoked ? 'Credential revoked' :
        isExpired ? 'Credential expired' :
        'Credential invalid',
      timestamp: Date.now(),
    };
  }

  /**
   * Query reputation across hApps
   */
  async queryReputation(query: ReputationQuery): Promise<ReputationResult> {
    assertValid(validateDID(query.did));

    const identity = this.identities.get(query.did);
    if (!identity) {
      return {
        did: query.did,
        aggregateScore: 0,
        contributingHapps: 0,
        lastUpdate: Date.now(),
      };
    }

    // In production, this would aggregate from multiple hApps
    return {
      did: query.did,
      aggregateScore: identity.trustScore,
      contributingHapps: 1,
      breakdown: query.includeBreakdown ? [{
        happId: 'identity',
        score: identity.trustScore,
        weight: 1.0,
        lastInteraction: Date.now(),
      }] : undefined,
      lastUpdate: Date.now(),
    };
  }

  /**
   * Register an identity (for testing/simulation)
   */
  registerIdentity(did: string, trustScore: number, credentials: VerifiedCredential[] = []): void {
    assertValid(combineResults(
      validateDID(did),
      validateTrustScore(trustScore)
    ));

    this.identities.set(did, {
      active: true,
      trustScore,
      credentials,
    });
  }

  /**
   * Deactivate an identity
   */
  deactivateIdentity(did: string): void {
    const identity = this.identities.get(did);
    if (identity) {
      identity.active = false;
      // Invalidate cache
      this.invalidateCache(did);
    }
  }

  /**
   * Add a credential to an identity
   */
  addCredential(did: string, credential: VerifiedCredential): void {
    const identity = this.identities.get(did);
    if (identity) {
      identity.credentials.push(credential);
      this.invalidateCache(did);
    }
  }

  /**
   * Update trust score
   */
  updateTrustScore(did: string, score: number): void {
    assertValid(validateTrustScore(score));
    const identity = this.identities.get(did);
    if (identity) {
      identity.trustScore = score;
      this.invalidateCache(did);
    }
  }

  /**
   * Get verification statistics
   */
  getStats(): {
    totalVerifications: number;
    cacheHits: number;
    cacheHitRate: number;
    registeredIdentities: number;
  } {
    return {
      totalVerifications: this.verificationCount,
      cacheHits: this.cacheHits,
      cacheHitRate: this.verificationCount > 0
        ? this.cacheHits / this.verificationCount
        : 0,
      registeredIdentities: this.identities.size,
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async performVerification(
    request: VerifyIdentityRequest,
    level: VerificationLevel
  ): Promise<VerifyIdentityResult> {
    const failureReasons: string[] = [];
    const verifiedCredentials: VerifiedCredential[] = [];

    // Get identity
    const identity = this.identities.get(request.did);
    if (!identity) {
      return {
        verified: false,
        did: request.did,
        trustScore: 0,
        level: VerificationLevel.BASIC,
        isActive: false,
        verifiedCredentials: [],
        failureReasons: ['Identity not found'],
        timestamp: Date.now(),
        ttlMs: 0,
      };
    }

    // Check if active
    if (!identity.active) {
      failureReasons.push('Identity deactivated');
    }

    // Check trust score based on level
    const minTrust = this.getMinTrustForLevel(level, request.minTrustScore);
    if (identity.trustScore < minTrust) {
      failureReasons.push(
        `Trust score ${identity.trustScore.toFixed(2)} below minimum ${minTrust.toFixed(2)}`
      );
    }

    // Check required credentials
    if (request.requiredCredentials && request.requiredCredentials.length > 0) {
      for (const required of request.requiredCredentials) {
        const cred = identity.credentials.find(c => c.type === required && c.isValid && !c.isRevoked);
        if (cred) {
          verifiedCredentials.push(cred);
        } else {
          failureReasons.push(`Missing or invalid credential: ${required}`);
        }
      }
    }

    // Determine achieved level
    const achievedLevel = this.determineAchievedLevel(identity.trustScore, verifiedCredentials.length);

    return {
      verified: failureReasons.length === 0,
      did: request.did,
      trustScore: identity.trustScore,
      level: achievedLevel,
      isActive: identity.active,
      verifiedCredentials,
      failureReasons,
      timestamp: Date.now(),
      ttlMs: this.config.cacheTtlMs,
    };
  }

  private getMinTrustForLevel(level: VerificationLevel, override?: number): number {
    if (override !== undefined) return override;

    switch (level) {
      case VerificationLevel.BASIC:
        return 0;
      case VerificationLevel.STANDARD:
        return this.config.standardMinTrust;
      case VerificationLevel.ENHANCED:
      case VerificationLevel.MAXIMUM:
        return this.config.enhancedMinTrust;
      default:
        return 0;
    }
  }

  private determineAchievedLevel(trustScore: number, verifiedCredCount: number): VerificationLevel {
    if (trustScore >= this.config.enhancedMinTrust && verifiedCredCount > 0) {
      return VerificationLevel.ENHANCED;
    }
    if (trustScore >= this.config.standardMinTrust) {
      return VerificationLevel.STANDARD;
    }
    return VerificationLevel.BASIC;
  }

  private getCacheKey(did: string, level: VerificationLevel): string {
    return `${did}:${level}`;
  }

  private getCached(did: string, level: VerificationLevel): VerifyIdentityResult | null {
    const key = this.getCacheKey(did, level);
    const entry = this.cache.get(key);

    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.result;
  }

  private setCache(did: string, level: VerificationLevel, result: VerifyIdentityResult): void {
    const key = this.getCacheKey(did, level);
    this.cache.set(key, {
      result,
      expiresAt: Date.now() + this.config.cacheTtlMs,
    });
  }

  private invalidateCache(did: string): void {
    for (const key of this.cache.keys()) {
      if (key.startsWith(did + ':')) {
        this.cache.delete(key);
      }
    }
  }
}

// ============================================================================
// Singleton & Helpers
// ============================================================================

/** Global identity provider instance */
let globalProvider: UnifiedIdentityProvider | null = null;

/**
 * Get the global unified identity provider
 */
export function getIdentityProvider(config?: IdentityProviderConfig): UnifiedIdentityProvider {
  if (!globalProvider) {
    globalProvider = new UnifiedIdentityProvider(config);
  }
  return globalProvider;
}

/**
 * Create a new isolated identity provider (for testing)
 */
export function createIdentityProvider(config?: IdentityProviderConfig): UnifiedIdentityProvider {
  return new UnifiedIdentityProvider(config);
}

/**
 * Quick identity verification helper
 */
export async function verifyIdentity(
  did: string,
  sourceHapp: string,
  level: VerificationLevel = VerificationLevel.STANDARD
): Promise<boolean> {
  const provider = getIdentityProvider();
  const result = await provider.verifyIdentity({ did, sourceHapp, level });
  return result.verified;
}

/**
 * Quick credential verification helper
 */
export async function hasCredential(
  did: string,
  credentialType: string,
  sourceHapp: string
): Promise<boolean> {
  const provider = getIdentityProvider();
  const result = await provider.verifyCredential({
    subjectDid: did,
    credentialType,
    sourceHapp,
  });
  return result.verified;
}

/**
 * Quick reputation query helper
 */
export async function getTrustScore(did: string): Promise<number> {
  const provider = getIdentityProvider();
  const result = await provider.queryReputation({ did });
  return result.aggregateScore;
}

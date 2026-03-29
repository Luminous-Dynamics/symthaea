// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Identity Client
 *
 * Shared client for identity operations across all Mycelix applications.
 * Provides DID resolution, credential verification, and assurance level checks.
 *
 * Designed for graceful degradation - if identity hApp is unavailable,
 * operations return warnings/defaults instead of hard failures.
 */

import type {
  IdentityClientOptions,
  DidDocument,
  DidResolutionResult,
  VerifiableCredential,
  CredentialVerificationResult,
  RevocationStatus,
  IdentityVerificationRequest,
  IdentityVerificationResponse,
  CrossHappReputation,
  AssuranceLevel,
  HighValueTransactionConfig,
} from './types';
import { ASSURANCE_LEVEL_VALUE } from './types';

/**
 * Default configuration
 */
const DEFAULT_OPTIONS: Partial<IdentityClientOptions> = {
  roleName: 'identity',
  enableCache: true,
  cacheTtl: 5 * 60 * 1000, // 5 minutes
  fallbackMode: 'warn',
};

/**
 * Simple in-memory cache for DID documents
 */
interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

/**
 * IdentityClient - Main client for identity operations
 *
 * Usage:
 * ```typescript
 * const client = new IdentityClient({ appClient });
 * const did = await client.resolveDID('did:mycelix:abc123');
 * const valid = await client.verifyCredential(credential);
 * const level = await client.getAssuranceLevel('did:mycelix:abc123');
 * ```
 */
export class IdentityClient {
  private appClient: any;
  private roleName: string;
  private enableCache: boolean;
  private cacheTtl: number;
  private fallbackMode: 'warn' | 'error' | 'silent';

  // Caches
  private didCache: Map<string, CacheEntry<DidDocument>> = new Map();
  private revocationCache: Map<string, CacheEntry<RevocationStatus>> = new Map();

  // Connection state
  private isConnected: boolean = false;
  private identityHappAvailable: boolean = false;
  private lastConnectionCheck: number = 0;
  private connectionCheckInterval: number = 30000; // 30 seconds

  constructor(options: IdentityClientOptions) {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    this.appClient = options.appClient;
    this.roleName = opts.roleName!;
    this.enableCache = opts.enableCache!;
    this.cacheTtl = opts.cacheTtl!;
    this.fallbackMode = opts.fallbackMode!;
  }

  /**
   * Check if identity hApp is available and connected
   */
  async checkConnection(): Promise<boolean> {
    const now = Date.now();

    // Rate limit connection checks
    if (now - this.lastConnectionCheck < this.connectionCheckInterval) {
      return this.identityHappAvailable;
    }

    this.lastConnectionCheck = now;

    try {
      const appInfo = await this.appClient.appInfo();
      this.isConnected = !!appInfo;

      // Check if identity role exists
      if (appInfo?.cell_info) {
        const identityCell = appInfo.cell_info[this.roleName];
        this.identityHappAvailable = !!identityCell && identityCell.length > 0;
      } else {
        this.identityHappAvailable = false;
      }
    } catch (error) {
      this.isConnected = false;
      this.identityHappAvailable = false;

      if (this.fallbackMode === 'error') {
        throw new Error(`Identity hApp connection failed: ${error}`);
      } else if (this.fallbackMode === 'warn') {
        console.warn('[IdentityClient] Identity hApp not available:', error);
      }
    }

    return this.identityHappAvailable;
  }

  /**
   * Get cell ID for identity DNA
   */
  private async getCellId(): Promise<[Uint8Array, Uint8Array] | null> {
    try {
      const appInfo = await this.appClient.appInfo();
      const cellInfo = appInfo?.cell_info?.[this.roleName]?.[0];

      if (cellInfo && 'provisioned' in cellInfo) {
        return cellInfo.provisioned.cell_id;
      }
    } catch (error) {
      if (this.fallbackMode !== 'silent') {
        console.warn('[IdentityClient] Failed to get cell ID:', error);
      }
    }
    return null;
  }

  /**
   * Make a zome call to the identity DNA
   */
  private async callIdentityZome<T>(
    zomeName: string,
    fnName: string,
    payload: any
  ): Promise<T | null> {
    const cellId = await this.getCellId();

    if (!cellId) {
      if (this.fallbackMode === 'error') {
        throw new Error('Identity DNA not available');
      }
      return null;
    }

    try {
      const result = await this.appClient.callZome({
        cap_secret: null,
        cell_id: cellId,
        zome_name: zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      if (this.fallbackMode === 'error') {
        throw error;
      } else if (this.fallbackMode === 'warn') {
        console.warn(`[IdentityClient] Zome call ${zomeName}.${fnName} failed:`, error);
      }
      return null;
    }
  }

  // =========================================================================
  // DID Resolution
  // =========================================================================

  /**
   * Resolve a DID to its document
   *
   * @param did - DID to resolve (e.g., "did:mycelix:abc123")
   * @returns Resolution result with DID document or error
   */
  async resolveDID(did: string): Promise<DidResolutionResult> {
    const startTime = Date.now();

    // Validate DID format
    if (!did.startsWith('did:mycelix:')) {
      return {
        success: false,
        didDocument: null,
        error: 'Invalid DID format. Must start with "did:mycelix:"',
      };
    }

    // Check cache first
    if (this.enableCache) {
      const cached = this.didCache.get(did);
      if (cached && cached.expiresAt > Date.now()) {
        return {
          success: true,
          didDocument: cached.value,
          cached: true,
          metadata: {
            resolveTime: Date.now() - startTime,
            source: 'local',
          },
        };
      }
    }

    // Try to resolve from identity hApp
    const available = await this.checkConnection();

    if (available) {
      try {
        const record = await this.callIdentityZome<any>('did_registry', 'resolve_did', did);

        if (record?.entry) {
          const didDoc = this.recordToDidDocument(record);

          // Cache the result
          if (this.enableCache && didDoc) {
            this.didCache.set(did, {
              value: didDoc,
              expiresAt: Date.now() + this.cacheTtl,
            });
          }

          return {
            success: true,
            didDocument: didDoc,
            metadata: {
              resolveTime: Date.now() - startTime,
              source: 'dht',
            },
          };
        }

        return {
          success: false,
          didDocument: null,
          error: 'DID not found',
          metadata: {
            resolveTime: Date.now() - startTime,
            source: 'dht',
          },
        };
      } catch (error) {
        if (this.fallbackMode === 'error') {
          throw error;
        }
      }
    }

    // Fallback: Return mock/degraded result
    return {
      success: false,
      didDocument: null,
      error: 'Identity hApp not available - using degraded mode',
      metadata: {
        resolveTime: Date.now() - startTime,
        source: 'fallback',
      },
    };
  }

  /**
   * Resolve DID from agent public key
   *
   * @param agentPubKey - Agent public key (base64 or Uint8Array)
   * @returns Resolution result
   */
  async resolveDIDFromAgent(agentPubKey: string | Uint8Array): Promise<DidResolutionResult> {
    const agentStr = typeof agentPubKey === 'string' ? agentPubKey : this.encodeAgentKey(agentPubKey);
    const did = `did:mycelix:${agentStr}`;
    return this.resolveDID(did);
  }

  /**
   * Get DID document for an agent
   */
  async getDidDocument(agentPubKey: string | Uint8Array): Promise<DidDocument | null> {
    const available = await this.checkConnection();

    if (!available) {
      return null;
    }

    const record = await this.callIdentityZome<any>('did_registry', 'get_did_document', agentPubKey);
    return record?.entry ? this.recordToDidDocument(record) : null;
  }

  /**
   * Create a DID for the current agent
   */
  async createDID(): Promise<DidDocument | null> {
    const available = await this.checkConnection();

    if (!available) {
      if (this.fallbackMode === 'error') {
        throw new Error('Cannot create DID - identity hApp not available');
      }
      return null;
    }

    const record = await this.callIdentityZome<any>('did_registry', 'create_did', null);
    return record?.entry ? this.recordToDidDocument(record) : null;
  }

  /**
   * Get my DID document
   */
  async getMyDID(): Promise<DidDocument | null> {
    const available = await this.checkConnection();

    if (!available) {
      return null;
    }

    const record = await this.callIdentityZome<any>('did_registry', 'get_my_did', null);
    return record?.entry ? this.recordToDidDocument(record) : null;
  }

  // =========================================================================
  // Credential Verification
  // =========================================================================

  /**
   * Verify a credential
   *
   * @param credential - Verifiable credential to verify
   * @returns Verification result
   */
  async verifyCredential(credential: VerifiableCredential): Promise<CredentialVerificationResult> {
    const result: CredentialVerificationResult = {
      valid: false,
      checks: {
        signatureValid: false,
        notExpired: false,
        notRevoked: false,
        issuerTrusted: false,
        schemaValid: false,
      },
    };

    try {
      // Check expiration
      if (credential.expirationDate) {
        const expiry = new Date(credential.expirationDate);
        result.checks.notExpired = expiry > new Date();
      } else {
        result.checks.notExpired = true;
      }

      // Check revocation
      if (credential.credentialStatus) {
        const revocationResult = await this.checkRevocationStatus(credential.id);
        result.checks.notRevoked = revocationResult.status === 'active';
      } else {
        result.checks.notRevoked = true;
      }

      // Verify issuer
      const issuerResolution = await this.resolveDID(credential.issuer);
      result.checks.issuerTrusted = issuerResolution.success && !!issuerResolution.didDocument;

      // Verify signature (simplified - would need actual crypto verification)
      if (credential.proof) {
        result.checks.signatureValid = await this.verifySignature(credential, issuerResolution.didDocument);
      } else {
        result.checks.signatureValid = false;
      }

      // Schema validation (simplified)
      result.checks.schemaValid = this.validateCredentialSchema(credential);

      // Overall validity
      result.valid = Object.values(result.checks).every((check) => check);

      if (result.valid) {
        result.credential = credential;
        result.assuranceLevel = this.deriveAssuranceLevel(credential);
      }
    } catch (error) {
      result.error = `Verification failed: ${error}`;
    }

    return result;
  }

  /**
   * Check if a credential is revoked
   *
   * @param credentialId - Credential ID to check
   * @returns Revocation status
   */
  async checkRevocationStatus(credentialId: string): Promise<RevocationStatus> {
    // Check cache
    if (this.enableCache) {
      const cached = this.revocationCache.get(credentialId);
      if (cached && cached.expiresAt > Date.now()) {
        return cached.value;
      }
    }

    const available = await this.checkConnection();

    if (available) {
      const result = await this.callIdentityZome<any>(
        'revocation',
        'check_revocation_status',
        credentialId
      );

      if (result) {
        const status: RevocationStatus = {
          credentialId: result.credential_id || credentialId,
          status: result.status?.toLowerCase() || 'active',
          reason: result.reason,
          checkedAt: Date.now(),
        };

        // Cache the result
        if (this.enableCache) {
          this.revocationCache.set(credentialId, {
            value: status,
            expiresAt: Date.now() + this.cacheTtl,
          });
        }

        return status;
      }
    }

    // Fallback - assume active (permissive)
    return {
      credentialId,
      status: 'active',
      checkedAt: Date.now(),
    };
  }

  /**
   * Batch check multiple credentials for revocation
   */
  async batchCheckRevocation(credentialIds: string[]): Promise<RevocationStatus[]> {
    const available = await this.checkConnection();

    if (available) {
      const result = await this.callIdentityZome<any[]>(
        'revocation',
        'batch_check_revocation',
        credentialIds
      );

      if (result) {
        return result.map((r) => ({
          credentialId: r.credential_id,
          status: r.status?.toLowerCase() || 'active',
          reason: r.reason,
          checkedAt: Date.now(),
        }));
      }
    }

    // Fallback
    return credentialIds.map((id) => ({
      credentialId: id,
      status: 'active',
      checkedAt: Date.now(),
    }));
  }

  // =========================================================================
  // Assurance Level
  // =========================================================================

  /**
   * Get the assurance level for a DID
   *
   * @param did - DID to check
   * @returns Assurance level (E0-E4)
   */
  async getAssuranceLevel(did: string): Promise<AssuranceLevel> {
    const available = await this.checkConnection();

    if (!available) {
      // Fallback to lowest level
      return 'E0';
    }

    try {
      // Get DID document
      const resolution = await this.resolveDID(did);
      if (!resolution.success || !resolution.didDocument) {
        return 'E0';
      }

      // Check for verification credentials
      const credentials = await this.getCredentialsForDID(did);

      // Determine level based on credentials
      let level: AssuranceLevel = 'E0';

      for (const cred of credentials) {
        const credLevel = this.deriveAssuranceLevel(cred);
        if (ASSURANCE_LEVEL_VALUE[credLevel] > ASSURANCE_LEVEL_VALUE[level]) {
          level = credLevel;
        }
      }

      return level;
    } catch (error) {
      if (this.fallbackMode === 'warn') {
        console.warn('[IdentityClient] Failed to get assurance level:', error);
      }
      return 'E0';
    }
  }

  /**
   * Check if a DID meets a minimum assurance level
   *
   * @param did - DID to check
   * @param minLevel - Minimum required level
   * @returns Whether the DID meets the minimum level
   */
  async meetsAssuranceLevel(did: string, minLevel: AssuranceLevel): Promise<boolean> {
    const currentLevel = await this.getAssuranceLevel(did);
    return ASSURANCE_LEVEL_VALUE[currentLevel] >= ASSURANCE_LEVEL_VALUE[minLevel];
  }

  /**
   * Verify identity for high-value transactions
   */
  async verifyForHighValueTransaction(
    did: string,
    config: HighValueTransactionConfig
  ): Promise<IdentityVerificationResponse> {
    const verification = await this.verifyIdentity({
      did,
      minAssuranceLevel: config.requiredAssuranceLevel,
      requiredCredentials: config.requiredCredentials,
      sourceHapp: 'marketplace',
    });

    return verification;
  }

  // =========================================================================
  // Cross-hApp Identity Verification
  // =========================================================================

  /**
   * Verify identity for cross-hApp operations
   */
  async verifyIdentity(request: IdentityVerificationRequest): Promise<IdentityVerificationResponse> {
    const available = await this.checkConnection();
    const now = Date.now();

    if (!available) {
      // Degraded response
      return {
        id: `verify_${now}`,
        did: request.did,
        isValid: false,
        isDeactivated: false,
        assuranceLevel: 'E0',
        matlScore: 0.5,
        credentialCount: 0,
        verifiedAt: now,
      };
    }

    try {
      // Try the identity bridge
      const result = await this.callIdentityZome<any>('identity_bridge', 'verify_identity', {
        did: request.did,
        min_assurance_level: request.minAssuranceLevel,
        required_credentials: request.requiredCredentials,
        source_happ: request.sourceHapp,
      });

      if (result) {
        return {
          id: result.id || `verify_${now}`,
          did: result.did || request.did,
          isValid: result.is_valid ?? false,
          isDeactivated: result.is_deactivated ?? false,
          assuranceLevel: (result.assurance_level as AssuranceLevel) || 'E0',
          matlScore: result.matl_score ?? 0.5,
          credentialCount: result.credential_count ?? 0,
          didCreated: result.did_created,
          verifiedAt: now,
        };
      }
    } catch (error) {
      if (this.fallbackMode === 'warn') {
        console.warn('[IdentityClient] Identity verification failed:', error);
      }
    }

    // Manual verification fallback
    const resolution = await this.resolveDID(request.did);
    const assurance = await this.getAssuranceLevel(request.did);
    const reputation = await this.getCrossHappReputation(request.did);

    return {
      id: `verify_${now}`,
      did: request.did,
      isValid: resolution.success,
      isDeactivated: false,
      assuranceLevel: assurance,
      matlScore: reputation?.aggregateScore ?? 0.5,
      credentialCount: 0,
      didCreated: resolution.didDocument?.created,
      verifiedAt: now,
    };
  }

  /**
   * Get cross-hApp reputation for a DID
   */
  async getCrossHappReputation(did: string): Promise<CrossHappReputation | null> {
    const available = await this.checkConnection();

    if (!available) {
      return null;
    }

    try {
      const result = await this.callIdentityZome<any>('identity_bridge', 'get_cross_happ_reputation', {
        did,
      });

      if (result) {
        return {
          did: result.did || did,
          scores: (result.scores || []).map((s: any) => ({
            happId: s.happ_id,
            happName: s.happ_name || s.happ_id,
            score: s.score ?? 0.5,
            interactions: s.interactions ?? 0,
            updatedAt: s.updated_at ?? Date.now(),
          })),
          aggregateScore: result.aggregate_score ?? result.aggregate ?? 0.5,
          lastUpdated: result.last_updated ?? Date.now(),
        };
      }
    } catch (error) {
      if (this.fallbackMode === 'warn') {
        console.warn('[IdentityClient] Failed to get cross-hApp reputation:', error);
      }
    }

    return null;
  }

  // =========================================================================
  // Helper Methods
  // =========================================================================

  /**
   * Convert Holochain record to DidDocument
   */
  private recordToDidDocument(record: any): DidDocument | null {
    try {
      const entry = record.entry?.Present?.entry || record.entry;

      if (!entry) return null;

      // Handle both direct entry and nested entry format
      const data = typeof entry === 'object' ? entry : JSON.parse(entry);

      return {
        id: data.id,
        controller: data.controller,
        verificationMethod: (data.verification_method || []).map((vm: any) => ({
          id: vm.id,
          type: vm.type_ || vm.type,
          controller: vm.controller,
          publicKeyMultibase: vm.public_key_multibase,
        })),
        authentication: data.authentication || [],
        service: (data.service || []).map((s: any) => ({
          id: s.id,
          type: s.type_ || s.type,
          serviceEndpoint: s.service_endpoint,
        })),
        created: data.created,
        updated: data.updated,
        version: data.version || 1,
      };
    } catch (error) {
      console.error('[IdentityClient] Failed to parse DID document:', error);
      return null;
    }
  }

  /**
   * Encode agent public key to string
   */
  private encodeAgentKey(key: Uint8Array): string {
    // Convert to base64
    return btoa(String.fromCharCode(...key));
  }

  /**
   * Get credentials for a DID (placeholder - would query credential_schema zome)
   */
  private async getCredentialsForDID(did: string): Promise<VerifiableCredential[]> {
    // This would query the credential store
    // For now, return empty array
    return [];
  }

  /**
   * Derive assurance level from a credential
   */
  private deriveAssuranceLevel(credential: VerifiableCredential): AssuranceLevel {
    const types = credential.type || [];

    if (types.includes('BiometricCredential') || types.includes('MultifactorCredential')) {
      return 'E4';
    }
    if (types.includes('GovernmentIdCredential') || types.includes('KYCCredential')) {
      return 'E3';
    }
    if (types.includes('PhoneCredential') || types.includes('RecoveryConfigured')) {
      return 'E2';
    }
    if (types.includes('EmailCredential')) {
      return 'E1';
    }

    return 'E0';
  }

  /**
   * Verify credential signature (simplified)
   */
  private async verifySignature(
    credential: VerifiableCredential,
    issuerDoc: DidDocument | null
  ): Promise<boolean> {
    if (!credential.proof || !issuerDoc) {
      return false;
    }

    // In production, this would:
    // 1. Get the verification method from issuer's DID document
    // 2. Verify the signature using the public key
    // For now, return true if proof exists and issuer is valid
    const verificationMethod = issuerDoc.verificationMethod.find(
      (vm) => vm.id === credential.proof?.verificationMethod
    );

    return !!verificationMethod;
  }

  /**
   * Validate credential schema (simplified)
   */
  private validateCredentialSchema(credential: VerifiableCredential): boolean {
    // Basic validation
    return !!(
      credential['@context'] &&
      credential.id &&
      credential.type &&
      credential.issuer &&
      credential.issuanceDate &&
      credential.credentialSubject
    );
  }

  /**
   * Clear all caches
   */
  clearCache(): void {
    this.didCache.clear();
    this.revocationCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { didCacheSize: number; revocationCacheSize: number } {
    return {
      didCacheSize: this.didCache.size,
      revocationCacheSize: this.revocationCache.size,
    };
  }
}

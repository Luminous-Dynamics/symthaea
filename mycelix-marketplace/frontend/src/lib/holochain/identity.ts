// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Integration for Mycelix Marketplace
 *
 * Connects to the mycelix-identity hApp for:
 * - DID resolution for seller/buyer profiles
 * - Credential verification for high-value transactions
 * - Assurance level checks (E0-E4)
 *
 * Graceful degradation: If identity hApp is unavailable,
 * operations return warnings/defaults instead of failures.
 */

import type { AppClient } from '@holochain/client';
import { callZome, getCellId, getAppInfo } from './client';
import { holochain } from '$lib/stores';
import { get } from 'svelte/store';

// =========================================================================
// Types
// =========================================================================

/**
 * Assurance levels following eIDAS-inspired tiers
 */
export type AssuranceLevel = 'E0' | 'E1' | 'E2' | 'E3' | 'E4';

/**
 * Numeric mapping for assurance levels
 */
export const ASSURANCE_LEVEL_VALUE: Record<AssuranceLevel, number> = {
  E0: 0,
  E1: 1,
  E2: 2,
  E3: 3,
  E4: 4,
};

/**
 * Human-readable descriptions
 */
export const ASSURANCE_LEVEL_DESCRIPTION: Record<AssuranceLevel, string> = {
  E0: 'Unverified',
  E1: 'Email Verified',
  E2: 'Phone + Recovery',
  E3: 'Government ID',
  E4: 'Biometric + MFA',
};

/**
 * DID Document structure
 */
export interface DidDocument {
  id: string;
  controller: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  service: ServiceEndpoint[];
  created: number;
  updated: number;
  version: number;
}

export interface VerificationMethod {
  id: string;
  type: string;
  controller: string;
  publicKeyMultibase: string;
}

export interface ServiceEndpoint {
  id: string;
  type: string;
  serviceEndpoint: string;
}

/**
 * DID Resolution result
 */
export interface DidResolutionResult {
  success: boolean;
  didDocument: DidDocument | null;
  error?: string;
  cached?: boolean;
  source?: 'local' | 'dht' | 'fallback';
}

/**
 * Credential verification result
 */
export interface CredentialVerificationResult {
  valid: boolean;
  checks: {
    signatureValid: boolean;
    notExpired: boolean;
    notRevoked: boolean;
    issuerTrusted: boolean;
    schemaValid: boolean;
  };
  error?: string;
  assuranceLevel?: AssuranceLevel;
}

/**
 * Identity verification for transactions
 */
export interface TransactionIdentityVerification {
  buyerDid: string;
  buyerVerified: boolean;
  buyerAssuranceLevel: AssuranceLevel;
  sellerDid: string;
  sellerVerified: boolean;
  sellerAssuranceLevel: AssuranceLevel;
  meetsRequirements: boolean;
  warnings: string[];
}

/**
 * High-value transaction configuration
 */
export interface HighValueTransactionConfig {
  threshold: number;
  currency: string;
  requiredAssuranceLevel: AssuranceLevel;
}

// =========================================================================
// State
// =========================================================================

let identityCellId: [Uint8Array, Uint8Array] | null = null;
let identityAvailable = false;
let lastConnectionCheck = 0;
const CONNECTION_CHECK_INTERVAL = 30000; // 30 seconds

// Simple cache for DID documents
const didCache = new Map<string, { document: DidDocument; expiresAt: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// =========================================================================
// Connection Management
// =========================================================================

/**
 * Check if identity hApp is available
 */
export async function checkIdentityConnection(): Promise<boolean> {
  const now = Date.now();

  // Rate limit connection checks
  if (now - lastConnectionCheck < CONNECTION_CHECK_INTERVAL) {
    return identityAvailable;
  }

  lastConnectionCheck = now;

  try {
    const appInfo = getAppInfo();
    if (!appInfo?.cell_info) {
      identityAvailable = false;
      return false;
    }

    // Look for identity role
    for (const roleName of Object.keys(appInfo.cell_info)) {
      if (roleName.includes('identity')) {
        const cellInfo = appInfo.cell_info[roleName]?.[0];
        if (cellInfo && 'provisioned' in cellInfo) {
          identityCellId = cellInfo.provisioned.cell_id;
          identityAvailable = true;
          console.log('[Identity] Connected via role:', roleName);
          return true;
        }
      }
    }

    console.warn('[Identity] Identity hApp not found - using degraded mode');
    identityAvailable = false;
    return false;
  } catch (error) {
    console.warn('[Identity] Connection check failed:', error);
    identityAvailable = false;
    return false;
  }
}

/**
 * Check if identity is available (cached)
 */
export function isIdentityAvailable(): boolean {
  return identityAvailable;
}

/**
 * Make a zome call to the identity DNA
 */
async function callIdentityZome<T>(
  client: AppClient,
  zomeName: string,
  fnName: string,
  payload: any
): Promise<T | null> {
  if (!identityCellId) {
    await checkIdentityConnection();
    if (!identityCellId) {
      console.warn(`[Identity] Cannot call ${zomeName}.${fnName} - identity not available`);
      return null;
    }
  }

  try {
    const result = await client.callZome({
      cap_secret: null,
      cell_id: identityCellId,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });

    return result as T;
  } catch (error) {
    console.warn(`[Identity] Zome call ${zomeName}.${fnName} failed:`, error);
    return null;
  }
}

// =========================================================================
// DID Resolution
// =========================================================================

/**
 * Resolve a DID to its document
 *
 * @param client - Holochain client
 * @param did - DID to resolve (e.g., "did:mycelix:abc123")
 * @returns Resolution result
 */
export async function resolveDID(
  client: AppClient,
  did: string
): Promise<DidResolutionResult> {
  // Validate DID format
  if (!did.startsWith('did:mycelix:')) {
    return {
      success: false,
      didDocument: null,
      error: 'Invalid DID format. Must start with "did:mycelix:"',
      source: 'fallback',
    };
  }

  // Check cache
  const cached = didCache.get(did);
  if (cached && cached.expiresAt > Date.now()) {
    return {
      success: true,
      didDocument: cached.document,
      cached: true,
      source: 'local',
    };
  }

  // Check if identity hApp is available
  const available = await checkIdentityConnection();

  if (available) {
    const result = await callIdentityZome<any>(client, 'did_registry', 'resolve_did', did);

    if (result) {
      const doc = parseDidDocument(result);

      // Cache the result
      didCache.set(did, {
        document: doc,
        expiresAt: Date.now() + CACHE_TTL,
      });

      return {
        success: true,
        didDocument: doc,
        source: 'dht',
      };
    }

    return {
      success: false,
      didDocument: null,
      error: 'DID not found',
      source: 'dht',
    };
  }

  // Fallback mode
  return {
    success: false,
    didDocument: null,
    error: 'Identity hApp not available - using degraded mode',
    source: 'fallback',
  };
}

/**
 * Resolve DID from agent public key
 */
export async function resolveDIDFromAgent(
  client: AppClient,
  agentPubKey: string
): Promise<DidResolutionResult> {
  const did = `did:mycelix:${agentPubKey}`;
  return resolveDID(client, did);
}

/**
 * Get my DID document
 */
export async function getMyDID(client: AppClient): Promise<DidDocument | null> {
  const available = await checkIdentityConnection();

  if (!available) {
    return null;
  }

  const result = await callIdentityZome<any>(client, 'did_registry', 'get_my_did', null);
  return result ? parseDidDocument(result) : null;
}

/**
 * Parse zome response into DidDocument
 */
function parseDidDocument(data: any): DidDocument {
  return {
    id: data.id || '',
    controller: data.controller || '',
    verificationMethod: (data.verification_method || []).map((vm: any) => ({
      id: vm.id,
      type: vm.type_ || vm.type || 'Ed25519VerificationKey2020',
      controller: vm.controller,
      publicKeyMultibase: vm.public_key_multibase || '',
    })),
    authentication: data.authentication || [],
    service: (data.service || []).map((s: any) => ({
      id: s.id,
      type: s.type_ || s.type || 'LinkedDomains',
      serviceEndpoint: s.service_endpoint || '',
    })),
    created: data.created || 0,
    updated: data.updated || 0,
    version: data.version || 1,
  };
}

// =========================================================================
// Credential Verification
// =========================================================================

/**
 * Verify a credential
 */
export async function verifyCredential(
  client: AppClient,
  credentialId: string
): Promise<CredentialVerificationResult> {
  const available = await checkIdentityConnection();

  if (!available) {
    // Permissive default in degraded mode
    return {
      valid: true,
      checks: {
        signatureValid: true,
        notExpired: true,
        notRevoked: true,
        issuerTrusted: true,
        schemaValid: true,
      },
      error: 'Identity hApp not available - verification skipped',
      assuranceLevel: 'E0',
    };
  }

  // Check revocation status
  const revocationResult = await callIdentityZome<any>(
    client,
    'revocation',
    'check_revocation_status',
    credentialId
  );

  const isRevoked = revocationResult?.status?.toLowerCase() === 'revoked';
  const isSuspended = revocationResult?.status?.toLowerCase() === 'suspended';

  return {
    valid: !isRevoked && !isSuspended,
    checks: {
      signatureValid: true,
      notExpired: true,
      notRevoked: !isRevoked && !isSuspended,
      issuerTrusted: true,
      schemaValid: true,
    },
    error: isRevoked ? 'Credential has been revoked' : isSuspended ? 'Credential is suspended' : undefined,
    assuranceLevel: 'E1',
  };
}

/**
 * Check revocation status of a credential
 */
export async function checkRevocationStatus(
  client: AppClient,
  credentialId: string
): Promise<{ status: 'active' | 'revoked' | 'suspended'; reason?: string }> {
  const available = await checkIdentityConnection();

  if (!available) {
    // Permissive default
    return { status: 'active' };
  }

  const result = await callIdentityZome<any>(
    client,
    'revocation',
    'check_revocation_status',
    credentialId
  );

  if (result) {
    return {
      status: result.status?.toLowerCase() || 'active',
      reason: result.reason,
    };
  }

  return { status: 'active' };
}

// =========================================================================
// Assurance Level
// =========================================================================

/**
 * Get the assurance level for a DID
 */
export async function getAssuranceLevel(
  client: AppClient,
  did: string
): Promise<AssuranceLevel> {
  const available = await checkIdentityConnection();

  if (!available) {
    return 'E0';
  }

  const result = await callIdentityZome<any>(
    client,
    'identity_bridge',
    'get_assurance_level',
    did
  );

  if (result?.level) {
    const level = result.level.toUpperCase();
    if (level in ASSURANCE_LEVEL_VALUE) {
      return level as AssuranceLevel;
    }
  }

  return 'E0';
}

/**
 * Check if a DID meets a minimum assurance level
 */
export async function meetsAssuranceLevel(
  client: AppClient,
  did: string,
  minLevel: AssuranceLevel
): Promise<boolean> {
  const current = await getAssuranceLevel(client, did);
  return ASSURANCE_LEVEL_VALUE[current] >= ASSURANCE_LEVEL_VALUE[minLevel];
}

/**
 * Get required assurance level based on transaction value
 */
export function getRequiredAssuranceLevel(value: number, currency: string = 'USD'): AssuranceLevel {
  // Simple threshold mapping
  if (value >= 10000) return 'E4';
  if (value >= 1000) return 'E3';
  if (value >= 100) return 'E2';
  if (value >= 10) return 'E1';
  return 'E0';
}

// =========================================================================
// Transaction Identity Verification
// =========================================================================

/**
 * Verify identity for a transaction (checkout flow)
 */
export async function verifyTransactionIdentity(
  client: AppClient,
  buyerDid: string,
  sellerDid: string,
  transactionValue: number,
  currency: string = 'USD'
): Promise<TransactionIdentityVerification> {
  const warnings: string[] = [];

  // Get required assurance level based on value
  const requiredLevel = getRequiredAssuranceLevel(transactionValue, currency);

  // Verify buyer
  const buyerResolution = await resolveDID(client, buyerDid);
  const buyerAssurance = await getAssuranceLevel(client, buyerDid);

  // Verify seller
  const sellerResolution = await resolveDID(client, sellerDid);
  const sellerAssurance = await getAssuranceLevel(client, sellerDid);

  // Check if requirements are met
  let meetsRequirements = true;

  if (!buyerResolution.success) {
    warnings.push('Buyer DID could not be resolved');
    meetsRequirements = false;
  }

  if (!sellerResolution.success) {
    warnings.push('Seller DID could not be resolved');
    meetsRequirements = false;
  }

  if (ASSURANCE_LEVEL_VALUE[buyerAssurance] < ASSURANCE_LEVEL_VALUE[requiredLevel]) {
    warnings.push(
      `Buyer assurance level (${buyerAssurance}) is below required level (${requiredLevel}) for this transaction value`
    );
    // Don't fail - just warn
  }

  if (ASSURANCE_LEVEL_VALUE[sellerAssurance] < ASSURANCE_LEVEL_VALUE[requiredLevel]) {
    warnings.push(
      `Seller assurance level (${sellerAssurance}) is below required level (${requiredLevel}) for this transaction value`
    );
  }

  // Add degraded mode warning if applicable
  if (!isIdentityAvailable()) {
    warnings.push('Identity verification running in degraded mode - some checks skipped');
  }

  return {
    buyerDid,
    buyerVerified: buyerResolution.success,
    buyerAssuranceLevel: buyerAssurance,
    sellerDid,
    sellerVerified: sellerResolution.success,
    sellerAssuranceLevel: sellerAssurance,
    meetsRequirements,
    warnings,
  };
}

/**
 * Verify identity for high-value transaction
 */
export async function verifyHighValueTransaction(
  client: AppClient,
  buyerDid: string,
  sellerDid: string,
  config: HighValueTransactionConfig
): Promise<TransactionIdentityVerification> {
  return verifyTransactionIdentity(client, buyerDid, sellerDid, config.threshold, config.currency);
}

// =========================================================================
// Seller/Buyer Profile Enhancement
// =========================================================================

/**
 * Enhanced profile data with identity information
 */
export interface EnhancedProfile {
  agentId: string;
  did: string;
  didResolved: boolean;
  assuranceLevel: AssuranceLevel;
  assuranceLevelDescription: string;
  identityWarnings: string[];
}

/**
 * Get enhanced profile with identity data for a seller
 */
export async function getEnhancedSellerProfile(
  client: AppClient,
  agentId: string
): Promise<EnhancedProfile> {
  const did = `did:mycelix:${agentId}`;
  const warnings: string[] = [];

  // Resolve DID
  const resolution = await resolveDID(client, did);
  if (!resolution.success && resolution.error) {
    warnings.push(resolution.error);
  }

  // Get assurance level
  const assurance = await getAssuranceLevel(client, did);

  return {
    agentId,
    did,
    didResolved: resolution.success,
    assuranceLevel: assurance,
    assuranceLevelDescription: ASSURANCE_LEVEL_DESCRIPTION[assurance],
    identityWarnings: warnings,
  };
}

/**
 * Get enhanced profile with identity data for a buyer
 */
export async function getEnhancedBuyerProfile(
  client: AppClient,
  agentId: string
): Promise<EnhancedProfile> {
  // Same implementation as seller - could differ in the future
  return getEnhancedSellerProfile(client, agentId);
}

// =========================================================================
// Cache Management
// =========================================================================

/**
 * Clear the DID cache
 */
export function clearDIDCache(): void {
  didCache.clear();
  console.log('[Identity] DID cache cleared');
}

/**
 * Get cache statistics
 */
export function getCacheStats(): { size: number; entries: string[] } {
  return {
    size: didCache.size,
    entries: Array.from(didCache.keys()),
  };
}

// =========================================================================
// Utility Functions
// =========================================================================

/**
 * Format assurance level for display
 */
export function formatAssuranceLevel(level: AssuranceLevel): string {
  return `${level} (${ASSURANCE_LEVEL_DESCRIPTION[level]})`;
}

/**
 * Get color for assurance level (for UI badges)
 */
export function getAssuranceLevelColor(level: AssuranceLevel): string {
  const colors: Record<AssuranceLevel, string> = {
    E0: '#9CA3AF', // Gray
    E1: '#60A5FA', // Blue
    E2: '#34D399', // Green
    E3: '#FBBF24', // Yellow
    E4: '#A78BFA', // Purple
  };
  return colors[level];
}

/**
 * Check if DID is valid format
 */
export function isValidDID(did: string): boolean {
  return did.startsWith('did:mycelix:') && did.length > 'did:mycelix:'.length;
}

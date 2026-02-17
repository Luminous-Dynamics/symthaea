/**
 * @mycelix/sdk Identity Integration
 *
 * hApp-specific adapter for Mycelix-Identity providing:
 * - Decentralized identifier (DID) management
 * - Verifiable credential issuance and verification
 * - Selective disclosure with zero-knowledge proofs
 * - Cross-hApp identity federation via Bridge
 * - Reputation-backed identity verification
 * - Bridge zome client for cross-hApp communication
 *
 * @packageDocumentation
 * @module integrations/identity
 */


import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  createReputation,
  recordPositive,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';
import { IdentityValidators } from '../../utils/validation.js';

// ============================================================================
// Bridge Zome Types (matching Rust identity_bridge zome)
// ============================================================================

/** hApp registration record */
export interface HappRegistration {
  happ_id: string;
  happ_name: string;
  capabilities: string[];
  matl_score: number;
  registered_at: number;
}

/** Identity query from another hApp */
export interface IdentityQuery {
  id: string;
  did: string;
  source_happ: string;
  requested_fields: string[];
  queried_at: number;
}

/** Identity verification result */
export interface IdentityVerificationResult {
  id: string;
  did: string;
  is_valid: boolean;
  is_deactivated: boolean;
  matl_score: number;
  credential_count: number;
  did_created?: number;
  verified_at: number;
}

/** Bridge event types */
export type BridgeEventType =
  | 'IdentityCreated'
  | 'IdentityUpdated'
  | 'CredentialIssued'
  | 'CredentialRevoked'
  | 'TrustAttested'
  | 'RecoveryInitiated';

/** Bridge event record */
export interface BridgeEvent {
  id: string;
  event_type: BridgeEventType;
  did?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Identity reputation from a specific hApp */
export interface IdentityReputation {
  id: string;
  did: string;
  source_happ: string;
  score: number;
  interactions: number;
  reported_at: number;
}

/** Aggregated reputation across all hApps */
export interface AggregatedReputation {
  did: string;
  sources: IdentityReputation[];
  aggregate_score: number;
  total_interactions: number;
}

/** Query identity input */
export interface QueryIdentityInput {
  did: string;
  source_happ: string;
  requested_fields?: string[];
}

/** Report reputation input */
export interface ReportReputationInput {
  did: string;
  source_happ: string;
  score: number;
  interactions: number;
}

/** Broadcast event input */
export interface BroadcastEventInput {
  event_type: BridgeEventType;
  did?: string;
  payload: string;
}

/** Register hApp input */
export interface RegisterHappInput {
  happ_name: string;
  capabilities: string[];
}

// ============================================================================
// Identity-Specific Types
// ============================================================================

/** DID document representation */
export interface DIDDocument {
  id: string;
  controller: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  assertionMethod: string[];
  keyAgreement?: string[];
  service?: ServiceEndpoint[];
  created: number;
  updated: number;
}

/** Verification method for DID */
export interface VerificationMethod {
  id: string;
  type: string;
  controller: string;
  publicKeyMultibase: string;
}

/** Service endpoint */
export interface ServiceEndpoint {
  id: string;
  type: string;
  serviceEndpoint: string;
}

/** Verifiable credential */
export interface VerifiableCredential {
  '@context': string[];
  id: string;
  type: string[];
  issuer: string;
  issuanceDate: string;
  expirationDate?: string;
  credentialSubject: Record<string, unknown>;
  proof?: CredentialProof;
}

/** Credential proof */
export interface CredentialProof {
  type: string;
  created: string;
  verificationMethod: string;
  proofPurpose: string;
  proofValue: string;
}

/** Identity verification level */
export type VerificationLevel = 'self_attested' | 'peer_verified' | 'credential_backed' | 'zkp_verified';

/** Identity profile */
export interface IdentityProfile {
  did: string;
  document: DIDDocument;
  credentials: VerifiableCredential[];
  reputation: ReputationScore;
  verificationLevel: VerificationLevel;
  trustedBy: string[];
  createdAt: number;
  lastActive: number;
}

/** Verification request */
export interface VerificationRequest {
  id: string;
  requesterId: string;
  subjectId: string;
  credentialType: string;
  attributes: string[];
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  createdAt: number;
  expiresAt: number;
}

// ============================================================================
// Identity Service
// ============================================================================

/**
 * Identity service for DID and credential management
 */
export class IdentityService {
  private profiles = new Map<string, IdentityProfile>();
  private credentials = new Map<string, VerifiableCredential>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('identity');
  }

  /**
   * Create a new DID and identity profile
   * @throws {MycelixError} If public key is invalid
   */
  createIdentity(publicKey: string): IdentityProfile {
    // Validate public key
    IdentityValidators.createIdentity(publicKey);

    const did = `did:mycelix:${publicKey.slice(0, 32)}`;

    const document: DIDDocument = {
      id: did,
      controller: did,
      verificationMethod: [
        {
          id: `${did}#key-1`,
          type: 'Ed25519VerificationKey2020',
          controller: did,
          publicKeyMultibase: `z${publicKey}`,
        },
      ],
      authentication: [`${did}#key-1`],
      assertionMethod: [`${did}#key-1`],
      created: Date.now(),
      updated: Date.now(),
    };

    const profile: IdentityProfile = {
      did,
      document,
      credentials: [],
      reputation: createReputation(did),
      verificationLevel: 'self_attested',
      trustedBy: [],
      createdAt: Date.now(),
      lastActive: Date.now(),
    };

    this.profiles.set(did, profile);
    return profile;
  }

  /**
   * Issue a verifiable credential
   * @throws {MycelixError} If issuer or subject DID is invalid
   * @throws {Error} If issuer or subject profile does not exist
   */
  issueCredential(
    issuerId: string,
    subjectId: string,
    type: string,
    claims: Record<string, unknown>,
    expirationDays?: number
  ): VerifiableCredential {
    // Validate DID formats
    IdentityValidators.credential(issuerId, subjectId);

    // Verify issuer exists
    if (!this.profiles.has(issuerId)) {
      throw new Error('Issuer profile does not exist');
    }

    // Verify subject exists
    if (!this.profiles.has(subjectId)) {
      throw new Error('Subject profile does not exist');
    }

    const id = `vc-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const now = new Date().toISOString();

    const credential: VerifiableCredential = {
      '@context': ['https://www.w3.org/2018/credentials/v1', 'https://mycelix.net/credentials/v1'],
      id,
      type: ['VerifiableCredential', type],
      issuer: issuerId,
      issuanceDate: now,
      expirationDate: expirationDays
        ? new Date(Date.now() + expirationDays * 24 * 60 * 60 * 1000).toISOString()
        : undefined,
      credentialSubject: {
        id: subjectId,
        ...claims,
      },
    };

    this.credentials.set(id, credential);

    // Update subject's profile
    const subjectProfile = this.profiles.get(subjectId);
    if (subjectProfile) {
      subjectProfile.credentials.push(credential);
      subjectProfile.reputation = recordPositive(subjectProfile.reputation);
      if (subjectProfile.verificationLevel === 'self_attested') {
        subjectProfile.verificationLevel = 'credential_backed';
      }
    }

    return credential;
  }

  /**
   * Verify a credential
   */
  verifyCredential(credentialId: string): { valid: boolean; reason?: string } {
    const credential = this.credentials.get(credentialId);

    if (!credential) {
      return { valid: false, reason: 'Credential not found' };
    }

    if (credential.expirationDate) {
      const expiry = new Date(credential.expirationDate).getTime();
      if (Date.now() > expiry) {
        return { valid: false, reason: 'Credential expired' };
      }
    }

    // Verify issuer is trusted
    const issuerProfile = this.profiles.get(credential.issuer);
    if (!issuerProfile || !isTrustworthy(issuerProfile.reputation)) {
      return { valid: false, reason: 'Issuer not trusted' };
    }

    return { valid: true };
  }

  /**
   * Add peer trust attestation
   */
  attestTrust(attesterId: string, subjectId: string): void {
    const subjectProfile = this.profiles.get(subjectId);
    const attesterProfile = this.profiles.get(attesterId);

    if (!subjectProfile || !attesterProfile) {
      throw new Error('Profile not found');
    }

    if (!subjectProfile.trustedBy.includes(attesterId)) {
      subjectProfile.trustedBy.push(attesterId);
      subjectProfile.reputation = recordPositive(subjectProfile.reputation);

      if (subjectProfile.trustedBy.length >= 3 && subjectProfile.verificationLevel === 'self_attested') {
        subjectProfile.verificationLevel = 'peer_verified';
      }
    }
  }

  /**
   * Get identity profile
   */
  getProfile(did: string): IdentityProfile | undefined {
    return this.profiles.get(did);
  }

  /**
   * Resolve DID to document
   */
  resolveDID(did: string): DIDDocument | undefined {
    return this.profiles.get(did)?.document;
  }

  /**
   * Get credentials for a subject
   */
  getCredentials(did: string): VerifiableCredential[] {
    return this.profiles.get(did)?.credentials || [];
  }

  /**
   * Cross-hApp identity verification
   */
  async verifyCrossHapp(did: string): Promise<{ verified: boolean; trustScore: number }> {
    const scores = this.bridge.getCrossHappReputation(did);
    const total = scores.reduce((sum, s) => sum + s.score, 0);
    const avgScore = scores.length > 0 ? total / scores.length : 0;
    return {
      verified: avgScore > 0.6,
      trustScore: avgScore,
    };
  }
}

// Singleton
let instance: IdentityService | null = null;

export function getIdentityService(): IdentityService {
  if (!instance) instance = new IdentityService();
  return instance;
}

export function resetIdentityService(): void {
  instance = null;
}

// ============================================================================
// Identity Bridge Client (Holochain Zome Calls)
// ============================================================================

const IDENTITY_ROLE = 'identity';
const BRIDGE_ZOME = 'identity_bridge';

/**
 * Identity Bridge Client - Direct Holochain zome calls for cross-hApp identity
 *
 * @example
 * ```typescript
 * import { IdentityBridgeClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-identity' });
 * await client.connect();
 *
 * const bridgeClient = new IdentityBridgeClient(client);
 *
 * // Register your hApp with the identity bridge
 * await bridgeClient.registerHapp({
 *   happ_name: 'my-marketplace',
 *   capabilities: ['credit_query', 'reputation_report'],
 * });
 *
 * // Query identity from another hApp
 * const verification = await bridgeClient.queryIdentity({
 *   did: 'did:mycelix:abc123',
 *   source_happ: 'my-marketplace',
 *   requested_fields: ['matl_score', 'credential_count'],
 * });
 * ```
 */
export class IdentityBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Register a hApp with the identity bridge
   */
  async registerHapp(input: RegisterHappInput): Promise<HappRegistration> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'register_happ',
      payload: input,
    });
  }

  /**
   * Get all registered hApps
   */
  async getRegisteredHapps(): Promise<HappRegistration[]> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_registered_happs',
      payload: null,
    });
  }

  /**
   * Query identity verification for a DID
   */
  async queryIdentity(input: QueryIdentityInput): Promise<IdentityVerificationResult> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_identity',
      payload: input,
    });
  }

  /**
   * Report reputation for a DID from your hApp
   */
  async reportReputation(input: ReportReputationInput): Promise<IdentityReputation> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'report_reputation',
      payload: input,
    });
  }

  /**
   * Get aggregated reputation for a DID across all hApps
   */
  async getReputation(did: string): Promise<AggregatedReputation> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_reputation',
      payload: did,
    });
  }

  /**
   * Broadcast an identity event to other hApps
   */
  async broadcastEvent(input: BroadcastEventInput): Promise<BridgeEvent> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_event',
      payload: input,
    });
  }

  /**
   * Get recent identity events
   */
  async getRecentEvents(limit?: number): Promise<BridgeEvent[]> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }

  /**
   * Get events for a specific DID
   */
  async getEventsByDid(did: string): Promise<BridgeEvent[]> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_did',
      payload: did,
    });
  }

  /**
   * Get events by type
   */
  async getEventsByType(eventType: BridgeEventType): Promise<BridgeEvent[]> {
    return this.client.callZome({
      role_name: IDENTITY_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_type',
      payload: eventType,
    });
  }
}

// Bridge client singleton
let bridgeInstance: IdentityBridgeClient | null = null;

/**
 * Get identity bridge client (requires connected MycelixClient)
 */
export function getIdentityBridgeClient(client: MycelixClient): IdentityBridgeClient {
  if (!bridgeInstance) {
    bridgeInstance = new IdentityBridgeClient(client);
  }
  return bridgeInstance;
}

/**
 * Reset identity bridge client
 */
export function resetIdentityBridgeClient(): void {
  bridgeInstance = null;
}

// ============================================================================
// Health-Identity Cross-Domain Linking Types
// ============================================================================

/** Input for linking a patient to a DID */
export interface LinkPatientIdentityInput {
  patient_hash: string;
  did: string;
  identity_provider: string;
  verification_method: string;
  confidence_score: number;
}

/** Input for getting a patient by DID */
export interface GetPatientByDIDInput {
  did: string;
  is_emergency: boolean;
  emergency_reason?: string;
}

/** Input for verifying a DID-Patient link */
export interface VerifyDIDPatientLinkInput {
  did: string;
  patient_hash: string;
}

/** Patient DID information */
export interface PatientDIDInfo {
  patient_hash: string;
  did: string;
  identity_provider: string;
  verified_at: number;
  confidence_score: number;
}

/** Patient identity link record */
export interface PatientIdentityLinkRecord {
  patient_hash: string;
  did: string;
  identity_provider: string;
  verified_at: number;
  verification_method: string;
  confidence_score: number;
}

// ============================================================================
// Health-Identity Bridge Client
// ============================================================================

const HEALTH_ROLE = 'health';
const PATIENT_ZOME = 'patient';

/**
 * Health-Identity Bridge Client - Cross-domain DID ↔ Patient linking
 *
 * Enables federated health data sharing across Mycelix hApps by providing
 * bidirectional identity resolution between DIDs and patient records.
 *
 * @example
 * ```typescript
 * import { getHealthIdentityBridgeClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-health' });
 * await client.connect();
 *
 * const bridge = getHealthIdentityBridgeClient(client);
 *
 * // Link a patient to their DID
 * const link = await bridge.linkPatientToIdentity({
 *   patient_hash: 'uhCkkABC123',
 *   did: 'did:mycelix:xyz789',
 *   identity_provider: 'Mycelix',
 *   verification_method: 'credential',
 *   confidence_score: 0.95,
 * });
 *
 * // Look up patient by DID
 * const patient = await bridge.getPatientByDID({
 *   did: 'did:mycelix:xyz789',
 *   is_emergency: false,
 * });
 * ```
 */
export class HealthIdentityBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Link a patient record to a DID
   */
  async linkPatientToIdentity(input: LinkPatientIdentityInput): Promise<PatientIdentityLinkRecord> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: PATIENT_ZOME,
      fn_name: 'link_patient_to_identity',
      payload: input,
    });
  }

  /**
   * Get patient record by DID
   */
  async getPatientByDID(input: GetPatientByDIDInput): Promise<unknown | null> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: PATIENT_ZOME,
      fn_name: 'get_patient_by_did',
      payload: input,
    });
  }

  /**
   * Get DID information for a patient
   */
  async getDIDForPatient(patientHash: string): Promise<PatientDIDInfo | null> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: PATIENT_ZOME,
      fn_name: 'get_did_for_patient',
      payload: patientHash,
    });
  }

  /**
   * Verify that a DID-Patient link exists and is valid
   */
  async verifyDIDPatientLink(input: VerifyDIDPatientLinkInput): Promise<boolean> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: PATIENT_ZOME,
      fn_name: 'verify_did_patient_link',
      payload: input,
    });
  }
}

// Health-Identity bridge client singleton
let healthIdentityBridgeInstance: HealthIdentityBridgeClient | null = null;

/**
 * Get health-identity bridge client (requires connected MycelixClient)
 */
export function getHealthIdentityBridgeClient(client: MycelixClient): HealthIdentityBridgeClient {
  if (!healthIdentityBridgeInstance) {
    healthIdentityBridgeInstance = new HealthIdentityBridgeClient(client);
  }
  return healthIdentityBridgeInstance;
}

/**
 * Reset health-identity bridge client
 */
export function resetHealthIdentityBridgeClient(): void {
  healthIdentityBridgeInstance = null;
}

// ============================================================================
// NEW SDK PATTERN EXPORTS (Following Health SDK architecture)
// ============================================================================

// Re-export types
export * from './types.js';

// Re-export zome clients
export * from './zomes/index.js';

// Re-export unified client
export { MycelixIdentityClient, type MycelixIdentityConfig } from './client.js';
export { default as IdentityClient } from './client.js';

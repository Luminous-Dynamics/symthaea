/**
 * Identity hApp Clients
 *
 * Comprehensive client library for the Mycelix-Identity hApp.
 * Provides access to all 8 identity zomes through the Master SDK.
 *
 * ## Zome Clients
 *
 * | Client | Zome | Description |
 * |--------|------|-------------|
 * | `DidClient` | did_registry | DID document management |
 * | `CredentialsClient` | verifiable_credential | W3C Verifiable Credentials |
 * | `RecoveryClient` | recovery | Social recovery with trustees |
 * | `BridgeClient` | identity_bridge | Cross-hApp identity |
 * | `SchemaClient` | credential_schema | Credential schema management |
 * | `RevocationClient` | revocation | Credential revocation |
 * | `TrustClient` | trust_credential | K-Vector trust credentials |
 * | `EducationClient` | education | Academic credentials |
 *
 * @example
 * ```typescript
 * import { AppWebsocket } from '@holochain/client';
 * import {
 *   DidClient,
 *   CredentialsClient,
 *   RecoveryClient,
 *   BridgeClient,
 *   SchemaClient,
 *   RevocationClient,
 *   TrustClient,
 *   EducationClient,
 * } from '@mycelix/sdk/clients/identity';
 *
 * // Connect to Holochain
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 *
 * // Initialize clients
 * const config = { roleName: 'mycelix_identity' };
 * const did = new DidClient(client, config);
 * const credentials = new CredentialsClient(client, config);
 * const recovery = new RecoveryClient(client, config);
 * const bridge = new BridgeClient(client, config);
 * const schema = new SchemaClient(client, config);
 * const revocation = new RevocationClient(client, config);
 * const trust = new TrustClient(client, config);
 * const education = new EducationClient(client, config);
 *
 * // Create a DID
 * const myDid = await did.createDid();
 * console.log(`Created: ${myDid.document.id}`);
 *
 * // Issue a credential
 * const credential = await credentials.issueCredential({
 *   subject_did: 'did:mycelix:bob...',
 *   schema_id: 'mycelix:schema:employment',
 *   claims: { role: 'Engineer', company: 'Acme Corp' },
 * });
 *
 * // Set up social recovery
 * await recovery.setupRecovery({
 *   did: myDid.document.id,
 *   trustees: ['did:mycelix:alice...', 'did:mycelix:carol...'],
 *   threshold: 2,
 * });
 * ```
 *
 * @module @mycelix/sdk/clients/identity
 */

// =============================================================================
// ZOME CLIENTS
// =============================================================================

export { DidClient } from './did.js';
export { CredentialsClient } from './credentials.js';
export { RecoveryClient } from './recovery.js';
export { BridgeClient } from './bridge.js';
export { SchemaClient } from './schema.js';
export { RevocationClient } from './revocation.js';
export { TrustClient } from './trust.js';
export { EducationClient } from './education.js';

// =============================================================================
// RECORD WRAPPERS (from individual clients)
// =============================================================================

export type { RecoveryConfigRecord, RecoveryRequestRecord, RecoveryVoteRecord } from './recovery.js';
export type { HappRegistrationRecord, BridgeEventRecord } from './bridge.js';
export type { SchemaRecord, EndorsementRecord } from './schema.js';
export type { RevocationRecord } from './revocation.js';
export type {
  TrustCredentialRecord,
  TrustPresentationRecord,
  AttestationRequestRecord,
} from './trust.js';
export type {
  AcademicCredentialRecord,
  VerifyDnsDidResult,
  DnsVerificationRecord,
  ImportCredentialResult,
  ImportError,
  EpistemicPosition,
  PublishCredentialAsClaimOutput,
} from './education.js';

// =============================================================================
// TYPES
// =============================================================================

export * from './types.js';

// =============================================================================
// UNIFIED CLIENT FACTORY
// =============================================================================

import { BridgeClient } from './bridge.js';
import { CredentialsClient } from './credentials.js';
import { DidClient } from './did.js';
import { EducationClient } from './education.js';
import { RecoveryClient } from './recovery.js';
import { RevocationClient } from './revocation.js';
import { SchemaClient } from './schema.js';
import { TrustClient } from './trust.js';

import type { IdentityClientConfig } from './types.js';
import type { AppClient } from '@holochain/client';

/**
 * Unified Identity Client
 *
 * Provides access to all Identity hApp zomes through a single interface.
 *
 * @example
 * ```typescript
 * import { createIdentityClient } from '@mycelix/sdk/clients/identity';
 * import { AppWebsocket } from '@holochain/client';
 *
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 * const identity = createIdentityClient(client);
 *
 * // Access individual zome clients
 * const myDid = await identity.did.createDid();
 * const credential = await identity.credentials.issueCredential({...});
 * await identity.recovery.setupRecovery({...});
 * ```
 */
export interface UnifiedIdentityClient {
  /** DID Registry - DID document management */
  did: DidClient;
  /** Verifiable Credentials - Credential issuance and verification */
  credentials: CredentialsClient;
  /** Recovery - Social recovery with trustees */
  recovery: RecoveryClient;
  /** Bridge - Cross-hApp identity operations */
  bridge: BridgeClient;
  /** Schema - Credential schema management */
  schema: SchemaClient;
  /** Revocation - Credential revocation management */
  revocation: RevocationClient;
  /** Trust - K-Vector trust credentials */
  trust: TrustClient;
  /** Education - Academic credentials */
  education: EducationClient;
}

/**
 * Create a unified Identity client with access to all zomes
 *
 * @param client - Holochain app client
 * @param config - Optional client configuration
 * @returns Unified client with all zome clients
 *
 * @example
 * ```typescript
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 * const identity = createIdentityClient(client, { roleName: 'mycelix_identity' });
 *
 * // Use individual clients
 * const myDid = await identity.did.createDid();
 * const verified = await identity.bridge.verifyDid(myDid.document.id);
 * ```
 */
export function createIdentityClient(
  client: AppClient,
  config: Partial<IdentityClientConfig> = {}
): UnifiedIdentityClient {
  return {
    did: new DidClient(client, config),
    credentials: new CredentialsClient(client, config),
    recovery: new RecoveryClient(client, config),
    bridge: new BridgeClient(client, config),
    schema: new SchemaClient(client, config),
    revocation: new RevocationClient(client, config),
    trust: new TrustClient(client, config),
    education: new EducationClient(client, config),
  };
}

/**
 * Identity Client class (alternative OOP interface)
 *
 * @example
 * ```typescript
 * const client = await AppWebsocket.connect({ url: 'ws://localhost:8888' });
 * const identity = new IdentityClient(client);
 *
 * const myDid = await identity.did.createDid();
 * ```
 */
export class IdentityClient implements UnifiedIdentityClient {
  public readonly did: DidClient;
  public readonly credentials: CredentialsClient;
  public readonly recovery: RecoveryClient;
  public readonly bridge: BridgeClient;
  public readonly schema: SchemaClient;
  public readonly revocation: RevocationClient;
  public readonly trust: TrustClient;
  public readonly education: EducationClient;

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    this.did = new DidClient(client, config);
    this.credentials = new CredentialsClient(client, config);
    this.recovery = new RecoveryClient(client, config);
    this.bridge = new BridgeClient(client, config);
    this.schema = new SchemaClient(client, config);
    this.revocation = new RevocationClient(client, config);
    this.trust = new TrustClient(client, config);
    this.education = new EducationClient(client, config);
  }
}

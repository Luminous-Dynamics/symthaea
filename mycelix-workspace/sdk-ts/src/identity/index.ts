/**
 * Mycelix Identity Module
 *
 * TypeScript client for the mycelix-identity hApp.
 * Provides DID management, credential schemas, revocation, and social recovery.
 *
 * @module @mycelix/sdk/identity
 */

// ============================================================================
// Types
// ============================================================================

/** DID Document following W3C DID Core spec with Mycelix extensions */
export interface DidDocument {
  /** The DID identifier (did:mycelix:<agent_pub_key>) */
  id: string;
  /** Controller (agent pub key) */
  controller: string;
  /** Verification methods (keys) */
  verificationMethod: VerificationMethod[];
  /** Authentication methods (key references) */
  authentication: string[];
  /** Service endpoints */
  service: ServiceEndpoint[];
  /** Creation timestamp (microseconds) */
  created: number;
  /** Last update timestamp (microseconds) */
  updated: number;
  /** Document version */
  version: number;
}

/** Verification method (public key) */
export interface VerificationMethod {
  /** Method ID (e.g., "did:mycelix:xxx#keys-1") */
  id: string;
  /** Key type (e.g., "Ed25519VerificationKey2020") */
  type: string;
  /** Controller DID */
  controller: string;
  /** Multibase-encoded public key */
  publicKeyMultibase: string;
  /** Algorithm identifier (multicodec u16). Absent defaults to Ed25519. */
  algorithm?: number;
}

/** Service endpoint */
export interface ServiceEndpoint {
  /** Service ID */
  id: string;
  /** Service type (e.g., "MycelixBridge", "LinkedDomains") */
  type: string;
  /** Service URL or endpoint */
  serviceEndpoint: string;
}

/** DID Deactivation record */
export interface DidDeactivation {
  /** The deactivated DID */
  did: string;
  /** Reason for deactivation */
  reason: string;
  /** Deactivation timestamp */
  deactivatedAt: number;
}

/** Input for updating a DID document */
export interface UpdateDidInput {
  /** New verification methods (optional) */
  verificationMethod?: VerificationMethod[];
  /** New authentication methods (optional) */
  authentication?: string[];
  /** New service endpoints (optional) */
  service?: ServiceEndpoint[];
}

// ============================================================================
// Credential Schema Types
// ============================================================================

/** Credential schema definition */
export interface CredentialSchema {
  /** Schema ID (e.g., "mycelix:schema:education:degree:v1") */
  id: string;
  /** Human-readable name */
  name: string;
  /** Description */
  description: string;
  /** Semver version */
  version: string;
  /** Author's DID */
  author: string;
  /** JSON Schema for credential subject */
  schema: string;
  /** Required fields */
  requiredFields: string[];
  /** Optional fields */
  optionalFields: string[];
  /** W3C credential types */
  credentialType: string[];
  /** Default expiration in seconds (0 = never) */
  defaultExpiration: number;
  /** Whether credentials can be revoked */
  revocable: boolean;
  /** Whether schema is active */
  active: boolean;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/** Schema endorsement */
export interface SchemaEndorsement {
  /** Schema being endorsed */
  schemaId: string;
  /** Endorser's DID */
  endorser: string;
  /** Trust level (0.0 - 1.0) */
  trustLevel: number;
  /** Optional comment */
  comment?: string;
  /** Endorsement timestamp */
  endorsedAt: number;
}

/** Schema category */
export type SchemaCategory =
  | 'Education'
  | 'Employment'
  | 'Identity'
  | 'Skills'
  | 'Governance'
  | 'Financial'
  | 'Energy'
  | { Custom: string };

// ============================================================================
// Revocation Types
// ============================================================================

/** Revocation status */
export type RevocationStatus = 'Active' | 'Suspended' | 'Revoked';

/** Revocation entry */
export interface RevocationEntry {
  /** Credential ID */
  credentialId: string;
  /** Issuer's DID */
  issuer: string;
  /** Current status */
  status: RevocationStatus;
  /** Reason for revocation/suspension */
  reason: string;
  /** When revocation takes effect */
  effectiveFrom: number;
  /** When revocation was recorded */
  recordedAt: number;
  /** When suspension ends (for Suspended status) */
  suspensionEnd?: number;
}

/** Revocation check result */
export interface RevocationCheckResult {
  /** Credential ID */
  credentialId: string;
  /** Current status */
  status: RevocationStatus;
  /** Reason (if revoked/suspended) */
  reason?: string;
  /** Check timestamp */
  checkedAt: number;
}

// ============================================================================
// Recovery Types
// ============================================================================

/** Recovery configuration */
export interface RecoveryConfig {
  /** DID being protected */
  did: string;
  /** Owner's agent pub key */
  owner: string;
  /** Trustee DIDs */
  trustees: string[];
  /** Minimum trustees required */
  threshold: number;
  /** Time lock in seconds */
  timeLock: number;
  /** Whether recovery is active */
  active: boolean;
  /** Creation timestamp */
  created: number;
  /** Last update timestamp */
  updated: number;
}

/** Recovery request */
export interface RecoveryRequest {
  /** Request ID */
  id: string;
  /** DID being recovered */
  did: string;
  /** New agent pub key */
  newAgent: string;
  /** Initiating trustee's DID */
  initiatedBy: string;
  /** Reason for recovery */
  reason: string;
  /** Current status */
  status: RecoveryStatus;
  /** Creation timestamp */
  created: number;
  /** Time lock expiration */
  timeLockExpires?: number;
}

/** Recovery status */
export type RecoveryStatus =
  | 'Pending'
  | 'Approved'
  | 'ReadyToExecute'
  | 'Completed'
  | 'Rejected'
  | 'Cancelled';

/** Vote decision */
export type VoteDecision = 'Approve' | 'Reject' | 'Abstain';

/** Recovery vote */
export interface RecoveryVote {
  /** Recovery request ID */
  requestId: string;
  /** Voting trustee's DID */
  trustee: string;
  /** Vote decision */
  vote: VoteDecision;
  /** Optional comment */
  comment?: string;
  /** Vote timestamp */
  votedAt: number;
}

// ============================================================================
// Bridge Types
// ============================================================================

/** Identity verification result */
export interface IdentityVerificationResult {
  /** Verification hash */
  verificationHash: string;
  /** DID that was verified */
  did: string;
  /** Whether DID is valid */
  isValid: boolean;
  /** MATL reputation score */
  matlScore: number;
  /** Number of credentials */
  credentialCount: number;
}

/** Aggregated reputation */
export interface AggregatedReputation {
  /** DID */
  did: string;
  /** Aggregate score (0.0 - 1.0) */
  aggregateScore: number;
  /** Reputation sources */
  sources: ReputationSource[];
  /** Total interactions across all hApps */
  totalInteractions: number;
}

/** Reputation source */
export interface ReputationSource {
  /** Source hApp */
  sourceHapp: string;
  /** Score from this hApp */
  score: number;
  /** Number of interactions */
  interactions: number;
}

// ============================================================================
// Client Interface
// ============================================================================

/** Zome call interface (compatible with @holochain/client) */
export interface ZomeCallable {
  callZome<T>(params: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

/** Record wrapper from Holochain */
export interface HolochainRecord<T = unknown> {
  signed_action: unknown;
  entry: { Present: { entry: T } } | { NotApplicable: null };
}

// ============================================================================
// Identity Client
// ============================================================================

/**
 * Client for DID management operations.
 *
 * @example
 * ```typescript
 * const identity = new IdentityClient(conductor);
 *
 * // Create a new DID
 * const did = await identity.createDid();
 * console.log(did.id); // did:mycelix:uhCAk...
 *
 * // Add a service endpoint
 * await identity.addServiceEndpoint({
 *   id: 'did:mycelix:xxx#mycelix-bridge',
 *   type: 'MycelixBridge',
 *   serviceEndpoint: 'https://bridge.mycelix.net'
 * });
 * ```
 */
export class IdentityClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a new DID for the calling agent */
  async createDid(): Promise<DidDocument> {
    const record = await this.client.callZome<HolochainRecord<DidDocument>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'create_did',
      payload: null,
    });
    return this.extractEntry(record);
  }

  /** Get DID document for an agent */
  async getDidDocument(agentPubKey: string): Promise<DidDocument | null> {
    const record = await this.client.callZome<HolochainRecord<DidDocument> | null>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'get_did_document',
      payload: agentPubKey,
    });
    return record ? this.extractEntry(record) : null;
  }

  /** Resolve a DID to its document */
  async resolveDid(did: string): Promise<DidDocument | null> {
    const record = await this.client.callZome<HolochainRecord<DidDocument> | null>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'resolve_did',
      payload: did,
    });
    return record ? this.extractEntry(record) : null;
  }

  /** Update DID document */
  async updateDidDocument(input: UpdateDidInput): Promise<DidDocument> {
    const record = await this.client.callZome<HolochainRecord<DidDocument>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'update_did_document',
      payload: {
        verificationMethod: input.verificationMethod,
        authentication: input.authentication,
        service: input.service,
      },
    });
    return this.extractEntry(record);
  }

  /** Add a service endpoint to DID */
  async addServiceEndpoint(service: ServiceEndpoint): Promise<DidDocument> {
    const record = await this.client.callZome<HolochainRecord<DidDocument>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'add_service_endpoint',
      payload: service,
    });
    return this.extractEntry(record);
  }

  /** Remove a service endpoint from DID */
  async removeServiceEndpoint(serviceId: string): Promise<DidDocument> {
    const record = await this.client.callZome<HolochainRecord<DidDocument>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'remove_service_endpoint',
      payload: serviceId,
    });
    return this.extractEntry(record);
  }

  /** Add a verification method to DID */
  async addVerificationMethod(method: VerificationMethod): Promise<DidDocument> {
    const record = await this.client.callZome<HolochainRecord<DidDocument>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'add_verification_method',
      payload: method,
    });
    return this.extractEntry(record);
  }

  /** Get calling agent's DID */
  async getMyDid(): Promise<DidDocument | null> {
    const record = await this.client.callZome<HolochainRecord<DidDocument> | null>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'get_my_did',
      payload: null,
    });
    return record ? this.extractEntry(record) : null;
  }

  /** Check if DID is active */
  async isDidActive(did: string): Promise<boolean> {
    return this.client.callZome<boolean>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'is_did_active',
      payload: did,
    });
  }

  /** Deactivate a DID */
  async deactivateDid(reason: string): Promise<DidDeactivation> {
    const record = await this.client.callZome<HolochainRecord<DidDeactivation>>({
      role_name: 'identity',
      zome_name: 'did_registry',
      fn_name: 'deactivate_did',
      payload: reason,
    });
    return this.extractEntry(record);
  }

  private extractEntry<T>(record: HolochainRecord<T>): T {
    if ('Present' in record.entry) {
      return record.entry.Present.entry;
    }
    throw new Error('Entry not found in record');
  }
}

// ============================================================================
// Credential Schema Client
// ============================================================================

/**
 * Client for credential schema management.
 *
 * @example
 * ```typescript
 * const schemas = new CredentialSchemaClient(conductor);
 *
 * // Create a degree credential schema
 * const schema = await schemas.createSchema({
 *   id: 'mycelix:schema:education:degree:v1',
 *   name: 'University Degree',
 *   description: 'Academic degree credential',
 *   version: '1.0.0',
 *   author: 'did:mycelix:xxx',
 *   schema: JSON.stringify({
 *     type: 'object',
 *     properties: {
 *       degree: { type: 'string' },
 *       institution: { type: 'string' },
 *       graduationDate: { type: 'string', format: 'date' }
 *     }
 *   }),
 *   requiredFields: ['degree', 'institution'],
 *   optionalFields: ['gpa', 'honors'],
 *   credentialType: ['VerifiableCredential', 'EducationCredential'],
 *   defaultExpiration: 0,
 *   revocable: true,
 *   active: true,
 *   created: Date.now() * 1000,
 *   updated: Date.now() * 1000
 * });
 * ```
 */
export class CredentialSchemaClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create a new credential schema */
  async createSchema(schema: CredentialSchema): Promise<CredentialSchema> {
    const record = await this.client.callZome<HolochainRecord<CredentialSchema>>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'create_schema',
      payload: schema,
    });
    return this.extractEntry(record);
  }

  /** Get schema by ID */
  async getSchema(schemaId: string): Promise<CredentialSchema | null> {
    const record = await this.client.callZome<HolochainRecord<CredentialSchema> | null>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'get_schema',
      payload: schemaId,
    });
    return record ? this.extractEntry(record) : null;
  }

  /** Get schemas by author */
  async getSchemasByAuthor(authorDid: string): Promise<CredentialSchema[]> {
    const records = await this.client.callZome<HolochainRecord<CredentialSchema>[]>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'get_schemas_by_author',
      payload: authorDid,
    });
    return records.map((r) => this.extractEntry(r));
  }

  /** List all active schemas */
  async listActiveSchemas(): Promise<CredentialSchema[]> {
    const records = await this.client.callZome<HolochainRecord<CredentialSchema>[]>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'list_active_schemas',
      payload: null,
    });
    return records.map((r) => this.extractEntry(r));
  }

  /** Endorse a schema */
  async endorseSchema(
    schemaId: string,
    endorserDid: string,
    trustLevel: number,
    comment?: string
  ): Promise<SchemaEndorsement> {
    const record = await this.client.callZome<HolochainRecord<SchemaEndorsement>>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'endorse_schema',
      payload: {
        schema_id: schemaId,
        endorser_did: endorserDid,
        trust_level: trustLevel,
        comment,
      },
    });
    return this.extractEntry(record);
  }

  /** Get endorsements for a schema */
  async getSchemaEndorsements(schemaId: string): Promise<SchemaEndorsement[]> {
    const records = await this.client.callZome<HolochainRecord<SchemaEndorsement>[]>({
      role_name: 'identity',
      zome_name: 'credential_schema',
      fn_name: 'get_schema_endorsements',
      payload: schemaId,
    });
    return records.map((r) => this.extractEntry(r));
  }

  private extractEntry<T>(record: HolochainRecord<T>): T {
    if ('Present' in record.entry) {
      return record.entry.Present.entry;
    }
    throw new Error('Entry not found in record');
  }
}

// ============================================================================
// Revocation Client
// ============================================================================

/**
 * Client for credential revocation operations.
 *
 * @example
 * ```typescript
 * const revocation = new RevocationClient(conductor);
 *
 * // Check if a credential is revoked
 * const status = await revocation.checkStatus('credential-123');
 * if (status.status === 'Revoked') {
 *   console.log('Credential revoked:', status.reason);
 * }
 *
 * // Revoke a credential
 * await revocation.revoke('credential-123', 'did:mycelix:xxx', 'Fraudulent use');
 * ```
 */
export class RevocationClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Revoke a credential permanently */
  async revoke(
    credentialId: string,
    issuerDid: string,
    reason: string,
    effectiveFrom?: number
  ): Promise<RevocationEntry> {
    const record = await this.client.callZome<HolochainRecord<RevocationEntry>>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'revoke_credential',
      payload: {
        credential_id: credentialId,
        issuer_did: issuerDid,
        reason,
        effective_from: effectiveFrom,
      },
    });
    return this.extractEntry(record);
  }

  /** Suspend a credential temporarily */
  async suspend(
    credentialId: string,
    issuerDid: string,
    reason: string,
    suspensionEnd: number
  ): Promise<RevocationEntry> {
    const record = await this.client.callZome<HolochainRecord<RevocationEntry>>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'suspend_credential',
      payload: {
        credential_id: credentialId,
        issuer_did: issuerDid,
        reason,
        suspension_end: suspensionEnd,
      },
    });
    return this.extractEntry(record);
  }

  /** Reinstate a suspended credential */
  async reinstate(
    credentialId: string,
    issuerDid: string,
    reason: string
  ): Promise<RevocationEntry> {
    const record = await this.client.callZome<HolochainRecord<RevocationEntry>>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'reinstate_credential',
      payload: {
        credential_id: credentialId,
        issuer_did: issuerDid,
        reason,
      },
    });
    return this.extractEntry(record);
  }

  /** Check revocation status */
  async checkStatus(credentialId: string): Promise<RevocationCheckResult> {
    return this.client.callZome<RevocationCheckResult>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'check_revocation_status',
      payload: credentialId,
    });
  }

  /** Batch check multiple credentials */
  async batchCheckStatus(credentialIds: string[]): Promise<RevocationCheckResult[]> {
    return this.client.callZome<RevocationCheckResult[]>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'batch_check_revocation',
      payload: credentialIds,
    });
  }

  /** Get all revocations by issuer */
  async getByIssuer(issuerDid: string): Promise<RevocationEntry[]> {
    const records = await this.client.callZome<HolochainRecord<RevocationEntry>[]>({
      role_name: 'identity',
      zome_name: 'revocation',
      fn_name: 'get_revocations_by_issuer',
      payload: issuerDid,
    });
    return records.map((r) => this.extractEntry(r));
  }

  private extractEntry<T>(record: HolochainRecord<T>): T {
    if ('Present' in record.entry) {
      return record.entry.Present.entry;
    }
    throw new Error('Entry not found in record');
  }
}

// ============================================================================
// Recovery Client
// ============================================================================

/**
 * Client for social recovery operations.
 *
 * @example
 * ```typescript
 * const recovery = new RecoveryClient(conductor);
 *
 * // Set up recovery with 3 trustees, requiring 2 approvals
 * await recovery.setupRecovery({
 *   did: 'did:mycelix:xxx',
 *   trustees: ['did:mycelix:trustee1', 'did:mycelix:trustee2', 'did:mycelix:trustee3'],
 *   threshold: 2,
 *   timeLock: 7 * 24 * 60 * 60 // 7 days in seconds
 * });
 *
 * // As a trustee, initiate recovery
 * const request = await recovery.initiateRecovery({
 *   did: 'did:mycelix:xxx',
 *   initiatorDid: 'did:mycelix:trustee1',
 *   newAgent: 'uhCAk...',
 *   reason: 'Lost device'
 * });
 * ```
 */
export class RecoveryClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Set up recovery for a DID */
  async setupRecovery(input: {
    did: string;
    trustees: string[];
    threshold: number;
    timeLock?: number;
  }): Promise<RecoveryConfig> {
    const record = await this.client.callZome<HolochainRecord<RecoveryConfig>>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'setup_recovery',
      payload: {
        did: input.did,
        trustees: input.trustees,
        threshold: input.threshold,
        time_lock: input.timeLock,
      },
    });
    return this.extractEntry(record);
  }

  /** Get recovery config for a DID */
  async getRecoveryConfig(did: string): Promise<RecoveryConfig | null> {
    const record = await this.client.callZome<HolochainRecord<RecoveryConfig> | null>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'get_recovery_config',
      payload: did,
    });
    return record ? this.extractEntry(record) : null;
  }

  /** Initiate a recovery request (trustee only) */
  async initiateRecovery(input: {
    did: string;
    initiatorDid: string;
    newAgent: string;
    reason: string;
  }): Promise<RecoveryRequest> {
    const record = await this.client.callZome<HolochainRecord<RecoveryRequest>>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'initiate_recovery',
      payload: {
        did: input.did,
        initiator_did: input.initiatorDid,
        new_agent: input.newAgent,
        reason: input.reason,
      },
    });
    return this.extractEntry(record);
  }

  /** Vote on a recovery request */
  async voteOnRecovery(input: {
    requestId: string;
    trusteeDid: string;
    vote: VoteDecision;
    comment?: string;
  }): Promise<RecoveryVote> {
    const record = await this.client.callZome<HolochainRecord<RecoveryVote>>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'vote_on_recovery',
      payload: {
        request_id: input.requestId,
        trustee_did: input.trusteeDid,
        vote: input.vote,
        comment: input.comment,
      },
    });
    return this.extractEntry(record);
  }

  /** Get votes for a recovery request */
  async getRecoveryVotes(requestId: string): Promise<RecoveryVote[]> {
    const records = await this.client.callZome<HolochainRecord<RecoveryVote>[]>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'get_recovery_votes',
      payload: requestId,
    });
    return records.map((r) => this.extractEntry(r));
  }

  /** Execute recovery (after time lock) */
  async executeRecovery(requestId: string): Promise<RecoveryRequest> {
    const record = await this.client.callZome<HolochainRecord<RecoveryRequest>>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'execute_recovery',
      payload: requestId,
    });
    return this.extractEntry(record);
  }

  /** Cancel a recovery request (owner only) */
  async cancelRecovery(requestId: string): Promise<RecoveryRequest> {
    const record = await this.client.callZome<HolochainRecord<RecoveryRequest>>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'cancel_recovery',
      payload: requestId,
    });
    return this.extractEntry(record);
  }

  /** Get pending recovery requests for a trustee */
  async getTrusteeResponsibilities(trusteeDid: string): Promise<RecoveryConfig[]> {
    const records = await this.client.callZome<HolochainRecord<RecoveryConfig>[]>({
      role_name: 'identity',
      zome_name: 'recovery',
      fn_name: 'get_trustee_responsibilities',
      payload: trusteeDid,
    });
    return records.map((r) => this.extractEntry(r));
  }

  private extractEntry<T>(record: HolochainRecord<T>): T {
    if ('Present' in record.entry) {
      return record.entry.Present.entry;
    }
    throw new Error('Entry not found in record');
  }
}

// ============================================================================
// Bridge Client
// ============================================================================

/**
 * Client for cross-hApp identity operations via the Bridge.
 *
 * @example
 * ```typescript
 * const bridge = new IdentityBridgeClient(conductor);
 *
 * // Verify a DID from another hApp
 * const result = await bridge.verifyIdentity({
 *   did: 'did:mycelix:xxx',
 *   sourceHapp: 'marketplace',
 *   requestedFields: ['name', 'email']
 * });
 *
 * // Check if DID is trustworthy
 * const trustworthy = await bridge.isTrustworthy('did:mycelix:xxx', 0.7);
 * ```
 */
export class IdentityBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Verify an identity from another hApp */
  async verifyIdentity(input: {
    did: string;
    sourceHapp: string;
    requestedFields: string[];
  }): Promise<IdentityVerificationResult> {
    return this.client.callZome<IdentityVerificationResult>({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'query_identity',
      payload: {
        did: input.did,
        source_happ: input.sourceHapp,
        requested_fields: input.requestedFields,
      },
    });
  }

  /** Quick DID verification */
  async verifyDid(did: string): Promise<boolean> {
    return this.client.callZome<boolean>({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'verify_did',
      payload: did,
    });
  }

  /** Get MATL score for a DID */
  async getMatlScore(did: string): Promise<number> {
    return this.client.callZome<number>({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'get_matl_score',
      payload: did,
    });
  }

  /** Check if DID meets trust threshold */
  async isTrustworthy(did: string, threshold: number): Promise<boolean> {
    return this.client.callZome<boolean>({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'is_trustworthy',
      payload: { did, threshold },
    });
  }

  /** Get aggregated reputation for a DID */
  async getReputation(did: string): Promise<AggregatedReputation> {
    return this.client.callZome<AggregatedReputation>({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'get_reputation',
      payload: did,
    });
  }

  /** Report reputation for a DID from your hApp */
  async reportReputation(input: {
    did: string;
    sourceHapp: string;
    score: number;
    interactions: number;
  }): Promise<void> {
    await this.client.callZome({
      role_name: 'identity',
      zome_name: 'identity_bridge',
      fn_name: 'report_reputation',
      payload: {
        did: input.did,
        source_happ: input.sourceHapp,
        score: input.score,
        interactions: input.interactions,
      },
    });
  }
}

// ============================================================================
// MFA (Multi-Factor Authentication) Types - MFDI Spec v1.0
// ============================================================================

/** Factor category for diversity scoring */
export type FactorCategory =
  | 'Cryptographic'
  | 'Biometric'
  | 'SocialProof'
  | 'ExternalVerification'
  | 'Knowledge';

/** Identity factor types from MFDI spec */
export type FactorType =
  | 'PrimaryKeyPair'
  | 'HardwareKey'
  | 'Biometric'
  | 'SocialRecovery'
  | 'ReputationAttestation'
  | 'GitcoinPassport'
  | 'VerifiableCredential'
  | 'RecoveryPhrase'
  | 'SecurityQuestions';

/** Assurance levels aligned with Epistemic E-Axis */
export type AssuranceLevel =
  | 'Anonymous'
  | 'Basic'
  | 'Verified'
  | 'HighlyAssured'
  | 'ConstitutionallyCritical';

/** Enrollment action type */
export type EnrollmentAction = 'Enroll' | 'Revoke' | 'Update' | 'Reverify';

/** An enrolled identity factor */
export interface EnrolledFactor {
  /** Type of factor */
  factorType: FactorType;
  /** Factor-specific identifier (hash of key, device ID, etc.) */
  factorId: string;
  /** When the factor was enrolled (microseconds) */
  enrolledAt: number;
  /** Last verification/use timestamp (microseconds) */
  lastVerified: number;
  /** Factor-specific metadata (JSON string) */
  metadata: string;
  /** Current effective strength (0.0-1.0) */
  effectiveStrength: number;
  /** Whether factor is currently active */
  active: boolean;
}

/** Multi-Factor Authentication state for an identity */
export interface MfaState {
  /** The DID this MFA state belongs to */
  did: string;
  /** Owner's agent pub key */
  owner: string;
  /** List of enrolled factors */
  factors: EnrolledFactor[];
  /** Current calculated assurance level */
  assuranceLevel: AssuranceLevel;
  /** Total effective strength (for MATL) */
  effectiveStrength: number;
  /** Number of unique factor categories */
  categoryCount: number;
  /** Creation timestamp (microseconds) */
  created: number;
  /** Last update timestamp (microseconds) */
  updated: number;
  /** Version number */
  version: number;
}

/** Assurance calculation output */
export interface AssuranceOutput {
  /** Current assurance level */
  level: AssuranceLevel;
  /** Numeric score (0.0-1.0) for MATL */
  score: number;
  /** Total effective strength */
  effectiveStrength: number;
  /** Number of unique factor categories */
  categoryCount: number;
  /** Factor IDs needing re-verification */
  staleFactors: string[];
}

/** MFA state with assurance calculation */
export interface MfaStateOutput {
  /** The MFA state */
  state: MfaState;
  /** Action hash for updates */
  actionHash: string;
  /** Calculated assurance (with decay applied) */
  assurance: AssuranceOutput;
}

/** Factor enrollment record (audit trail) */
export interface FactorEnrollment {
  /** The DID this enrollment belongs to */
  did: string;
  /** The factor type */
  factorType: FactorType;
  /** Factor-specific identifier */
  factorId: string;
  /** Action type */
  action: EnrollmentAction;
  /** Timestamp (microseconds) */
  timestamp: number;
  /** Reason for the action */
  reason: string;
}

/** FL (Federated Learning) eligibility result */
export interface FlEligibilityResult {
  /** Whether eligible for FL participation */
  eligible: boolean;
  /** Current assurance level */
  assuranceLevel: AssuranceLevel;
  /** Effective identity strength */
  effectiveStrength: number;
  /** Reasons for denial (if any) */
  denialReasons: string[];
}

// ============================================================================
// MFA Client
// ============================================================================

/**
 * Client for Multi-Factor Authentication operations.
 *
 * Implements the MFDI (Multi-Factor Decentralized Identity) spec v1.0.
 *
 * @example
 * ```typescript
 * const mfa = new MfaClient(conductor);
 *
 * // Create MFA state for a DID
 * const state = await mfa.createMfaState({
 *   did: 'did:mycelix:xxx',
 *   primaryKeyHash: 'sha256:...'
 * });
 *
 * // Enroll additional factors
 * await mfa.enrollFactor({
 *   did: 'did:mycelix:xxx',
 *   factorType: 'GitcoinPassport',
 *   factorId: 'passport-0x...',
 *   metadata: '{"score": 42}',
 *   reason: 'Gitcoin verification'
 * });
 *
 * // Check FL eligibility
 * const eligible = await mfa.checkFlEligibility('did:mycelix:xxx');
 * ```
 */
export class MfaClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Create initial MFA state for a DID */
  async createMfaState(input: {
    did: string;
    primaryKeyHash: string;
  }): Promise<MfaStateOutput> {
    return this.client.callZome<MfaStateOutput>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'create_mfa_state',
      payload: {
        did: input.did,
        primary_key_hash: input.primaryKeyHash,
      },
    });
  }

  /** Enroll a new identity factor */
  async enrollFactor(input: {
    did: string;
    factorType: FactorType;
    factorId: string;
    metadata: string;
    reason: string;
  }): Promise<MfaStateOutput> {
    return this.client.callZome<MfaStateOutput>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'enroll_factor',
      payload: {
        did: input.did,
        factor_type: input.factorType,
        factor_id: input.factorId,
        metadata: input.metadata,
        reason: input.reason,
      },
    });
  }

  /** Revoke an existing factor */
  async revokeFactor(input: {
    did: string;
    factorId: string;
    reason: string;
  }): Promise<MfaStateOutput> {
    return this.client.callZome<MfaStateOutput>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'revoke_factor',
      payload: {
        did: input.did,
        factor_id: input.factorId,
        reason: input.reason,
      },
    });
  }

  /** Verify a factor to reset its decay timer */
  async verifyFactor(input: { did: string; factorId: string }): Promise<MfaStateOutput> {
    return this.client.callZome<MfaStateOutput>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'verify_factor',
      payload: {
        did: input.did,
        factor_id: input.factorId,
      },
    });
  }

  /** Get MFA state for a DID */
  async getMfaState(did: string): Promise<MfaStateOutput | null> {
    return this.client.callZome<MfaStateOutput | null>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'get_mfa_state',
      payload: did,
    });
  }

  /** Calculate current assurance level (with decay applied) */
  async calculateAssurance(did: string): Promise<AssuranceOutput> {
    return this.client.callZome<AssuranceOutput>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'calculate_assurance',
      payload: did,
    });
  }

  /** Get enrollment history for a DID */
  async getEnrollmentHistory(did: string): Promise<FactorEnrollment[]> {
    return this.client.callZome<FactorEnrollment[]>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'get_enrollment_history',
      payload: did,
    });
  }

  /** Check if identity meets FL participation requirements */
  async checkFlEligibility(did: string): Promise<FlEligibilityResult> {
    return this.client.callZome<FlEligibilityResult>({
      role_name: 'identity',
      zome_name: 'mfa',
      fn_name: 'check_fl_eligibility',
      payload: did,
    });
  }

  /** Get assurance level as numeric score (0.0-1.0) */
  getAssuranceScore(level: AssuranceLevel): number {
    const scores: Record<AssuranceLevel, number> = {
      Anonymous: 0.0,
      Basic: 0.25,
      Verified: 0.5,
      HighlyAssured: 0.75,
      ConstitutionallyCritical: 1.0,
    };
    return scores[level];
  }

  /** Get factor category for a factor type */
  getFactorCategory(factorType: FactorType): FactorCategory {
    const categories: Record<FactorType, FactorCategory> = {
      PrimaryKeyPair: 'Cryptographic',
      HardwareKey: 'Cryptographic',
      Biometric: 'Biometric',
      SocialRecovery: 'SocialProof',
      ReputationAttestation: 'SocialProof',
      GitcoinPassport: 'ExternalVerification',
      VerifiableCredential: 'ExternalVerification',
      RecoveryPhrase: 'Knowledge',
      SecurityQuestions: 'Knowledge',
    };
    return categories[factorType];
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all identity clients from a single conductor connection.
 *
 * @example
 * ```typescript
 * const { identity, schemas, revocation, recovery, bridge, mfa } = createIdentityClients(conductor);
 *
 * // Use each client
 * const did = await identity.createDid();
 * const isActive = await identity.isDidActive(did.id);
 * const reputation = await bridge.getReputation(did.id);
 *
 * // MFA operations
 * const mfaState = await mfa.createMfaState({ did: did.id, primaryKeyHash: '...' });
 * const eligible = await mfa.checkFlEligibility(did.id);
 * ```
 */
export function createIdentityClients(client: ZomeCallable): {
  identity: IdentityClient;
  schemas: CredentialSchemaClient;
  revocation: RevocationClient;
  recovery: RecoveryClient;
  bridge: IdentityBridgeClient;
  mfa: MfaClient;
} {
  return {
    identity: new IdentityClient(client),
    schemas: new CredentialSchemaClient(client),
    revocation: new RevocationClient(client),
    recovery: new RecoveryClient(client),
    bridge: new IdentityBridgeClient(client),
    mfa: new MfaClient(client),
  };
}

// All classes and types are exported inline above

/**
 * @mycelix/sdk Identity Validated Clients
 *
 * Provides input-validated versions of all Identity clients.
 * All inputs are validated using Zod schemas before being sent to the conductor.
 *
 * @packageDocumentation
 * @module identity/validated
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  IdentityClient,
  CredentialSchemaClient,
  RevocationClient,
  RecoveryClient,
  IdentityBridgeClient,
  type ZomeCallable,
  type DidDocument,
  type VerificationMethod,
  type ServiceEndpoint,
  type UpdateDidInput,
  type CredentialSchema,
  type SchemaEndorsement,
  type RevocationEntry,
  type RevocationCheckResult,
  type RecoveryConfig,
  type RecoveryRequest,
  type RecoveryVote,
  type VoteDecision,
  type DidDeactivation,
  type IdentityVerificationResult,
  type AggregatedReputation,
} from './index.js';

// ============================================================================
// Validation Schemas
// ============================================================================

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID (start with "did:")' });

const publicKeySchema = z.string().min(32, 'Public key must be at least 32 characters');

const verificationMethodSchema = z.object({
  id: z.string().min(1),
  type: z.string().min(1),
  controller: didSchema,
  publicKeyMultibase: z.string().min(1),
});

const serviceEndpointSchema = z.object({
  id: z.string().min(1),
  type: z.string().min(1),
  serviceEndpoint: z.string().min(1),
});

const updateDidInputSchema = z.object({
  verificationMethod: z.array(verificationMethodSchema).optional(),
  authentication: z.array(z.string()).optional(),
  service: z.array(serviceEndpointSchema).optional(),
});

const credentialSchemaSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1, 'Schema name is required'),
  description: z.string(),
  version: z.string().min(1, 'Version is required'),
  author: didSchema,
  schema: z.string().min(1),
  requiredFields: z.array(z.string()),
  optionalFields: z.array(z.string()),
  credentialType: z.array(z.string()),
  defaultExpiration: z.number().min(0),
  revocable: z.boolean(),
  active: z.boolean(),
  created: z.number().positive(),
  updated: z.number().positive(),
});

const setupRecoveryInputSchema = z.object({
  did: didSchema,
  trustees: z.array(didSchema).min(1, 'At least 1 trustee required'),
  threshold: z.number().min(1),
  timeLock: z.number().min(0).optional(),
});

const initiateRecoveryInputSchema = z.object({
  did: didSchema,
  initiatorDid: didSchema,
  newAgent: publicKeySchema,
  reason: z.string().min(1, 'Recovery reason is required'),
});

const voteOnRecoveryInputSchema = z.object({
  requestId: z.string().min(1),
  trusteeDid: didSchema,
  vote: z.enum(['Approve', 'Reject', 'Abstain']),
  comment: z.string().optional(),
});

// ============================================================================
// Validation Utility
// ============================================================================

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

// ============================================================================
// Validated Identity Client
// ============================================================================

/**
 * Validated Identity Client - All inputs are validated before zome calls
 */
export class ValidatedIdentityClient {
  private client: IdentityClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new IdentityClient(zomeClient);
  }

  async createDid(): Promise<DidDocument> {
    return this.client.createDid();
  }

  async getDidDocument(agentPubKey: string): Promise<DidDocument | null> {
    validateOrThrow(publicKeySchema, agentPubKey, 'agentPubKey');
    return this.client.getDidDocument(agentPubKey);
  }

  async resolveDid(did: string): Promise<DidDocument | null> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.resolveDid(did);
  }

  async updateDidDocument(input: UpdateDidInput): Promise<DidDocument> {
    validateOrThrow(updateDidInputSchema, input, 'updateDidInput');
    return this.client.updateDidDocument(input);
  }

  async deactivateDid(reason: string): Promise<DidDeactivation> {
    validateOrThrow(z.string().min(1, 'Deactivation reason is required'), reason, 'reason');
    return this.client.deactivateDid(reason);
  }

  async isDidActive(did: string): Promise<boolean> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.isDidActive(did);
  }

  async addServiceEndpoint(service: ServiceEndpoint): Promise<DidDocument> {
    validateOrThrow(serviceEndpointSchema, service, 'service');
    return this.client.addServiceEndpoint(service);
  }

  async removeServiceEndpoint(serviceId: string): Promise<DidDocument> {
    validateOrThrow(z.string().min(1), serviceId, 'serviceId');
    return this.client.removeServiceEndpoint(serviceId);
  }

  async addVerificationMethod(method: VerificationMethod): Promise<DidDocument> {
    validateOrThrow(verificationMethodSchema, method, 'method');
    return this.client.addVerificationMethod(method);
  }

  async getMyDid(): Promise<DidDocument | null> {
    return this.client.getMyDid();
  }
}

// ============================================================================
// Validated Credential Schema Client
// ============================================================================

/**
 * Validated Credential Schema Client
 */
export class ValidatedCredentialSchemaClient {
  private client: CredentialSchemaClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new CredentialSchemaClient(zomeClient);
  }

  async createSchema(schema: CredentialSchema): Promise<CredentialSchema> {
    validateOrThrow(credentialSchemaSchema, schema, 'schema');
    return this.client.createSchema(schema);
  }

  async getSchema(schemaId: string): Promise<CredentialSchema | null> {
    validateOrThrow(z.string().min(1), schemaId, 'schemaId');
    return this.client.getSchema(schemaId);
  }

  async getSchemasByAuthor(authorDid: string): Promise<CredentialSchema[]> {
    validateOrThrow(didSchema, authorDid, 'authorDid');
    return this.client.getSchemasByAuthor(authorDid);
  }

  async listActiveSchemas(): Promise<CredentialSchema[]> {
    return this.client.listActiveSchemas();
  }

  async endorseSchema(
    schemaId: string,
    endorserDid: string,
    trustLevel: number,
    comment?: string
  ): Promise<SchemaEndorsement> {
    validateOrThrow(z.string().min(1), schemaId, 'schemaId');
    validateOrThrow(didSchema, endorserDid, 'endorserDid');
    validateOrThrow(z.number().min(0).max(1), trustLevel, 'trustLevel');
    if (comment !== undefined) {
      validateOrThrow(z.string(), comment, 'comment');
    }
    return this.client.endorseSchema(schemaId, endorserDid, trustLevel, comment);
  }

  async getSchemaEndorsements(schemaId: string): Promise<SchemaEndorsement[]> {
    validateOrThrow(z.string().min(1), schemaId, 'schemaId');
    return this.client.getSchemaEndorsements(schemaId);
  }
}

// ============================================================================
// Validated Revocation Client
// ============================================================================

/**
 * Validated Revocation Client
 */
export class ValidatedRevocationClient {
  private client: RevocationClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new RevocationClient(zomeClient);
  }

  async revoke(
    credentialId: string,
    issuerDid: string,
    reason: string,
    effectiveFrom?: number
  ): Promise<RevocationEntry> {
    validateOrThrow(z.string().min(1), credentialId, 'credentialId');
    validateOrThrow(didSchema, issuerDid, 'issuerDid');
    validateOrThrow(z.string().min(1, 'Revocation reason is required'), reason, 'reason');
    if (effectiveFrom !== undefined) {
      validateOrThrow(z.number().positive(), effectiveFrom, 'effectiveFrom');
    }
    return this.client.revoke(credentialId, issuerDid, reason, effectiveFrom);
  }

  async suspend(
    credentialId: string,
    issuerDid: string,
    reason: string,
    suspensionEnd: number
  ): Promise<RevocationEntry> {
    validateOrThrow(z.string().min(1), credentialId, 'credentialId');
    validateOrThrow(didSchema, issuerDid, 'issuerDid');
    validateOrThrow(z.string().min(1), reason, 'reason');
    validateOrThrow(z.number().positive(), suspensionEnd, 'suspensionEnd');
    return this.client.suspend(credentialId, issuerDid, reason, suspensionEnd);
  }

  async reinstate(
    credentialId: string,
    issuerDid: string,
    reason: string
  ): Promise<RevocationEntry> {
    validateOrThrow(z.string().min(1), credentialId, 'credentialId');
    validateOrThrow(didSchema, issuerDid, 'issuerDid');
    validateOrThrow(z.string().min(1), reason, 'reason');
    return this.client.reinstate(credentialId, issuerDid, reason);
  }

  async checkStatus(credentialId: string): Promise<RevocationCheckResult> {
    validateOrThrow(z.string().min(1), credentialId, 'credentialId');
    return this.client.checkStatus(credentialId);
  }

  async batchCheckStatus(credentialIds: string[]): Promise<RevocationCheckResult[]> {
    validateOrThrow(z.array(z.string().min(1)).min(1), credentialIds, 'credentialIds');
    return this.client.batchCheckStatus(credentialIds);
  }

  async getByIssuer(issuerDid: string): Promise<RevocationEntry[]> {
    validateOrThrow(didSchema, issuerDid, 'issuerDid');
    return this.client.getByIssuer(issuerDid);
  }
}

// ============================================================================
// Validated Recovery Client
// ============================================================================

/**
 * Validated Recovery Client
 */
export class ValidatedRecoveryClient {
  private client: RecoveryClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new RecoveryClient(zomeClient);
  }

  async setupRecovery(input: {
    did: string;
    trustees: string[];
    threshold: number;
    timeLock?: number;
  }): Promise<RecoveryConfig> {
    validateOrThrow(setupRecoveryInputSchema, input, 'setupRecovery input');
    return this.client.setupRecovery(input);
  }

  async getRecoveryConfig(did: string): Promise<RecoveryConfig | null> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getRecoveryConfig(did);
  }

  async initiateRecovery(input: {
    did: string;
    initiatorDid: string;
    newAgent: string;
    reason: string;
  }): Promise<RecoveryRequest> {
    validateOrThrow(initiateRecoveryInputSchema, input, 'initiateRecovery input');
    return this.client.initiateRecovery(input);
  }

  async voteOnRecovery(input: {
    requestId: string;
    trusteeDid: string;
    vote: VoteDecision;
    comment?: string;
  }): Promise<RecoveryVote> {
    validateOrThrow(voteOnRecoveryInputSchema, input, 'voteOnRecovery input');
    return this.client.voteOnRecovery(input);
  }

  async getRecoveryVotes(requestId: string): Promise<RecoveryVote[]> {
    validateOrThrow(z.string().min(1), requestId, 'requestId');
    return this.client.getRecoveryVotes(requestId);
  }

  async executeRecovery(requestId: string): Promise<RecoveryRequest> {
    validateOrThrow(z.string().min(1), requestId, 'requestId');
    return this.client.executeRecovery(requestId);
  }

  async cancelRecovery(requestId: string): Promise<RecoveryRequest> {
    validateOrThrow(z.string().min(1), requestId, 'requestId');
    return this.client.cancelRecovery(requestId);
  }

  async getTrusteeResponsibilities(trusteeDid: string): Promise<RecoveryConfig[]> {
    validateOrThrow(didSchema, trusteeDid, 'trusteeDid');
    return this.client.getTrusteeResponsibilities(trusteeDid);
  }
}

// ============================================================================
// Validated Identity Bridge Client
// ============================================================================

/** Input validation schema for verifyIdentity */
const verifyIdentityInputSchema = z.object({
  did: didSchema,
  sourceHapp: z.string().min(1),
  requestedFields: z.array(z.string()),
});

/** Input validation schema for reportReputation */
const reportReputationInputSchema = z.object({
  did: didSchema,
  sourceHapp: z.string().min(1),
  score: z.number().min(0).max(1),
  interactions: z.number().min(0),
});

/**
 * Validated Identity Bridge Client
 */
export class ValidatedIdentityBridgeClient {
  private client: IdentityBridgeClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new IdentityBridgeClient(zomeClient);
  }

  async verifyIdentity(input: {
    did: string;
    sourceHapp: string;
    requestedFields: string[];
  }): Promise<IdentityVerificationResult> {
    validateOrThrow(verifyIdentityInputSchema, input, 'verifyIdentity input');
    return this.client.verifyIdentity(input);
  }

  async verifyDid(did: string): Promise<boolean> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.verifyDid(did);
  }

  async getMatlScore(did: string): Promise<number> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getMatlScore(did);
  }

  async isTrustworthy(did: string, threshold: number): Promise<boolean> {
    validateOrThrow(didSchema, did, 'did');
    validateOrThrow(z.number().min(0).max(1), threshold, 'threshold');
    return this.client.isTrustworthy(did, threshold);
  }

  async getReputation(did: string): Promise<AggregatedReputation> {
    validateOrThrow(didSchema, did, 'did');
    return this.client.getReputation(did);
  }

  async reportReputation(input: {
    did: string;
    sourceHapp: string;
    score: number;
    interactions: number;
  }): Promise<void> {
    validateOrThrow(reportReputationInputSchema, input, 'reportReputation input');
    return this.client.reportReputation(input);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all validated Identity hApp clients
 */
export function createValidatedIdentityClients(client: ZomeCallable) {
  return {
    identity: new ValidatedIdentityClient(client),
    schemas: new ValidatedCredentialSchemaClient(client),
    revocation: new ValidatedRevocationClient(client),
    recovery: new ValidatedRecoveryClient(client),
    bridge: new ValidatedIdentityBridgeClient(client),
  };
}

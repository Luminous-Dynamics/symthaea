// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Module Tests
 *
 * Tests for the Identity hApp TypeScript clients:
 * - IdentityClient (DID management)
 * - CredentialSchemaClient (schema management)
 * - RevocationClient (revocation registry)
 * - RecoveryClient (social recovery)
 * - IdentityBridgeClient (cross-hApp operations)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  IdentityClient,
  CredentialSchemaClient,
  RevocationClient,
  RecoveryClient,
  IdentityBridgeClient,
  MfaClient,
  createIdentityClients,
  type DidDocument,
  type CredentialSchema,
  type RevocationEntry,
  type RecoveryConfig,
  type MfaState,
  type MfaStateOutput,
  type AssuranceOutput,
  type EnrolledFactor,
  type FlEligibilityResult,
  type FactorType,
  type AssuranceLevel,
  type ZomeCallable,
  type HolochainRecord,
} from '../src/identity/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {},
    entry: { Present: { entry } },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// IdentityClient Tests
// ============================================================================

describe('IdentityClient', () => {
  let client: IdentityClient;
  let mockZome: ZomeCallable;

  const mockDidDocument: DidDocument = {
    id: 'did:mycelix:uhCAk123abc',
    controller: 'uhCAk123abc',
    verificationMethod: [
      {
        id: 'did:mycelix:uhCAk123abc#keys-1',
        type: 'Ed25519VerificationKey2020',
        controller: 'did:mycelix:uhCAk123abc',
        publicKeyMultibase: 'zuhCAk123abc',
      },
    ],
    authentication: ['did:mycelix:uhCAk123abc#keys-1'],
    service: [],
    created: Date.now() * 1000,
    updated: Date.now() * 1000,
    version: 1,
  };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('did_registry:create_did', createMockRecord(mockDidDocument));
    responses.set('did_registry:get_did_document', createMockRecord(mockDidDocument));
    responses.set('did_registry:resolve_did', createMockRecord(mockDidDocument));
    responses.set('did_registry:get_my_did', createMockRecord(mockDidDocument));
    responses.set('did_registry:is_did_active', true);
    responses.set(
      'did_registry:update_did_document',
      createMockRecord({
        ...mockDidDocument,
        version: 2,
        updated: Date.now() * 1000,
      })
    );
    responses.set(
      'did_registry:add_service_endpoint',
      createMockRecord({
        ...mockDidDocument,
        service: [
          {
            id: 'did:mycelix:uhCAk123abc#bridge',
            type: 'MycelixBridge',
            serviceEndpoint: 'https://bridge.mycelix.net',
          },
        ],
      })
    );
    responses.set('did_registry:remove_service_endpoint', createMockRecord(mockDidDocument));
    responses.set(
      'did_registry:add_verification_method',
      createMockRecord({
        ...mockDidDocument,
        verificationMethod: [
          ...mockDidDocument.verificationMethod,
          {
            id: 'did:mycelix:uhCAk123abc#keys-2',
            type: 'X25519KeyAgreementKey2020',
            controller: 'did:mycelix:uhCAk123abc',
            publicKeyMultibase: 'z456def',
          },
        ],
      })
    );
    responses.set(
      'did_registry:deactivate_did',
      createMockRecord({
        did: 'did:mycelix:uhCAk123abc',
        reason: 'Key compromised',
        deactivatedAt: Date.now() * 1000,
      })
    );

    mockZome = createMockClient(responses);
    client = new IdentityClient(mockZome);
  });

  it('should create a new DID', async () => {
    const did = await client.createDid();

    expect(did.id).toBe('did:mycelix:uhCAk123abc');
    expect(did.verificationMethod).toHaveLength(1);
    expect(mockZome.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        fn_name: 'create_did',
      })
    );
  });

  it('should get DID document by agent pub key', async () => {
    const did = await client.getDidDocument('uhCAk123abc');

    expect(did).not.toBeNull();
    expect(did!.id).toBe('did:mycelix:uhCAk123abc');
  });

  it('should resolve DID to document', async () => {
    const did = await client.resolveDid('did:mycelix:uhCAk123abc');

    expect(did).not.toBeNull();
    expect(did!.id).toBe('did:mycelix:uhCAk123abc');
  });

  it('should get calling agent DID', async () => {
    const did = await client.getMyDid();

    expect(did).not.toBeNull();
    expect(did!.id).toContain('did:mycelix:');
  });

  it('should check if DID is active', async () => {
    const isActive = await client.isDidActive('did:mycelix:uhCAk123abc');

    expect(isActive).toBe(true);
  });

  it('should add service endpoint', async () => {
    const did = await client.addServiceEndpoint({
      id: 'did:mycelix:uhCAk123abc#bridge',
      type: 'MycelixBridge',
      serviceEndpoint: 'https://bridge.mycelix.net',
    });

    expect(did.service).toHaveLength(1);
    expect(did.service[0].type).toBe('MycelixBridge');
  });

  it('should add verification method', async () => {
    const did = await client.addVerificationMethod({
      id: 'did:mycelix:uhCAk123abc#keys-2',
      type: 'X25519KeyAgreementKey2020',
      controller: 'did:mycelix:uhCAk123abc',
      publicKeyMultibase: 'z456def',
    });

    expect(did.verificationMethod).toHaveLength(2);
  });

  it('should update DID document', async () => {
    const did = await client.updateDidDocument({
      service: [
        {
          id: 'did:mycelix:uhCAk123abc#linked',
          type: 'LinkedDomains',
          serviceEndpoint: 'https://example.com',
        },
      ],
    });

    expect(did.version).toBe(2);
  });
});

// ============================================================================
// CredentialSchemaClient Tests
// ============================================================================

describe('CredentialSchemaClient', () => {
  let client: CredentialSchemaClient;
  let mockZome: ZomeCallable;

  const mockSchema: CredentialSchema = {
    id: 'mycelix:schema:education:degree:v1',
    name: 'University Degree',
    description: 'Academic degree credential',
    version: '1.0.0',
    author: 'did:mycelix:uhCAk123abc',
    schema: JSON.stringify({
      type: 'object',
      properties: {
        degree: { type: 'string' },
        institution: { type: 'string' },
      },
    }),
    requiredFields: ['degree', 'institution'],
    optionalFields: ['gpa'],
    credentialType: ['VerifiableCredential', 'EducationCredential'],
    defaultExpiration: 0,
    revocable: true,
    active: true,
    created: Date.now() * 1000,
    updated: Date.now() * 1000,
  };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('credential_schema:create_schema', createMockRecord(mockSchema));
    responses.set('credential_schema:get_schema', createMockRecord(mockSchema));
    responses.set('credential_schema:get_schemas_by_author', [createMockRecord(mockSchema)]);
    responses.set('credential_schema:list_active_schemas', [createMockRecord(mockSchema)]);
    responses.set(
      'credential_schema:endorse_schema',
      createMockRecord({
        schemaId: mockSchema.id,
        endorser: 'did:mycelix:endorser',
        trustLevel: 0.9,
        endorsedAt: Date.now() * 1000,
      })
    );
    responses.set('credential_schema:get_schema_endorsements', [
      createMockRecord({
        schemaId: mockSchema.id,
        endorser: 'did:mycelix:endorser',
        trustLevel: 0.9,
        endorsedAt: Date.now() * 1000,
      }),
    ]);

    mockZome = createMockClient(responses);
    client = new CredentialSchemaClient(mockZome);
  });

  it('should create a schema', async () => {
    const schema = await client.createSchema(mockSchema);

    expect(schema.id).toBe('mycelix:schema:education:degree:v1');
    expect(schema.name).toBe('University Degree');
  });

  it('should get schema by ID', async () => {
    const schema = await client.getSchema('mycelix:schema:education:degree:v1');

    expect(schema).not.toBeNull();
    expect(schema!.id).toBe('mycelix:schema:education:degree:v1');
  });

  it('should get schemas by author', async () => {
    const schemas = await client.getSchemasByAuthor('did:mycelix:uhCAk123abc');

    expect(schemas).toHaveLength(1);
    expect(schemas[0].author).toBe('did:mycelix:uhCAk123abc');
  });

  it('should list active schemas', async () => {
    const schemas = await client.listActiveSchemas();

    expect(schemas).toHaveLength(1);
    expect(schemas[0].active).toBe(true);
  });

  it('should endorse a schema', async () => {
    const endorsement = await client.endorseSchema(
      mockSchema.id,
      'did:mycelix:endorser',
      0.9,
      'Verified by trusted authority'
    );

    expect(endorsement.trustLevel).toBe(0.9);
  });
});

// ============================================================================
// RevocationClient Tests
// ============================================================================

describe('RevocationClient', () => {
  let client: RevocationClient;
  let mockZome: ZomeCallable;

  const mockRevocation: RevocationEntry = {
    credentialId: 'vc-123',
    issuer: 'did:mycelix:issuer',
    status: 'Revoked',
    reason: 'Fraudulent use',
    effectiveFrom: Date.now() * 1000,
    recordedAt: Date.now() * 1000,
  };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('revocation:revoke_credential', createMockRecord(mockRevocation));
    responses.set(
      'revocation:suspend_credential',
      createMockRecord({
        ...mockRevocation,
        status: 'Suspended',
        reason: 'Under investigation',
        suspensionEnd: Date.now() * 1000 + 86400000000,
      })
    );
    responses.set(
      'revocation:reinstate_credential',
      createMockRecord({
        ...mockRevocation,
        status: 'Active',
        reason: 'Investigation cleared',
      })
    );
    responses.set('revocation:check_revocation_status', {
      credentialId: 'vc-123',
      status: 'Active',
      checkedAt: Date.now() * 1000,
    });
    responses.set('revocation:batch_check_revocation', [
      { credentialId: 'vc-123', status: 'Active', checkedAt: Date.now() * 1000 },
      {
        credentialId: 'vc-456',
        status: 'Revoked',
        reason: 'Expired',
        checkedAt: Date.now() * 1000,
      },
    ]);
    responses.set('revocation:get_revocations_by_issuer', [createMockRecord(mockRevocation)]);

    mockZome = createMockClient(responses);
    client = new RevocationClient(mockZome);
  });

  it('should revoke a credential', async () => {
    const entry = await client.revoke('vc-123', 'did:mycelix:issuer', 'Fraudulent use');

    expect(entry.status).toBe('Revoked');
    expect(entry.reason).toBe('Fraudulent use');
  });

  it('should suspend a credential', async () => {
    const futureDate = Date.now() * 1000 + 86400000000;
    const entry = await client.suspend(
      'vc-123',
      'did:mycelix:issuer',
      'Under investigation',
      futureDate
    );

    expect(entry.status).toBe('Suspended');
    expect(entry.suspensionEnd).toBeDefined();
  });

  it('should check revocation status', async () => {
    const result = await client.checkStatus('vc-123');

    expect(result.status).toBe('Active');
    expect(result.credentialId).toBe('vc-123');
  });

  it('should batch check revocation status', async () => {
    const results = await client.batchCheckStatus(['vc-123', 'vc-456']);

    expect(results).toHaveLength(2);
    expect(results[0].status).toBe('Active');
    expect(results[1].status).toBe('Revoked');
  });

  it('should get revocations by issuer', async () => {
    const entries = await client.getByIssuer('did:mycelix:issuer');

    expect(entries).toHaveLength(1);
    expect(entries[0].issuer).toBe('did:mycelix:issuer');
  });
});

// ============================================================================
// RecoveryClient Tests
// ============================================================================

describe('RecoveryClient', () => {
  let client: RecoveryClient;
  let mockZome: ZomeCallable;

  const mockConfig: RecoveryConfig = {
    did: 'did:mycelix:uhCAk123abc',
    owner: 'uhCAk123abc',
    trustees: ['did:mycelix:trustee1', 'did:mycelix:trustee2', 'did:mycelix:trustee3'],
    threshold: 2,
    timeLock: 604800, // 7 days
    active: true,
    created: Date.now() * 1000,
    updated: Date.now() * 1000,
  };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('recovery:setup_recovery', createMockRecord(mockConfig));
    responses.set('recovery:get_recovery_config', createMockRecord(mockConfig));
    responses.set(
      'recovery:initiate_recovery',
      createMockRecord({
        id: 'recovery:123',
        did: mockConfig.did,
        newAgent: 'uhCAk456def',
        initiatedBy: 'did:mycelix:trustee1',
        reason: 'Lost device',
        status: 'Pending',
        created: Date.now() * 1000,
      })
    );
    responses.set(
      'recovery:vote_on_recovery',
      createMockRecord({
        requestId: 'recovery:123',
        trustee: 'did:mycelix:trustee2',
        vote: 'Approve',
        votedAt: Date.now() * 1000,
      })
    );
    responses.set('recovery:get_recovery_votes', [
      createMockRecord({
        requestId: 'recovery:123',
        trustee: 'did:mycelix:trustee1',
        vote: 'Approve',
        votedAt: Date.now() * 1000,
      }),
    ]);
    responses.set(
      'recovery:execute_recovery',
      createMockRecord({
        id: 'recovery:123',
        did: mockConfig.did,
        status: 'Completed',
      })
    );
    responses.set(
      'recovery:cancel_recovery',
      createMockRecord({
        id: 'recovery:123',
        did: mockConfig.did,
        status: 'Cancelled',
      })
    );
    responses.set('recovery:get_trustee_responsibilities', [createMockRecord(mockConfig)]);

    mockZome = createMockClient(responses);
    client = new RecoveryClient(mockZome);
  });

  it('should setup recovery', async () => {
    const config = await client.setupRecovery({
      did: 'did:mycelix:uhCAk123abc',
      trustees: mockConfig.trustees,
      threshold: 2,
      timeLock: 604800,
    });

    expect(config.trustees).toHaveLength(3);
    expect(config.threshold).toBe(2);
  });

  it('should get recovery config', async () => {
    const config = await client.getRecoveryConfig('did:mycelix:uhCAk123abc');

    expect(config).not.toBeNull();
    expect(config!.timeLock).toBe(604800);
  });

  it('should initiate recovery', async () => {
    const request = await client.initiateRecovery({
      did: 'did:mycelix:uhCAk123abc',
      initiatorDid: 'did:mycelix:trustee1',
      newAgent: 'uhCAk456def',
      reason: 'Lost device',
    });

    expect(request.status).toBe('Pending');
    expect(request.initiatedBy).toBe('did:mycelix:trustee1');
  });

  it('should vote on recovery', async () => {
    const vote = await client.voteOnRecovery({
      requestId: 'recovery:123',
      trusteeDid: 'did:mycelix:trustee2',
      vote: 'Approve',
    });

    expect(vote.vote).toBe('Approve');
  });

  it('should get recovery votes', async () => {
    const votes = await client.getRecoveryVotes('recovery:123');

    expect(votes).toHaveLength(1);
    expect(votes[0].vote).toBe('Approve');
  });

  it('should get trustee responsibilities', async () => {
    const configs = await client.getTrusteeResponsibilities('did:mycelix:trustee1');

    expect(configs).toHaveLength(1);
  });
});

// ============================================================================
// IdentityBridgeClient Tests
// ============================================================================

describe('IdentityBridgeClient', () => {
  let client: IdentityBridgeClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('identity_bridge:verify_did', true);
    responses.set('identity_bridge:get_matl_score', 0.85);
    responses.set('identity_bridge:is_trustworthy', true);
    responses.set('identity_bridge:query_identity', {
      verificationHash: 'uhCXk123',
      did: 'did:mycelix:uhCAk123abc',
      isValid: true,
      matlScore: 0.85,
      credentialCount: 5,
    });
    responses.set('identity_bridge:get_reputation', {
      did: 'did:mycelix:uhCAk123abc',
      aggregateScore: 0.85,
      sources: [
        { sourceHapp: 'marketplace', score: 0.9, interactions: 50 },
        { sourceHapp: 'governance', score: 0.8, interactions: 20 },
      ],
      totalInteractions: 70,
    });
    responses.set('identity_bridge:report_reputation', undefined);

    mockZome = createMockClient(responses);
    client = new IdentityBridgeClient(mockZome);
  });

  it('should verify DID', async () => {
    const isValid = await client.verifyDid('did:mycelix:uhCAk123abc');

    expect(isValid).toBe(true);
  });

  it('should get MATL score', async () => {
    const score = await client.getMatlScore('did:mycelix:uhCAk123abc');

    expect(score).toBe(0.85);
  });

  it('should check trustworthiness', async () => {
    const trustworthy = await client.isTrustworthy('did:mycelix:uhCAk123abc', 0.7);

    expect(trustworthy).toBe(true);
  });

  it('should verify identity from another hApp', async () => {
    const result = await client.verifyIdentity({
      did: 'did:mycelix:uhCAk123abc',
      sourceHapp: 'marketplace',
      requestedFields: ['matl_score', 'credential_count'],
    });

    expect(result.isValid).toBe(true);
    expect(result.matlScore).toBe(0.85);
    expect(result.credentialCount).toBe(5);
  });

  it('should get aggregated reputation', async () => {
    const rep = await client.getReputation('did:mycelix:uhCAk123abc');

    expect(rep.aggregateScore).toBe(0.85);
    expect(rep.sources).toHaveLength(2);
    expect(rep.totalInteractions).toBe(70);
  });
});

// ============================================================================
// MfaClient Tests
// ============================================================================

describe('MfaClient', () => {
  let client: MfaClient;
  let mockZome: ZomeCallable;

  const mockMfaState: MfaState = {
    did: 'did:mycelix:uhCAk123abc',
    owner: 'uhCAk123abc',
    factors: [
      {
        factorType: 'PrimaryKeyPair',
        factorId: 'primary-key-hash',
        enrolledAt: Date.now() * 1000,
        lastVerified: Date.now() * 1000,
        metadata: '{}',
        effectiveStrength: 1.0,
        active: true,
      },
    ],
    assuranceLevel: 'Basic',
    effectiveStrength: 1.0,
    categoryCount: 1,
    created: Date.now() * 1000,
    updated: Date.now() * 1000,
    version: 1,
  };

  const mockAssurance: AssuranceOutput = {
    level: 'Basic',
    score: 0.25,
    effectiveStrength: 1.0,
    categoryCount: 1,
    staleFactors: [],
  };

  const mockMfaStateOutput: MfaStateOutput = {
    state: mockMfaState,
    actionHash: 'uhCXk123abc',
    assurance: mockAssurance,
  };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('mfa:create_mfa_state', mockMfaStateOutput);
    responses.set('mfa:get_mfa_state', mockMfaStateOutput);
    responses.set('mfa:enroll_factor', {
      ...mockMfaStateOutput,
      state: {
        ...mockMfaState,
        factors: [
          ...mockMfaState.factors,
          {
            factorType: 'GitcoinPassport',
            factorId: 'passport-123',
            enrolledAt: Date.now() * 1000,
            lastVerified: Date.now() * 1000,
            metadata: '{"score":42}',
            effectiveStrength: 1.0,
            active: true,
          },
        ],
        version: 2,
      },
      assurance: {
        level: 'Verified',
        score: 0.5,
        effectiveStrength: 1.8,
        categoryCount: 2,
        staleFactors: [],
      },
    });
    responses.set('mfa:revoke_factor', mockMfaStateOutput);
    responses.set('mfa:verify_factor', mockMfaStateOutput);
    responses.set('mfa:calculate_assurance', mockAssurance);
    responses.set('mfa:get_enrollment_history', [
      {
        did: 'did:mycelix:uhCAk123abc',
        factorType: 'PrimaryKeyPair',
        factorId: 'primary-key-hash',
        action: 'Enroll',
        timestamp: Date.now() * 1000,
        reason: 'Initial creation',
      },
    ]);
    responses.set('mfa:check_fl_eligibility', {
      eligible: false,
      assuranceLevel: 'Basic',
      effectiveStrength: 1.0,
      denialReasons: ['Missing ExternalVerification factor'],
    });

    mockZome = createMockClient(responses);
    client = new MfaClient(mockZome);
  });

  it('should create MFA state', async () => {
    const result = await client.createMfaState({
      did: 'did:mycelix:uhCAk123abc',
      primaryKeyHash: 'primary-key-hash',
    });

    expect(result.state.did).toBe('did:mycelix:uhCAk123abc');
    expect(result.state.factors).toHaveLength(1);
    expect(result.state.factors[0].factorType).toBe('PrimaryKeyPair');
    expect(result.assurance.level).toBe('Basic');
  });

  it('should get MFA state', async () => {
    const result = await client.getMfaState('did:mycelix:uhCAk123abc');

    expect(result).not.toBeNull();
    expect(result!.state.version).toBe(1);
  });

  it('should enroll a new factor', async () => {
    const result = await client.enrollFactor({
      did: 'did:mycelix:uhCAk123abc',
      factorType: 'GitcoinPassport',
      factorId: 'passport-123',
      metadata: '{"score":42}',
      reason: 'Verified via Gitcoin',
    });

    expect(result.state.factors).toHaveLength(2);
    expect(result.state.version).toBe(2);
    expect(result.assurance.level).toBe('Verified');
    expect(result.assurance.categoryCount).toBe(2);
  });

  it('should revoke a factor', async () => {
    const result = await client.revokeFactor({
      did: 'did:mycelix:uhCAk123abc',
      factorId: 'some-factor-id',
      reason: 'No longer needed',
    });

    expect(result.state.did).toBe('did:mycelix:uhCAk123abc');
  });

  it('should verify a factor to reset decay', async () => {
    const result = await client.verifyFactor({
      did: 'did:mycelix:uhCAk123abc',
      factorId: 'primary-key-hash',
    });

    expect(result.state.factors[0].effectiveStrength).toBe(1.0);
  });

  it('should calculate assurance', async () => {
    const result = await client.calculateAssurance('did:mycelix:uhCAk123abc');

    expect(result.level).toBe('Basic');
    expect(result.score).toBe(0.25);
    expect(result.staleFactors).toHaveLength(0);
  });

  it('should get enrollment history', async () => {
    const history = await client.getEnrollmentHistory('did:mycelix:uhCAk123abc');

    expect(history).toHaveLength(1);
    expect(history[0].action).toBe('Enroll');
    expect(history[0].factorType).toBe('PrimaryKeyPair');
  });

  it('should check FL eligibility', async () => {
    const result = await client.checkFlEligibility('did:mycelix:uhCAk123abc');

    expect(result.eligible).toBe(false);
    expect(result.denialReasons).toContain('Missing ExternalVerification factor');
  });

  it('should get assurance score from level', () => {
    expect(client.getAssuranceScore('Anonymous')).toBe(0.0);
    expect(client.getAssuranceScore('Basic')).toBe(0.25);
    expect(client.getAssuranceScore('Verified')).toBe(0.5);
    expect(client.getAssuranceScore('HighlyAssured')).toBe(0.75);
    expect(client.getAssuranceScore('ConstitutionallyCritical')).toBe(1.0);
  });

  it('should get factor category from type', () => {
    expect(client.getFactorCategory('PrimaryKeyPair')).toBe('Cryptographic');
    expect(client.getFactorCategory('HardwareKey')).toBe('Cryptographic');
    expect(client.getFactorCategory('Biometric')).toBe('Biometric');
    expect(client.getFactorCategory('SocialRecovery')).toBe('SocialProof');
    expect(client.getFactorCategory('GitcoinPassport')).toBe('ExternalVerification');
    expect(client.getFactorCategory('RecoveryPhrase')).toBe('Knowledge');
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createIdentityClients', () => {
  it('should create all identity clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createIdentityClients(mockZome);

    expect(clients.identity).toBeInstanceOf(IdentityClient);
    expect(clients.schemas).toBeInstanceOf(CredentialSchemaClient);
    expect(clients.revocation).toBeInstanceOf(RevocationClient);
    expect(clients.recovery).toBeInstanceOf(RecoveryClient);
    expect(clients.bridge).toBeInstanceOf(IdentityBridgeClient);
    expect(clients.mfa).toBeInstanceOf(MfaClient);
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce DID format in types', () => {
    const did: DidDocument = {
      id: 'did:mycelix:test',
      controller: 'test',
      verificationMethod: [],
      authentication: [],
      service: [],
      created: 0,
      updated: 0,
      version: 1,
    };

    expect(did.id).toMatch(/^did:mycelix:/);
  });

  it('should enforce schema ID format', () => {
    const schema: CredentialSchema = {
      id: 'mycelix:schema:test:v1',
      name: 'Test',
      description: 'Test schema',
      version: '1.0.0',
      author: 'did:mycelix:test',
      schema: '{}',
      requiredFields: [],
      optionalFields: [],
      credentialType: ['VerifiableCredential'],
      defaultExpiration: 0,
      revocable: true,
      active: true,
      created: 0,
      updated: 0,
    };

    expect(schema.id).toMatch(/^mycelix:schema:/);
  });

  it('should enforce recovery threshold constraints', () => {
    const config: RecoveryConfig = {
      did: 'did:mycelix:test',
      owner: 'test',
      trustees: ['t1', 't2', 't3'],
      threshold: 2,
      timeLock: 86400,
      active: true,
      created: 0,
      updated: 0,
    };

    // Threshold must be <= trustees.length
    expect(config.threshold).toBeLessThanOrEqual(config.trustees.length);
    // Threshold should be majority
    expect(config.threshold).toBeGreaterThanOrEqual(Math.ceil(config.trustees.length / 2));
  });
});

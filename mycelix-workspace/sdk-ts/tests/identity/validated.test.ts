/**
 * Tests for identity/validated.ts
 *
 * Tests validation schemas and validated client wrappers for Identity hApp.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ValidatedIdentityClient,
  ValidatedCredentialSchemaClient,
  ValidatedRevocationClient,
  ValidatedRecoveryClient,
  ValidatedIdentityBridgeClient,
  createValidatedIdentityClients,
} from '../../src/identity/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Helper to create Holochain record structure
const createHolochainRecord = <T>(entry: T) => ({
  entry: { Present: { entry } },
  signed_action: {
    hashed: {
      content: {
        entry_hash: 'test-entry-hash',
        entry_type: 'test-type',
      },
      hash: 'test-action-hash',
    },
  },
});

// Mock zome callable that returns proper Holochain record format
const createMockZomeCallable = () => ({
  callZome: vi.fn().mockImplementation(async ({ fn_name }) => {
    // Return array for list-type functions
    const isArrayReturn =
      fn_name.startsWith('list_') ||
      fn_name.includes('_by_') ||
      fn_name === 'get_recovery_votes' ||
      fn_name === 'get_trustee_responsibilities' ||
      fn_name === 'get_schema_endorsements' ||
      fn_name === 'get_revocations_by_issuer';
    if (isArrayReturn) {
      return [createHolochainRecord({})];
    }
    // Return single record for others
    return createHolochainRecord({});
  }),
});

describe('Identity Validated Clients', () => {
  let mockZomeClient: ReturnType<typeof createMockZomeCallable>;

  beforeEach(() => {
    mockZomeClient = createMockZomeCallable();
    vi.clearAllMocks();
  });

  describe('ValidatedIdentityClient', () => {
    let client: ValidatedIdentityClient;

    beforeEach(() => {
      client = new ValidatedIdentityClient(mockZomeClient);
    });

    describe('createDid', () => {
      it('should not require any validation', async () => {
        await expect(client.createDid()).resolves.toBeDefined();
      });
    });

    describe('getDidDocument', () => {
      it('should accept valid public key (32+ chars)', async () => {
        const validKey = 'a'.repeat(32);
        await expect(client.getDidDocument(validKey)).resolves.toBeDefined();
      });

      it('should reject short public key', async () => {
        await expect(client.getDidDocument('shortkey')).rejects.toThrow(MycelixError);
      });
    });

    describe('resolveDid', () => {
      it('should accept valid DID', async () => {
        await expect(client.resolveDid('did:example:123')).resolves.toBeDefined();
      });

      it('should reject non-DID string', async () => {
        await expect(client.resolveDid('not-a-did')).rejects.toThrow(MycelixError);
      });
    });

    describe('updateDidDocument', () => {
      it('should accept empty update input', async () => {
        await expect(client.updateDidDocument({})).resolves.toBeDefined();
      });

      it('should accept valid verificationMethod', async () => {
        await expect(
          client.updateDidDocument({
            verificationMethod: [
              {
                id: 'key-1',
                type: 'Ed25519VerificationKey2020',
                controller: 'did:example:123',
                publicKeyMultibase: 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
              },
            ],
          })
        ).resolves.toBeDefined();
      });

      it('should reject verificationMethod with invalid controller', async () => {
        await expect(
          client.updateDidDocument({
            verificationMethod: [
              {
                id: 'key-1',
                type: 'Ed25519',
                controller: 'not-a-did',
                publicKeyMultibase: 'key123',
              },
            ],
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept valid service endpoint', async () => {
        await expect(
          client.updateDidDocument({
            service: [
              {
                id: 'service-1',
                type: 'LinkedDomains',
                serviceEndpoint: 'https://example.com',
              },
            ],
          })
        ).resolves.toBeDefined();
      });

      it('should reject service with empty id', async () => {
        await expect(
          client.updateDidDocument({
            service: [
              {
                id: '',
                type: 'LinkedDomains',
                serviceEndpoint: 'https://example.com',
              },
            ],
          })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('deactivateDid', () => {
      it('should accept valid reason', async () => {
        await expect(client.deactivateDid('Compromised key')).resolves.toBeDefined();
      });

      it('should reject empty reason', async () => {
        await expect(client.deactivateDid('')).rejects.toThrow(MycelixError);
      });
    });

    describe('isDidActive', () => {
      it('should accept valid DID', async () => {
        await expect(client.isDidActive('did:example:123')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.isDidActive('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('addServiceEndpoint', () => {
      const validService = {
        id: 'service-1',
        type: 'LinkedDomains',
        serviceEndpoint: 'https://example.com',
      };

      it('should accept valid service', async () => {
        await expect(client.addServiceEndpoint(validService)).resolves.toBeDefined();
      });

      it('should reject empty id', async () => {
        await expect(
          client.addServiceEndpoint({ ...validService, id: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty type_', async () => {
        await expect(
          client.addServiceEndpoint({ ...validService, type: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty serviceEndpoint', async () => {
        await expect(
          client.addServiceEndpoint({ ...validService, serviceEndpoint: '' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('removeServiceEndpoint', () => {
      it('should accept valid serviceId', async () => {
        await expect(client.removeServiceEndpoint('service-1')).resolves.toBeDefined();
      });

      it('should reject empty serviceId', async () => {
        await expect(client.removeServiceEndpoint('')).rejects.toThrow(MycelixError);
      });
    });

    describe('addVerificationMethod', () => {
      const validMethod = {
        id: 'key-1',
        type: 'Ed25519VerificationKey2020',
        controller: 'did:example:controller',
        publicKeyMultibase: 'z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
      };

      it('should accept valid method', async () => {
        await expect(client.addVerificationMethod(validMethod)).resolves.toBeDefined();
      });

      it('should reject invalid controller', async () => {
        await expect(
          client.addVerificationMethod({ ...validMethod, controller: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty id', async () => {
        await expect(
          client.addVerificationMethod({ ...validMethod, id: '' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getMyDid', () => {
      it('should not require validation', async () => {
        await expect(client.getMyDid()).resolves.toBeDefined();
      });
    });
  });

  describe('ValidatedCredentialSchemaClient', () => {
    let client: ValidatedCredentialSchemaClient;

    beforeEach(() => {
      client = new ValidatedCredentialSchemaClient(mockZomeClient);
    });

    describe('createSchema', () => {
      const validSchema = {
        id: 'schema-1',
        name: 'Test Schema',
        description: 'A test schema',
        version: '1.0.0',
        author: 'did:example:author',
        schema: '{}',
        requiredFields: ['field1'],
        optionalFields: ['field2'],
        credentialType: ['VerifiableCredential'],
        defaultExpiration: 86400,
        revocable: true,
        active: true,
        created: Date.now(),
        updated: Date.now(),
      };

      it('should accept valid schema', async () => {
        await expect(client.createSchema(validSchema)).resolves.toBeDefined();
      });

      it('should reject empty name', async () => {
        await expect(
          client.createSchema({ ...validSchema, name: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty version', async () => {
        await expect(
          client.createSchema({ ...validSchema, version: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid author DID', async () => {
        await expect(
          client.createSchema({ ...validSchema, author: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative defaultExpiration', async () => {
        await expect(
          client.createSchema({ ...validSchema, defaultExpiration: -1 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getSchema', () => {
      it('should accept valid schemaId', async () => {
        await expect(client.getSchema('schema-1')).resolves.toBeDefined();
      });

      it('should reject empty schemaId', async () => {
        await expect(client.getSchema('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getSchemasByAuthor', () => {
      it('should accept valid DID', async () => {
        await expect(client.getSchemasByAuthor('did:example:author')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getSchemasByAuthor('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('listActiveSchemas', () => {
      it('should not require validation', async () => {
        await expect(client.listActiveSchemas()).resolves.toBeDefined();
      });
    });

    describe('endorseSchema', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.endorseSchema('schema-1', 'did:example:endorser', 0.8)
        ).resolves.toBeDefined();
      });

      it('should accept optional comment', async () => {
        await expect(
          client.endorseSchema('schema-1', 'did:example:endorser', 0.8, 'Good schema')
        ).resolves.toBeDefined();
      });

      it('should reject empty schemaId', async () => {
        await expect(
          client.endorseSchema('', 'did:example:endorser', 0.8)
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid endorserDid', async () => {
        await expect(
          client.endorseSchema('schema-1', 'invalid', 0.8)
        ).rejects.toThrow(MycelixError);
      });

      it('should reject trustLevel below 0', async () => {
        await expect(
          client.endorseSchema('schema-1', 'did:example:endorser', -0.1)
        ).rejects.toThrow(MycelixError);
      });

      it('should reject trustLevel above 1', async () => {
        await expect(
          client.endorseSchema('schema-1', 'did:example:endorser', 1.1)
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getSchemaEndorsements', () => {
      it('should accept valid schemaId', async () => {
        await expect(client.getSchemaEndorsements('schema-1')).resolves.toBeDefined();
      });

      it('should reject empty schemaId', async () => {
        await expect(client.getSchemaEndorsements('')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedRevocationClient', () => {
    let client: ValidatedRevocationClient;

    beforeEach(() => {
      client = new ValidatedRevocationClient(mockZomeClient);
    });

    describe('revoke', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.revoke('cred-1', 'did:example:issuer', 'Compromised')
        ).resolves.toBeDefined();
      });

      it('should accept optional effectiveFrom', async () => {
        await expect(
          client.revoke('cred-1', 'did:example:issuer', 'Compromised', Date.now())
        ).resolves.toBeDefined();
      });

      it('should reject empty credentialId', async () => {
        await expect(
          client.revoke('', 'did:example:issuer', 'Compromised')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid issuerDid', async () => {
        await expect(
          client.revoke('cred-1', 'invalid', 'Compromised')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty reason', async () => {
        await expect(
          client.revoke('cred-1', 'did:example:issuer', '')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject non-positive effectiveFrom', async () => {
        await expect(
          client.revoke('cred-1', 'did:example:issuer', 'Compromised', 0)
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('suspend', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.suspend('cred-1', 'did:example:issuer', 'Pending review', Date.now() + 86400000)
        ).resolves.toBeDefined();
      });

      it('should reject empty credentialId', async () => {
        await expect(
          client.suspend('', 'did:example:issuer', 'Pending review', Date.now())
        ).rejects.toThrow(MycelixError);
      });

      it('should reject non-positive suspensionEnd', async () => {
        await expect(
          client.suspend('cred-1', 'did:example:issuer', 'Pending review', 0)
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('reinstate', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.reinstate('cred-1', 'did:example:issuer', 'Review passed')
        ).resolves.toBeDefined();
      });

      it('should reject empty reason', async () => {
        await expect(
          client.reinstate('cred-1', 'did:example:issuer', '')
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('checkStatus', () => {
      it('should accept valid credentialId', async () => {
        await expect(client.checkStatus('cred-1')).resolves.toBeDefined();
      });

      it('should reject empty credentialId', async () => {
        await expect(client.checkStatus('')).rejects.toThrow(MycelixError);
      });
    });

    describe('batchCheckStatus', () => {
      it('should accept valid credentialIds array', async () => {
        await expect(client.batchCheckStatus(['cred-1', 'cred-2'])).resolves.toBeDefined();
      });

      it('should reject empty array', async () => {
        await expect(client.batchCheckStatus([])).rejects.toThrow(MycelixError);
      });

      it('should reject array with empty string', async () => {
        await expect(client.batchCheckStatus(['cred-1', ''])).rejects.toThrow(MycelixError);
      });
    });

    describe('getByIssuer', () => {
      it('should accept valid DID', async () => {
        await expect(client.getByIssuer('did:example:issuer')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getByIssuer('invalid')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedRecoveryClient', () => {
    let client: ValidatedRecoveryClient;

    beforeEach(() => {
      client = new ValidatedRecoveryClient(mockZomeClient);
    });

    describe('setupRecovery', () => {
      const validInput = {
        did: 'did:example:user',
        trustees: ['did:example:trustee1', 'did:example:trustee2'],
        threshold: 2,
      };

      it('should accept valid input', async () => {
        await expect(client.setupRecovery(validInput)).resolves.toBeDefined();
      });

      it('should accept optional timeLock', async () => {
        await expect(
          client.setupRecovery({ ...validInput, timeLock: 86400 })
        ).resolves.toBeDefined();
      });

      it('should reject invalid did', async () => {
        await expect(
          client.setupRecovery({ ...validInput, did: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty trustees array', async () => {
        await expect(
          client.setupRecovery({ ...validInput, trustees: [] })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid trustee DID', async () => {
        await expect(
          client.setupRecovery({ ...validInput, trustees: ['invalid'] })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject threshold less than 1', async () => {
        await expect(
          client.setupRecovery({ ...validInput, threshold: 0 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative timeLock', async () => {
        await expect(
          client.setupRecovery({ ...validInput, timeLock: -1 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getRecoveryConfig', () => {
      it('should accept valid DID', async () => {
        await expect(client.getRecoveryConfig('did:example:user')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getRecoveryConfig('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('initiateRecovery', () => {
      const validInput = {
        did: 'did:example:user',
        initiatorDid: 'did:example:trustee',
        newAgent: 'a'.repeat(32),
        reason: 'Lost access to device',
      };

      it('should accept valid input', async () => {
        await expect(client.initiateRecovery(validInput)).resolves.toBeDefined();
      });

      it('should reject invalid did', async () => {
        await expect(
          client.initiateRecovery({ ...validInput, did: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject short newAgent', async () => {
        await expect(
          client.initiateRecovery({ ...validInput, newAgent: 'short' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty reason', async () => {
        await expect(
          client.initiateRecovery({ ...validInput, reason: '' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('voteOnRecovery', () => {
      const validInput = {
        requestId: 'request-1',
        trusteeDid: 'did:example:trustee',
        vote: 'Approve' as const,
      };

      it('should accept valid input', async () => {
        await expect(client.voteOnRecovery(validInput)).resolves.toBeDefined();
      });

      it('should accept Reject vote', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, vote: 'Reject' })
        ).resolves.toBeDefined();
      });

      it('should accept Abstain vote', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, vote: 'Abstain' })
        ).resolves.toBeDefined();
      });

      it('should accept optional comment', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, comment: 'Verified via call' })
        ).resolves.toBeDefined();
      });

      it('should reject empty requestId', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, requestId: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid trusteeDid', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, trusteeDid: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid vote', async () => {
        await expect(
          client.voteOnRecovery({ ...validInput, vote: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getRecoveryVotes', () => {
      it('should accept valid requestId', async () => {
        await expect(client.getRecoveryVotes('request-1')).resolves.toBeDefined();
      });

      it('should reject empty requestId', async () => {
        await expect(client.getRecoveryVotes('')).rejects.toThrow(MycelixError);
      });
    });

    describe('executeRecovery', () => {
      it('should accept valid requestId', async () => {
        await expect(client.executeRecovery('request-1')).resolves.toBeDefined();
      });

      it('should reject empty requestId', async () => {
        await expect(client.executeRecovery('')).rejects.toThrow(MycelixError);
      });
    });

    describe('cancelRecovery', () => {
      it('should accept valid requestId', async () => {
        await expect(client.cancelRecovery('request-1')).resolves.toBeDefined();
      });

      it('should reject empty requestId', async () => {
        await expect(client.cancelRecovery('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getTrusteeResponsibilities', () => {
      it('should accept valid DID', async () => {
        await expect(
          client.getTrusteeResponsibilities('did:example:trustee')
        ).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getTrusteeResponsibilities('invalid')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedIdentityBridgeClient', () => {
    let client: ValidatedIdentityBridgeClient;

    beforeEach(() => {
      client = new ValidatedIdentityBridgeClient(mockZomeClient);
    });

    describe('verifyIdentity', () => {
      const validInput = {
        did: 'did:example:user',
        sourceHapp: 'marketplace',
        requestedFields: ['name', 'email'],
      };

      it('should accept valid input', async () => {
        await expect(client.verifyIdentity(validInput)).resolves.toBeDefined();
      });

      it('should reject invalid did', async () => {
        await expect(
          client.verifyIdentity({ ...validInput, did: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty sourceHapp', async () => {
        await expect(
          client.verifyIdentity({ ...validInput, sourceHapp: '' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('verifyDid', () => {
      it('should accept valid DID', async () => {
        await expect(client.verifyDid('did:example:user')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.verifyDid('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('getMatlScore', () => {
      it('should accept valid DID', async () => {
        await expect(client.getMatlScore('did:example:user')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getMatlScore('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('isTrustworthy', () => {
      it('should accept valid inputs', async () => {
        await expect(client.isTrustworthy('did:example:user', 0.7)).resolves.toBeDefined();
      });

      it('should reject invalid DID', async () => {
        await expect(client.isTrustworthy('invalid', 0.7)).rejects.toThrow(MycelixError);
      });

      it('should reject threshold below 0', async () => {
        await expect(client.isTrustworthy('did:example:user', -0.1)).rejects.toThrow(MycelixError);
      });

      it('should reject threshold above 1', async () => {
        await expect(client.isTrustworthy('did:example:user', 1.5)).rejects.toThrow(MycelixError);
      });
    });

    describe('getReputation', () => {
      it('should accept valid DID', async () => {
        await expect(client.getReputation('did:example:user')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getReputation('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('reportReputation', () => {
      const validInput = {
        did: 'did:example:user',
        sourceHapp: 'marketplace',
        score: 0.85,
        interactions: 10,
      };

      it('should accept valid input', async () => {
        await expect(client.reportReputation(validInput)).resolves.toBeUndefined();
      });

      it('should reject invalid did', async () => {
        await expect(
          client.reportReputation({ ...validInput, did: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty sourceHapp', async () => {
        await expect(
          client.reportReputation({ ...validInput, sourceHapp: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject score below 0', async () => {
        await expect(
          client.reportReputation({ ...validInput, score: -0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject score above 1', async () => {
        await expect(
          client.reportReputation({ ...validInput, score: 1.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative interactions', async () => {
        await expect(
          client.reportReputation({ ...validInput, interactions: -1 })
        ).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('createValidatedIdentityClients', () => {
    it('should create all validated clients', () => {
      const clients = createValidatedIdentityClients(mockZomeClient);

      expect(clients.identity).toBeInstanceOf(ValidatedIdentityClient);
      expect(clients.schemas).toBeInstanceOf(ValidatedCredentialSchemaClient);
      expect(clients.revocation).toBeInstanceOf(ValidatedRevocationClient);
      expect(clients.recovery).toBeInstanceOf(ValidatedRecoveryClient);
      expect(clients.bridge).toBeInstanceOf(ValidatedIdentityBridgeClient);
    });
  });

  describe('Error handling', () => {
    let client: ValidatedIdentityClient;

    beforeEach(() => {
      client = new ValidatedIdentityClient(mockZomeClient);
    });

    it('should have INVALID_ARGUMENT error code', async () => {
      try {
        await client.resolveDid('invalid');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(MycelixError);
        expect((error as MycelixError).code).toBe(ErrorCode.INVALID_ARGUMENT);
      }
    });

    it('should include context in error message', async () => {
      try {
        await client.resolveDid('invalid');
        expect.fail('Should have thrown');
      } catch (error) {
        expect((error as MycelixError).message).toContain('did');
      }
    });
  });
});

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity hApp Conductor Integration Tests
 *
 * These tests verify the Identity clients work correctly with a real
 * Holochain conductor. They require the conductor harness to be available.
 *
 * Run with: npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  IdentityClient,
  CredentialSchemaClient,
  RevocationClient,
  RecoveryClient,
  IdentityBridgeClient,
  createIdentityClients,
  type ZomeCallable,
  type DidDocument,
} from '../../src/identity/index.js';
import {
  createValidatedIdentityClients,
  ValidatedIdentityClient,
} from '../../src/identity/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Conductor harness (dynamically imported if available)
let conductorHarness: {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  createClient: () => ZomeCallable;
  getAgentPubKey: () => string;
} | null = null;

let client: ZomeCallable;
let agentPubKey: string;

// Skip all tests if conductor is not available
const describeConductor = process.env.HOLOCHAIN_CONDUCTOR_AVAILABLE ? describe : describe.skip;

describeConductor('Identity Conductor Integration Tests', () => {
  beforeAll(async () => {
    try {
      // Try to import the conductor harness
      const harness = await import('./conductor-harness.js');
      conductorHarness = harness.default || harness;
      await conductorHarness.start();
      client = conductorHarness.createClient();
      agentPubKey = conductorHarness.getAgentPubKey();
    } catch (error) {
      console.log('Conductor not available, skipping integration tests');
    }
  }, 60000); // 60 second timeout for conductor startup

  afterAll(async () => {
    if (conductorHarness) {
      await conductorHarness.stop();
    }
  }, 30000);

  describe('IdentityClient', () => {
    let identityClient: IdentityClient;

    beforeAll(() => {
      identityClient = new IdentityClient(client);
    });

    it('should create a DID', async () => {
      const result = await identityClient.createDid();

      expect(result).toBeDefined();
      expect(result.signed_action).toBeDefined();
      expect(result.entry).toBeDefined();

      const doc = result.entry.Present as DidDocument;
      expect(doc.id).toMatch(/^did:mycelix:/);
      expect(doc.controller).toBe(doc.id);
      expect(doc.verificationMethod).toBeDefined();
      expect(doc.verificationMethod.length).toBeGreaterThan(0);
    });

    it('should get my DID document', async () => {
      // First create a DID if none exists
      await identityClient.createDid().catch(() => {});

      const result = await identityClient.getMyDid();

      expect(result).toBeDefined();
      if (result) {
        const doc = result.entry.Present as DidDocument;
        expect(doc.id).toMatch(/^did:mycelix:/);
      }
    });

    it('should resolve a DID', async () => {
      // First create a DID
      const created = await identityClient.createDid();
      const did = (created.entry.Present as DidDocument).id;

      const result = await identityClient.resolveDid(did);

      expect(result).toBeDefined();
      if (result) {
        const doc = result.entry.Present as DidDocument;
        expect(doc.id).toBe(did);
      }
    });

    it('should check if DID is active', async () => {
      const created = await identityClient.createDid();
      const did = (created.entry.Present as DidDocument).id;

      const isActive = await identityClient.isDidActive(did);

      expect(typeof isActive).toBe('boolean');
      expect(isActive).toBe(true);
    });

    it('should add a service endpoint', async () => {
      const service = {
        id: `${agentPubKey}#my-service`,
        type: 'MycelixService',
        serviceEndpoint: 'https://example.com/api',
      };

      const result = await identityClient.addServiceEndpoint(service);

      expect(result).toBeDefined();
      const doc = result.entry.Present as DidDocument;
      expect(doc.service).toBeDefined();
      expect(doc.service?.some((s) => s.id === service.id)).toBe(true);
    });

    it('should add a verification method', async () => {
      const myDid = await identityClient.getMyDid();
      const did = (myDid?.entry.Present as DidDocument).id;

      const method = {
        id: `${did}#key-2`,
        type: 'Ed25519VerificationKey2020',
        controller: did,
        publicKeyMultibase: 'zH3C2AVvLMv6gmMNam3uVAjZpfkcJCwDwnZn6z3wXmqPV',
      };

      const result = await identityClient.addVerificationMethod(method);

      expect(result).toBeDefined();
      const doc = result.entry.Present as DidDocument;
      expect(doc.verificationMethod.some((m: { id: string }) => m.id === method.id)).toBe(true);
    });
  });

  describe('CredentialSchemaClient', () => {
    let schemaClient: CredentialSchemaClient;

    beforeAll(() => {
      schemaClient = new CredentialSchemaClient(client);
    });

    it('should register a credential schema', async () => {
      const schema = {
        id: `schema-${Date.now()}`,
        name: 'Test Credential',
        version: '1.0.0',
        description: 'A test credential schema',
        category: 'Custom' as const,
        properties: {
          name: { type: 'string' },
          age: { type: 'number' },
        },
        required_properties: ['name'],
        author_did: `did:mycelix:${agentPubKey}`,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      const result = await schemaClient.registerSchema(schema);

      expect(result).toBeDefined();
      expect(result.entry.Present).toMatchObject({
        id: schema.id,
        name: schema.name,
        version: schema.version,
      });
    });

    it('should get schemas by category', async () => {
      const result = await schemaClient.getSchemasByCategory('Custom');

      expect(Array.isArray(result)).toBe(true);
    });

    it('should endorse a schema', async () => {
      // First create a schema
      const schema = {
        id: `schema-endorse-${Date.now()}`,
        name: 'Endorsable Schema',
        version: '1.0.0',
        description: 'Schema for endorsement test',
        category: 'Education' as const,
        properties: {},
        required_properties: [],
        author_did: `did:mycelix:${agentPubKey}`,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      const created = await schemaClient.registerSchema(schema);

      // Now endorse it
      const endorsement = await schemaClient.endorseSchema(schema.id);

      expect(endorsement).toBeDefined();
    });
  });

  describe('RevocationClient', () => {
    let revocationClient: RevocationClient;

    beforeAll(() => {
      revocationClient = new RevocationClient(client);
    });

    it('should revoke a credential', async () => {
      const input = {
        credential_id: `cred-${Date.now()}`,
        reason: 'Testing revocation',
        status: 'Revoked' as const,
      };

      const result = await revocationClient.revokeCredential(input);

      expect(result).toBeDefined();
    });

    it('should check revocation status', async () => {
      const credId = `cred-check-${Date.now()}`;

      // First revoke
      await revocationClient.revokeCredential({
        credential_id: credId,
        reason: 'Test',
        status: 'Suspended',
      });

      // Then check
      const result = await revocationClient.checkRevocation(credId);

      expect(result).toBeDefined();
    });

    it('should update revocation status', async () => {
      const credId = `cred-update-${Date.now()}`;

      // First revoke with suspended status
      await revocationClient.revokeCredential({
        credential_id: credId,
        reason: 'Initial suspension',
        status: 'Suspended',
      });

      // Then update to fully revoked
      const result = await revocationClient.updateRevocationStatus({
        credential_id: credId,
        new_status: 'Revoked',
        reason: 'Changed to full revocation',
      });

      expect(result).toBeDefined();
    });
  });

  describe('RecoveryClient', () => {
    let recoveryClient: RecoveryClient;

    beforeAll(() => {
      recoveryClient = new RecoveryClient(client);
    });

    it('should configure recovery', async () => {
      const config = {
        did: `did:mycelix:${agentPubKey}`,
        trustees: ['did:mycelix:trustee1', 'did:mycelix:trustee2', 'did:mycelix:trustee3'],
        threshold: 67, // 2 of 3
        timeout_days: 30,
        created: Date.now() * 1000,
      };

      const result = await recoveryClient.configureRecovery(config);

      expect(result).toBeDefined();
    });

    it('should get recovery config', async () => {
      const did = `did:mycelix:${agentPubKey}`;

      const result = await recoveryClient.getRecoveryConfig(did);

      // May or may not exist depending on previous test
      if (result) {
        expect(result.entry.Present).toHaveProperty('did');
        expect(result.entry.Present).toHaveProperty('trustees');
      }
    });
  });

  describe('IdentityBridgeClient', () => {
    let bridgeClient: IdentityBridgeClient;

    beforeAll(() => {
      bridgeClient = new IdentityBridgeClient(client);
    });

    it('should register a hApp', async () => {
      const result = await bridgeClient.registerHapp({
        happ_name: 'test-marketplace',
        capabilities: ['identity_query', 'reputation_report'],
      });

      expect(result).toBeDefined();
    });

    it('should get registered hApps', async () => {
      const result = await bridgeClient.getRegisteredHapps();

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get reputation for a DID', async () => {
      const did = `did:mycelix:${agentPubKey}`;

      const result = await bridgeClient.getReputation(did);

      expect(result).toBeDefined();
    });

    it('should get recent events', async () => {
      const result = await bridgeClient.getRecentEvents(10);

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Validated Clients', () => {
    it('should reject invalid DID format', async () => {
      const clients = createValidatedIdentityClients(client);

      await expect(clients.identity.resolveDid('invalid-did')).rejects.toThrow(MycelixError);
    });

    it('should reject invalid public key', async () => {
      const clients = createValidatedIdentityClients(client);

      await expect(clients.identity.getDidDocument('short')).rejects.toThrow(MycelixError);
    });

    it('should accept valid inputs', async () => {
      const clients = createValidatedIdentityClients(client);

      // This should not throw validation errors
      const result = await clients.identity.resolveDid(`did:mycelix:${agentPubKey}`);

      // Result may be null if DID doesn't exist, but should not throw
      expect(result === null || result !== undefined).toBe(true);
    });
  });

  describe('createIdentityClients factory', () => {
    it('should create all clients', () => {
      const clients = createIdentityClients(client);

      expect(clients.identity).toBeInstanceOf(IdentityClient);
      expect(clients.schemas).toBeInstanceOf(CredentialSchemaClient);
      expect(clients.revocation).toBeInstanceOf(RevocationClient);
      expect(clients.recovery).toBeInstanceOf(RecoveryClient);
      expect(clients.bridge).toBeInstanceOf(IdentityBridgeClient);
    });
  });
});

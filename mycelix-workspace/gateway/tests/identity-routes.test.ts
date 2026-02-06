/**
 * Identity Routes Tests
 *
 * Tests for the identity hApp REST API endpoints.
 */

import { describe, expect, it, beforeAll, afterAll, vi } from 'vitest';
import { Hono } from 'hono';
import { identityRoutes } from '../src/routes/identity';
import { MockHolochainService, setHolochainService } from '../src/services/holochain';

// Mock the rate limiters to avoid Redis dependency
vi.mock('../src/middleware/rateLimit', () => ({
  standardLimit: vi.fn((c: any, next: any) => next()),
  strictLimit: vi.fn((c: any, next: any) => next()),
  authLimit: vi.fn((c: any, next: any) => next()),
}));

// Mock auth middleware for testing
vi.mock('../src/middleware/auth', () => ({
  authRequired: vi.fn((c: any, next: any) => {
    c.set('authDid', 'did:mycelix:testuser');
    return next();
  }),
  authOptional: vi.fn((c: any, next: any) => next()),
  requireDid: vi.fn(() => (c: any, next: any) => next()),
  getAuthDid: vi.fn(() => 'did:mycelix:testuser'),
  createToken: vi.fn(async () => 'mock-jwt-token'),
}));

describe('Identity Routes', () => {
  let app: Hono;
  let mockService: MockHolochainService;

  beforeAll(async () => {
    mockService = new MockHolochainService();
    await mockService.connect();
    setHolochainService(mockService);

    app = new Hono();
    app.route('/identity', identityRoutes);
  });

  afterAll(() => {
    vi.clearAllMocks();
  });

  // ==========================================================================
  // Profile Operations
  // ==========================================================================

  describe('GET /identity/profile/:did', () => {
    it('returns profile for valid DID', async () => {
      const res = await app.request('/identity/profile/did:mycelix:mock123');

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data.did).toBe('did:mycelix:mock123');
      expect(body.data.displayName).toBe('Test User');
    });

    it('includes meta information in response', async () => {
      const res = await app.request('/identity/profile/did:mycelix:mock123');

      const body = await res.json();
      expect(body.meta).toBeDefined();
      expect(body.meta.requestId).toBeDefined();
      expect(body.meta.timestamp).toBeGreaterThan(0);
    });
  });

  describe('POST /identity/profile', () => {
    it('creates profile with valid data', async () => {
      const res = await app.request('/identity/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          displayName: 'New User',
          bio: 'A new test user',
        }),
      });

      expect(res.status).toBe(201);
      const body = await res.json();
      expect(body.success).toBe(true);
    });

    it('accepts optional avatar URL', async () => {
      const res = await app.request('/identity/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          displayName: 'User With Avatar',
          avatar: 'https://example.com/avatar.png',
        }),
      });

      expect(res.status).toBe(201);
    });

    it('rejects displayName over 100 characters', async () => {
      const res = await app.request('/identity/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          displayName: 'A'.repeat(101),
        }),
      });

      expect(res.status).toBe(400);
    });

    it('rejects bio over 500 characters', async () => {
      const res = await app.request('/identity/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          displayName: 'Valid Name',
          bio: 'B'.repeat(501),
        }),
      });

      expect(res.status).toBe(400);
    });
  });

  // ==========================================================================
  // Credential Operations
  // ==========================================================================

  describe('GET /identity/credentials/:did', () => {
    it('returns credentials for authorized user', async () => {
      const res = await app.request('/identity/credentials/did:mycelix:testuser');

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.success).toBe(true);
    });
  });

  describe('POST /identity/credentials/issue', () => {
    it('issues credential with valid input', async () => {
      const res = await app.request('/identity/credentials/issue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subjectDid: 'did:mycelix:subject123',
          credentialType: ['VerifiableCredential', 'MycelixMemberCredential'],
          claims: { memberSince: '2024-01-01' },
        }),
      });

      expect(res.status).toBe(201);
      const body = await res.json();
      expect(body.success).toBe(true);
    });

    it('requires at least one credential type', async () => {
      const res = await app.request('/identity/credentials/issue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subjectDid: 'did:mycelix:subject123',
          credentialType: [],
          claims: {},
        }),
      });

      expect(res.status).toBe(400);
    });

    it('validates DID format', async () => {
      const res = await app.request('/identity/credentials/issue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subjectDid: 'invalid-did-format',
          credentialType: ['VerifiableCredential'],
          claims: {},
        }),
      });

      expect(res.status).toBe(400);
    });
  });

  describe('POST /identity/credentials/verify', () => {
    it('verifies valid credential structure', async () => {
      const res = await app.request('/identity/credentials/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          credential: {
            id: 'urn:uuid:3978344f-8596-4c3a-a978-8fcaba3903c5',
            type: ['VerifiableCredential'],
            issuer: 'did:mycelix:issuer123',
            issuanceDate: '2024-01-01T00:00:00Z',
            credentialSubject: {
              id: 'did:mycelix:subject123',
            },
            proof: {
              type: 'Ed25519Signature2020',
              proofPurpose: 'assertionMethod',
              verificationMethod: 'did:mycelix:issuer123#key-1',
              created: '2024-01-01T00:00:00Z',
              proofValue: 'mockProofValue',
            },
          },
        }),
      });

      expect(res.status).toBe(200);
    });
  });

  // ==========================================================================
  // Authentication
  // ==========================================================================

  describe('POST /identity/auth/challenge', () => {
    it('returns challenge with expiration', async () => {
      const res = await app.request('/identity/auth/challenge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          did: 'did:mycelix:user123',
          pubkey: 'uhCAkMOCKPUBKEY123456789012345678901234567890',
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data.challenge).toBeDefined();
      expect(body.data.expiresAt).toBeGreaterThan(Date.now());
    });
  });

  describe('POST /identity/auth/verify', () => {
    it('returns token on successful verification', async () => {
      const res = await app.request('/identity/auth/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          did: 'did:mycelix:user123',
          pubkey: 'uhCAkMOCKPUBKEY123456789012345678901234567890',
          challenge: 'test-challenge',
          signature: 'mock-signature',
        }),
      });

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.success).toBe(true);
      expect(body.data.token).toBeDefined();
      expect(body.data.expiresIn).toBe(86400);
    });
  });

  // ==========================================================================
  // MATL Reputation
  // ==========================================================================

  describe('GET /identity/reputation/:did', () => {
    it('returns reputation for valid DID', async () => {
      const res = await app.request('/identity/reputation/did:mycelix:mock123');

      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.success).toBe(true);
    });
  });
});

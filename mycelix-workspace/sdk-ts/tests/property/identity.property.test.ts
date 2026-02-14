/**
 * Property-Based Tests for Identity Module
 *
 * Uses fast-check to verify Identity invariants hold across all valid inputs.
 * Tests DID format validation, credential schemas, recovery thresholds, and more.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import fc from 'fast-check';
import {
  IdentityService,
  resetIdentityService,
  type IdentityProfile,
  type DIDDocument,
  type VerifiableCredential,
  type VerificationLevel,
} from '../../src/integrations/identity/index.js';

describe('Identity Property-Based Tests', () => {
  let service: IdentityService;

  // Generate valid public key (32+ hex characters)
  // Use array of hex chars to avoid filter timeouts
  const hexChars = '0123456789abcdef';
  const publicKeyArb = fc
    .array(fc.constantFrom(...hexChars.split('')), { minLength: 32, maxLength: 64 })
    .map((chars) => chars.join(''));

  // Generate valid credential type (use constant choices for reliability)
  const credentialTypeArb = fc.constantFrom(
    'MembershipCredential',
    'EducationCredential',
    'EmploymentCredential',
    'SkillCredential',
    'IdentityCredential'
  );

  // Generate claim keys (use constant choices for reliability)
  const claimKeyArb = fc.constantFrom(
    'name',
    'email',
    'value',
    'score',
    'level',
    'status',
    'date',
    'role'
  );

  // Generate claim values
  const claimValueArb = fc.oneof(
    fc.string({ minLength: 1, maxLength: 50 }),
    fc.integer({ min: 0, max: 1000 }),
    fc.boolean()
  );

  // Generate claims object (simpler to avoid filter issues)
  const claimsArb = fc
    .tuple(claimKeyArb, claimValueArb)
    .map(([key, value]) => ({ [key]: value }));

  beforeEach(() => {
    resetIdentityService();
    service = new IdentityService();
  });

  describe('DID Format Properties', () => {
    it('all DIDs should follow did:mycelix format', { timeout: 30000 }, () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);

          // DID must start with did:mycelix:
          expect(profile.did).toMatch(/^did:mycelix:/);

          // DID must be derived from public key
          expect(profile.did).toContain(pubKey.slice(0, 32));
        }),
        { numRuns: 20 }
      );
    });

    it('identical public keys should produce same DID', { timeout: 30000 }, () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          // Reset service to ensure clean state
          resetIdentityService();
          const service1 = new IdentityService();
          const profile1 = service1.createIdentity(pubKey);

          resetIdentityService();
          const service2 = new IdentityService();
          const profile2 = service2.createIdentity(pubKey);

          expect(profile1.did).toBe(profile2.did);
        }),
        { numRuns: 20 }
      );
    });

    it('different public keys should produce different DIDs', { timeout: 30000 }, () => {
      fc.assert(
        fc.property(publicKeyArb, publicKeyArb, (pubKey1, pubKey2) => {
          fc.pre(pubKey1.slice(0, 32) !== pubKey2.slice(0, 32));

          resetIdentityService();
          const svc = new IdentityService();
          const profile1 = svc.createIdentity(pubKey1);
          const profile2 = svc.createIdentity(pubKey2);

          expect(profile1.did).not.toBe(profile2.did);
        }),
        { numRuns: 20 }
      );
    });
  });

  describe('DID Document Properties', () => {
    it('DID document should have valid structure', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);
          const doc = profile.document;

          // Required fields
          expect(doc.id).toBe(profile.did);
          expect(doc.controller).toBe(profile.did);
          expect(doc.verificationMethod).toBeDefined();
          expect(Array.isArray(doc.verificationMethod)).toBe(true);
          expect(doc.verificationMethod.length).toBeGreaterThan(0);
          expect(doc.authentication).toBeDefined();
          expect(Array.isArray(doc.authentication)).toBe(true);
          expect(doc.authentication.length).toBeGreaterThan(0);
        }),
        { numRuns: 15 }
      );
    });

    it('verification method should have proper key format', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);
          const method = profile.document.verificationMethod[0];

          // ID should reference the DID
          expect(method.id).toContain(profile.did);

          // Type should be Ed25519
          expect(method.type).toBe('Ed25519VerificationKey2020');

          // Controller should match DID
          expect(method.controller).toBe(profile.did);

          // Public key should be multibase encoded (starts with 'z')
          expect(method.publicKeyMultibase).toMatch(/^z/);
        }),
        { numRuns: 15 }
      );
    });

    it('created timestamp should be valid', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const before = Date.now();
          const profile = service.createIdentity(pubKey);
          const after = Date.now();

          expect(profile.document.created).toBeGreaterThanOrEqual(before);
          expect(profile.document.created).toBeLessThanOrEqual(after);
        }),
        { numRuns: 20 }
      );
    });
  });

  describe('Credential Properties', () => {
    it('issued credentials should have valid W3C structure', () => {
      fc.assert(
        fc.property(
          publicKeyArb,
          publicKeyArb,
          credentialTypeArb,
          claimsArb,
          (issuerKey, subjectKey, credType, claims) => {
            fc.pre(issuerKey.slice(0, 32) !== subjectKey.slice(0, 32));

            resetIdentityService();
            const svc = new IdentityService();
            const issuer = svc.createIdentity(issuerKey);
            const subject = svc.createIdentity(subjectKey);

            const credential = svc.issueCredential(issuer.did, subject.did, credType, claims);

            // W3C Verifiable Credential structure
            expect(credential['@context']).toBeDefined();
            expect(Array.isArray(credential['@context'])).toBe(true);
            expect(credential['@context']).toContain('https://www.w3.org/2018/credentials/v1');

            // ID should be unique
            expect(credential.id).toMatch(/^vc-/);

            // Type array should include VerifiableCredential
            expect(Array.isArray(credential.type)).toBe(true);
            expect(credential.type).toContain('VerifiableCredential');
            expect(credential.type).toContain(credType);

            // Issuer should match
            expect(credential.issuer).toBe(issuer.did);

            // Issuance date should be valid ISO string
            expect(new Date(credential.issuanceDate).getTime()).toBeLessThanOrEqual(Date.now());

            // Credential subject should have id
            expect(credential.credentialSubject.id).toBe(subject.did);
          }
        ),
        { numRuns: 20 }
      );
    });

    it('credentials with expiration should have valid future date', () => {
      fc.assert(
        fc.property(
          publicKeyArb,
          publicKeyArb,
          fc.integer({ min: 1, max: 365 }),
          (issuerKey, subjectKey, days) => {
            fc.pre(issuerKey.slice(0, 32) !== subjectKey.slice(0, 32));

            resetIdentityService();
            const svc = new IdentityService();
            const issuer = svc.createIdentity(issuerKey);
            const subject = svc.createIdentity(subjectKey);

            const credential = svc.issueCredential(
              issuer.did,
              subject.did,
              'ExpiringCredential',
              { value: 1 },
              days
            );

            expect(credential.expirationDate).toBeDefined();
            const expiry = new Date(credential.expirationDate!).getTime();
            expect(expiry).toBeGreaterThan(Date.now());
            expect(expiry).toBeLessThanOrEqual(Date.now() + days * 24 * 60 * 60 * 1000 + 1000);
          }
        ),
        { numRuns: 20 }
      );
    });

    it('issued credentials should be added to subject profile', () => {
      fc.assert(
        fc.property(
          publicKeyArb,
          publicKeyArb,
          fc.integer({ min: 1, max: 5 }),
          (issuerKey, subjectKey, numCreds) => {
            fc.pre(issuerKey.slice(0, 32) !== subjectKey.slice(0, 32));

            resetIdentityService();
            const svc = new IdentityService();
            const issuer = svc.createIdentity(issuerKey);
            const subject = svc.createIdentity(subjectKey);

            for (let i = 0; i < numCreds; i++) {
              svc.issueCredential(issuer.did, subject.did, `TestCredential${i}`, { index: i });
            }

            const profile = svc.getProfile(subject.did);
            expect(profile!.credentials.length).toBe(numCreds);
          }
        ),
        { numRuns: 15 }
      );
    });
  });

  describe('Reputation Properties', () => {
    it('new identities should start with neutral reputation (Laplace prior)', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);

          expect(profile.reputation).toBeDefined();
          expect(profile.reputation.agentId).toBe(profile.did);
          // Laplace smoothing: starts with 1/1 prior (neutral 0.5)
          expect(profile.reputation.positiveCount).toBe(1);
          expect(profile.reputation.negativeCount).toBe(1);
        }),
        { numRuns: 15 }
      );
    });

    it('credentials should increase positive reputation', () => {
      fc.assert(
        fc.property(publicKeyArb, publicKeyArb, (issuerKey, subjectKey) => {
          fc.pre(issuerKey.slice(0, 32) !== subjectKey.slice(0, 32));

          resetIdentityService();
          const svc = new IdentityService();
          const issuer = svc.createIdentity(issuerKey);
          const subject = svc.createIdentity(subjectKey);

          const repBefore = subject.reputation.positiveCount;
          svc.issueCredential(issuer.did, subject.did, 'BoostCredential', {});

          const profile = svc.getProfile(subject.did);
          expect(profile!.reputation.positiveCount).toBeGreaterThan(repBefore);
        }),
        { numRuns: 20 }
      );
    });

    it('trust attestations should increase positive reputation', () => {
      fc.assert(
        fc.property(publicKeyArb, publicKeyArb, (attesterKey, subjectKey) => {
          fc.pre(attesterKey.slice(0, 32) !== subjectKey.slice(0, 32));

          resetIdentityService();
          const svc = new IdentityService();
          const attester = svc.createIdentity(attesterKey);
          const subject = svc.createIdentity(subjectKey);

          const repBefore = subject.reputation.positiveCount;
          svc.attestTrust(attester.did, subject.did);

          const profile = svc.getProfile(subject.did);
          expect(profile!.reputation.positiveCount).toBeGreaterThan(repBefore);
        }),
        { numRuns: 20 }
      );
    });
  });

  describe('Verification Level Properties', () => {
    it('new identities should start as self_attested', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);
          expect(profile.verificationLevel).toBe('self_attested');
        }),
        { numRuns: 15 }
      );
    });

    it('credentials should upgrade to credential_backed', () => {
      fc.assert(
        fc.property(publicKeyArb, publicKeyArb, (issuerKey, subjectKey) => {
          fc.pre(issuerKey.slice(0, 32) !== subjectKey.slice(0, 32));

          resetIdentityService();
          const svc = new IdentityService();
          const issuer = svc.createIdentity(issuerKey);
          const subject = svc.createIdentity(subjectKey);

          expect(subject.verificationLevel).toBe('self_attested');

          svc.issueCredential(issuer.did, subject.did, 'UpgradeCredential', {});

          const profile = svc.getProfile(subject.did);
          expect(profile!.verificationLevel).toBe('credential_backed');
        }),
        { numRuns: 20 }
      );
    });

    it('3+ attestations should upgrade to peer_verified', () => {
      fc.assert(
        fc.property(publicKeyArb, fc.array(publicKeyArb, { minLength: 3, maxLength: 5 }), (subjectKey, attesterKeys) => {
          // Ensure all keys are different
          const uniqueKeys = new Set([subjectKey.slice(0, 32), ...attesterKeys.map(k => k.slice(0, 32))]);
          fc.pre(uniqueKeys.size === attesterKeys.length + 1);

          resetIdentityService();
          const svc = new IdentityService();
          const subject = svc.createIdentity(subjectKey);

          for (const attesterKey of attesterKeys) {
            const attester = svc.createIdentity(attesterKey);
            svc.attestTrust(attester.did, subject.did);
          }

          const profile = svc.getProfile(subject.did);
          expect(profile!.verificationLevel).toBe('peer_verified');
        }),
        { numRuns: 15 }
      );
    });

    it('verification level should never decrease', () => {
      fc.assert(
        fc.property(
          publicKeyArb,
          publicKeyArb,
          fc.array(publicKeyArb, { minLength: 3, maxLength: 5 }),
          (issuerKey, subjectKey, attesterKeys) => {
            const uniqueKeys = new Set([
              issuerKey.slice(0, 32),
              subjectKey.slice(0, 32),
              ...attesterKeys.map(k => k.slice(0, 32))
            ]);
            fc.pre(uniqueKeys.size === attesterKeys.length + 2);

            resetIdentityService();
            const svc = new IdentityService();
            const issuer = svc.createIdentity(issuerKey);
            const subject = svc.createIdentity(subjectKey);

            const levels: VerificationLevel[] = ['self_attested', 'peer_verified', 'credential_backed', 'zkp_verified'];
            const levelOrder = (l: VerificationLevel) => levels.indexOf(l);

            let maxLevel = levelOrder(subject.verificationLevel);

            // Issue credential
            svc.issueCredential(issuer.did, subject.did, 'TestCred', {});
            let currentLevel = levelOrder(svc.getProfile(subject.did)!.verificationLevel);
            expect(currentLevel).toBeGreaterThanOrEqual(maxLevel);
            maxLevel = Math.max(maxLevel, currentLevel);

            // Add attestations
            for (const attesterKey of attesterKeys) {
              const attester = svc.createIdentity(attesterKey);
              svc.attestTrust(attester.did, subject.did);
              currentLevel = levelOrder(svc.getProfile(subject.did)!.verificationLevel);
              expect(currentLevel).toBeGreaterThanOrEqual(maxLevel);
              maxLevel = Math.max(maxLevel, currentLevel);
            }
          }
        ),
        { numRuns: 20 }
      );
    });
  });

  describe('Trust Attestation Properties', () => {
    it('attestations should be idempotent', () => {
      fc.assert(
        fc.property(publicKeyArb, publicKeyArb, fc.integer({ min: 1, max: 10 }), (attesterKey, subjectKey, times) => {
          fc.pre(attesterKey.slice(0, 32) !== subjectKey.slice(0, 32));

          resetIdentityService();
          const svc = new IdentityService();
          const attester = svc.createIdentity(attesterKey);
          const subject = svc.createIdentity(subjectKey);

          for (let i = 0; i < times; i++) {
            svc.attestTrust(attester.did, subject.did);
          }

          const profile = svc.getProfile(subject.did);
          // Attester should only appear once
          expect(profile!.trustedBy.filter(t => t === attester.did).length).toBe(1);
        }),
        { numRuns: 20 }
      );
    });

    it('trustedBy should contain all unique attesters', () => {
      fc.assert(
        fc.property(publicKeyArb, fc.array(publicKeyArb, { minLength: 1, maxLength: 5 }), (subjectKey, attesterKeys) => {
          const uniqueKeys = [...new Set([subjectKey.slice(0, 32), ...attesterKeys.map(k => k.slice(0, 32))])];
          fc.pre(uniqueKeys.length === attesterKeys.length + 1);

          resetIdentityService();
          const svc = new IdentityService();
          const subject = svc.createIdentity(subjectKey);

          const attesterDids: string[] = [];
          for (const attesterKey of attesterKeys) {
            const attester = svc.createIdentity(attesterKey);
            attesterDids.push(attester.did);
            svc.attestTrust(attester.did, subject.did);
          }

          const profile = svc.getProfile(subject.did);
          expect(profile!.trustedBy.length).toBe(attesterDids.length);
          for (const did of attesterDids) {
            expect(profile!.trustedBy).toContain(did);
          }
        }),
        { numRuns: 15 }
      );
    });
  });

  describe('Profile Resolution Properties', () => {
    it('created profiles should be retrievable', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const created = service.createIdentity(pubKey);
          const retrieved = service.getProfile(created.did);

          expect(retrieved).toBeDefined();
          expect(retrieved!.did).toBe(created.did);
          expect(retrieved!.document.id).toBe(created.document.id);
        }),
        { numRuns: 15 }
      );
    });

    it('DID resolution should return same document', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          const profile = service.createIdentity(pubKey);
          const resolved = service.resolveDID(profile.did);

          expect(resolved).toBeDefined();
          expect(resolved!.id).toBe(profile.document.id);
          expect(resolved!.controller).toBe(profile.document.controller);
          expect(resolved!.verificationMethod.length).toBe(profile.document.verificationMethod.length);
        }),
        { numRuns: 15 }
      );
    });

    it('unknown DIDs should return undefined', () => {
      fc.assert(
        fc.property(publicKeyArb, (pubKey) => {
          // Don't create the identity
          const fakeDid = `did:mycelix:${pubKey.slice(0, 32)}`;

          const profile = service.getProfile(fakeDid);
          const doc = service.resolveDID(fakeDid);

          expect(profile).toBeUndefined();
          expect(doc).toBeUndefined();
        }),
        { numRuns: 20 }
      );
    });
  });

  describe('Credential Verification Properties', () => {
    it('non-existent credentials should fail verification', () => {
      fc.assert(
        fc.property(fc.uuid(), (randomId) => {
          const result = service.verifyCredential(`vc-${randomId}`);

          expect(result.valid).toBe(false);
          expect(result.reason).toBe('Credential not found');
        }),
        { numRuns: 20 }
      );
    });
  });

  describe('Edge Case Properties', () => {
    it('should handle maximum length public keys', () => {
      const maxKey = 'a'.repeat(64);
      const profile = service.createIdentity(maxKey);

      expect(profile).toBeDefined();
      expect(profile.did).toMatch(/^did:mycelix:/);
    });

    it('should handle minimum length public keys', () => {
      const minKey = 'b'.repeat(32);
      const profile = service.createIdentity(minKey);

      expect(profile).toBeDefined();
      expect(profile.did).toMatch(/^did:mycelix:/);
    });
  });
});

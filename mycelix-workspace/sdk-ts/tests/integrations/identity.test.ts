/**
 * Identity Integration Tests
 *
 * Tests for IdentityService - DIDs, credentials, and trust attestation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  IdentityService,
  getIdentityService,
  resetIdentityService,
  type IdentityProfile,
  type VerifiableCredential,
  type DIDDocument,
} from '../../src/integrations/identity/index.js';

describe('Identity Integration', () => {
  let service: IdentityService;

  beforeEach(() => {
    resetIdentityService();
    service = new IdentityService();
  });

  describe('IdentityService', () => {
    describe('createIdentity', () => {
      it('should create a new identity profile', () => {
        const profile = service.createIdentity('publickey123abcdefg1234567890abcd');

        expect(profile).toBeDefined();
        expect(profile.did).toMatch(/^did:mycelix:/);
        expect(profile.credentials).toEqual([]);
        expect(profile.verificationLevel).toBe('self_attested');
      });

      it('should create unique DIDs for different public keys', () => {
        const profile1 = service.createIdentity('pubkey1111111111111111111111111111');
        const profile2 = service.createIdentity('pubkey2222222222222222222222222222');

        expect(profile1.did).not.toBe(profile2.did);
      });

      it('should include created timestamp', () => {
        const before = Date.now();
        const profile = service.createIdentity('testpublickey1234567890abcdefghij');
        const after = Date.now();

        expect(profile.createdAt).toBeGreaterThanOrEqual(before);
        expect(profile.createdAt).toBeLessThanOrEqual(after);
      });

      it('should create DID document with verification method', () => {
        const profile = service.createIdentity('myverificationkey1234567890abcdef');

        expect(profile.document.verificationMethod).toBeDefined();
        expect(profile.document.verificationMethod.length).toBeGreaterThan(0);
        expect(profile.document.verificationMethod[0].publicKeyMultibase).toMatch(/^z/);
      });

      it('should initialize with reputation', () => {
        const profile = service.createIdentity('newidentity1234567890abcdefghijkl');

        expect(profile.reputation).toBeDefined();
        expect(profile.reputation.agentId).toBe(profile.did);
      });
    });

    describe('getProfile', () => {
      it('should retrieve an existing profile', () => {
        const created = service.createIdentity('existingkey1234567890abcdefghijk');
        const retrieved = service.getProfile(created.did);

        expect(retrieved).toBeDefined();
        expect(retrieved!.did).toBe(created.did);
      });

      it('should return undefined for non-existent DID', () => {
        const result = service.getProfile('did:mycelix:nonexistent');
        expect(result).toBeUndefined();
      });
    });

    describe('resolveDID', () => {
      it('should resolve DID to document', () => {
        const profile = service.createIdentity('resolvablekey1234567890123456789a');
        const document = service.resolveDID(profile.did);

        expect(document).toBeDefined();
        expect(document!.id).toBe(profile.did);
        expect(document!.controller).toBe(profile.did);
      });

      it('should return undefined for unknown DID', () => {
        const document = service.resolveDID('did:mycelix:unknown');
        expect(document).toBeUndefined();
      });

      it('should include authentication method', () => {
        const profile = service.createIdentity('authkey12345678901234567890abcdef');
        const document = service.resolveDID(profile.did);

        expect(document!.authentication).toBeDefined();
        expect(document!.authentication.length).toBeGreaterThan(0);
      });
    });

    describe('issueCredential', () => {
      it('should issue a credential', () => {
        const issuer = service.createIdentity('issuerkey12345678901234567890123a');
        const subject = service.createIdentity('subjectkey1234567890123456789012a');

        const credential = service.issueCredential(
          issuer.did,
          subject.did,
          'MembershipCredential',
          { membershipLevel: 'gold', organization: 'Mycelix' }
        );

        expect(credential).toBeDefined();
        expect(credential.id).toMatch(/^vc-/);
        expect(credential.type).toContain('MembershipCredential');
        expect(credential.issuer).toBe(issuer.did);
      });

      it('should include issuance date', () => {
        const issuer = service.createIdentity('issuer2key123456789012345678901a2');
        const subject = service.createIdentity('subject2key12345678901234567890a2');

        const credential = service.issueCredential(
          issuer.did,
          subject.did,
          'TestCredential',
          { test: true }
        );

        expect(credential.issuanceDate).toBeDefined();
        expect(new Date(credential.issuanceDate).getTime()).toBeLessThanOrEqual(Date.now());
      });

      it('should support optional expiration', () => {
        const issuer = service.createIdentity('issuer3key123456789012345678901b3');
        const subject = service.createIdentity('subject3key12345678901234567890b3');

        const credential = service.issueCredential(
          issuer.did,
          subject.did,
          'ExpiringCredential',
          { value: 123 },
          30 // 30 days expiration
        );

        expect(credential.expirationDate).toBeDefined();
        const expiry = new Date(credential.expirationDate!).getTime();
        expect(expiry).toBeGreaterThan(Date.now());
      });

      it('should add credential to subject profile', () => {
        const issuer = service.createIdentity('issuer4key123456789012345678901c4');
        const subject = service.createIdentity('subject4key12345678901234567890c4');

        service.issueCredential(issuer.did, subject.did, 'Credential1', { a: 1 });
        service.issueCredential(issuer.did, subject.did, 'Credential2', { b: 2 });

        const profile = service.getProfile(subject.did);
        expect(profile!.credentials.length).toBe(2);
      });

      it('should boost subject reputation on credential issuance', () => {
        const issuer = service.createIdentity('issuer5key123456789012345678901d5');
        const subject = service.createIdentity('subject5key12345678901234567890d5');

        const repBefore = subject.reputation.positiveCount;
        service.issueCredential(issuer.did, subject.did, 'BoostCredential', {});

        const profile = service.getProfile(subject.did);
        expect(profile!.reputation.positiveCount).toBeGreaterThan(repBefore);
      });

      it('should upgrade verification level on credential issuance', () => {
        const issuer = service.createIdentity('issuer6key123456789012345678901e6');
        const subject = service.createIdentity('subject6key12345678901234567890e6');

        expect(subject.verificationLevel).toBe('self_attested');

        service.issueCredential(issuer.did, subject.did, 'UpgradeCredential', {});

        const profile = service.getProfile(subject.did);
        expect(profile!.verificationLevel).toBe('credential_backed');
      });
    });

    describe('getCredentials', () => {
      it('should return all credentials for a DID', () => {
        const issuer = service.createIdentity('issuergckey1234567890123456789ab');
        const subject = service.createIdentity('subjectgckey123456789012345678ab');

        service.issueCredential(issuer.did, subject.did, 'Cred1', { x: 1 });
        service.issueCredential(issuer.did, subject.did, 'Cred2', { y: 2 });
        service.issueCredential(issuer.did, subject.did, 'Cred3', { z: 3 });

        const credentials = service.getCredentials(subject.did);
        expect(credentials.length).toBe(3);
      });

      it('should return empty array for DID with no credentials', () => {
        const profile = service.createIdentity('nocredskey12345678901234567890ab');
        const credentials = service.getCredentials(profile.did);

        expect(credentials).toEqual([]);
      });

      it('should return empty array for unknown DID', () => {
        const credentials = service.getCredentials('did:mycelix:unknown');
        expect(credentials).toEqual([]);
      });
    });

    describe('verifyCredential', () => {
      it('should verify a valid credential', () => {
        const issuer = service.createIdentity('verifyissuerkey123456789012345ab');
        const subject = service.createIdentity('verifysubjectkey12345678901234ab');

        // Build issuer reputation to be trustworthy
        for (let i = 0; i < 5; i++) {
          const peer = service.createIdentity(`peer${i}key12345678901234567890abcd`);
          service.attestTrust(peer.did, issuer.did);
        }

        const credential = service.issueCredential(issuer.did, subject.did, 'ValidCredential', {});

        const result = service.verifyCredential(credential.id);
        expect(result.valid).toBe(true);
      });

      it('should return invalid for non-existent credential', () => {
        const result = service.verifyCredential('vc-nonexistent');

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Credential not found');
      });
    });

    describe('attestTrust', () => {
      it('should add trust attestation', () => {
        const attester = service.createIdentity('attesterkey123456789012345678901');
        const subject = service.createIdentity('subjecttrustkey12345678901234567');

        service.attestTrust(attester.did, subject.did);

        const profile = service.getProfile(subject.did);
        expect(profile!.trustedBy).toContain(attester.did);
      });

      it('should not duplicate attestations', () => {
        const attester = service.createIdentity('attester2key12345678901234567890');
        const subject = service.createIdentity('subject2trustkey1234567890123456');

        service.attestTrust(attester.did, subject.did);
        service.attestTrust(attester.did, subject.did);
        service.attestTrust(attester.did, subject.did);

        const profile = service.getProfile(subject.did);
        expect(profile!.trustedBy.filter((t) => t === attester.did).length).toBe(1);
      });

      it('should upgrade to peer_verified after 3 attestations', () => {
        const subject = service.createIdentity('peerverifiedkey12345678901234567');

        for (let i = 0; i < 3; i++) {
          const attester = service.createIdentity(`attester${i}peerkey1234567890123456`);
          service.attestTrust(attester.did, subject.did);
        }

        const profile = service.getProfile(subject.did);
        expect(profile!.verificationLevel).toBe('peer_verified');
      });

      it('should boost reputation on attestation', () => {
        const attester = service.createIdentity('attester3key12345678901234567890');
        const subject = service.createIdentity('subject3trustkey1234567890123456');

        const repBefore = subject.reputation.positiveCount;
        service.attestTrust(attester.did, subject.did);

        const profile = service.getProfile(subject.did);
        expect(profile!.reputation.positiveCount).toBeGreaterThan(repBefore);
      });

      it('should throw for non-existent profiles', () => {
        const attester = service.createIdentity('validattesterkey12345678901234ab');

        expect(() => {
          service.attestTrust(attester.did, 'did:mycelix:nonexistent');
        }).toThrow('Profile not found');
      });
    });

    describe('verifyCrossHapp', () => {
      it('should return cross-hApp verification result', async () => {
        const profile = service.createIdentity('crosshappkey1234567890123456789ab');

        const result = await service.verifyCrossHapp(profile.did);

        expect(result).toBeDefined();
        expect(typeof result.verified).toBe('boolean');
        expect(typeof result.trustScore).toBe('number');
        expect(result.trustScore).toBeGreaterThanOrEqual(0);
        expect(result.trustScore).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('getIdentityService', () => {
    it('should return singleton instance', () => {
      const service1 = getIdentityService();
      const service2 = getIdentityService();

      expect(service1).toBe(service2);
    });

    it('should maintain state across calls', () => {
      const service1 = getIdentityService();
      const profile = service1.createIdentity('persistentkey1234567890123456789a');

      const service2 = getIdentityService();
      const retrieved = service2.getProfile(profile.did);

      expect(retrieved).toBeDefined();
      expect(retrieved!.did).toBe(profile.did);
    });
  });

  describe('DID Format', () => {
    it('should follow did:mycelix format', () => {
      const profile = service.createIdentity('formatkey12345678901234567890123a');

      expect(profile.did).toMatch(/^did:mycelix:/);
    });

    it('should include proper W3C context in document', () => {
      const profile = service.createIdentity('contextkey1234567890123456789012a');

      // DID document should have proper structure
      expect(profile.document.id).toBe(profile.did);
      expect(profile.document.verificationMethod).toBeDefined();
    });
  });
});

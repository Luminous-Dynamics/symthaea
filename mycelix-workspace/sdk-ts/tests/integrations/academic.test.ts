/**
 * Academic Credentials Integration Tests
 *
 * Tests for AcademicCredentialService - W3C VC 2.0 academic credential
 * verification, ZK commitment validation, institution reputation, and
 * epistemic claim publication.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  AcademicCredentialService,
  getAcademicService,
  resetAcademicService,
  type AcademicCredential,
  type CredentialRevocation,
  type DnssecStatus,
} from '../../src/integrations/academic/index.js';

/** Helper: build a valid AcademicCredential for testing */
function makeCredential(overrides: Partial<AcademicCredential> = {}): AcademicCredential {
  return {
    '@context': ['https://www.w3.org/ns/credentials/v2'],
    id: 'urn:uuid:test-cred-001',
    type: ['VerifiableCredential', 'AcademicCredential'],
    issuer: {
      id: 'did:dns:university.edu',
      name: 'Test University',
      type: ['EducationalOrganization'],
    },
    validFrom: '2024-06-01T00:00:00Z',
    credentialSubject: {
      id: 'did:mycelix:student-001',
    },
    proof: {
      type: 'DataIntegrityProof',
      created: '2024-06-01T00:00:00Z',
      verificationMethod: 'did:dns:university.edu#key-1',
      proofPurpose: 'assertionMethod',
      proofValue: 'z3FXQw6M7v9r4k',
    },
    zkCommitment: new Uint8Array(32),
    revocationRegistryId: 'reg-001',
    revocationIndex: 0,
    dnsDid: {
      domain: 'university.edu',
      did: 'did:dns:university.edu',
      txtRecord: '_did.university.edu TXT "did=did:dns:university.edu"',
      dnssec: 'Validated',
      lastVerified: Date.now(),
      verificationChain: [],
    },
    achievement: {
      degreeType: 'Bachelor',
      degreeName: 'Bachelor of Science',
      fieldOfStudy: 'Computer Science',
      conferralDate: '2024-06-01',
    },
    mycelixSchemaId: 'schema-academic-v1',
    mycelixCreated: Date.now(),
    ...overrides,
  };
}

describe('Academic Credentials Integration', () => {
  let service: AcademicCredentialService;

  beforeEach(() => {
    resetAcademicService();
    service = new AcademicCredentialService();
  });

  describe('verifyCredentialStructure', () => {
    it('should verify a valid credential', () => {
      const cred = makeCredential();
      const result = service.verifyCredentialStructure(cred);

      expect(result.overallValid).toBe(true);
      expect(result.structureValid).toBe(true);
      expect(result.issuerValid).toBe(true);
      expect(result.proofValid).toBe(true);
      expect(result.commitmentValid).toBe(true);
      expect(result.isRevoked).toBe(false);
      expect(result.errors).toEqual([]);
    });

    it('should reject credential missing W3C VC 2.0 context', () => {
      const cred = makeCredential({ '@context': ['https://example.com'] });
      const result = service.verifyCredentialStructure(cred);

      expect(result.structureValid).toBe(false);
      expect(result.overallValid).toBe(false);
      expect(result.errors).toContain('Missing W3C VC 2.0 context');
    });

    it('should reject credential missing required types', () => {
      const cred = makeCredential({ type: ['VerifiableCredential'] });
      const result = service.verifyCredentialStructure(cred);

      expect(result.structureValid).toBe(false);
      expect(result.errors).toContain('Missing required credential types');
    });

    it('should reject credential with invalid ID format', () => {
      const cred = makeCredential({ id: 'not-a-urn' });
      const result = service.verifyCredentialStructure(cred);

      expect(result.structureValid).toBe(false);
      expect(result.errors).toContain('Invalid credential ID format');
    });

    it('should reject credential with invalid issuer', () => {
      const cred = makeCredential({
        issuer: { id: 'not-a-did', name: '', type: [] },
      });
      const result = service.verifyCredentialStructure(cred);

      expect(result.issuerValid).toBe(false);
      expect(result.overallValid).toBe(false);
    });

    it('should reject credential with invalid proof', () => {
      const cred = makeCredential({
        proof: {
          type: 'InvalidProof',
          created: '2024-06-01T00:00:00Z',
          verificationMethod: 'did:dns:uni.edu#k1',
          proofPurpose: 'assertionMethod',
          proofValue: 'not-z-prefix',
        },
      });
      const result = service.verifyCredentialStructure(cred);

      expect(result.proofValid).toBe(false);
    });

    it('should reject credential with wrong-length zkCommitment', () => {
      const cred = makeCredential({ zkCommitment: new Uint8Array(16) });
      const result = service.verifyCredentialStructure(cred);

      expect(result.commitmentValid).toBe(false);
    });

    it('should flag a revoked credential', () => {
      const cred = makeCredential();
      service.registerCredential(cred);
      service.registerRevocation({
        credentialId: cred.id,
        revocationRegistryId: 'reg-001',
        revocationIndex: 0,
        reason: 'AcademicFraud',
        revokedAt: new Date().toISOString(),
        revokedBy: 'did:mycelix:admin',
      });

      const result = service.verifyCredentialStructure(cred);

      expect(result.isRevoked).toBe(true);
      expect(result.overallValid).toBe(false);
    });

    it('should include an epistemic claim when valid', () => {
      const cred = makeCredential();
      const result = service.verifyCredentialStructure(cred);

      expect(result.epistemicClaim).toBeDefined();
      expect(result.epistemicClaim!.content).toContain('Academic credential');
    });

    it('should not include an epistemic claim when invalid', () => {
      const cred = makeCredential({ id: 'bad-id' });
      const result = service.verifyCredentialStructure(cred);

      expect(result.epistemicClaim).toBeUndefined();
    });
  });

  describe('registerCredential', () => {
    it('should store credential and return it by ID', () => {
      const cred = makeCredential();
      service.registerCredential(cred);

      const retrieved = service.getCredential(cred.id);
      expect(retrieved).toBeDefined();
      expect(retrieved!.id).toBe(cred.id);
    });

    it('should initialize institution reputation on first credential', () => {
      const cred = makeCredential();
      service.registerCredential(cred);

      const rep = service.getInstitutionReputation('did:dns:university.edu');
      expect(rep).toBeGreaterThan(0);
    });
  });

  describe('getCredentialsByInstitution', () => {
    it('should return all credentials for an institution', () => {
      const cred1 = makeCredential({ id: 'urn:uuid:cred-1' });
      const cred2 = makeCredential({ id: 'urn:uuid:cred-2' });
      const other = makeCredential({
        id: 'urn:uuid:cred-3',
        issuer: { id: 'did:dns:other.edu', name: 'Other', type: [] },
      });

      service.registerCredential(cred1);
      service.registerCredential(cred2);
      service.registerCredential(other);

      const results = service.getCredentialsByInstitution('did:dns:university.edu');
      expect(results).toHaveLength(2);
    });

    it('should return empty array for unknown institution', () => {
      const results = service.getCredentialsByInstitution('did:dns:nonexistent.edu');
      expect(results).toEqual([]);
    });
  });

  describe('getCredentialsBySubject', () => {
    it('should return all credentials for a subject', () => {
      const cred1 = makeCredential({ id: 'urn:uuid:s-1' });
      const cred2 = makeCredential({
        id: 'urn:uuid:s-2',
        credentialSubject: { id: 'did:mycelix:student-002' },
      });

      service.registerCredential(cred1);
      service.registerCredential(cred2);

      const results = service.getCredentialsBySubject('did:mycelix:student-001');
      expect(results).toHaveLength(1);
      expect(results[0].id).toBe('urn:uuid:s-1');
    });
  });

  describe('isRevoked', () => {
    it('should return false for non-revoked credential', () => {
      expect(service.isRevoked('urn:uuid:unknown')).toBe(false);
    });

    it('should return true after registering a revocation', () => {
      service.registerRevocation({
        credentialId: 'urn:uuid:rev-test',
        revocationRegistryId: 'reg-1',
        revocationIndex: 0,
        reason: 'IssuedInError',
        revokedAt: new Date().toISOString(),
        revokedBy: 'did:mycelix:admin',
      });

      expect(service.isRevoked('urn:uuid:rev-test')).toBe(true);
    });
  });

  describe('isDnssecAcceptable', () => {
    it('should accept Validated status', () => {
      expect(service.isDnssecAcceptable('Validated')).toBe(true);
    });

    it('should accept Insecure status', () => {
      expect(service.isDnssecAcceptable('Insecure')).toBe(true);
    });

    it('should reject Invalid status', () => {
      expect(service.isDnssecAcceptable('Invalid')).toBe(false);
    });

    it('should reject Unknown status', () => {
      expect(service.isDnssecAcceptable('Unknown')).toBe(false);
    });
  });

  describe('publishAsEpistemicClaim', () => {
    it('should publish a valid credential as an epistemic claim', () => {
      const cred = makeCredential();
      const claim = service.publishAsEpistemicClaim(cred);

      expect(claim).toBeDefined();
      expect(claim.content).toContain('Bachelor');
      expect(claim.content).toContain('Computer Science');
    });

    it('should throw for an invalid credential', () => {
      const cred = makeCredential({ id: 'invalid-id' });

      expect(() => service.publishAsEpistemicClaim(cred)).toThrow(
        'Cannot publish invalid credential',
      );
    });
  });

  describe('getEpistemicPosition', () => {
    it('should return classification for a valid credential', () => {
      const cred = makeCredential();
      const pos = service.getEpistemicPosition(cred);

      expect(pos.empirical).toBeGreaterThan(0);
      expect(pos.normative).toBeGreaterThan(0);
      expect(pos.materiality).toBeGreaterThan(0);
      expect(pos.label).toContain('E3');
    });

    it('should return zeros for an invalid credential', () => {
      const cred = makeCredential({ id: 'bad' });
      const pos = service.getEpistemicPosition(cred);

      expect(pos.empirical).toBe(0);
      expect(pos.normative).toBe(0);
      expect(pos.materiality).toBe(0);
      expect(pos.label).toContain('Invalid');
    });
  });

  describe('getStats', () => {
    it('should return correct statistics', () => {
      service.registerCredential(makeCredential({ id: 'urn:uuid:st-1' }));
      service.registerCredential(
        makeCredential({
          id: 'urn:uuid:st-2',
          achievement: {
            degreeType: 'Master',
            degreeName: 'Master of Arts',
            fieldOfStudy: 'History',
            conferralDate: '2024-01-01',
          },
        }),
      );
      service.registerRevocation({
        credentialId: 'urn:uuid:st-1',
        revocationRegistryId: 'reg-1',
        revocationIndex: 0,
        reason: 'HolderRequest',
        revokedAt: new Date().toISOString(),
        revokedBy: 'did:mycelix:admin',
      });

      const stats = service.getStats();

      expect(stats.totalCredentials).toBe(2);
      expect(stats.totalRevocations).toBe(1);
      expect(stats.institutionCount).toBe(1);
      expect(stats.degreeBreakdown['Bachelor']).toBe(1);
      expect(stats.degreeBreakdown['Master']).toBe(1);
    });

    it('should return empty stats for a fresh service', () => {
      const stats = service.getStats();

      expect(stats.totalCredentials).toBe(0);
      expect(stats.totalRevocations).toBe(0);
      expect(stats.institutionCount).toBe(0);
    });
  });

  describe('getAcademicService', () => {
    it('should return singleton instance', () => {
      const s1 = getAcademicService();
      const s2 = getAcademicService();

      expect(s1).toBe(s2);
    });

    it('should return a fresh instance after reset', () => {
      const s1 = getAcademicService();
      resetAcademicService();
      const s2 = getAcademicService();

      expect(s1).not.toBe(s2);
    });
  });
});

/**
 * Knowledge Integration Tests
 *
 * Tests for KnowledgeService - claims, evidence, and knowledge graph
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  KnowledgeService,
  getKnowledgeService,
  resetKnowledgeService,
  type KnowledgeEntry,
  type EvidenceSubmission,
  type ContributorProfile,
} from '../../src/integrations/knowledge/index.js';
import { EmpiricalLevel, NormativeLevel } from '../../src/epistemic/index.js';

describe('Knowledge Integration', () => {
  let service: KnowledgeService;

  beforeEach(() => {
    resetKnowledgeService();
    service = new KnowledgeService();
  });

  describe('KnowledgeService', () => {
    describe('submitClaim', () => {
      it('should create a new knowledge claim', () => {
        const claim = service.submitClaim(
          'did:mycelix:author1',
          'Climate Change is Real',
          'Scientific consensus confirms anthropogenic climate change',
          EmpiricalLevel.E2_PrivateVerify,
          NormativeLevel.N1_Descriptive,
          ['climate', 'science']
        );

        expect(claim).toBeDefined();
        expect(claim.id).toMatch(/^claim-/);
        expect(claim.title).toBe('Climate Change is Real');
        expect(claim.type).toBe('claim');
        expect(claim.status).toBe('proposed');
        expect(claim.tags).toContain('climate');
      });

      it('should initialize with zero endorsements and challenges', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Test Claim',
          'Test content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        expect(claim.endorsements).toBe(0);
        expect(claim.challenges).toBe(0);
      });

      it('should set proper epistemic classification', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Cryptographic Claim',
          'Verified by cryptographic proof',
          EmpiricalLevel.E3_Cryptographic,
          NormativeLevel.N2_Normative
        );

        expect(claim.classification.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
        expect(claim.classification.normative).toBe(NormativeLevel.N2_Normative);
      });

      it('should create contributor profile for new author', () => {
        service.submitClaim(
          'did:mycelix:newauthor',
          'First Claim',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        const contributor = service.getContributor('did:mycelix:newauthor');
        expect(contributor).toBeDefined();
        expect(contributor!.contributions).toBe(1);
      });
    });

    describe('submitEvidence', () => {
      it('should add evidence to an existing claim', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Evidence Claim',
          'Needs evidence',
          EmpiricalLevel.E0_Unknown,
          NormativeLevel.N1_Descriptive
        );

        const evidence = service.submitEvidence(
          claim.id,
          'did:mycelix:researcher',
          'empirical',
          'Study results show...',
          'https://example.com/study'
        );

        expect(evidence).toBeDefined();
        expect(evidence.id).toMatch(/^evidence-/);
        expect(evidence.evidenceType).toBe('empirical');
        expect(evidence.sourceUrl).toBe('https://example.com/study');
      });

      it('should throw for non-existent claim', () => {
        expect(() => {
          service.submitEvidence(
            'claim-nonexistent',
            'did:mycelix:researcher',
            'empirical',
            'Evidence content'
          );
        }).toThrow('Claim not found');
      });

      it('should upgrade claim empirical level with cryptographic evidence', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Upgradable Claim',
          'Content',
          EmpiricalLevel.E0_Unknown,
          NormativeLevel.N1_Descriptive
        );

        service.submitEvidence(
          claim.id,
          'did:mycelix:verifier',
          'cryptographic',
          'ZK proof verification'
        );

        const updated = service.getClaim(claim.id);
        expect(updated!.classification.empirical).toBe(EmpiricalLevel.E3_Cryptographic);
      });

      it('should boost submitter reputation', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Test',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        service.submitEvidence(
          claim.id,
          'did:mycelix:helper',
          'testimonial',
          'I witnessed this'
        );

        const contributor = service.getContributor('did:mycelix:helper');
        expect(contributor).toBeDefined();
        // Reputation should be positive after submitting evidence
        expect(contributor!.reputation.positiveCount).toBeGreaterThan(1);
      });
    });

    describe('endorseClaim', () => {
      it('should increment endorsement count', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Endorsable',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        service.endorseClaim(claim.id, 'did:mycelix:endorser1');
        service.endorseClaim(claim.id, 'did:mycelix:endorser2');

        const updated = service.getClaim(claim.id);
        expect(updated!.endorsements).toBe(2);
      });

      it('should verify claim after 5 endorsements', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'To Verify',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        for (let i = 0; i < 5; i++) {
          service.endorseClaim(claim.id, `did:mycelix:endorser${i}`);
        }

        const updated = service.getClaim(claim.id);
        expect(updated!.status).toBe('verified');
      });

      it('should throw for non-existent claim', () => {
        expect(() => {
          service.endorseClaim('claim-fake', 'did:mycelix:endorser');
        }).toThrow('Claim not found');
      });
    });

    describe('challengeClaim', () => {
      it('should increment challenge count', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Challengeable',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        service.challengeClaim(claim.id, 'did:mycelix:challenger', 'Factual error');

        const updated = service.getClaim(claim.id);
        expect(updated!.challenges).toBe(1);
      });

      it('should dispute verified claim after 3 challenges', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Will Be Disputed',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        // First verify the claim
        for (let i = 0; i < 5; i++) {
          service.endorseClaim(claim.id, `did:mycelix:endorser${i}`);
        }
        expect(service.getClaim(claim.id)!.status).toBe('verified');

        // Then challenge it
        for (let i = 0; i < 3; i++) {
          service.challengeClaim(claim.id, `did:mycelix:challenger${i}`, 'Reason');
        }

        const updated = service.getClaim(claim.id);
        expect(updated!.status).toBe('disputed');
      });
    });

    describe('search', () => {
      it('should find claims by title', () => {
        service.submitClaim(
          'did:mycelix:author',
          'Quantum Computing Advances',
          'New developments in quantum computing',
          EmpiricalLevel.E2_PrivateVerify,
          NormativeLevel.N1_Descriptive,
          ['quantum', 'computing']
        );

        const results = service.search('quantum');

        expect(results.length).toBe(1);
        expect(results[0].title).toContain('Quantum');
      });

      it('should find claims by content', () => {
        service.submitClaim(
          'did:mycelix:author',
          'Research Paper',
          'This paper discusses machine learning applications',
          EmpiricalLevel.E2_PrivateVerify,
          NormativeLevel.N1_Descriptive
        );

        const results = service.search('machine learning');

        expect(results.length).toBe(1);
      });

      it('should filter by tags', () => {
        service.submitClaim('did:mycelix:a1', 'AI Article', 'Content', EmpiricalLevel.E1_PublicVerify, NormativeLevel.N1_Descriptive, ['ai', 'tech']);
        service.submitClaim('did:mycelix:a2', 'AI Paper', 'Content', EmpiricalLevel.E1_PublicVerify, NormativeLevel.N1_Descriptive, ['ai', 'research']);
        service.submitClaim('did:mycelix:a3', 'Bio Article', 'Content', EmpiricalLevel.E1_PublicVerify, NormativeLevel.N1_Descriptive, ['biology']);

        const results = service.search('Article', ['ai']);

        expect(results.length).toBe(1);
        expect(results[0].title).toBe('AI Article');
      });
    });

    describe('getEvidence', () => {
      it('should return all evidence for a claim', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'Multi Evidence',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        service.submitEvidence(claim.id, 'did:mycelix:r1', 'empirical', 'Evidence 1');
        service.submitEvidence(claim.id, 'did:mycelix:r2', 'testimonial', 'Evidence 2');

        const evidence = service.getEvidence(claim.id);
        expect(evidence.length).toBe(2);
      });

      it('should return empty array for claim with no evidence', () => {
        const claim = service.submitClaim(
          'did:mycelix:author',
          'No Evidence',
          'Content',
          EmpiricalLevel.E1_PublicVerify,
          NormativeLevel.N1_Descriptive
        );

        const evidence = service.getEvidence(claim.id);
        expect(evidence).toEqual([]);
      });
    });

    describe('requestSynthesis', () => {
      it('should create a synthesis request', () => {
        const claim1 = service.submitClaim('did:mycelix:a1', 'C1', 'Content', EmpiricalLevel.E1_PublicVerify, NormativeLevel.N1_Descriptive);
        const claim2 = service.submitClaim('did:mycelix:a2', 'C2', 'Content', EmpiricalLevel.E1_PublicVerify, NormativeLevel.N1_Descriptive);

        const request = service.requestSynthesis(
          'Climate Change Summary',
          [claim1.id, claim2.id],
          'did:mycelix:researcher'
        );

        expect(request).toBeDefined();
        expect(request.id).toMatch(/^synthesis-req-/);
        expect(request.topic).toBe('Climate Change Summary');
        expect(request.sourceClaimIds.length).toBe(2);
        expect(request.status).toBe('pending');
      });
    });
  });

  describe('getKnowledgeService', () => {
    it('should return singleton instance', () => {
      const service1 = getKnowledgeService();
      const service2 = getKnowledgeService();

      expect(service1).toBe(service2);
    });
  });
});

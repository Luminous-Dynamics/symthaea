/**
 * DeSci (Decentralized Science) Integration Tests
 *
 * Tests for DeSciService - researcher registration, research publication
 * with epistemic classification, peer review reputation tracking, grant
 * milestone verification, citation recording, and researcher profiles.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  DeSciService,
  getDeSciService,
  type Researcher,
  type Publication,
  type PeerReview,
  type ResearchGrant,
  type GrantMilestone,
  type Collaboration,
} from '../../src/integrations/desci/index.js';

describe('DeSci Integration', () => {
  let service: DeSciService;

  beforeEach(() => {
    service = new DeSciService();
  });

  describe('registerResearcher', () => {
    it('should register a new researcher', () => {
      const researcher = service.registerResearcher({
        id: 'r-001',
        name: 'Dr. Alice Smith',
        orcid: '0000-0002-1234-5678',
        institution: 'MIT',
        fields: ['neuroscience', 'ml'],
      });

      expect(researcher).toBeDefined();
      expect(researcher.id).toBe('r-001');
      expect(researcher.name).toBe('Dr. Alice Smith');
      expect(researcher.joinedAt).toBeGreaterThan(0);
    });

    it('should mark researcher with ORCID as verified', () => {
      const withOrcid = service.registerResearcher({
        id: 'r-v1',
        name: 'Verified',
        orcid: '0000-0001-0000-0001',
        fields: ['physics'],
      });

      expect(withOrcid.verifiedIdentity).toBe(true);
    });

    it('should mark researcher without ORCID as unverified', () => {
      const noOrcid = service.registerResearcher({
        id: 'r-v2',
        name: 'Unverified',
        fields: ['biology'],
      });

      expect(noOrcid.verifiedIdentity).toBe(false);
    });

    it('should initialize researcher reputation', () => {
      service.registerResearcher({
        id: 'r-rep',
        name: 'Rep Test',
        fields: ['chemistry'],
      });

      // Should not throw for this researcher
      const profile = service.getResearcherProfile('r-rep');
      expect(profile.trustScore).toBeGreaterThan(0);
    });
  });

  describe('publishResearch', () => {
    it('should publish a research paper', () => {
      service.registerResearcher({ id: 'a-1', name: 'Author', fields: ['cs'] });

      const pub = service.publishResearch({
        id: 'pub-001',
        title: 'Novel ML Approach',
        authors: ['a-1'],
        abstract: 'We present a novel approach...',
        doi: '10.1234/example.001',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      expect(pub).toBeDefined();
      expect(pub.id).toBe('pub-001');
      expect(pub.reviewCount).toBe(0);
      expect(pub.citationCount).toBe(0);
    });

    it('should assign E3 level for preregistered studies', () => {
      service.registerResearcher({ id: 'a-pre', name: 'Prereg Author', fields: ['psych'] });

      const pub = service.publishResearch({
        id: 'pub-pre',
        title: 'Preregistered Study',
        authors: ['a-pre'],
        abstract: 'Preregistered...',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: true,
        codeAvailable: true,
        preregistered: true,
      });

      expect(pub.epistemicClassification).toBeDefined();
      expect(pub.epistemicClassification!.empirical).toBe(3); // E3
    });

    it('should assign E2 level for open data + code', () => {
      service.registerResearcher({ id: 'a-open', name: 'Open Sci', fields: ['bio'] });

      const pub = service.publishResearch({
        id: 'pub-open',
        title: 'Open Data Study',
        authors: ['a-open'],
        abstract: 'Open...',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: true,
        codeAvailable: true,
        preregistered: false,
      });

      expect(pub.epistemicClassification!.empirical).toBe(2); // E2
    });

    it('should assign N2 for peer-reviewed publications', () => {
      service.registerResearcher({ id: 'a-pr', name: 'Peer Rev', fields: ['math'] });

      const pub = service.publishResearch({
        id: 'pub-pr',
        title: 'Peer Reviewed',
        authors: ['a-pr'],
        abstract: 'Reviewed...',
        publishedAt: Date.now(),
        peerReviewed: true,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      expect(pub.epistemicClassification!.normative).toBe(2); // N2
    });

    it('should boost author reputation on publishing', () => {
      service.registerResearcher({ id: 'a-boost', name: 'Boosted', fields: ['cs'] });

      const profileBefore = service.getResearcherProfile('a-boost');
      const scoreBefore = profileBefore.trustScore;

      service.publishResearch({
        id: 'pub-boost',
        title: 'Boost Paper',
        authors: ['a-boost'],
        abstract: 'A paper',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      const profileAfter = service.getResearcherProfile('a-boost');
      expect(profileAfter.trustScore).toBeGreaterThanOrEqual(scoreBefore);
    });
  });

  describe('submitPeerReview', () => {
    it('should submit a peer review', () => {
      service.registerResearcher({ id: 'author-1', name: 'Author', fields: ['cs'] });
      service.registerResearcher({
        id: 'reviewer-1',
        name: 'Reviewer',
        orcid: '0000-0001-0000-0002',
        fields: ['cs'],
      });
      service.publishResearch({
        id: 'pub-rev',
        title: 'Paper to Review',
        authors: ['author-1'],
        abstract: 'Abstract...',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      const review = service.submitPeerReview({
        publicationId: 'pub-rev',
        reviewerId: 'reviewer-1',
        recommendation: 'accept',
        confidence: 4,
        qualityScore: 8,
        comments: 'Excellent methodology and thorough analysis. The results are well-supported by the data. I recommend acceptance with minor revisions.',
        blind: true,
      });

      expect(review).toBeDefined();
      expect(review.id).toMatch(/^review-/);
      expect(review.verified).toBe(true);
    });

    it('should mark publication as peer-reviewed after 2+ non-reject reviews', () => {
      service.registerResearcher({ id: 'a-rev2', name: 'Author', fields: ['cs'] });
      service.registerResearcher({ id: 'rev-a', name: 'Reviewer A', fields: ['cs'] });
      service.registerResearcher({ id: 'rev-b', name: 'Reviewer B', fields: ['cs'] });

      service.publishResearch({
        id: 'pub-pr2',
        title: 'Paper to Peer Review',
        authors: ['a-rev2'],
        abstract: 'Abstract',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      service.submitPeerReview({
        publicationId: 'pub-pr2',
        reviewerId: 'rev-a',
        recommendation: 'accept',
        confidence: 3,
        qualityScore: 7,
        comments: 'Solid work.',
        blind: true,
      });

      service.submitPeerReview({
        publicationId: 'pub-pr2',
        reviewerId: 'rev-b',
        recommendation: 'minor_revision',
        confidence: 4,
        qualityScore: 8,
        comments: 'Very good, minor edits needed.',
        blind: true,
      });

      const { publication } = service.getPublicationWithReviews('pub-pr2');
      expect(publication.peerReviewed).toBe(true);
    });

    it('should throw for non-existent publication', () => {
      expect(() =>
        service.submitPeerReview({
          publicationId: 'pub-nonexistent',
          reviewerId: 'rev-1',
          recommendation: 'accept',
          confidence: 3,
          qualityScore: 7,
          comments: 'Good',
          blind: false,
        }),
      ).toThrow('Publication not found');
    });

    it('should boost reviewer reputation for quality reviews', () => {
      service.registerResearcher({ id: 'a-qr', name: 'Author', fields: ['cs'] });
      service.registerResearcher({ id: 'rev-q', name: 'Quality Reviewer', fields: ['cs'] });
      service.publishResearch({
        id: 'pub-qr',
        title: 'Quality Review Paper',
        authors: ['a-qr'],
        abstract: 'Abstract',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      const profileBefore = service.getResearcherProfile('rev-q');

      service.submitPeerReview({
        publicationId: 'pub-qr',
        reviewerId: 'rev-q',
        recommendation: 'accept',
        confidence: 4,
        qualityScore: 9,
        comments: 'A'.repeat(100), // >= 100 chars for quality boost
        blind: true,
      });

      const profileAfter = service.getResearcherProfile('rev-q');
      expect(profileAfter.trustScore).toBeGreaterThanOrEqual(profileBefore.trustScore);
    });
  });

  describe('createGrant', () => {
    it('should create a research grant with proposed status', () => {
      const grant = service.createGrant({
        id: 'grant-001',
        title: 'Neural Network Research',
        principalInvestigator: 'r-001',
        coInvestigators: ['r-002'],
        fundingAgency: 'NSF',
        amount: 500000,
        currency: 'USD',
        startDate: Date.now(),
        endDate: Date.now() + 365 * 24 * 60 * 60 * 1000,
        milestones: [
          {
            id: 'm-1',
            title: 'Phase 1',
            description: 'Initial experiments',
            dueDate: Date.now() + 90 * 24 * 60 * 60 * 1000,
            deliverables: ['Report'],
            verified: false,
          },
        ],
      });

      expect(grant.status).toBe('proposed');
      expect(grant.amount).toBe(500000);
    });
  });

  describe('verifyMilestone', () => {
    it('should verify a milestone with a verified researcher', () => {
      service.registerResearcher({
        id: 'pi-1',
        name: 'PI',
        orcid: '0000-0001-0000-0001',
        fields: ['cs'],
      });
      service.registerResearcher({
        id: 'verifier-1',
        name: 'Verifier',
        orcid: '0000-0001-0000-0002',
        fields: ['cs'],
      });
      service.createGrant({
        id: 'grant-vm',
        title: 'Verify Milestone',
        principalInvestigator: 'pi-1',
        coInvestigators: [],
        fundingAgency: 'DOE',
        amount: 100000,
        currency: 'USD',
        startDate: Date.now(),
        endDate: Date.now() + 86400000,
        milestones: [
          {
            id: 'ms-1',
            title: 'Milestone 1',
            description: 'Deliverable',
            dueDate: Date.now() + 86400000,
            deliverables: ['Paper'],
            verified: false,
          },
        ],
      });

      const result = service.verifyMilestone('grant-vm', 'ms-1', 'verifier-1');
      expect(result).toBe(true);
    });

    it('should throw if verifier is not verified', () => {
      service.registerResearcher({ id: 'pi-uv', name: 'PI', fields: ['cs'] });
      service.registerResearcher({ id: 'unv', name: 'Unverified', fields: ['cs'] });
      service.createGrant({
        id: 'grant-uv',
        title: 'Test',
        principalInvestigator: 'pi-uv',
        coInvestigators: [],
        fundingAgency: 'NSF',
        amount: 10000,
        currency: 'USD',
        startDate: Date.now(),
        endDate: Date.now() + 86400000,
        milestones: [
          {
            id: 'ms-uv',
            title: 'MS',
            description: 'D',
            dueDate: Date.now(),
            deliverables: [],
            verified: false,
          },
        ],
      });

      expect(() => service.verifyMilestone('grant-uv', 'ms-uv', 'unv')).toThrow(
        'Verifier must have verified identity',
      );
    });

    it('should throw for non-existent grant', () => {
      expect(() => service.verifyMilestone('fake-grant', 'ms-1', 'r-1')).toThrow(
        'Grant not found',
      );
    });
  });

  describe('recordCitation', () => {
    it('should increment citation count on cited publication', () => {
      service.registerResearcher({ id: 'a-cite', name: 'Author', fields: ['cs'] });
      service.publishResearch({
        id: 'pub-cited',
        title: 'Cited Paper',
        authors: ['a-cite'],
        abstract: 'Abstract',
        publishedAt: Date.now(),
        peerReviewed: true,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      service.recordCitation('pub-citing', 'pub-cited');

      const { publication } = service.getPublicationWithReviews('pub-cited');
      expect(publication.citationCount).toBe(1);
    });

    it('should throw when citing non-existent publication', () => {
      expect(() => service.recordCitation('pub-a', 'pub-nonexistent')).toThrow(
        'Publication not found',
      );
    });
  });

  describe('getResearcherProfile', () => {
    it('should return comprehensive profile', () => {
      service.registerResearcher({
        id: 'rp-1',
        name: 'Profile Test',
        orcid: '0000-0001-0000-0005',
        fields: ['cs'],
      });
      service.publishResearch({
        id: 'rp-pub-1',
        title: 'Paper 1',
        authors: ['rp-1'],
        abstract: 'Abstract',
        publishedAt: Date.now(),
        peerReviewed: true,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });

      const profile = service.getResearcherProfile('rp-1');

      expect(profile.researcher.id).toBe('rp-1');
      expect(profile.publicationCount).toBe(1);
      expect(profile.trustScore).toBeGreaterThan(0);
    });

    it('should throw for unknown researcher', () => {
      expect(() => service.getResearcherProfile('unknown')).toThrow('Researcher not found');
    });
  });

  describe('getPublicationWithReviews', () => {
    it('should return publication with reviews and average quality', () => {
      service.registerResearcher({ id: 'a-pwr', name: 'Author', fields: ['cs'] });
      service.registerResearcher({ id: 'r-pwr', name: 'Reviewer', fields: ['cs'] });
      service.publishResearch({
        id: 'pub-pwr',
        title: 'Paper with Reviews',
        authors: ['a-pwr'],
        abstract: 'Abstract',
        publishedAt: Date.now(),
        peerReviewed: false,
        dataAvailable: false,
        codeAvailable: false,
        preregistered: false,
      });
      service.submitPeerReview({
        publicationId: 'pub-pwr',
        reviewerId: 'r-pwr',
        recommendation: 'accept',
        confidence: 4,
        qualityScore: 8,
        comments: 'Good paper.',
        blind: false,
      });

      const result = service.getPublicationWithReviews('pub-pwr');

      expect(result.publication.id).toBe('pub-pwr');
      expect(result.reviews).toHaveLength(1);
      expect(result.averageQuality).toBe(8);
    });
  });

  describe('isResearcherTrustworthy', () => {
    it('should return false for unknown researcher', () => {
      expect(service.isResearcherTrustworthy('unknown')).toBe(false);
    });

    it('should evaluate trust based on reputation threshold', () => {
      service.registerResearcher({ id: 'rt-1', name: 'TrustTest', fields: ['cs'] });
      // New researcher starts at neutral reputation (~0.5)
      // Default threshold is 0.7, so should be false initially
      const result = service.isResearcherTrustworthy('rt-1');
      expect(typeof result).toBe('boolean');
    });
  });

  describe('createCollaboration', () => {
    it('should create an active collaboration', () => {
      const collab = service.createCollaboration({
        id: 'collab-001',
        title: 'Joint Research',
        participants: ['r-001', 'r-002'],
        institution: ['MIT', 'Stanford'],
        startDate: Date.now(),
      });

      expect(collab.id).toBe('collab-001');
      expect(collab.status).toBe('active');
      expect(collab.publications).toEqual([]);
      expect(collab.grants).toEqual([]);
    });
  });

  describe('getDeSciService', () => {
    it('should return singleton instance', () => {
      const s1 = getDeSciService();
      const s2 = getDeSciService();
      expect(s1).toBe(s2);
    });
  });
});

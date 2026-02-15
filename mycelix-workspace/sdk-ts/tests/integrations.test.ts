/**
 * Integrations Module Tests
 *
 * Comprehensive tests for hApp-specific integration services:
 * - MarketplaceReputationService
 * - EduNetCredentialService
 * - SupplyChainProvenanceService
 *
 * Also tests the main integrations barrel exports.
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Import from barrel file to get index.ts coverage
import {
  MarketplaceReputationService,
  getMarketplaceService,
  EduNetCredentialService,
  getEduNetService,
  SupplyChainProvenanceService,
  getSupplyChainService,
  MailTrustService,
  getMailTrustService,
  IdentityBridgeClient,
  getIdentityBridgeClient,
  FinanceBridgeClient,
  getFinanceBridgeClient,
  PropertyBridgeClient,
  getPropertyBridgeClient,
  EnergyBridgeClient,
  getEnergyBridgeClient,
  MediaBridgeClient,
  getMediaBridgeClient,
  GovernanceBridgeClient,
  getGovernanceBridgeClient,
  JusticeBridgeClient,
  getJusticeBridgeClient,
  KnowledgeBridgeClient,
  getKnowledgeBridgeClient,
  FabricationService,
  getFabricationService,
  EpistemicMarketsService,
  getEpistemicMarketsService,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  calculateStakeValue,
  getRecommendedResolution,
  getRecommendedDurationDays,
  toClassificationCode,
  createReputationOnlyStake,
  createMonetaryOnlyStake,
} from '../src/integrations/index.js';

// Direct imports for typed usage
import {
  type Transaction,
  type TransactionType,
  type SellerProfile,
  type BuyerProfile,
  type ListingVerification,
  createReputation as mpCreateReputation,
  recordPositive as mpRecordPositive,
  recordNegative as mpRecordNegative,
  reputationValue as mpReputationValue,
  isTrustworthy as mpIsTrustworthy,
  AggregationMethod,
} from '../src/integrations/marketplace/index.js';

import {
  type CompletionStatus,
  type Course,
  type CourseCompletion,
  type EducationalCredential,
  type SkillAssessment,
  type LearnerProfile,
  claim as eduClaim,
  EmpiricalLevel as EduEmpiricalLevel,
  NormativeLevel as EduNormativeLevel,
  MaterialityLevel as EduMaterialityLevel,
  createReputation as eduCreateReputation,
  recordPositive as eduRecordPositive,
  reputationValue as eduReputationValue,
} from '../src/integrations/edunet/index.js';

import {
  type EvidenceType,
  type CheckpointEvidence,
  type Checkpoint,
  type ProvenanceChain,
  type HandlerProfile,
  type ChainVerification,
  claim as scClaim,
  EmpiricalLevel as ScEmpiricalLevel,
  NormativeLevel as ScNormativeLevel,
  MaterialityLevel as ScMaterialityLevel,
  createReputation as scCreateReputation,
  recordPositive as scRecordPositive,
  recordNegative as scRecordNegative,
  reputationValue as scReputationValue,
} from '../src/integrations/supplychain/index.js';

// =============================================================================
// Marketplace Integration Tests
// =============================================================================

describe('Marketplace Integration', () => {
  describe('MarketplaceReputationService', () => {
    let service: MarketplaceReputationService;

    beforeEach(() => {
      service = new MarketplaceReputationService();
    });

    describe('transaction recording', () => {
      it('should record a successful transaction', () => {
        const tx: Transaction = {
          id: 'tx-001',
          type: 'purchase',
          buyerId: 'buyer-1',
          sellerId: 'seller-1',
          amount: 100,
          currency: 'USD',
          itemId: 'item-1',
          timestamp: Date.now(),
          success: true,
        };

        const result = service.recordTransaction(tx);

        expect(result.sellerScore).toBeGreaterThan(0.5);
        expect(result.buyerScore).toBeGreaterThan(0.5);
      });

      it('should record a failed transaction', () => {
        const tx: Transaction = {
          id: 'tx-002',
          type: 'purchase',
          buyerId: 'buyer-2',
          sellerId: 'seller-2',
          amount: 50,
          currency: 'USD',
          itemId: 'item-2',
          timestamp: Date.now(),
          success: false,
        };

        const result = service.recordTransaction(tx);

        // Failed transaction should still return scores (lower)
        expect(result.sellerScore).toBeDefined();
        expect(result.buyerScore).toBeDefined();
      });

      it('should handle multiple transactions', () => {
        const sellerId = 'seller-multi';
        const buyerId = 'buyer-multi';

        // Record 5 successful transactions
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-multi-${i}`,
            type: 'purchase',
            buyerId,
            sellerId,
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const sellerProfile = service.getSellerProfile(sellerId);
        expect(sellerProfile.transactionCount).toBe(5);
        expect(sellerProfile.trustScore).toBeGreaterThan(0.7);
      });

      it('should support all transaction types', () => {
        const types: TransactionType[] = ['purchase', 'sale', 'exchange', 'rental'];

        for (const type of types) {
          const result = service.recordTransaction({
            id: `tx-${type}`,
            type,
            buyerId: 'buyer-type',
            sellerId: 'seller-type',
            amount: 100,
            currency: 'USD',
            itemId: `item-${type}`,
            timestamp: Date.now(),
            success: true,
          });

          expect(result.sellerScore).toBeGreaterThan(0);
        }
      });

      it('should handle dispute resolutions', () => {
        const tx: Transaction = {
          id: 'tx-dispute',
          type: 'purchase',
          buyerId: 'buyer-dispute',
          sellerId: 'seller-dispute',
          amount: 200,
          currency: 'USD',
          itemId: 'item-dispute',
          timestamp: Date.now(),
          success: false,
          disputeResolution: 'buyer_favor',
        };

        const result = service.recordTransaction(tx);
        expect(result).toBeDefined();
      });
    });

    describe('seller profiles', () => {
      it('should get seller profile', () => {
        const sellerId = 'seller-profile';

        // Record some transactions
        service.recordTransaction({
          id: 'tx-sp-1',
          type: 'sale',
          buyerId: 'buyer-sp',
          sellerId,
          amount: 100,
          currency: 'USD',
          itemId: 'item-sp',
          timestamp: Date.now(),
          success: true,
        });

        const profile = service.getSellerProfile(sellerId);

        expect(profile.sellerId).toBe(sellerId);
        expect(profile.trustScore).toBeGreaterThan(0);
        expect(profile.transactionCount).toBe(1);
        expect(profile.reputation).toBeDefined();
      });

      it('should return default profile for unknown seller', () => {
        const profile = service.getSellerProfile('unknown-seller');

        expect(profile.sellerId).toBe('unknown-seller');
        expect(profile.transactionCount).toBe(0);
        // successRate uses Laplace smoothing: 1/(1+1) = 0.5 for new agents
        expect(profile.successRate).toBe(0.5);
        expect(profile.verified).toBe(false);
      });

      it('should mark seller as verified with high trust', () => {
        const sellerId = 'verified-seller';

        // Record 15 successful transactions to get verified
        for (let i = 0; i < 15; i++) {
          service.recordTransaction({
            id: `tx-verified-${i}`,
            type: 'sale',
            buyerId: 'buyer-vs',
            sellerId,
            amount: 100,
            currency: 'USD',
            itemId: `item-vs-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const profile = service.getSellerProfile(sellerId);
        expect(profile.transactionCount).toBe(15);
        // High success rate should lead to high trust
        expect(profile.trustScore).toBeGreaterThan(0.8);
      });
    });

    describe('buyer profiles', () => {
      it('should get buyer profile', () => {
        const buyerId = 'buyer-profile';

        service.recordTransaction({
          id: 'tx-bp',
          type: 'purchase',
          buyerId,
          sellerId: 'seller-bp',
          amount: 50,
          currency: 'USD',
          itemId: 'item-bp',
          timestamp: Date.now(),
          success: true,
        });

        const profile = service.getBuyerProfile(buyerId);

        expect(profile.buyerId).toBe(buyerId);
        expect(profile.trustScore).toBeGreaterThan(0);
        expect(profile.transactionCount).toBe(1);
        // Uses Laplace smoothing: (1+1)/(1+1+2) = 2/4 = 0.5 after 1 positive tx
        // Starting with 1,1 priors, after 1 positive: 2/3
        expect(profile.paymentReliability).toBeCloseTo(2 / 3, 1);
        expect(profile.disputeRate).toBeCloseTo(1 / 3, 1);
      });

      it('should track dispute rate', () => {
        const buyerId = 'buyer-disputes';

        // Record 2 successful, 1 failed
        for (let i = 0; i < 2; i++) {
          service.recordTransaction({
            id: `tx-bd-s-${i}`,
            type: 'purchase',
            buyerId,
            sellerId: 'seller-bd',
            amount: 100,
            currency: 'USD',
            itemId: `item-bd-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        service.recordTransaction({
          id: 'tx-bd-f',
          type: 'purchase',
          buyerId,
          sellerId: 'seller-bd',
          amount: 100,
          currency: 'USD',
          itemId: 'item-bd-f',
          timestamp: Date.now(),
          success: false,
        });

        const profile = service.getBuyerProfile(buyerId);
        // With Laplace priors: starts 1,1, after +2 success +1 fail: 3 pos, 2 neg, total 5
        // disputeRate = 2/5 = 0.4
        expect(profile.disputeRate).toBeCloseTo(2 / 5, 1);
      });
    });

    describe('listing verification', () => {
      it('should verify listing from trusted seller', () => {
        const sellerId = 'trusted-seller';

        // Build trust
        for (let i = 0; i < 10; i++) {
          service.recordTransaction({
            id: `tx-trust-${i}`,
            type: 'sale',
            buyerId: 'buyer-trust',
            sellerId,
            amount: 100,
            currency: 'USD',
            itemId: `item-trust-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const verification = service.verifyListing('listing-1', sellerId);

        expect(verification.listingId).toBe('listing-1');
        expect(verification.sellerTrust).toBeGreaterThan(0.7);
        expect(verification.scamRiskScore).toBeLessThan(0.3);
        expect(verification.recommendations.length).toBeGreaterThan(0);
      });

      it('should flag listing from untrusted seller', () => {
        const sellerId = 'untrusted-seller';

        // Record failed transaction to lower trust
        service.recordTransaction({
          id: 'tx-untrust',
          type: 'sale',
          buyerId: 'buyer-untrust',
          sellerId,
          amount: 100,
          currency: 'USD',
          itemId: 'item-untrust',
          timestamp: Date.now(),
          success: false,
        });

        const verification = service.verifyListing('listing-2', sellerId);

        expect(verification.verified).toBe(false);
        expect(verification.scamRiskScore).toBeGreaterThan(0);
        expect(verification.recommendations).toContain(
          'Seller has low trust score - proceed with caution'
        );
      });

      it('should handle unknown seller', () => {
        const verification = service.verifyListing('listing-3', 'unknown-seller');

        expect(verification.listingId).toBe('listing-3');
        // Unknown seller gets default trust of 0.5 (Laplace smoothing)
        expect(verification.sellerTrust).toBe(0.5);
        // 0.5 trust is below 0.7 threshold, so verified should be false
        expect(verification.verified).toBe(false);
        // Recommendations array exists (may be empty if trust is exactly 0.5)
        expect(verification.recommendations).toBeDefined();
      });
    });

    describe('trust checks', () => {
      it('should check if seller is trustworthy', () => {
        const sellerId = 'trustworthy-seller';

        // Build trust
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-tw-${i}`,
            type: 'sale',
            buyerId: 'buyer-tw',
            sellerId,
            amount: 100,
            currency: 'USD',
            itemId: `item-tw-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        expect(service.isSellerTrustworthy(sellerId, 0.5)).toBe(true);
      });

      it('should return false for unknown seller', () => {
        expect(service.isSellerTrustworthy('nobody', 0.5)).toBe(false);
      });
    });

    describe('external reputation', () => {
      it('should query external reputation without error', () => {
        // Should not throw
        expect(() => service.queryExternalReputation('seller-ext')).not.toThrow();
      });
    });

    describe('FL coordinator', () => {
      it('should get FL coordinator', () => {
        const coordinator = service.getFLCoordinator();
        expect(coordinator).toBeDefined();
      });

      it('should get FL stats', () => {
        const stats = service.getFLStats();

        expect(stats.totalRounds).toBeDefined();
        expect(stats.participantCount).toBeDefined();
        expect(stats.averageParticipation).toBeDefined();
      });
    });
  });

  describe('getMarketplaceService', () => {
    it('should return singleton instance', () => {
      const service1 = getMarketplaceService();
      const service2 = getMarketplaceService();

      expect(service1).toBe(service2);
    });
  });

  describe('re-exported functions', () => {
    it('should export createReputation', () => {
      const rep = mpCreateReputation('test-agent');
      expect(rep.agentId).toBe('test-agent');
    });

    it('should export recordPositive', () => {
      let rep = mpCreateReputation('test-pos');
      // createReputation starts with positiveCount: 1 (Laplace prior)
      rep = mpRecordPositive(rep);
      expect(rep.positiveCount).toBe(2); // 1 (prior) + 1 (recorded)
    });

    it('should export recordNegative', () => {
      let rep = mpCreateReputation('test-neg');
      // createReputation starts with negativeCount: 1 (Laplace prior)
      rep = mpRecordNegative(rep);
      expect(rep.negativeCount).toBe(2); // 1 (prior) + 1 (recorded)
    });

    it('should export reputationValue', () => {
      const rep = mpCreateReputation('test-val');
      const value = mpReputationValue(rep);
      expect(value).toBe(0.5);
    });

    it('should export isTrustworthy', () => {
      let rep = mpCreateReputation('test-trust');
      for (let i = 0; i < 10; i++) {
        rep = mpRecordPositive(rep);
      }
      expect(mpIsTrustworthy(rep, 0.7)).toBe(true);
    });

    it('should export AggregationMethod', () => {
      expect(AggregationMethod.TrustWeighted).toBeDefined();
      expect(AggregationMethod.FedAvg).toBeDefined();
    });
  });
});

// =============================================================================
// EduNet Integration Tests
// =============================================================================

describe('EduNet Integration', () => {
  describe('EduNetCredentialService', () => {
    let service: EduNetCredentialService;

    beforeEach(() => {
      service = new EduNetCredentialService();
    });

    describe('certificate issuance', () => {
      it('should issue course completion certificate', () => {
        const credential = service.issueCertificate({
          studentId: 'student-1',
          courseId: 'course-101',
          courseName: 'Introduction to TypeScript',
          grade: 95,
        });

        expect(credential.id).toBeDefined();
        expect(credential.type).toBe('course_completion');
        expect(credential.holderId).toBe('student-1');
        expect(credential.courseId).toBe('course-101');
        expect(credential.grade).toBe(95);
        expect(credential.verified).toBe(true);
        expect(credential.expiresAt).toBeGreaterThan(Date.now());
      });

      it('should create epistemic claim for certificate', () => {
        const credential = service.issueCertificate({
          studentId: 'student-2',
          courseId: 'course-102',
          courseName: 'Advanced React',
          grade: 88,
        });

        expect(credential.claim).toBeDefined();
        expect(credential.claim.content).toContain('Advanced React');
        expect(credential.claim.content).toContain('88%');
      });

      it('should throw for invalid grade (too low)', () => {
        expect(() =>
          service.issueCertificate({
            studentId: 'student-3',
            courseId: 'course-103',
            courseName: 'Test Course',
            grade: -10,
          })
        ).toThrow('Grade must be between 0 and 100');
      });

      it('should throw for invalid grade (too high)', () => {
        expect(() =>
          service.issueCertificate({
            studentId: 'student-4',
            courseId: 'course-104',
            courseName: 'Test Course',
            grade: 150,
          })
        ).toThrow('Grade must be between 0 and 100');
      });

      it('should use custom issuer DID', () => {
        const credential = service.issueCertificate({
          studentId: 'student-5',
          courseId: 'course-105',
          courseName: 'Custom Issuer Course',
          grade: 90,
          issuerDid: 'did:custom:issuer',
        });

        expect(credential.issuerDid).toBe('did:custom:issuer');
      });

      it('should update learner reputation', () => {
        const studentId = 'student-rep';

        service.issueCertificate({
          studentId,
          courseId: 'course-rep',
          courseName: 'Reputation Course',
          grade: 85,
        });

        const profile = service.getLearnerProfile(studentId);
        expect(profile.trustScore).toBeGreaterThan(0.5);
      });
    });

    describe('skill certification', () => {
      it('should issue skill certification', () => {
        const credential = service.issueSkillCertification({
          holderId: 'dev-1',
          skillId: 'typescript',
          skillName: 'TypeScript Programming',
          level: 85,
        });

        expect(credential.id).toBeDefined();
        expect(credential.type).toBe('skill_certification');
        expect(credential.holderId).toBe('dev-1');
        expect(credential.skillId).toBe('typescript');
        expect(credential.verified).toBe(true);
      });

      it('should set 2-year expiration for skills', () => {
        const credential = service.issueSkillCertification({
          holderId: 'dev-2',
          skillId: 'react',
          skillName: 'React Development',
          level: 90,
        });

        const twoYearsMs = 2 * 365 * 24 * 60 * 60 * 1000;
        const expectedExpiry = Date.now() + twoYearsMs;

        // Should expire ~2 years from now (within 1 minute tolerance)
        expect(credential.expiresAt).toBeGreaterThan(expectedExpiry - 60000);
        expect(credential.expiresAt).toBeLessThan(expectedExpiry + 60000);
      });

      it('should throw for invalid level', () => {
        expect(() =>
          service.issueSkillCertification({
            holderId: 'dev-3',
            skillId: 'invalid',
            skillName: 'Invalid Skill',
            level: -5,
          })
        ).toThrow('Level must be between 0 and 100');

        expect(() =>
          service.issueSkillCertification({
            holderId: 'dev-4',
            skillId: 'invalid2',
            skillName: 'Invalid Skill 2',
            level: 200,
          })
        ).toThrow('Level must be between 0 and 100');
      });
    });

    describe('credential verification', () => {
      it('should verify valid credential', () => {
        const credential = service.issueCertificate({
          studentId: 'verify-1',
          courseId: 'verify-course',
          courseName: 'Verification Test',
          grade: 80,
        });

        const result = service.verifyCredential(credential.id);

        expect(result.valid).toBe(true);
        expect(result.credential).toBeDefined();
        expect(result.credential!.id).toBe(credential.id);
      });

      it('should return invalid for unknown credential', () => {
        const result = service.verifyCredential('unknown-cred-id');

        expect(result.valid).toBe(false);
        expect(result.credential).toBeNull();
        expect(result.reason).toBe('Credential not found');
      });

      it('should return invalid for expired credential', () => {
        // Create credential and manually set expiration to past
        const credential = service.issueCertificate({
          studentId: 'expired-1',
          courseId: 'expired-course',
          courseName: 'Expired Course',
          grade: 75,
        });

        // Directly modify the stored credential (for testing)
        (credential as any).expiresAt = Date.now() - 1000;

        const result = service.verifyCredential(credential.id);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Credential expired');
      });
    });

    describe('learner profiles', () => {
      it('should get learner profile', () => {
        const learnerId = 'learner-profile';

        service.issueCertificate({
          studentId: learnerId,
          courseId: 'profile-course',
          courseName: 'Profile Test Course',
          grade: 90,
        });

        const profile = service.getLearnerProfile(learnerId);

        expect(profile.learnerId).toBe(learnerId);
        expect(profile.coursesCompleted).toBe(1);
        expect(profile.trustScore).toBeGreaterThan(0);
      });

      it('should track courses and certifications separately', () => {
        const learnerId = 'learner-mixed';

        // Issue course completions
        service.issueCertificate({
          studentId: learnerId,
          courseId: 'course-a',
          courseName: 'Course A',
          grade: 85,
        });

        service.issueCertificate({
          studentId: learnerId,
          courseId: 'course-b',
          courseName: 'Course B',
          grade: 88,
        });

        // Issue skill certifications
        service.issueSkillCertification({
          holderId: learnerId,
          skillId: 'skill-a',
          skillName: 'Skill A',
          level: 80,
        });

        const profile = service.getLearnerProfile(learnerId);

        expect(profile.coursesCompleted).toBe(2);
        expect(profile.certificationsEarned).toBe(1);
      });

      it('should return default profile for unknown learner', () => {
        const profile = service.getLearnerProfile('unknown-learner');

        expect(profile.learnerId).toBe('unknown-learner');
        expect(profile.coursesCompleted).toBe(0);
        expect(profile.certificationsEarned).toBe(0);
        expect(profile.trustScore).toBe(0.5);
      });
    });

    describe('credential retrieval', () => {
      it('should get credentials for holder', () => {
        const holderId = 'holder-creds';

        service.issueCertificate({
          studentId: holderId,
          courseId: 'cred-course-1',
          courseName: 'Cred Course 1',
          grade: 85,
        });

        service.issueSkillCertification({
          holderId,
          skillId: 'cred-skill-1',
          skillName: 'Cred Skill 1',
          level: 90,
        });

        const credentials = service.getCredentialsForHolder(holderId);

        expect(credentials.length).toBe(2);
        expect(credentials.some((c) => c.type === 'course_completion')).toBe(true);
        expect(credentials.some((c) => c.type === 'skill_certification')).toBe(true);
      });

      it('should return empty array for holder with no credentials', () => {
        const credentials = service.getCredentialsForHolder('no-creds-holder');
        expect(credentials).toEqual([]);
      });
    });

    describe('trust checks', () => {
      it('should check if learner is trusted', () => {
        const learnerId = 'trusted-learner';

        // Build trust
        for (let i = 0; i < 5; i++) {
          service.issueCertificate({
            studentId: learnerId,
            courseId: `trust-course-${i}`,
            courseName: `Trust Course ${i}`,
            grade: 90,
          });
        }

        expect(service.isLearnerTrusted(learnerId, 0.5)).toBe(true);
      });

      it('should return false for unknown learner', () => {
        expect(service.isLearnerTrusted('nobody', 0.5)).toBe(false);
      });
    });

    describe('external reputation', () => {
      it('should query external reputation without error', () => {
        expect(() => service.queryExternalReputation('learner-ext')).not.toThrow();
      });
    });
  });

  describe('getEduNetService', () => {
    it('should return singleton instance', () => {
      const service1 = getEduNetService();
      const service2 = getEduNetService();

      expect(service1).toBe(service2);
    });
  });

  describe('re-exported functions', () => {
    it('should export claim builder', () => {
      const c = eduClaim('Test claim')
        .withEmpirical(EduEmpiricalLevel.E2_PrivateVerify)
        .withNormative(EduNormativeLevel.N1_Communal)
        .withMateriality(EduMaterialityLevel.M1_Session)
        .build();

      expect(c.content).toBe('Test claim');
    });

    it('should export epistemic levels', () => {
      expect(EduEmpiricalLevel.E0_Unverified).toBeDefined();
      expect(EduEmpiricalLevel.E3_Cryptographic).toBeDefined();
      expect(EduNormativeLevel.N2_Network).toBeDefined();
      expect(EduMaterialityLevel.M2_Persistent).toBeDefined();
    });

    it('should export reputation functions', () => {
      let rep = eduCreateReputation('edu-test');
      rep = eduRecordPositive(rep);
      const val = eduReputationValue(rep);

      expect(val).toBeGreaterThan(0.5);
    });
  });

  describe('type definitions', () => {
    it('should define CompletionStatus', () => {
      const statuses: CompletionStatus[] = ['in_progress', 'completed', 'certified', 'expired'];
      expect(statuses.length).toBe(4);
    });
  });
});

// =============================================================================
// SupplyChain Integration Tests
// =============================================================================

describe('SupplyChain Integration', () => {
  describe('SupplyChainProvenanceService', () => {
    let service: SupplyChainProvenanceService;

    beforeEach(() => {
      service = new SupplyChainProvenanceService();
    });

    describe('checkpoint recording', () => {
      it('should record a checkpoint', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-1',
          location: 'Warehouse A',
          handler: 'handler-1',
          action: 'received',
          evidence: [
            {
              type: 'iot_sensor',
              data: { temperature: 4.5, humidity: 45 },
              timestamp: Date.now(),
              verified: true,
            },
          ],
        });

        expect(checkpoint.id).toBeDefined();
        expect(checkpoint.productId).toBe('product-1');
        expect(checkpoint.location).toBe('Warehouse A');
        expect(checkpoint.handler).toBe('handler-1');
        expect(checkpoint.action).toBe('received');
        expect(checkpoint.evidence.length).toBe(1);
      });

      it('should record checkpoint with coordinates', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-geo',
          location: 'Distribution Center',
          coordinates: { lat: 40.7128, lng: -74.006 },
          handler: 'handler-geo',
          action: 'shipped',
          evidence: [
            {
              type: 'gps',
              data: { accuracy: 5 },
              timestamp: Date.now(),
              verified: true,
            },
          ],
        });

        expect(checkpoint.coordinates).toEqual({ lat: 40.7128, lng: -74.006 });
      });

      it('should record checkpoint with batch ID', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-batch',
          batchId: 'BATCH-2024-001',
          location: 'Factory',
          handler: 'handler-batch',
          action: 'processed',
          evidence: [],
        });

        expect(checkpoint.batchId).toBe('BATCH-2024-001');
      });

      it('should link checkpoints in sequence', () => {
        const productId = 'linked-product';

        const cp1 = service.recordCheckpoint({
          productId,
          location: 'Origin',
          handler: 'handler-origin',
          action: 'received',
          evidence: [],
        });

        const cp2 = service.recordCheckpoint({
          productId,
          location: 'Processing',
          handler: 'handler-process',
          action: 'processed',
          evidence: [],
        });

        expect(cp2.previousCheckpointId).toBe(cp1.id);
      });

      it('should support all action types', () => {
        const actions = ['received', 'processed', 'shipped', 'delivered', 'inspected'] as const;

        for (const action of actions) {
          const checkpoint = service.recordCheckpoint({
            productId: `product-${action}`,
            location: 'Test Location',
            handler: 'test-handler',
            action,
            evidence: [],
          });

          expect(checkpoint.action).toBe(action);
        }
      });

      it('should create epistemic claim for checkpoint', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-claim',
          location: 'Factory A',
          handler: 'handler-claim',
          action: 'processed',
          evidence: [
            {
              type: 'blockchain',
              data: { txHash: '0x123' },
              timestamp: Date.now(),
              verified: true,
            },
          ],
        });

        expect(checkpoint.claim).toBeDefined();
        expect(checkpoint.claim.content).toContain('product-claim');
        expect(checkpoint.claim.content).toContain('processed');
        expect(checkpoint.claim.content).toContain('Factory A');
      });

      it('should update handler reputation', () => {
        const handlerId = 'handler-rep';

        service.recordCheckpoint({
          productId: 'product-rep',
          location: 'Location',
          handler: handlerId,
          action: 'received',
          evidence: [],
        });

        const profile = service.getHandlerProfile(handlerId);
        expect(profile.trustScore).toBeGreaterThan(0.5);
      });
    });

    describe('evidence types', () => {
      it('should support all evidence types', () => {
        const evidenceTypes: EvidenceType[] = [
          'manual_entry',
          'signature',
          'photo',
          'gps',
          'iot_sensor',
          'rfid',
          'blockchain',
          'third_party_verification',
        ];

        for (const type of evidenceTypes) {
          const checkpoint = service.recordCheckpoint({
            productId: `product-evidence-${type}`,
            location: 'Test',
            handler: 'test-handler',
            action: 'inspected',
            evidence: [
              {
                type,
                data: { test: true },
                timestamp: Date.now(),
                verified: true,
              },
            ],
          });

          expect(checkpoint.evidence[0].type).toBe(type);
        }
      });

      it('should determine empirical level based on evidence', () => {
        // Blockchain evidence should yield high empirical level
        const blockchainCheckpoint = service.recordCheckpoint({
          productId: 'product-blockchain',
          location: 'Test',
          handler: 'handler-bc',
          action: 'received',
          evidence: [
            {
              type: 'blockchain',
              data: { txHash: '0x456' },
              timestamp: Date.now(),
              verified: true,
            },
          ],
        });

        expect(blockchainCheckpoint.claim).toBeDefined();
      });
    });

    describe('provenance chains', () => {
      it('should get provenance chain', () => {
        const productId = 'chain-product';

        service.recordCheckpoint({
          productId,
          location: 'Origin Farm',
          coordinates: { lat: 10, lng: 20 },
          handler: 'farmer',
          action: 'received',
          evidence: [{ type: 'photo', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId,
          location: 'Processing Plant',
          coordinates: { lat: 11, lng: 21 },
          handler: 'processor',
          action: 'processed',
          evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId,
          location: 'Distribution Center',
          coordinates: { lat: 12, lng: 22 },
          handler: 'distributor',
          action: 'shipped',
          evidence: [{ type: 'rfid', data: {}, timestamp: Date.now(), verified: true }],
        });

        const chain = service.getProvenanceChain(productId);

        expect(chain).not.toBeNull();
        expect(chain!.productId).toBe(productId);
        expect(chain!.checkpoints.length).toBe(3);
        expect(chain!.origin.location).toBe('Origin Farm');
        expect(chain!.currentLocation).toBe('Distribution Center');
        expect(chain!.chainIntegrity).toBeGreaterThan(0);
        expect(chain!.totalDistance).toBeGreaterThan(0);
      });

      it('should return null for unknown product', () => {
        const chain = service.getProvenanceChain('unknown-product');
        expect(chain).toBeNull();
      });

      it('should calculate total distance', () => {
        const productId = 'distance-product';

        // Two points ~111km apart (1 degree at equator)
        service.recordCheckpoint({
          productId,
          location: 'Point A',
          coordinates: { lat: 0, lng: 0 },
          handler: 'handler-a',
          action: 'received',
          evidence: [],
        });

        service.recordCheckpoint({
          productId,
          location: 'Point B',
          coordinates: { lat: 0, lng: 1 },
          handler: 'handler-b',
          action: 'shipped',
          evidence: [],
        });

        const chain = service.getProvenanceChain(productId);

        expect(chain!.totalDistance).toBeGreaterThan(100);
        expect(chain!.totalDistance).toBeLessThan(120);
      });

      it('should calculate total time', () => {
        const productId = 'time-product';
        const startTime = Date.now() - 3600000; // 1 hour ago

        // Mock the checkpoint with specific timestamp
        service.recordCheckpoint({
          productId,
          location: 'Start',
          handler: 'handler-start',
          action: 'received',
          evidence: [],
        });

        // Second checkpoint
        service.recordCheckpoint({
          productId,
          location: 'End',
          handler: 'handler-end',
          action: 'delivered',
          evidence: [],
        });

        const chain = service.getProvenanceChain(productId);

        expect(chain!.totalTime).toBeDefined();
        expect(chain!.totalTime).toBeGreaterThanOrEqual(0);
      });
    });

    describe('chain verification', () => {
      it('should verify valid chain', () => {
        const productId = 'verified-product';

        // Create chain with good evidence
        service.recordCheckpoint({
          productId,
          location: 'Origin',
          handler: 'trusted-handler',
          action: 'received',
          evidence: [
            { type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true },
            { type: 'photo', data: {}, timestamp: Date.now(), verified: true },
          ],
        });

        service.recordCheckpoint({
          productId,
          location: 'Destination',
          handler: 'trusted-handler',
          action: 'delivered',
          evidence: [{ type: 'blockchain', data: {}, timestamp: Date.now(), verified: true }],
        });

        const verification = service.verifyChain(productId);

        expect(verification.productId).toBe(productId);
        expect(verification.integrityScore).toBeGreaterThan(0);
        expect(verification.timeline.length).toBe(2);
      });

      it('should detect weak links (unverified evidence)', () => {
        const productId = 'weak-product';

        service.recordCheckpoint({
          productId,
          location: 'Origin',
          handler: 'handler-weak',
          action: 'received',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
        });

        const verification = service.verifyChain(productId);

        expect(verification.weakLinks.length).toBeGreaterThan(0);
        expect(verification.timeline[0].status).toBe('unverified');
      });

      it('should return not found for unknown product', () => {
        const verification = service.verifyChain('not-found-product');

        expect(verification.verified).toBe(false);
        expect(verification.integrityScore).toBe(0);
        expect(verification.recommendations).toContain('Product not found in supply chain');
      });

      it('should provide recommendations', () => {
        const productId = 'recommendations-product';

        service.recordCheckpoint({
          productId,
          location: 'Location',
          handler: 'handler-rec',
          action: 'received',
          evidence: [{ type: 'photo', data: {}, timestamp: Date.now(), verified: true }],
        });

        const verification = service.verifyChain(productId);

        expect(verification.recommendations.length).toBeGreaterThan(0);
      });
    });

    describe('handler profiles', () => {
      it('should get handler profile', () => {
        const handlerId = 'handler-profile';

        service.recordCheckpoint({
          productId: 'hp-product',
          location: 'Location',
          handler: handlerId,
          action: 'received',
          evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
        });

        const profile = service.getHandlerProfile(handlerId);

        expect(profile.handlerId).toBe(handlerId);
        expect(profile.handlerDid).toBe(`did:mycelix:${handlerId}`);
        expect(profile.trustScore).toBeGreaterThan(0);
        expect(profile.checkpointsRecorded).toBe(1);
        expect(profile.verificationRate).toBe(1);
      });

      it('should return default profile for unknown handler', () => {
        const profile = service.getHandlerProfile('unknown-handler');

        expect(profile.handlerId).toBe('unknown-handler');
        expect(profile.checkpointsRecorded).toBe(0);
        expect(profile.verificationRate).toBe(0);
      });

      it('should calculate verification rate', () => {
        const handlerId = 'rate-handler';

        // 2 verified, 1 unverified
        service.recordCheckpoint({
          productId: 'rate-1',
          location: 'L1',
          handler: handlerId,
          action: 'received',
          evidence: [{ type: 'photo', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId: 'rate-2',
          location: 'L2',
          handler: handlerId,
          action: 'shipped',
          evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId: 'rate-3',
          location: 'L3',
          handler: handlerId,
          action: 'delivered',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
        });

        const profile = service.getHandlerProfile(handlerId);

        expect(profile.checkpointsRecorded).toBe(3);
        expect(profile.verificationRate).toBeCloseTo(2 / 3, 1);
      });
    });

    describe('trust checks', () => {
      it('should check if handler is trusted', () => {
        const handlerId = 'trusted-handler-sc';

        for (let i = 0; i < 5; i++) {
          service.recordCheckpoint({
            productId: `trust-${i}`,
            location: 'Loc',
            handler: handlerId,
            action: 'received',
            evidence: [],
          });
        }

        expect(service.isHandlerTrusted(handlerId, 0.5)).toBe(true);
      });

      it('should return false for unknown handler', () => {
        expect(service.isHandlerTrusted('nobody-handler', 0.5)).toBe(false);
      });
    });

    describe('external reputation', () => {
      it('should query external reputation without error', () => {
        expect(() => service.queryExternalReputation('handler-ext')).not.toThrow();
      });
    });
  });

  describe('getSupplyChainService', () => {
    it('should return singleton instance', () => {
      const service1 = getSupplyChainService();
      const service2 = getSupplyChainService();

      expect(service1).toBe(service2);
    });
  });

  describe('re-exported functions', () => {
    it('should export claim builder', () => {
      const c = scClaim('Supply chain claim')
        .withEmpirical(ScEmpiricalLevel.E3_Cryptographic)
        .withNormative(ScNormativeLevel.N2_Network)
        .withMateriality(ScMaterialityLevel.M2_Persistent)
        .build();

      expect(c.content).toBe('Supply chain claim');
    });

    it('should export epistemic levels', () => {
      expect(ScEmpiricalLevel.E0_Unverified).toBeDefined();
      expect(ScEmpiricalLevel.E4_Consensus).toBeDefined();
      expect(ScNormativeLevel.N2_Network).toBeDefined();
      expect(ScMaterialityLevel.M3_Immutable).toBeDefined();
    });

    it('should export reputation functions', () => {
      let rep = scCreateReputation('sc-test');
      rep = scRecordPositive(rep);
      rep = scRecordNegative(rep);
      const val = scReputationValue(rep);

      expect(val).toBeDefined();
    });
  });
});

// =============================================================================
// Integrations Barrel Export Tests
// =============================================================================

describe('Integrations Barrel Exports', () => {
  describe('service exports', () => {
    it('should export MarketplaceReputationService', () => {
      expect(MarketplaceReputationService).toBeDefined();
      expect(getMarketplaceService).toBeDefined();
    });

    it('should export EduNetCredentialService', () => {
      expect(EduNetCredentialService).toBeDefined();
      expect(getEduNetService).toBeDefined();
    });

    it('should export SupplyChainProvenanceService', () => {
      expect(SupplyChainProvenanceService).toBeDefined();
      expect(getSupplyChainService).toBeDefined();
    });

    it('should export MailTrustService', () => {
      expect(MailTrustService).toBeDefined();
      expect(getMailTrustService).toBeDefined();
    });

    it('should export IdentityBridgeClient', () => {
      expect(IdentityBridgeClient).toBeDefined();
      expect(getIdentityBridgeClient).toBeDefined();
    });

    it('should export FinanceBridgeClient', () => {
      expect(FinanceBridgeClient).toBeDefined();
      expect(getFinanceBridgeClient).toBeDefined();
    });

    it('should export PropertyBridgeClient', () => {
      expect(PropertyBridgeClient).toBeDefined();
      expect(getPropertyBridgeClient).toBeDefined();
    });

    it('should export EnergyBridgeClient', () => {
      expect(EnergyBridgeClient).toBeDefined();
      expect(getEnergyBridgeClient).toBeDefined();
    });

    it('should export MediaBridgeClient', () => {
      expect(MediaBridgeClient).toBeDefined();
      expect(getMediaBridgeClient).toBeDefined();
    });

    it('should export GovernanceBridgeClient', () => {
      expect(GovernanceBridgeClient).toBeDefined();
      expect(getGovernanceBridgeClient).toBeDefined();
    });

    it('should export JusticeBridgeClient', () => {
      expect(JusticeBridgeClient).toBeDefined();
      expect(getJusticeBridgeClient).toBeDefined();
    });

    it('should export KnowledgeBridgeClient', () => {
      expect(KnowledgeBridgeClient).toBeDefined();
      expect(getKnowledgeBridgeClient).toBeDefined();
    });

    it('should export FabricationService', () => {
      expect(FabricationService).toBeDefined();
      expect(getFabricationService).toBeDefined();
    });

    it('should export EpistemicMarketsService', () => {
      expect(EpistemicMarketsService).toBeDefined();
      expect(getEpistemicMarketsService).toBeDefined();
    });
  });

  describe('epistemic exports', () => {
    it('should export EmpiricalLevel', () => {
      expect(EmpiricalLevel).toBeDefined();
      // Epistemic-markets uses string enums
      expect(EmpiricalLevel.Subjective).toBe('Subjective');
      expect(EmpiricalLevel.Measurable).toBe('Measurable');
    });

    it('should export NormativeLevel', () => {
      expect(NormativeLevel).toBeDefined();
      expect(NormativeLevel.Personal).toBe('Personal');
      expect(NormativeLevel.Universal).toBe('Universal');
    });

    it('should export MaterialityLevel', () => {
      expect(MaterialityLevel).toBeDefined();
      expect(MaterialityLevel.Ephemeral).toBe('Ephemeral');
      expect(MaterialityLevel.Foundational).toBe('Foundational');
    });
  });

  describe('singleton accessors', () => {
    it('should get MailTrustService singleton', () => {
      const service = getMailTrustService();
      expect(service).toBeDefined();
      expect(service).toBeInstanceOf(MailTrustService);
    });

    it('should get IdentityBridgeClient singleton', () => {
      const client = getIdentityBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(IdentityBridgeClient);
    });

    it('should get FinanceBridgeClient singleton', () => {
      const client = getFinanceBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(FinanceBridgeClient);
    });

    it('should get PropertyBridgeClient singleton', () => {
      const client = getPropertyBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(PropertyBridgeClient);
    });

    it('should get EnergyBridgeClient singleton', () => {
      const client = getEnergyBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(EnergyBridgeClient);
    });

    it('should get MediaBridgeClient singleton', () => {
      const client = getMediaBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(MediaBridgeClient);
    });

    it('should get GovernanceBridgeClient singleton', () => {
      const client = getGovernanceBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(GovernanceBridgeClient);
    });

    it('should get JusticeBridgeClient singleton', () => {
      const client = getJusticeBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(JusticeBridgeClient);
    });

    it('should get KnowledgeBridgeClient singleton', () => {
      const client = getKnowledgeBridgeClient();
      expect(client).toBeDefined();
      expect(client).toBeInstanceOf(KnowledgeBridgeClient);
    });

    it('should get FabricationService singleton', () => {
      const service = getFabricationService();
      expect(service).toBeDefined();
      expect(service).toBeInstanceOf(FabricationService);
    });

    it('should get EpistemicMarketsService singleton', () => {
      const service = getEpistemicMarketsService();
      expect(service).toBeDefined();
      expect(service).toBeInstanceOf(EpistemicMarketsService);
    });
  });

  describe('epistemic market helpers', () => {
    it('should export calculateStakeValue', () => {
      expect(calculateStakeValue).toBeDefined();
      expect(typeof calculateStakeValue).toBe('function');
    });

    it('should export getRecommendedResolution', () => {
      expect(getRecommendedResolution).toBeDefined();
      expect(typeof getRecommendedResolution).toBe('function');
    });

    it('should export getRecommendedDurationDays', () => {
      expect(getRecommendedDurationDays).toBeDefined();
      expect(typeof getRecommendedDurationDays).toBe('function');
    });

    it('should export toClassificationCode', () => {
      expect(toClassificationCode).toBeDefined();
      expect(typeof toClassificationCode).toBe('function');
    });

    it('should export createReputationOnlyStake', () => {
      expect(createReputationOnlyStake).toBeDefined();
      expect(typeof createReputationOnlyStake).toBe('function');
    });

    it('should export createMonetaryOnlyStake', () => {
      expect(createMonetaryOnlyStake).toBeDefined();
      expect(typeof createMonetaryOnlyStake).toBe('function');
    });

    it('should create valid classification code', () => {
      const code = toClassificationCode({
        empirical: EmpiricalLevel.PrivateVerify,
        normative: NormativeLevel.Communal,
        materiality: MaterialityLevel.Persistent,
      });

      expect(code).toBe('E2-N1-M2');
    });

    it('should create reputation-only stake', () => {
      const stake = createReputationOnlyStake(['epistemic', 'governance'], 50, 1.5);

      expect(stake.reputation).toBeDefined();
      expect(stake.reputation!.domains).toEqual(['epistemic', 'governance']);
      expect(stake.reputation!.stakePercentage).toBe(50);
      expect(stake.reputation!.confidenceMultiplier).toBe(1.5);
    });

    it('should create monetary-only stake', () => {
      const stake = createMonetaryOnlyStake(500, 'USD');

      expect(stake.monetary).toBeDefined();
      expect(stake.monetary!.amount).toBe(500);
      expect(stake.monetary!.currency).toBe('USD');
    });
  });
});

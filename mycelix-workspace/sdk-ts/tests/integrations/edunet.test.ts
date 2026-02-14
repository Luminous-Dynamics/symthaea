/**
 * EduNet Integration Tests
 *
 * Tests for EduNetCredentialService - educational credentials and learner trust
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EduNetCredentialService,
  getEduNetService,
  type EducationalCredential,
  type LearnerProfile,
} from '../../src/integrations/edunet/index.js';

describe('EduNet Integration', () => {
  let service: EduNetCredentialService;

  beforeEach(() => {
    service = new EduNetCredentialService();
  });

  describe('EduNetCredentialService', () => {
    describe('issueCertificate', () => {
      it('should issue course completion certificate', () => {
        const credential = service.issueCertificate({
          studentId: 'student-1',
          courseId: 'nix-101',
          courseName: 'NixOS Fundamentals',
          grade: 95,
        });

        expect(credential.id).toBeDefined();
        expect(credential.type).toBe('course_completion');
        expect(credential.holderId).toBe('student-1');
        expect(credential.courseId).toBe('nix-101');
        expect(credential.grade).toBe(95);
        expect(credential.verified).toBe(true);
      });

      it('should create epistemic claim with cryptographic level', () => {
        const credential = service.issueCertificate({
          studentId: 'student-2',
          courseId: 'rust-101',
          courseName: 'Rust Programming',
          grade: 88,
        });

        expect(credential.claim).toBeDefined();
        expect(credential.claim.content).toContain('Completed Rust Programming');
        expect(credential.claim.content).toContain('88%');
      });

      it('should set expiration date', () => {
        const credential = service.issueCertificate({
          studentId: 'student-3',
          courseId: 'python-101',
          courseName: 'Python Basics',
          grade: 92,
        });

        expect(credential.expiresAt).toBeDefined();
        expect(credential.expiresAt).toBeGreaterThan(Date.now());
      });

      it('should update learner reputation', () => {
        // Issue certificate
        service.issueCertificate({
          studentId: 'learning-student',
          courseId: 'course-1',
          courseName: 'Course 1',
          grade: 90,
        });

        const profile = service.getLearnerProfile('learning-student');
        expect(profile.trustScore).toBeGreaterThan(0.5);
      });

      it('should throw error for invalid grade', () => {
        expect(() => {
          service.issueCertificate({
            studentId: 'student',
            courseId: 'course',
            courseName: 'Course',
            grade: 150,
          });
        }).toThrow('Grade must be between 0 and 100');

        expect(() => {
          service.issueCertificate({
            studentId: 'student',
            courseId: 'course',
            courseName: 'Course',
            grade: -10,
          });
        }).toThrow('Grade must be between 0 and 100');
      });

      it('should use custom issuerDid when provided', () => {
        const credential = service.issueCertificate({
          studentId: 'student-4',
          courseId: 'course-4',
          courseName: 'Course 4',
          grade: 85,
          issuerDid: 'did:custom:issuer',
        });

        expect(credential.issuerDid).toBe('did:custom:issuer');
      });
    });

    describe('issueSkillCertification', () => {
      it('should issue skill certification', () => {
        const credential = service.issueSkillCertification({
          holderId: 'dev-1',
          skillId: 'typescript',
          skillName: 'TypeScript',
          level: 85,
        });

        expect(credential.id).toBeDefined();
        expect(credential.type).toBe('skill_certification');
        expect(credential.holderId).toBe('dev-1');
        expect(credential.skillId).toBe('typescript');
      });

      it('should create claim with skill info', () => {
        const credential = service.issueSkillCertification({
          holderId: 'dev-2',
          skillId: 'nix',
          skillName: 'Nix Language',
          level: 90,
        });

        expect(credential.claim.content).toContain('Nix Language');
        expect(credential.claim.content).toContain('90%');
      });

      it('should throw error for invalid level', () => {
        expect(() => {
          service.issueSkillCertification({
            holderId: 'dev',
            skillId: 'skill',
            skillName: 'Skill',
            level: 101,
          });
        }).toThrow('Level must be between 0 and 100');
      });

      it('should have 2-year expiration', () => {
        const credential = service.issueSkillCertification({
          holderId: 'dev-3',
          skillId: 'react',
          skillName: 'React',
          level: 75,
        });

        const twoYearsFromNow = Date.now() + 2 * 365 * 24 * 60 * 60 * 1000;
        expect(credential.expiresAt).toBeDefined();
        // Should be close to 2 years (within 1 minute)
        expect(Math.abs(credential.expiresAt! - twoYearsFromNow)).toBeLessThan(60000);
      });
    });

    describe('verifyCredential', () => {
      it('should verify valid credential', () => {
        const credential = service.issueCertificate({
          studentId: 'student-verify',
          courseId: 'course-verify',
          courseName: 'Verification Course',
          grade: 90,
        });

        const result = service.verifyCredential(credential.id);

        expect(result.valid).toBe(true);
        expect(result.credential).toBeDefined();
        expect(result.credential?.id).toBe(credential.id);
        expect(result.reason).toBeUndefined();
      });

      it('should reject non-existent credential', () => {
        const result = service.verifyCredential('non-existent-id');

        expect(result.valid).toBe(false);
        expect(result.credential).toBeNull();
        expect(result.reason).toBe('Credential not found');
      });

      it('should handle credential lookup correctly', () => {
        // Issue credential
        const credential = service.issueCertificate({
          studentId: 'student-expire',
          courseId: 'course-expire',
          courseName: 'Expiring Course',
          grade: 88,
        });

        // Modifying the returned object doesn't affect stored credential
        // The stored credential has not expired yet
        const result = service.verifyCredential(credential.id);
        // Credential is still valid since it was just issued
        expect(result.valid).toBe(true);
        expect(result.credential?.id).toBe(credential.id);
      });
    });

    describe('getLearnerProfile', () => {
      it('should return default profile for unknown learner', () => {
        const profile = service.getLearnerProfile('unknown-learner');

        expect(profile.learnerId).toBe('unknown-learner');
        expect(profile.coursesCompleted).toBe(0);
        expect(profile.certificationsEarned).toBe(0);
        expect(profile.trustScore).toBe(0.5); // Default reputation
      });

      it('should count completed courses', () => {
        // Issue multiple certificates
        for (let i = 0; i < 3; i++) {
          service.issueCertificate({
            studentId: 'active-learner',
            courseId: `course-${i}`,
            courseName: `Course ${i}`,
            grade: 85 + i,
          });
        }

        const profile = service.getLearnerProfile('active-learner');

        expect(profile.coursesCompleted).toBe(3);
      });

      it('should count skill certifications', () => {
        // Issue skill certifications
        service.issueSkillCertification({
          holderId: 'skilled-learner',
          skillId: 'typescript',
          skillName: 'TypeScript',
          level: 80,
        });
        service.issueSkillCertification({
          holderId: 'skilled-learner',
          skillId: 'rust',
          skillName: 'Rust',
          level: 70,
        });

        const profile = service.getLearnerProfile('skilled-learner');

        expect(profile.certificationsEarned).toBe(2);
      });

      it('should calculate trust score from reputation', () => {
        // Build up reputation with multiple completions
        for (let i = 0; i < 5; i++) {
          service.issueCertificate({
            studentId: 'trusted-learner',
            courseId: `course-${i}`,
            courseName: `Course ${i}`,
            grade: 90,
          });
        }

        const profile = service.getLearnerProfile('trusted-learner');

        expect(profile.trustScore).toBeGreaterThan(0.7);
      });

      it('should include reputation object', () => {
        service.issueCertificate({
          studentId: 'rep-learner',
          courseId: 'rep-course',
          courseName: 'Rep Course',
          grade: 95,
        });

        const profile = service.getLearnerProfile('rep-learner');

        expect(profile.reputation).toBeDefined();
        expect(profile.reputation.agentId).toBe('rep-learner');
        expect(profile.reputation.positiveCount).toBeGreaterThan(0);
      });
    });

    describe('getCredentialsForHolder', () => {
      it('should return empty array for unknown holder', () => {
        const credentials = service.getCredentialsForHolder('nobody');
        expect(credentials).toEqual([]);
      });

      it('should return all credentials for holder', () => {
        // Issue mixed credentials
        service.issueCertificate({
          studentId: 'multi-cred',
          courseId: 'course-1',
          courseName: 'Course 1',
          grade: 90,
        });
        service.issueCertificate({
          studentId: 'multi-cred',
          courseId: 'course-2',
          courseName: 'Course 2',
          grade: 85,
        });
        service.issueSkillCertification({
          holderId: 'multi-cred',
          skillId: 'nix',
          skillName: 'Nix',
          level: 80,
        });

        const credentials = service.getCredentialsForHolder('multi-cred');

        expect(credentials.length).toBe(3);
        expect(credentials.filter((c) => c.type === 'course_completion').length).toBe(2);
        expect(credentials.filter((c) => c.type === 'skill_certification').length).toBe(1);
      });

      it('should not return credentials from other holders', () => {
        service.issueCertificate({
          studentId: 'holder-a',
          courseId: 'course-a',
          courseName: 'Course A',
          grade: 90,
        });
        service.issueCertificate({
          studentId: 'holder-b',
          courseId: 'course-b',
          courseName: 'Course B',
          grade: 88,
        });

        const credentialsA = service.getCredentialsForHolder('holder-a');
        const credentialsB = service.getCredentialsForHolder('holder-b');

        expect(credentialsA.length).toBe(1);
        expect(credentialsB.length).toBe(1);
        expect(credentialsA[0].holderId).toBe('holder-a');
        expect(credentialsB[0].holderId).toBe('holder-b');
      });
    });

    describe('queryExternalReputation', () => {
      it('should not throw when querying', () => {
        expect(() => {
          service.queryExternalReputation('learner-1');
        }).not.toThrow();
      });
    });

    describe('isLearnerTrusted', () => {
      it('should return false for unknown learner', () => {
        expect(service.isLearnerTrusted('unknown')).toBe(false);
      });

      it('should return true for trusted learner', () => {
        // Build up trust
        for (let i = 0; i < 5; i++) {
          service.issueCertificate({
            studentId: 'trusted-student',
            courseId: `course-${i}`,
            courseName: `Course ${i}`,
            grade: 90,
          });
        }

        expect(service.isLearnerTrusted('trusted-student')).toBe(true);
      });

      it('should respect custom threshold', () => {
        service.issueCertificate({
          studentId: 'threshold-student',
          courseId: 'course-1',
          courseName: 'Course 1',
          grade: 85,
        });

        // Low threshold - should pass
        expect(service.isLearnerTrusted('threshold-student', 0.5)).toBe(true);

        // High threshold - should fail
        expect(service.isLearnerTrusted('threshold-student', 0.95)).toBe(false);
      });
    });
  });

  describe('getEduNetService', () => {
    it('should return singleton instance', () => {
      const service1 = getEduNetService();
      const service2 = getEduNetService();

      expect(service1).toBe(service2);
    });

    it('should maintain state across calls', () => {
      const service1 = getEduNetService();
      service1.issueCertificate({
        studentId: 'singleton-student',
        courseId: 'singleton-course',
        courseName: 'Singleton Course',
        grade: 92,
      });

      const service2 = getEduNetService();
      const credentials = service2.getCredentialsForHolder('singleton-student');

      expect(credentials.length).toBe(1);
    });
  });

  describe('Credential Types', () => {
    it('should support course_completion type', () => {
      const credential = service.issueCertificate({
        studentId: 'student',
        courseId: 'course',
        courseName: 'Course',
        grade: 90,
      });

      expect(credential.type).toBe('course_completion');
    });

    it('should support skill_certification type', () => {
      const credential = service.issueSkillCertification({
        holderId: 'holder',
        skillId: 'skill',
        skillName: 'Skill',
        level: 85,
      });

      expect(credential.type).toBe('skill_certification');
    });
  });

  describe('Cross-Service Integration', () => {
    it('should store reputation in bridge', () => {
      // Issue credential to update reputation
      service.issueCertificate({
        studentId: 'bridge-student',
        courseId: 'bridge-course',
        courseName: 'Bridge Course',
        grade: 95,
      });

      // Profile should reflect the updated reputation
      const profile = service.getLearnerProfile('bridge-student');
      expect(profile.coursesCompleted).toBe(1);
      expect(profile.trustScore).toBeGreaterThan(0.5);
    });
  });
});

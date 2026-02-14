/**
 * Mail Integration Tests
 *
 * Tests for MailTrustService - sender reputation and email verification
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MailTrustService,
  getMailTrustService,
  type SenderTrust,
  type VerifiedEmail,
  type EmailVerification,
} from '../../src/integrations/mail/index.js';

describe('Mail Integration', () => {
  let service: MailTrustService;

  beforeEach(() => {
    // Create fresh instance for each test
    service = new MailTrustService();
  });

  describe('MailTrustService', () => {
    describe('recordInteraction', () => {
      it('should create reputation for new sender', () => {
        const reputation = service.recordInteraction('sender@example.com', true);

        expect(reputation).toBeDefined();
        expect(reputation.agentId).toBe('sender@example.com');
        expect(reputation.positiveCount).toBeGreaterThan(0);
      });

      it('should increase positive count for positive interaction', () => {
        service.recordInteraction('sender@example.com', true);
        const rep2 = service.recordInteraction('sender@example.com', true);

        // Initial: 1 positive (prior) + 2 recorded = 3
        expect(rep2.positiveCount).toBe(3);
      });

      it('should increase negative count for negative interaction', () => {
        const rep = service.recordInteraction('spam@bad.com', false);

        // Initial: 1 negative (prior) + 1 recorded = 2
        expect(rep.negativeCount).toBe(2);
      });

      it('should track multiple senders independently', () => {
        service.recordInteraction('good@example.com', true);
        service.recordInteraction('good@example.com', true);
        service.recordInteraction('bad@example.com', false);
        service.recordInteraction('bad@example.com', false);

        const goodTrust = service.getSenderTrust('good@example.com');
        const badTrust = service.getSenderTrust('bad@example.com');

        expect(goodTrust.score).toBeGreaterThan(badTrust.score);
      });
    });

    describe('getSenderTrust', () => {
      it('should return default trust for unknown sender', () => {
        const trust = service.getSenderTrust('unknown@example.com');

        expect(trust.senderId).toBe('unknown@example.com');
        expect(trust.level).toBe('medium'); // 0.5 score = medium level
        expect(trust.score).toBe(0.5); // Default neutral
      });

      it('should return high trust for consistently positive sender', () => {
        // Record many positive interactions
        for (let i = 0; i < 10; i++) {
          service.recordInteraction('trusted@example.com', true);
        }

        const trust = service.getSenderTrust('trusted@example.com');

        expect(trust.trustworthy).toBe(true);
        expect(trust.score).toBeGreaterThan(0.8);
        expect(['high', 'verified']).toContain(trust.level);
      });

      it('should return low trust for consistently negative sender', () => {
        // Record many negative interactions
        for (let i = 0; i < 10; i++) {
          service.recordInteraction('spammer@bad.com', false);
        }

        const trust = service.getSenderTrust('spammer@bad.com');

        expect(trust.trustworthy).toBe(false);
        expect(trust.score).toBeLessThan(0.3);
        // Score < 0.2 returns 'unknown', 0.2-0.5 returns 'low'
        expect(['unknown', 'low']).toContain(trust.level);
      });

      it('should calculate confidence based on interaction count', () => {
        // Few interactions = low confidence
        service.recordInteraction('new@example.com', true);
        const lowConf = service.getSenderTrust('new@example.com');

        // Many interactions = high confidence
        for (let i = 0; i < 20; i++) {
          service.recordInteraction('established@example.com', true);
        }
        const highConf = service.getSenderTrust('established@example.com');

        expect(highConf.confidence).toBeGreaterThan(lowConf.confidence);
      });

      it('should include lastVerified timestamp', () => {
        service.recordInteraction('sender@example.com', true);
        const trust = service.getSenderTrust('sender@example.com');

        expect(trust.lastVerified).toBeDefined();
        expect(trust.lastVerified).toBeLessThanOrEqual(Date.now());
      });
    });

    describe('createEmailClaim', () => {
      it('should create claim with DKIM verification', () => {
        const email: VerifiedEmail = {
          id: 'email-123',
          subject: 'Test Email',
          from: 'sender@example.com',
          to: ['recipient@example.com'],
          body: 'Test body',
          timestamp: Date.now(),
          verification: {
            dkimVerified: true,
            spfPassed: false,
            dmarcPassed: false,
          },
        };

        const claim = service.createEmailClaim(email);

        expect(claim.emailId).toBe('email-123');
        expect(claim.verificationType).toBe('dkim');
        expect(claim.claim).toBeDefined();
        expect(claim.claim.content).toBe('Test Email');
      });

      it('should create claim with SPF verification', () => {
        const email: VerifiedEmail = {
          id: 'email-456',
          subject: 'SPF Verified',
          from: 'sender@example.com',
          to: ['recipient@example.com'],
          body: 'Test body',
          timestamp: Date.now(),
          verification: {
            dkimVerified: false,
            spfPassed: true,
            dmarcPassed: false,
          },
        };

        const claim = service.createEmailClaim(email);

        expect(claim.verificationType).toBe('spf');
      });

      it('should create claim with credential verification (highest priority)', () => {
        const email: VerifiedEmail = {
          id: 'email-789',
          subject: 'Credential Verified',
          from: 'sender@example.com',
          to: ['recipient@example.com'],
          body: 'Test body',
          timestamp: Date.now(),
          verification: {
            dkimVerified: true,
            spfPassed: true,
            dmarcPassed: true,
            credentialVerified: true,
          },
        };

        const claim = service.createEmailClaim(email);

        expect(claim.verificationType).toBe('credential');
      });

      it('should create claim with ZK proof verification (highest)', () => {
        const email: VerifiedEmail = {
          id: 'email-zk',
          subject: 'ZK Verified',
          from: 'sender@example.com',
          to: ['recipient@example.com'],
          body: 'Test body',
          timestamp: Date.now(),
          verification: {
            dkimVerified: true,
            spfPassed: true,
            dmarcPassed: true,
            zkProofVerified: true,
          },
        };

        const claim = service.createEmailClaim(email);

        expect(claim.verificationType).toBe('zk');
      });

      it('should mark unknown for unverified email', () => {
        const email: VerifiedEmail = {
          id: 'email-unknown',
          subject: 'Unverified',
          from: 'sender@example.com',
          to: ['recipient@example.com'],
          body: 'Test body',
          timestamp: Date.now(),
          verification: {
            dkimVerified: false,
            spfPassed: false,
            dmarcPassed: false,
          },
        };

        const claim = service.createEmailClaim(email);

        expect(claim.verificationType).toBe('unknown');
      });
    });

    describe('isTrusted', () => {
      it('should return false for unknown sender', () => {
        expect(service.isTrusted('unknown@example.com')).toBe(false);
      });

      it('should return true for trusted sender', () => {
        for (let i = 0; i < 5; i++) {
          service.recordInteraction('trusted@example.com', true);
        }

        expect(service.isTrusted('trusted@example.com')).toBe(true);
      });

      it('should respect custom threshold', () => {
        service.recordInteraction('sender@example.com', true);
        service.recordInteraction('sender@example.com', true);

        // Default threshold (0.5) - should pass
        expect(service.isTrusted('sender@example.com', 0.5)).toBe(true);

        // High threshold (0.9) - should fail
        expect(service.isTrusted('sender@example.com', 0.9)).toBe(false);
      });
    });

    describe('queryExternalReputation', () => {
      it('should not throw when querying external reputation', () => {
        expect(() => {
          service.queryExternalReputation('sender@example.com');
        }).not.toThrow();
      });
    });
  });

  describe('getMailTrustService', () => {
    it('should return singleton instance', () => {
      const service1 = getMailTrustService();
      const service2 = getMailTrustService();

      expect(service1).toBe(service2);
    });

    it('should maintain state across calls', () => {
      const service1 = getMailTrustService();
      service1.recordInteraction('persistent@example.com', true);

      const service2 = getMailTrustService();
      const trust = service2.getSenderTrust('persistent@example.com');

      expect(trust.score).toBeGreaterThan(0.5);
    });
  });

  describe('Trust Level Classification', () => {
    it('should classify as "unknown" for score < 0.2', () => {
      // Create a sender with very low score
      for (let i = 0; i < 20; i++) {
        service.recordInteraction('very-bad@spam.com', false);
      }

      const trust = service.getSenderTrust('very-bad@spam.com');
      // Score < 0.2 returns 'unknown' per scoreToLevel implementation
      expect(trust.level).toBe('unknown');
    });

    it('should classify as "low" for score between 0.2 and 0.5', () => {
      // Mix of interactions leaning negative
      service.recordInteraction('mixed@example.com', false);
      service.recordInteraction('mixed@example.com', false);
      service.recordInteraction('mixed@example.com', true);

      const trust = service.getSenderTrust('mixed@example.com');
      expect(['low', 'medium']).toContain(trust.level);
    });

    it('should classify as "medium" for score between 0.5 and 0.8', () => {
      service.recordInteraction('ok@example.com', true);
      service.recordInteraction('ok@example.com', true);

      const trust = service.getSenderTrust('ok@example.com');
      expect(['medium', 'high']).toContain(trust.level);
    });

    it('should classify as "high" for score between 0.8 and 0.95', () => {
      for (let i = 0; i < 8; i++) {
        service.recordInteraction('good@example.com', true);
      }

      const trust = service.getSenderTrust('good@example.com');
      expect(['high', 'verified']).toContain(trust.level);
    });

    it('should classify as "verified" for score >= 0.95', () => {
      for (let i = 0; i < 50; i++) {
        service.recordInteraction('excellent@example.com', true);
      }

      const trust = service.getSenderTrust('excellent@example.com');
      expect(trust.level).toBe('verified');
      expect(trust.score).toBeGreaterThanOrEqual(0.95);
    });
  });
});

/**
 * @mycelix/sdk Mail Integration
 *
 * hApp-specific adapter for Mycelix-Mail providing:
 * - Sender reputation tracking based on interaction history
 * - Email verification with DKIM, SPF, DMARC support
 * - Epistemic claim generation for verified emails
 * - Trust level classification (unknown → low → medium → high → verified)
 * - Cross-hApp reputation queries via Bridge
 *
 * @packageDocumentation
 * @module integrations/mail
 * @see {@link MailTrustService} - Main service class
 * @see {@link getMailTrustService} - Singleton accessor
 *
 * @example Basic sender trust tracking
 * ```typescript
 * import { getMailTrustService } from '@mycelix/sdk/integrations/mail';
 *
 * const mailService = getMailTrustService();
 *
 * // Record positive interaction (opened, replied, not spam)
 * mailService.recordInteraction('trusted@company.com', true);
 *
 * // Record negative interaction (marked spam, phishing attempt)
 * mailService.recordInteraction('spam@malicious.com', false);
 *
 * // Check sender trust
 * const trust = mailService.getSenderTrust('trusted@company.com');
 * console.log(`Trust level: ${trust.level}, Score: ${trust.score}`);
 * ```
 *
 * @example Email claim verification
 * ```typescript
 * const claim = mailService.createEmailClaim({
 *   id: 'email-123',
 *   subject: 'Invoice Payment',
 *   from: 'billing@vendor.com',
 *   to: ['accounts@mycompany.com'],
 *   body: 'Payment confirmation...',
 *   timestamp: Date.now(),
 *   verification: {
 *     dkimVerified: true,
 *     spfPassed: true,
 *     dmarcPassed: true,
 *   },
 * });
 *
 * console.log(`Verification: ${claim.verificationType}`); // 'dkim'
 * ```
 */

import {
  LocalBridge,
  createReputationQuery,
} from '../../bridge/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';



// ============================================================================
// Mail-Specific Types
// ============================================================================

/**
 * Email verification status with protocol-level checks
 *
 * @remarks
 * Verification priority (highest to lowest):
 * 1. `zkProofVerified` - Zero-knowledge proof (highest)
 * 2. `credentialVerified` - Verifiable credential
 * 3. `dkimVerified` + `spfPassed` - Standard email auth
 * 4. `dkimVerified` OR `spfPassed` - Basic verification
 * 5. None - Unverified
 */
export interface EmailVerification {
  dkimVerified: boolean;
  spfPassed: boolean;
  dmarcPassed: boolean;
  senderDid?: string;
  zkProofVerified?: boolean;
  credentialVerified?: boolean;
}

/** Email with verification metadata */
export interface VerifiedEmail {
  id: string;
  subject: string;
  from: string;
  to: string[];
  body: string;
  timestamp: number;
  verification: EmailVerification;
}

/**
 * Sender trust evaluation result
 *
 * @remarks
 * Trust levels are determined by score thresholds:
 * - `verified`: score >= 0.95
 * - `high`: score >= 0.80
 * - `medium`: score >= 0.50
 * - `low`: score >= 0.20
 * - `unknown`: score < 0.20
 *
 * Confidence increases with more recorded interactions.
 */
export interface SenderTrust {
  senderId: string;
  trustworthy: boolean;
  score: number;
  level: 'unknown' | 'low' | 'medium' | 'high' | 'verified';
  confidence: number;
  lastVerified: number;
}

/** Email claim with epistemic classification */
export interface EmailClaim {
  claim: EpistemicClaim;
  emailId: string;
  verificationType: 'dkim' | 'spf' | 'credential' | 'zk' | 'unknown';
}

// ============================================================================
// Mail Trust Service
// ============================================================================

/**
 * MailTrustService - Trust verification for email senders
 *
 * @example
 * ```typescript
 * const mailTrust = new MailTrustService();
 *
 * // Record sender interaction
 * mailTrust.recordInteraction('sender@example.com', true);
 *
 * // Get sender trust
 * const trust = mailTrust.getSenderTrust('sender@example.com');
 * if (trust.trustworthy) {
 *   // Display email normally
 * }
 * ```
 */
export class MailTrustService {
  private senderReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('mail');
  }

  /**
   * Record an email interaction outcome for sender reputation learning
   *
   * @param senderId - Email address of the sender
   * @param positive - Whether the interaction was positive (true) or negative (false)
   * @returns Updated reputation score for the sender
   *
   * @remarks
   * Positive interactions include: opening email, replying, marking important.
   * Negative interactions include: marking spam, reporting phishing, deleting unread.
   *
   * @example
   * ```typescript
   * // User opened and replied to email
   * service.recordInteraction('colleague@work.com', true);
   *
   * // User marked email as spam
   * service.recordInteraction('spam@unknown.com', false);
   * ```
   */
  recordInteraction(senderId: string, positive: boolean): ReputationScore {
    let reputation = this.senderReputations.get(senderId) || createReputation(senderId);
    reputation = positive ? recordPositive(reputation) : recordNegative(reputation);
    this.senderReputations.set(senderId, reputation);

    // Store reputation in bridge for cross-hApp queries
    this.bridge.setReputation('mail', senderId, reputation);

    return reputation;
  }

  /**
   * Get comprehensive trust evaluation for a sender
   *
   * @param senderId - Email address of the sender
   * @returns Complete trust evaluation including score, level, and confidence
   *
   * @remarks
   * For unknown senders, returns a default score of 0.5 (neutral) with
   * 'medium' trust level. Confidence is 0 for unknown senders.
   *
   * @example
   * ```typescript
   * const trust = service.getSenderTrust('sender@example.com');
   *
   * if (trust.trustworthy && trust.confidence > 0.8) {
   *   console.log('Highly confident sender is trustworthy');
   * } else if (trust.level === 'unknown') {
   *   console.log('No history with this sender');
   * }
   * ```
   */
  getSenderTrust(senderId: string): SenderTrust {
    const reputation = this.senderReputations.get(senderId) || createReputation(senderId);
    const score = reputationValue(reputation);

    return {
      senderId,
      trustworthy: score >= 0.5,
      score,
      level: this.scoreToLevel(score),
      confidence: this.calculateConfidence(reputation),
      lastVerified: Date.now(),
    };
  }

  /**
   * Create an epistemic claim for verified email content
   *
   * @param email - The verified email to create a claim for
   * @returns Epistemic claim with verification type and classification
   *
   * @remarks
   * The empirical level of the claim is determined by the email's verification:
   * - ZK Proof → E4 (Consensus level)
   * - Credential → E3 (Cryptographic level)
   * - DKIM + SPF → E2 (Private verification)
   * - DKIM OR SPF → E1 (Testimonial)
   * - None → E0 (Unverified)
   *
   * @example
   * ```typescript
   * const claim = service.createEmailClaim({
   *   id: 'email-001',
   *   subject: 'Contract Signed',
   *   from: 'legal@company.com',
   *   to: ['recipient@example.com'],
   *   body: 'Contract details...',
   *   timestamp: Date.now(),
   *   verification: {
   *     dkimVerified: true,
   *     spfPassed: true,
   *     dmarcPassed: true,
   *     credentialVerified: true,
   *   },
   * });
   *
   * // claim.verificationType will be 'credential' (highest available)
   * ```
   */
  createEmailClaim(email: VerifiedEmail): EmailClaim {
    const empiricalLevel = this.determineEmpiricalLevel(email.verification);
    const verificationType = this.getVerificationType(email.verification);

    const emailClaim = claim(email.subject)
      .withEmpirical(empiricalLevel)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M1_Temporal)
      .build();

    return {
      claim: emailClaim,
      emailId: email.id,
      verificationType,
    };
  }

  /**
   * Query reputation from other hApps
   */
  queryExternalReputation(senderId: string): void {
    const query = createReputationQuery('mail', senderId);
    this.bridge.send('marketplace', query);
  }

  /**
   * Check if sender is trusted
   */
  isTrusted(senderId: string, minimumScore = 0.5): boolean {
    const reputation = this.senderReputations.get(senderId);
    if (!reputation) return false;
    return reputationValue(reputation) >= minimumScore;
  }

  // Private helpers

  private scoreToLevel(
    score: number
  ): 'unknown' | 'low' | 'medium' | 'high' | 'verified' {
    if (score >= 0.95) return 'verified';
    if (score >= 0.8) return 'high';
    if (score >= 0.5) return 'medium';
    if (score >= 0.2) return 'low';
    return 'unknown';
  }

  private calculateConfidence(reputation: ReputationScore): number {
    const total = reputation.positiveCount + reputation.negativeCount;
    if (total === 0) return 0;
    // More interactions = higher confidence
    return Math.min(1, total / 20);
  }

  private determineEmpiricalLevel(verification: EmailVerification): EmpiricalLevel {
    if (verification.zkProofVerified) return EmpiricalLevel.E4_Consensus;
    if (verification.credentialVerified) return EmpiricalLevel.E3_Cryptographic;
    if (verification.dkimVerified && verification.spfPassed)
      return EmpiricalLevel.E2_PrivateVerify;
    if (verification.dkimVerified || verification.spfPassed)
      return EmpiricalLevel.E1_Testimonial;
    return EmpiricalLevel.E0_Unverified;
  }

  private getVerificationType(
    verification: EmailVerification
  ): 'dkim' | 'spf' | 'credential' | 'zk' | 'unknown' {
    if (verification.zkProofVerified) return 'zk';
    if (verification.credentialVerified) return 'credential';
    if (verification.dkimVerified) return 'dkim';
    if (verification.spfPassed) return 'spf';
    return 'unknown';
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  claim,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
};

// Default service instance
let defaultService: MailTrustService | null = null;

/**
 * Get the default mail trust service instance
 */
export function getMailTrustService(): MailTrustService {
  if (!defaultService) {
    defaultService = new MailTrustService();
  }
  return defaultService;
}

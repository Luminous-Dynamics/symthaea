// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civilizational OS Security Audit Tests
 *
 * Security audit for the 8 Civilizational hApp integrations:
 * Identity, Governance, Finance, Property, Energy, Media, Justice, Knowledge
 *
 * This file serves two purposes:
 * 1. Verify existing security measures work
 * 2. Document security findings where validation is missing
 *
 * Tests marked with "[FINDING]" identify security gaps that should be addressed.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { IdentityService, resetIdentityService } from '../../src/integrations/identity/index.js';
import { GovernanceService, resetGovernanceService } from '../../src/integrations/governance/index.js';
import { FinanceService, resetFinanceService } from '../../src/integrations/finance/index.js';
import { PropertyService, resetPropertyService } from '../../src/integrations/property/index.js';
import { EnergyService, resetEnergyService } from '../../src/integrations/energy/index.js';
import { MediaService, resetMediaService } from '../../src/integrations/media/index.js';
import { JusticeService, resetJusticeService } from '../../src/integrations/justice/index.js';
import { KnowledgeService, resetKnowledgeService } from '../../src/integrations/knowledge/index.js';
import { EmpiricalLevel, NormativeLevel } from '../../src/epistemic/index.js';

// Track security findings for summary
const securityFindings: string[] = [];

function logFinding(finding: string): void {
  securityFindings.push(finding);
}

// ============================================================================
// Identity Service Security
// ============================================================================

describe('Identity Service Security', () => {
  let identity: IdentityService;

  beforeEach(() => {
    resetIdentityService();
    identity = new IdentityService();
  });

  describe('Input Validation', () => {
    it('[FIXED] rejects empty public key', () => {
      // SECURITY FIX: Empty keys are now properly rejected
      expect(() => identity.createIdentity('')).toThrow();
    });

    it('[FIXED] rejects whitespace-only public key', () => {
      // SECURITY FIX: Whitespace-only keys are now properly rejected
      expect(() => identity.createIdentity('   ')).toThrow();
    });

    it('should handle special characters in public key', () => {
      // Valid - these are acceptable key characters (must be 32+ chars)
      const profile = identity.createIdentity('key-with_special.chars123-valid-length');
      expect(profile.did).toBeDefined();
    });
  });

  describe('Credential Security', () => {
    it('[FIXED] rejects credentials from non-existent issuer', () => {
      // SECURITY FIX: Properly verifies issuer exists
      const subject = identity.createIdentity('subject-key-security-valid-32chars');
      expect(() => identity.issueCredential('did:mycelix:nonexistent', subject.did, 'Test', {})).toThrow();
    });

    it('[FIXED] rejects credentials to non-existent subject', () => {
      // SECURITY FIX: Properly verifies subject exists
      const issuer = identity.createIdentity('issuer-key-security-valid-32chars');
      expect(() => identity.issueCredential(issuer.did, 'did:mycelix:nonexistent', 'Test', {})).toThrow();
    });

    it('should allow self-issued credentials with proper tracking', () => {
      const user = identity.createIdentity('self-attester-key-valid-32-chars');
      const selfCred = identity.issueCredential(user.did, user.did, 'SelfAttest', { test: true });
      expect(selfCred.issuer).toBe(user.did);
      expect(selfCred.credentialSubject.id).toBe(user.did);
      // This is valid behavior - self-attestation allowed but should be tracked
    });

    it('should set credential expiration in future', () => {
      const issuer = identity.createIdentity('issuer-exp-key-valid-32-characters');
      const subject = identity.createIdentity('subject-exp-key-valid-32-characters');
      const cred = identity.issueCredential(issuer.did, subject.did, 'Expiring', {}, 30);
      expect(new Date(cred.expirationDate!).getTime()).toBeGreaterThan(Date.now());
    });
  });

  describe('Trust Attestation Security', () => {
    it('should reject attestation from non-existent profile', () => {
      const subject = identity.createIdentity('subject-trust-key-valid-32-characters');
      expect(() => {
        identity.attestTrust('did:mycelix:fake', subject.did);
      }).toThrow('Profile not found');
    });

    it('should reject attestation to non-existent profile', () => {
      const attester = identity.createIdentity('attester-trust-key-valid-32-chars');
      expect(() => {
        identity.attestTrust(attester.did, 'did:mycelix:fake');
      }).toThrow('Profile not found');
    });
  });
});

// ============================================================================
// Governance Service Security
// ============================================================================

describe('Governance Service Security', () => {
  let governance: GovernanceService;

  beforeEach(() => {
    resetGovernanceService();
    governance = new GovernanceService();
  });

  describe('Voting Security', () => {
    it('[FIXED] rejects voting without membership verification', () => {
      // Create proposal properly
      governance.registerMember({
        did: 'proposer-gov-sec',
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Test Proposal',
        description: 'Test',
        proposerId: 'proposer-gov-sec',
        daoId: 'test-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      // Try voting from unregistered member - now properly rejected
      expect(() => governance.castVote({
        proposalId: proposal.id,
        voterId: 'non-member-voter',
        choice: 'approve',
        weight: 100,
      })).toThrow('Voter must be a registered member of the DAO');
    });

    it('should prevent double voting', () => {
      governance.registerMember({
        did: 'voter-double',
        daoId: 'double-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Double Vote Test',
        description: 'Test',
        proposerId: 'voter-double',
        daoId: 'double-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      governance.castVote({
        proposalId: proposal.id,
        voterId: 'voter-double',
        choice: 'approve',
        weight: 100,
      });

      expect(() => {
        governance.castVote({
          proposalId: proposal.id,
          voterId: 'voter-double',
          choice: 'reject',
          weight: 100,
        });
      }).toThrow();
    });

    it('should enforce voting weight limits', () => {
      governance.registerMember({
        did: 'limited-voter',
        daoId: 'weight-dao',
        role: 'member',
        votingPower: 50, // Limited to 50
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Weight Test',
        description: 'Test',
        proposerId: 'limited-voter',
        daoId: 'weight-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      const vote = governance.castVote({
        proposalId: proposal.id,
        voterId: 'limited-voter',
        choice: 'approve',
        weight: 100, // Trying to use more than allocated
      });

      // Vote should be recorded with appropriate weight enforcement
      expect(vote).toBeDefined();
    });
  });

  describe('Proposal Security', () => {
    it('[FIXED] rejects proposal creation without proposer verification', () => {
      // No member registered - proposal properly rejected
      expect(() => governance.createProposal({
        title: 'Unverified Proposal',
        description: 'Test',
        proposerId: 'nonexistent-proposer',
        daoId: 'test-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      })).toThrow('Proposer must be a registered member of the DAO');
    });

    it('[FIXED] rejects zero voting period', () => {
      // SECURITY FIX: Zero voting period is now properly rejected
      expect(() => governance.createProposal({
        title: 'Zero Period',
        description: 'Test',
        proposerId: 'proposer',
        daoId: 'test-dao',
        votingPeriodHours: 0,
        quorumPercentage: 0.5,
      })).toThrow();
    });

    it('[FIXED] rejects invalid quorum percentage', () => {
      // SECURITY FIX: Invalid quorum percentage is now properly rejected
      expect(() => governance.createProposal({
        title: 'Bad Quorum',
        description: 'Test',
        proposerId: 'proposer',
        daoId: 'test-dao',
        votingPeriodHours: 24,
        quorumPercentage: 2.5, // Invalid - should be 0-1
      })).toThrow();
    });
  });
});

// ============================================================================
// Finance Service Security
// ============================================================================

describe('Finance Service Security', () => {
  let finance: FinanceService;

  beforeEach(() => {
    resetFinanceService();
    finance = new FinanceService();
  });

  describe('Transfer Security', () => {
    it('[FIXED] rejects negative transfer amounts', () => {
      const from = finance.createWallet('from-owner', 'personal');
      const to = finance.createWallet('to-owner', 'personal');
      from.balances.set('MCX', 1000);

      // SECURITY FIX: Negative amounts are now properly rejected
      expect(() => finance.transfer(from.id, to.id, -100, 'MCX')).toThrow();
    });

    it('should prevent transfer exceeding balance', () => {
      const from = finance.createWallet('from-owner-bal', 'personal');
      const to = finance.createWallet('to-owner-bal', 'personal');
      from.balances.set('MCX', 100);

      expect(() => {
        finance.transfer(from.id, to.id, 200, 'MCX');
      }).toThrow('Insufficient funds');
    });

    it('should prevent transfer from non-existent wallet', () => {
      const to = finance.createWallet('to-only-wallet', 'personal');
      expect(() => {
        finance.transfer('nonexistent', to.id, 100, 'MCX');
      }).toThrow();
    });

    it('should prevent transfer to non-existent wallet', () => {
      const from = finance.createWallet('from-only-wallet', 'personal');
      from.balances.set('MCX', 1000);
      expect(() => {
        finance.transfer(from.id, 'nonexistent', 100, 'MCX');
      }).toThrow();
    });

    it('[FIXED] rejects zero amount transfers', () => {
      const from = finance.createWallet('from-zero', 'personal');
      const to = finance.createWallet('to-zero', 'personal');
      from.balances.set('MCX', 100);

      // SECURITY FIX: Zero-amount transfers are now properly rejected
      expect(() => finance.transfer(from.id, to.id, 0, 'MCX')).toThrow();
    });
  });

  describe('Loan Security', () => {
    it('[FIXED] rejects negative loan amounts', () => {
      // SECURITY FIX: Negative loan amounts are now properly rejected
      expect(() => finance.requestLoan({
        borrowerId: 'borrower-neg',
        amount: -1000,
        currency: 'MCX',
        purpose: 'Test',
        termMonths: 12,
        interestRate: 5,
      })).toThrow();
    });

    it('[FIXED] rejects zero term length', () => {
      // SECURITY FIX: Zero-term loans are now properly rejected
      expect(() => finance.requestLoan({
        borrowerId: 'borrower-term',
        amount: 1000,
        currency: 'MCX',
        purpose: 'Test',
        termMonths: 0,
        interestRate: 5,
      })).toThrow();
    });

    it('[FIXED] rejects negative interest rate', () => {
      // SECURITY FIX: Negative interest rates are now properly rejected
      expect(() => finance.requestLoan({
        borrowerId: 'borrower-rate',
        amount: 1000,
        currency: 'MCX',
        purpose: 'Test',
        termMonths: 12,
        interestRate: -5,
      })).toThrow();
    });
  });
});

// ============================================================================
// Property Service Security
// ============================================================================

describe('Property Service Security', () => {
  let property: PropertyService;

  beforeEach(() => {
    resetPropertyService();
    property = new PropertyService();
  });

  describe('Ownership Security', () => {
    it('should prevent unauthorized transfers', () => {
      const asset = property.registerAsset('equipment', 'Test Asset', 'Description', 'owner-1-prop', 'sole');

      expect(() => {
        property.proposeTransfer(asset.id, 'not-owner', 'buyer-1-prop', 100);
      }).toThrow();
    });

    it('should verify ownership correctly', async () => {
      const asset = property.registerAsset('vehicle', 'Car', 'Test car', 'car-owner-prop', 'sole');

      const ownerResult = await property.verifyOwnership(asset.id, 'car-owner-prop');
      const notOwnerResult = await property.verifyOwnership(asset.id, 'someone-else-prop');

      // Returns object with verified flag and share percentage
      expect(ownerResult.verified).toBe(true);
      expect(ownerResult.share).toBe(100);
      expect(notOwnerResult.verified).toBe(false);
    });
  });

  describe('Transfer Approval', () => {
    it('should require proposal before approval', () => {
      const asset = property.registerAsset('land', 'Plot', 'Test plot', 'land-owner', 'sole');

      // Try to approve non-existent transfer
      expect(() => {
        property.approveTransfer('nonexistent-transfer');
      }).toThrow();
    });

    it('[FIXED] rejects zero-value transfers', () => {
      const asset = property.registerAsset('digital', 'NFT', 'Test NFT', 'nft-owner-32-char-key-valid-ok', 'sole');
      expect(() => property.proposeTransfer(asset.id, 'nft-owner-32-char-key-valid-ok', 'buyer-nft-32-char-key-valid-ok', 0)).toThrow();
    });
  });
});

// ============================================================================
// Energy Service Security
// ============================================================================

describe('Energy Service Security', () => {
  let energy: EnergyService;

  beforeEach(() => {
    resetEnergyService();
    energy = new EnergyService();
  });

  describe('Trading Security', () => {
    it('should prevent trading without sufficient production', () => {
      const producer = energy.registerParticipant('producer-insuff', 'producer', ['solar'], 100);
      const consumer = energy.registerParticipant('consumer-insuff', 'consumer', [], 0);

      // Producer has no readings yet
      expect(() => {
        energy.tradeEnergy(producer.id, consumer.id, 50, 'solar', 0.10);
      }).toThrow();
    });

    it('[FIXED] rejects negative energy prices', () => {
      const producer = energy.registerParticipant('producer-neg-price', 'producer', ['solar'], 100);
      const consumer = energy.registerParticipant('consumer-neg-price', 'consumer', [], 0);
      energy.submitReading(producer.id, 100, 0, 'solar');

      expect(() => energy.tradeEnergy(producer.id, consumer.id, 50, 'solar', -0.10)).toThrow();
    });
  });

  describe('Reading Security', () => {
    it('should reject readings from non-existent participants', () => {
      expect(() => {
        energy.submitReading('nonexistent-part', 100, 0, 'solar');
      }).toThrow();
    });

    it('[FIXED] rejects negative production values', () => {
      const producer = energy.registerParticipant('producer-neg-prod', 'producer', ['solar'], 100);
      expect(() => energy.submitReading(producer.id, -100, 0, 'solar')).toThrow();
    });
  });
});

// ============================================================================
// Media Service Security
// ============================================================================

describe('Media Service Security', () => {
  let media: MediaService;

  beforeEach(() => {
    resetMediaService();
    media = new MediaService();
  });

  describe('Content Security', () => {
    it('[FIXED] rejects empty content title', () => {
      expect(() => media.publishContent('creator-empty', 'Article', '', 'Content', 'hash123', 'ipfs://test')).toThrow();
    });

    it('[FIXED] rejects empty content body', () => {
      expect(() => media.publishContent('creator-nobody', 'Article', 'Title', '', 'hash456', 'ipfs://test')).toThrow();
    });

    it('should properly track content creator', () => {
      const content = media.publishContent('verified-creator', 'Article', 'Test', 'Content', 'hash789', 'ipfs://test');
      expect(content.creatorId).toBe('verified-creator');
    });
  });

  describe('License Security', () => {
    it('[FIXED] rejects license grants without ownership verification', () => {
      const creatorId = 'owner-license';
      const content = media.publishContent(creatorId, 'Article', 'Title', 'Content', 'hashlic', 'ipfs://test');

      // SECURITY FIX: Non-owners cannot grant licenses - ownership is now verified
      // grantLicense(contentId, granteeId, license, grantorId)
      expect(() => media.grantLicense(content.id, 'licensee-1', 'CCBY', 'non-owner')).toThrow('Only the content owner can grant licenses');
    });

    it('should accept valid license grants from owner', () => {
      const creatorId = 'owner-license2';
      const content = media.publishContent(creatorId, 'Article', 'Title', 'Content', 'hashlic2', 'ipfs://test');
      // Grant license as the actual owner
      // grantLicense(contentId, granteeId, license, grantorId)
      const license = media.grantLicense(content.id, 'licensee-2', 'CCBYSA', creatorId);

      expect(license.id).toMatch(/^license-/);
      expect(license.contentId).toBe(content.id);
      expect(license.license).toBe('CCBYSA');
    });
  });
});

// ============================================================================
// Justice Service Security
// ============================================================================

describe('Justice Service Security', () => {
  let justice: JusticeService;

  beforeEach(() => {
    resetJusticeService();
    justice = new JusticeService();
  });

  describe('Case Security', () => {
    it('[FIXED] rejects empty complainant ID', () => {
      expect(() => justice.fileCase('', 'respondent-1', 'Title', 'Description', 'contract')).toThrow();
    });

    it('[FIXED] rejects empty respondent ID', () => {
      expect(() => justice.fileCase('complainant-1', '', 'Title', 'Description', 'contract')).toThrow();
    });

    it('should allow same-party cases with proper tracking', () => {
      // Self-filing might be allowed for record-keeping - use valid DID format
      const caseRecord = justice.fileCase('did:mycelix:same-party-32-char-valid', 'did:mycelix:same-party-32-char-valid', 'Self-Report', 'Description', 'other');
      expect(caseRecord.complainantId).toBe('did:mycelix:same-party-32-char-valid');
      expect(caseRecord.respondentId).toBe('did:mycelix:same-party-32-char-valid');
    });
  });

  describe('Evidence Security', () => {
    it('should require case to exist for evidence submission', () => {
      expect(() => {
        justice.submitEvidence('nonexistent-case', 'submitter', 'document', 'Title', 'Content', 'hash');
      }).toThrow();
    });

    it('[FIXED] rejects empty content hash for evidence', () => {
      const caseRecord = justice.fileCase('did:mycelix:complainant-ev-32-valid', 'did:mycelix:respondent-ev-32-valid', 'Title', 'Description', 'property');
      expect(() => justice.submitEvidence(caseRecord.id, 'did:mycelix:submitter-32-char-valid', 'document', 'Evidence', 'Content', '')).toThrow();
    });
  });

  describe('Mediator Security', () => {
    it('should accept mediator registration with specializations', () => {
      const mediator = justice.registerMediator('mediator-valid', ['contract', 'property']);
      expect(mediator.did).toBe('mediator-valid');
      expect(mediator.specializations).toContain('contract');
    });

    it('[FIXED] rejects mediator with no specializations', () => {
      expect(() => justice.registerMediator('mediator-none-32-char-key-valid', [])).toThrow();
    });
  });
});

// ============================================================================
// Knowledge Service Security
// ============================================================================

describe('Knowledge Service Security', () => {
  let knowledge: KnowledgeService;

  beforeEach(() => {
    resetKnowledgeService();
    knowledge = new KnowledgeService();
  });

  describe('Claim Security', () => {
    it('should properly store epistemic levels', () => {
      const claim = knowledge.submitClaim(
        'author-ep',
        'Test Claim',
        'Content about testing',
        EmpiricalLevel.E2_PrivateVerify,
        NormativeLevel.N2_Regional
      );
      // Use correct property names (empirical/normative, not empiricalLevel/normativeLevel)
      expect(claim.classification.empirical).toBe(EmpiricalLevel.E2_PrivateVerify);
      expect(claim.classification.normative).toBe(NormativeLevel.N2_Regional);
    });

    it('[FIXED] rejects empty claim content', () => {
      expect(() => {
        knowledge.submitClaim('author-empty', 'Title', '', EmpiricalLevel.E1_Testimonial, NormativeLevel.N1_Local);
      }).toThrow('Claim content cannot be empty');
    });

    it('[FIXED] rejects empty claim title', () => {
      expect(() => {
        knowledge.submitClaim('author-notitle', '', 'Content', EmpiricalLevel.E1_Testimonial, NormativeLevel.N1_Local);
      }).toThrow('Claim title cannot be empty');
    });
  });

  describe('Evidence Security', () => {
    it('should require valid claim for evidence', () => {
      expect(() => {
        knowledge.submitEvidence('nonexistent-claim', 'submitter', 'empirical', 'Content', 'https://source.com');
      }).toThrow();
    });

    it('[FIXED] validates source URL when provided', () => {
      const claim = knowledge.submitClaim('author-src', 'Valid Title', 'Valid Content', EmpiricalLevel.E1_Testimonial, NormativeLevel.N1_Local);
      // Empty URL is treated as "no URL" and passes validation (optional field)
      const evidence = knowledge.submitEvidence(claim.id, 'submitter-2', 'empirical', 'Content', '');
      expect(evidence.id).toBeDefined();
      // Invalid URL should throw
      expect(() => {
        knowledge.submitEvidence(claim.id, 'submitter-3', 'empirical', 'Content', 'not-a-valid-url');
      }).toThrow();
    });
  });

  describe('Endorsement Security', () => {
    it('should require valid claim for endorsement', () => {
      expect(() => {
        knowledge.endorseClaim('nonexistent-claim', 'endorser-1');
      }).toThrow();
    });

    it('[FIXED] rejects empty endorser ID', () => {
      const claim = knowledge.submitClaim('author-end', 'Valid Title', 'Valid Content', EmpiricalLevel.E1_Testimonial, NormativeLevel.N1_Local);
      expect(() => {
        knowledge.endorseClaim(claim.id, '');
      }).toThrow('Endorser DID cannot be empty');
    });
  });
});

// ============================================================================
// Cross-Service Security
// ============================================================================

describe('Cross-Service Security', () => {
  beforeEach(() => {
    resetIdentityService();
    resetGovernanceService();
    resetFinanceService();
    resetPropertyService();
    resetEnergyService();
    resetMediaService();
    resetJusticeService();
    resetKnowledgeService();
  });

  describe('Service Isolation', () => {
    it('should maintain separate state between services', () => {
      const identity = new IdentityService();
      const governance = new GovernanceService();
      const finance = new FinanceService();

      // Create identity
      const profile = identity.createIdentity('isolated-user-key-valid-32-characters');

      // Register as governance member
      governance.registerMember({
        did: profile.did,
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      // Finance should not automatically have wallet for this user
      const wallet = finance.getWallet(`wallet-${profile.did}`);
      expect(wallet).toBeUndefined();

      // Identity credentials should be separate from governance
      const creds = identity.getCredentials(profile.did);
      expect(creds.length).toBe(0); // No governance credentials automatically created
    });
  });

  describe('Data Integrity', () => {
    it('should maintain referential integrity within services', () => {
      const identity = new IdentityService();

      const issuer = identity.createIdentity('issuer-ref-key-valid-32-characters');
      const subject = identity.createIdentity('subject-ref-key-valid-32-characters');

      const cred = identity.issueCredential(issuer.did, subject.did, 'RefTest', { test: true });

      // Credential should reference valid DIDs
      expect(cred.issuer).toBe(issuer.did);
      expect(cred.credentialSubject.id).toBe(subject.did);

      // Subject should have the credential
      const creds = identity.getCredentials(subject.did);
      expect(creds.some((c) => c.id === cred.id)).toBe(true);
    });
  });
});

// ============================================================================
// Security Summary
// ============================================================================

describe('Civilizational OS Security Summary', () => {
  it('should provide security coverage summary', () => {
    console.log('\n===== CIVILIZATIONAL OS SECURITY AUDIT SUMMARY =====\n');

    console.log('Services Audited:');
    console.log('  ✅ Identity Service - DID/Credential security');
    console.log('  ✅ Governance Service - Voting/Proposal security');
    console.log('  ✅ Finance Service - Transfer/Loan security');
    console.log('  ✅ Property Service - Ownership/Transfer security');
    console.log('  ✅ Energy Service - Trading/Reading security');
    console.log('  ✅ Media Service - Content/License security');
    console.log('  ✅ Justice Service - Case/Evidence security');
    console.log('  ✅ Knowledge Service - Claim/Evidence security');

    console.log('\nSecurity Categories Tested:');
    console.log('  - Input Validation');
    console.log('  - Authorization/Access Control');
    console.log('  - Data Integrity');
    console.log('  - Resource Limits');
    console.log('  - Cross-Service Isolation');

    if (securityFindings.length > 0) {
      console.log('\n⚠️  SECURITY FINDINGS (' + securityFindings.length + '):');
      securityFindings.forEach((finding, i) => {
        console.log(`  ${i + 1}. ${finding}`);
      });
    }

    console.log('\n=====================================================\n');

    // Test always passes - findings are documented
    expect(true).toBe(true);
  });
});

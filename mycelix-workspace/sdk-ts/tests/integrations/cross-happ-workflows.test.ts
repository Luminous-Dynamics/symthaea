// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-hApp Workflow Integration Tests
 *
 * Tests core workflows that span multiple hApps in the Mycelix ecosystem.
 * These tests verify that the SDK integrations work together correctly.
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Import all core hApp services
import {
  getGovernanceService,
  resetGovernanceService,
} from '../../src/integrations/governance/index.js';

import {
  getFinanceService,
  resetFinanceService,
} from '../../src/integrations/finance/index.js';

import {
  getIdentityService,
  resetIdentityService,
} from '../../src/integrations/identity/index.js';

import {
  getKnowledgeService,
  resetKnowledgeService,
} from '../../src/integrations/knowledge/index.js';

import {
  getPropertyService,
  resetPropertyService,
} from '../../src/integrations/property/index.js';

import {
  getEnergyService,
  resetEnergyService,
} from '../../src/integrations/energy/index.js';

import {
  getMediaService,
  resetMediaService,
} from '../../src/integrations/media/index.js';

import {
  getJusticeService,
  resetJusticeService,
} from '../../src/integrations/justice/index.js';

import { EmpiricalLevel, NormativeLevel } from '../../src/epistemic/index.js';

describe('Cross-hApp Workflows', () => {
  beforeEach(() => {
    // Reset all services between tests
    resetGovernanceService();
    resetFinanceService();
    resetIdentityService();
    resetKnowledgeService();
    resetPropertyService();
    resetEnergyService();
    resetMediaService();
    resetJusticeService();
  });

  describe('Identity → Governance Workflow', () => {
    it('should allow verified identity to participate in governance', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();

      // Create identity
      const profile = identity.createIdentity('pubkey123abc1234567890abcdefghij');
      expect(profile.did).toMatch(/^did:mycelix:/);

      // Add peer attestations
      const peer1 = identity.createIdentity('peer1key1234567890123456789012ab');
      const peer2 = identity.createIdentity('peer2key1234567890123456789012cd');
      const peer3 = identity.createIdentity('peer3key1234567890123456789012ef');

      identity.attestTrust(peer1.did, profile.did);
      identity.attestTrust(peer2.did, profile.did);
      identity.attestTrust(peer3.did, profile.did);

      // Profile should now be peer verified
      const verified = identity.getProfile(profile.did);
      expect(verified?.verificationLevel).toBe('peer_verified');

      // Register as DAO member
      const member = governance.registerMember({
        did: profile.did,
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      // Create and vote on proposal
      const proposal = governance.createProposal({
        title: 'Test Proposal',
        description: 'A test proposal',
        proposerId: profile.did,
        daoId: 'test-dao',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      });

      expect(proposal.status).toBe('active');
    });
  });

  describe('Finance → Property Workflow', () => {
    it('should enable property transfer with financial settlement', () => {
      const finance = getFinanceService();
      const property = getPropertyService();

      // Create wallets
      const sellerWallet = finance.createWallet('seller-did', 'personal');
      const buyerWallet = finance.createWallet('buyer-did', 'personal');

      // Fund buyer wallet (in real scenario, this would be from external source)
      buyerWallet.balances.set('MCX', 1000);

      // Register property
      const asset = property.registerAsset(
        'real_estate',
        'Test House',
        'A beautiful house',
        'seller-did',
        'sole'
      );

      // Propose transfer
      const transfer = property.proposeTransfer(
        asset.id,
        'seller-did',
        'buyer-did',
        100, // 100% ownership
        500 // consideration
      );

      expect(transfer.status).toBe('approved'); // Sole owner auto-approves

      // Execute transfer
      const updatedAsset = property.executeTransfer(transfer.id);

      // Verify ownership changed
      expect(updatedAsset.owners.find((o) => o.ownerId === 'buyer-did')?.percentage).toBe(100);

      // Execute payment
      const payment = finance.transfer(
        buyerWallet.id,
        sellerWallet.id,
        500,
        'MCX',
        'Property purchase'
      );

      expect(payment.status).toBe('confirmed');
      expect(finance.getBalance(sellerWallet.id, 'MCX')).toBe(500);
    });
  });

  describe('Knowledge → Governance Workflow', () => {
    it('should use verified knowledge claims in governance decisions', () => {
      const knowledge = getKnowledgeService();
      const governance = getGovernanceService();

      // Submit a knowledge claim
      const claim = knowledge.submitClaim(
        'researcher-did',
        'Energy Policy Impact',
        'Renewable energy policies increase grid stability by 23%',
        EmpiricalLevel.E2_Observed,
        NormativeLevel.N2_Network,
        ['energy', 'policy']
      );

      // Submit evidence
      knowledge.submitEvidence(
        claim.id,
        'analyst-did',
        'empirical',
        'Grid stability analysis shows 23% improvement over 5 years',
        'https://data.example.com/study'
      );

      // Endorse claim
      for (let i = 0; i < 5; i++) {
        knowledge.endorseClaim(claim.id, `did:mycelix:endorser-${i}-padding-32chars`);
      }

      // Claim should now be verified
      const verifiedClaim = knowledge.getClaim(claim.id);
      expect(verifiedClaim?.status).toBe('verified');

      // Use claim in governance proposal
      governance.registerMember({
        did: 'proposer-did',
        daoId: 'energy-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Adopt Renewable Energy Policy',
        description: `Based on verified claim ${claim.id}: ${claim.title}`,
        proposerId: 'proposer-did',
        daoId: 'energy-dao',
        votingPeriodHours: 48,
        quorumPercentage: 0.6,
      });

      expect(proposal.description).toContain(claim.id);
    });
  });

  describe('Energy → Finance Workflow', () => {
    it('should enable energy credit trading with financial settlement', () => {
      const energy = getEnergyService();
      const finance = getFinanceService();

      // Register producer and consumer
      const producer = energy.registerParticipant(
        'producer-did',
        'producer',
        ['solar'],
        100, // 100 kWh capacity
        { lat: 37.7749, lng: -122.4194 }
      );

      const consumer = energy.registerParticipant(
        'consumer-did',
        'consumer',
        [],
        0
      );

      // Producer generates energy
      energy.submitReading(producer.id, 50, 10, 'solar'); // 50 kWh produced, 10 consumed

      // Check producer has credits
      const credits = energy.getCredits(producer.id);
      expect(credits.length).toBeGreaterThan(0);

      // Create wallets for payment
      const producerWallet = finance.createWallet('producer-did', 'business');
      const consumerWallet = finance.createWallet('consumer-did', 'personal');
      consumerWallet.balances.set('MCX', 500);

      // Trade energy
      const trade = energy.tradeEnergy(
        producer.id,
        consumer.id,
        20, // 20 kWh
        'solar',
        5 // $5 per kWh
      );

      expect(trade.settlementStatus).toBe('settled');

      // Execute financial payment
      const payment = finance.transfer(
        consumerWallet.id,
        producerWallet.id,
        100, // 20 kWh * $5
        'MCX',
        'Energy purchase'
      );

      expect(payment.status).toBe('confirmed');
    });
  });

  describe('Media → Justice Workflow', () => {
    it('should handle content disputes through justice system', () => {
      const media = getMediaService();
      const justice = getJusticeService();

      // Publish content
      const content = media.publishContent(
        'did:mycelix:creator-32-char-key-valid',
        'Article',
        'Original Article',
        'My original work on consciousness',
        'hash123',
        'storage://content/123',
        'CCBY',
        ['philosophy']
      );

      // Someone reports copyright infringement
      const report = media.reportContent(
        content.id,
        'did:mycelix:reporter-32-char-key-valid',
        'This content copies my work without attribution',
        'copyright'
      );

      // File case in justice system
      const disputeCase = justice.fileCase(
        'did:mycelix:reporter-32-char-key-valid',
        'did:mycelix:creator-32-char-key-valid',
        'Copyright Infringement Dispute',
        `Content ${content.id} allegedly infringes on reporter's copyright`,
        'property'
      );

      expect(disputeCase.status).toBe('Active');

      // Submit evidence
      justice.submitEvidence(
        disputeCase.id,
        'did:mycelix:reporter-32-char-key-valid',
        'document',
        'Original Work',
        'My original article published 6 months ago',
        'hash456'
      );

      // Register mediator and initiate mediation
      justice.registerMediator('did:mycelix:mediator-32-char-valid', ['PropertyDispute']);
      justice.initiateMediation(disputeCase.id, 'did:mycelix:mediator-32-char-valid');

      const updatedCase = justice.getCase(disputeCase.id);
      expect(updatedCase?.phase).toBe('Mediation');
    });
  });

  describe('Property → Energy → Finance Workflow', () => {
    it('should enable community solar cooperative with shared ownership', () => {
      const property = getPropertyService();
      const energy = getEnergyService();
      const finance = getFinanceService();

      // Register solar installation as cooperative property
      const solarFarm = property.registerAsset(
        'equipment',
        'Community Solar Farm',
        '1MW community solar installation',
        'cooperative-did',
        'cooperative'
      );

      // Members join the cooperative
      property.joinCommons(solarFarm.id, 'member1-did', ['generate', 'consume']);
      property.joinCommons(solarFarm.id, 'member2-did', ['generate', 'consume']);

      // Register as energy producer
      const producer = energy.registerParticipant(
        'cooperative-did',
        'producer',
        ['solar'],
        1000, // 1000 kWh capacity
        { lat: 35.0, lng: -100.0 }
      );

      // Generate energy
      energy.submitReading(producer.id, 500, 50, 'solar');

      // Create cooperative wallet
      const coopWallet = finance.createWallet('cooperative-did', 'cooperative');

      // Consumer buys energy
      const consumer = energy.registerParticipant('consumer-did', 'consumer', [], 0);
      const consumerWallet = finance.createWallet('consumer-did', 'personal');
      consumerWallet.balances.set('MCX', 1000);

      // Trade and pay
      energy.tradeEnergy(producer.id, consumer.id, 100, 'solar', 3);
      finance.transfer(consumerWallet.id, coopWallet.id, 300, 'MCX', 'Energy purchase');

      // Verify cooperative earnings
      expect(finance.getBalance(coopWallet.id, 'MCX')).toBe(300);
    });
  });

  describe('Justice → Finance Enforcement Workflow', () => {
    it('should enforce financial remedy from justice decision', () => {
      const justice = getJusticeService();
      const finance = getFinanceService();

      // Create wallets
      const offenderWallet = finance.createWallet('did:mycelix:offender-32-char-valid', 'personal');
      const victimWallet = finance.createWallet('did:mycelix:victim-32-char-key-valid', 'personal');
      offenderWallet.balances.set('MCX', 500);

      // File case
      const disputeCase = justice.fileCase(
        'did:mycelix:victim-32-char-key-valid',
        'did:mycelix:offender-32-char-valid',
        'Contract Breach',
        'Offender failed to deliver services as agreed',
        'contract'
      );

      // Register arbitrator
      justice.registerArbitrator('did:mycelix:arbitrator-32-char-ok', ['contract'], 2);

      // Escalate to arbitration
      justice.escalateToArbitration(disputeCase.id, ['did:mycelix:arbitrator-32-char-ok']);

      // Render decision with compensation remedy
      const decision = justice.renderDecision(
        disputeCase.id,
        'complainant_favor',
        'Evidence shows clear breach of contract',
        [
          {
            type: 'compensation',
            targetId: 'did:mycelix:offender-32-char-valid',
            description: 'Pay 200 MCX to victim',
            amount: 200,
            deadline: Date.now() + 7 * 24 * 60 * 60 * 1000,
            completed: false,
          },
        ],
        ['did:mycelix:arbitrator-32-char-ok']
      );

      expect(decision.outcome).toBe('complainant_favor');

      // Execute enforcement (transfer funds)
      finance.transfer(offenderWallet.id, victimWallet.id, 200, 'MCX', 'Court-ordered compensation');

      // Verify transfer
      expect(finance.getBalance(victimWallet.id, 'MCX')).toBe(200);
      expect(finance.getBalance(offenderWallet.id, 'MCX')).toBe(300);
    });
  });

  describe('Full DAO Lifecycle', () => {
    it('should support complete DAO proposal lifecycle', () => {
      const identity = getIdentityService();
      const governance = getGovernanceService();

      // Create members
      const alice = identity.createIdentity('alice-key123456789012345678901234');
      const bob = identity.createIdentity('bob-key1234567890123456789012345a');
      const charlie = identity.createIdentity('charlie-key12345678901234567890ab');

      // Register all as DAO members
      governance.registerMember({
        did: alice.did,
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: bob.did,
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      governance.registerMember({
        did: charlie.did,
        daoId: 'test-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      // Create proposal
      const proposal = governance.createProposal({
        title: 'Increase Treasury Allocation',
        description: 'Allocate 10% of treasury to development',
        proposerId: alice.did,
        daoId: 'test-dao',
        votingPeriodHours: 1, // Short for testing
        quorumPercentage: 0.5,
      });

      // Vote
      governance.castVote({
        proposalId: proposal.id,
        voterId: alice.did,
        choice: 'approve',
        weight: 100,
      });

      governance.castVote({
        proposalId: proposal.id,
        voterId: bob.did,
        choice: 'approve',
        weight: 100,
      });

      governance.castVote({
        proposalId: proposal.id,
        voterId: charlie.did,
        choice: 'reject',
        weight: 100,
      });

      // Check vote tallies
      const updatedProposal = governance.getProposal(proposal.id);
      expect(updatedProposal?.approvesWeight).toBe(200);
      expect(updatedProposal?.rejectsWeight).toBe(100);

      // Finalize
      const result = governance.finalizeProposal(proposal.id);
      expect(result.passed).toBe(true);
      expect(result.quorumMet).toBe(true);
      expect(result.approvalPercentage).toBeCloseTo(0.667, 2);
    });
  });
});

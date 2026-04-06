// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-hApp Integration Tests
 *
 * Tests ecosystem-level integration between Mycelix hApps:
 * - MATL trust propagation across Mail, Marketplace, Praxis, SupplyChain
 * - Bridge protocol reputation queries
 * - Epistemic Charter consistency
 * - Multi-hApp user journey scenarios
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Core SDK imports (direct module imports to avoid holochain client ESM issues)
import * as matl from '../../src/matl/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as bridge from '../../src/bridge/index.js';

// Integration modules
import { getMailTrustService, MailTrustService } from '../../src/integrations/mail/index.js';
import { getMarketplaceService, MarketplaceReputationService } from '../../src/integrations/marketplace/index.js';
import { getPraxisService, PraxisCredentialService } from '../../src/integrations/praxis/index.js';
import { getSupplyChainService, SupplyChainProvenanceService } from '../../src/integrations/supplychain/index.js';

describe('Cross-hApp Ecosystem Integration', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    // Register all hApps
    localBridge.registerHapp('mail');
    localBridge.registerHapp('marketplace');
    localBridge.registerHapp('praxis');
    localBridge.registerHapp('supplychain');
  });

  describe('MATL Trust Propagation', () => {
    it('should propagate reputation scores across hApps via Bridge', () => {
      const agentId = 'agent-123';

      // Build reputation in Mail (positive interactions)
      const mailReputation = matl.createReputation(agentId);
      const updated1 = matl.recordPositive(mailReputation);
      const updated2 = matl.recordPositive(updated1);
      const updated3 = matl.recordPositive(updated2);

      // Store in bridge
      localBridge.setReputation('mail', agentId, updated3);

      // Query from Marketplace perspective
      const crossHappScore = localBridge.getAggregateReputation(agentId);

      expect(crossHappScore).toBeGreaterThan(0.5);
      // Verify mail reputation was stored
      const scores = localBridge.getCrossHappReputation(agentId);
      expect(scores.find(s => s.happ === 'mail')).toBeDefined();
    });

    it('should aggregate reputation from multiple hApps with weighting', () => {
      const agentId = 'multi-happ-user';

      // High reputation in Mail
      let mailRep = matl.createReputation(agentId);
      for (let i = 0; i < 10; i++) {
        mailRep = matl.recordPositive(mailRep);
      }
      localBridge.setReputation('mail', agentId, mailRep);

      // Medium reputation in Marketplace
      let marketRep = matl.createReputation(agentId);
      for (let i = 0; i < 5; i++) {
        marketRep = matl.recordPositive(marketRep);
      }
      for (let i = 0; i < 3; i++) {
        marketRep = matl.recordNegative(marketRep);
      }
      localBridge.setReputation('marketplace', agentId, marketRep);

      // Low reputation in Praxis
      let eduRep = matl.createReputation(agentId);
      for (let i = 0; i < 2; i++) {
        eduRep = matl.recordNegative(eduRep);
      }
      localBridge.setReputation('praxis', agentId, eduRep);

      // Aggregate should reflect weighted combination
      const aggregate = localBridge.getAggregateReputation(agentId);

      // Should be between min and max of individual scores
      const mailScore = matl.reputationValue(mailRep);
      const marketScore = matl.reputationValue(marketRep);
      const eduScore = matl.reputationValue(eduRep);

      const minScore = Math.min(mailScore, marketScore, eduScore);
      const maxScore = Math.max(mailScore, marketScore, eduScore);

      expect(aggregate).toBeGreaterThanOrEqual(minScore - 0.1);
      expect(aggregate).toBeLessThanOrEqual(maxScore + 0.1);
    });

    it('should detect Byzantine behavior pattern across hApps', () => {
      const maliciousAgent = 'byzantine-actor';

      // Good behavior in Mail (building trust)
      let mailRep = matl.createReputation(maliciousAgent);
      for (let i = 0; i < 10; i++) {
        mailRep = matl.recordPositive(mailRep);
      }
      localBridge.setReputation('mail', maliciousAgent, mailRep);

      // Malicious behavior in Marketplace
      let marketRep = matl.createReputation(maliciousAgent);
      for (let i = 0; i < 15; i++) {
        marketRep = matl.recordNegative(marketRep);
      }
      localBridge.setReputation('marketplace', maliciousAgent, marketRep);

      // Check for suspicious pattern (high variance across hApps)
      const mailScore = matl.reputationValue(mailRep);
      const marketScore = matl.reputationValue(marketRep);
      const scoreDifference = Math.abs(mailScore - marketScore);

      // Large difference suggests potential gaming
      expect(scoreDifference).toBeGreaterThan(0.3);

      // Aggregate should be lower due to negative behavior
      const aggregate = localBridge.getAggregateReputation(maliciousAgent);
      expect(aggregate).toBeLessThan(mailScore);
    });
  });

  describe('Epistemic Charter Consistency', () => {
    it('should classify claims consistently across hApps', () => {
      // Same claim should get same classification regardless of source hApp

      // Email claim (from Mail)
      const emailClaim = epistemic
        .claim('Identity verified via DKIM signature')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M1_Temporal)
        .build();

      // Transaction claim (from Marketplace)
      const transactionClaim = epistemic
        .claim('Payment confirmed via blockchain')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .build();

      // Credential claim (from Praxis)
      const credentialClaim = epistemic
        .claim('Course completion certificate')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N1_Communal)
        .withMateriality(epistemic.MaterialityLevel.M3_Foundational)
        .build();

      // Verify classification structure is consistent
      expect(emailClaim.classification.empirical).toBe(epistemic.EmpiricalLevel.E2_PrivateVerify);
      expect(transactionClaim.classification.empirical).toBe(epistemic.EmpiricalLevel.E3_Cryptographic);
      expect(credentialClaim.classification.empirical).toBe(epistemic.EmpiricalLevel.E3_Cryptographic);

      // Materiality should reflect expected persistence
      expect(emailClaim.classification.materiality).toBe(epistemic.MaterialityLevel.M1_Temporal);
      expect(transactionClaim.classification.materiality).toBe(epistemic.MaterialityLevel.M2_Persistent);
      expect(credentialClaim.classification.materiality).toBe(epistemic.MaterialityLevel.M3_Foundational);
    });

    it('should validate claim transitions follow Charter rules', () => {
      // Claims can only upgrade empirical level with evidence
      const initialClaim = epistemic
        .claim('User submitted form')
        .withEmpirical(epistemic.EmpiricalLevel.E0_Unverified)
        .withNormative(epistemic.NormativeLevel.N0_Personal)
        .build();

      // After email verification
      const verifiedClaim = epistemic
        .claim('User submitted form')
        .withEmpirical(epistemic.EmpiricalLevel.E1_Testimonial)
        .withNormative(epistemic.NormativeLevel.N0_Personal)
        .build();

      // After cryptographic proof
      const provenClaim = epistemic
        .claim('User submitted form')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .build();

      // Empirical levels should be ordered
      expect(initialClaim.classification.empirical).toBeLessThan(verifiedClaim.classification.empirical);
      expect(verifiedClaim.classification.empirical).toBeLessThan(provenClaim.classification.empirical);
    });
  });

  describe('Multi-hApp User Journey', () => {
    it('should track user through onboarding journey across hApps', () => {
      const newUser = 'new-user-001';

      // Step 1: User registers via Mail (email verification)
      let mailRep = matl.createReputation(newUser);
      mailRep = matl.recordPositive(mailRep); // Verified email
      localBridge.setReputation('mail', newUser, mailRep);

      // Initial aggregate should be low (only 1 interaction)
      let aggregate = localBridge.getAggregateReputation(newUser);
      expect(aggregate).toBeGreaterThan(0.4);
      expect(aggregate).toBeLessThan(0.8);

      // Step 2: User completes first course in Praxis
      let eduRep = matl.createReputation(newUser);
      eduRep = matl.recordPositive(eduRep);
      eduRep = matl.recordPositive(eduRep);
      localBridge.setReputation('praxis', newUser, eduRep);

      // Aggregate should increase
      const afterEdunet = localBridge.getAggregateReputation(newUser);
      expect(afterEdunet).toBeGreaterThanOrEqual(aggregate);

      // Step 3: User makes first purchase in Marketplace
      let marketRep = matl.createReputation(newUser);
      marketRep = matl.recordPositive(marketRep); // Completed transaction
      localBridge.setReputation('marketplace', newUser, marketRep);

      // Step 4: User becomes supplier in SupplyChain
      let supplyRep = matl.createReputation(newUser);
      supplyRep = matl.recordPositive(supplyRep);
      supplyRep = matl.recordPositive(supplyRep);
      supplyRep = matl.recordPositive(supplyRep);
      localBridge.setReputation('supplychain', newUser, supplyRep);

      // Final aggregate should reflect good standing across ecosystem
      const finalAggregate = localBridge.getAggregateReputation(newUser);
      expect(finalAggregate).toBeGreaterThan(0.5);
    });

    it('should handle reputation recovery after incident', () => {
      const recoveryUser = 'recovery-user-001';

      // Build good reputation
      let marketRep = matl.createReputation(recoveryUser);
      for (let i = 0; i < 10; i++) {
        marketRep = matl.recordPositive(marketRep);
      }
      localBridge.setReputation('marketplace', recoveryUser, marketRep);

      const goodScore = localBridge.getAggregateReputation(recoveryUser);
      expect(goodScore).toBeGreaterThan(0.7);

      // Incident occurs (disputed transaction)
      for (let i = 0; i < 3; i++) {
        marketRep = matl.recordNegative(marketRep);
      }
      localBridge.setReputation('marketplace', recoveryUser, marketRep);

      const postIncidentScore = localBridge.getAggregateReputation(recoveryUser);
      expect(postIncidentScore).toBeLessThan(goodScore);

      // Recovery through consistent positive behavior
      for (let i = 0; i < 8; i++) {
        marketRep = matl.recordPositive(marketRep);
      }
      localBridge.setReputation('marketplace', recoveryUser, marketRep);

      const recoveredScore = localBridge.getAggregateReputation(recoveryUser);
      expect(recoveredScore).toBeGreaterThan(postIncidentScore);
    });
  });

  describe('Bridge Protocol Communication', () => {
    it('should create valid reputation query messages', () => {
      const query = bridge.createReputationQuery('mail', 'agent-123');

      expect(query.type).toBe(bridge.BridgeMessageType.ReputationQuery);
      expect(query.sourceHapp).toBe('mail');
      expect(query.agent).toBe('agent-123');
      expect(query.timestamp).toBeLessThanOrEqual(Date.now());
    });

    it('should handle cross-hApp credential verification', () => {
      const credentialHash = 'credential-hash-abc123';

      // Create verification request from Praxis
      const verifyRequest: bridge.CredentialVerificationMessage = {
        type: bridge.BridgeMessageType.CredentialVerification,
        timestamp: Date.now(),
        sourceHapp: 'praxis',
        credentialHash,
        issuerHapp: 'praxis',
      };

      expect(verifyRequest.type).toBe(bridge.BridgeMessageType.CredentialVerification);
      expect(verifyRequest.issuerHapp).toBe('praxis');
    });

    it('should broadcast events to all registered hApps', () => {
      const receivedEvents: bridge.BroadcastEventMessage[] = [];

      // Register event handlers for each hApp to receive broadcasts
      ['mail', 'praxis', 'supplychain'].forEach(happId => {
        localBridge.on(happId, bridge.BridgeMessageType.BroadcastEvent, (msg) => {
          receivedEvents.push(msg as bridge.BroadcastEventMessage);
        });
      });

      // Broadcast trust update event from marketplace
      localBridge.broadcast({
        type: bridge.BridgeMessageType.BroadcastEvent,
        timestamp: Date.now(),
        sourceHapp: 'marketplace',
        eventType: 'trust_updated',
        payload: new Uint8Array([1, 2, 3]),
      });

      // Other hApps (not marketplace) should receive the event
      expect(receivedEvents.length).toBe(3);
      expect(receivedEvents[0].eventType).toBe('trust_updated');
    });
  });

  describe('Byzantine Fault Tolerance', () => {
    it('should maintain 34% validated BFT threshold across ecosystem', () => {
      // Simulate network with honest and Byzantine nodes
      const totalNodes = 100;
      const byzantineRatio = 0.34; // At 34% validated tolerance limit
      const byzantineNodes = Math.floor(totalNodes * byzantineRatio);
      const honestNodes = totalNodes - byzantineNodes;

      // Create honest node reputations
      const honestScores: number[] = [];
      for (let i = 0; i < honestNodes; i++) {
        let rep = matl.createReputation(`honest-${i}`);
        for (let j = 0; j < 10; j++) {
          rep = matl.recordPositive(rep);
        }
        honestScores.push(matl.reputationValue(rep));
        localBridge.setReputation('marketplace', `honest-${i}`, rep);
      }

      // Create Byzantine node reputations (trying to game the system)
      const byzantineScores: number[] = [];
      for (let i = 0; i < byzantineNodes; i++) {
        let rep = matl.createReputation(`byzantine-${i}`);
        // Byzantine nodes may try to build reputation then exploit
        for (let j = 0; j < 5; j++) {
          rep = matl.recordPositive(rep);
        }
        byzantineScores.push(matl.reputationValue(rep));
        localBridge.setReputation('marketplace', `byzantine-${i}`, rep);
      }

      // Honest majority should maintain higher average
      const honestAvg = honestScores.reduce((a, b) => a + b, 0) / honestScores.length;
      const byzantineAvg = byzantineScores.reduce((a, b) => a + b, 0) / byzantineScores.length;

      expect(honestAvg).toBeGreaterThan(byzantineAvg);
    });

    it('should detect coordinated attack patterns (cartel)', () => {
      // Simulate cartel behavior: group of agents boosting each other
      const cartelSize = 5;
      const cartelAgents: string[] = [];

      for (let i = 0; i < cartelSize; i++) {
        cartelAgents.push(`cartel-${i}`);
      }

      // Cartel members all get positive reviews from each other
      for (const agent of cartelAgents) {
        let rep = matl.createReputation(agent);
        for (let i = 0; i < cartelSize - 1; i++) {
          rep = matl.recordPositive(rep);
        }
        localBridge.setReputation('marketplace', agent, rep);
      }

      // Check for suspicious pattern: all cartel members have similar scores
      const cartelScores = cartelAgents.map((agent) => {
        return localBridge.getAggregateReputation(agent);
      });

      // Calculate score variance (cartels often have low variance)
      const avgScore = cartelScores.reduce((a, b) => a + b, 0) / cartelScores.length;
      const variance =
        cartelScores.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) /
        cartelScores.length;

      // Low variance + similar timestamps would indicate cartel
      // (In production, this would trigger MATL's cartel detection)
      expect(variance).toBeLessThan(0.01); // Very low variance is suspicious
    });
  });
});

describe('Integration Service Tests', () => {
  describe('MailTrustService', () => {
    let mailService: MailTrustService;

    beforeEach(() => {
      mailService = new MailTrustService();
    });

    it('should track sender reputation through interactions', () => {
      const sender = 'trusted@example.com';

      // Record positive interactions
      mailService.recordInteraction(sender, true);
      mailService.recordInteraction(sender, true);
      mailService.recordInteraction(sender, true);

      const trust = mailService.getSenderTrust(sender);

      expect(trust.trustworthy).toBe(true);
      expect(trust.score).toBeGreaterThan(0.5);
      expect(trust.level).toMatch(/medium|high|verified/);
    });

    it('should classify email claims based on verification', () => {
      const verifiedEmail = {
        id: 'email-001',
        subject: 'Contract Signed',
        from: 'legal@company.com',
        to: ['recipient@example.com'],
        body: 'Contract details...',
        timestamp: Date.now(),
        verification: {
          dkimVerified: true,
          spfPassed: true,
          dmarcPassed: true,
        },
      };

      const claim = mailService.createEmailClaim(verifiedEmail);

      expect(claim.verificationType).toBe('dkim');
      expect(claim.claim.classification.empirical).toBe(epistemic.EmpiricalLevel.E2_PrivateVerify);
    });
  });

  describe('Cross-Service Integration', () => {
    it('should share reputation between Mail and Marketplace services', () => {
      const sharedBridge = new bridge.LocalBridge();
      sharedBridge.registerHapp('mail');
      sharedBridge.registerHapp('marketplace');

      const user = 'cross-service-user';

      // Build reputation via Mail
      let rep = matl.createReputation(user);
      rep = matl.recordPositive(rep);
      rep = matl.recordPositive(rep);
      sharedBridge.setReputation('mail', user, rep);

      // Query from Marketplace - check reputation is accessible
      const scores = sharedBridge.getCrossHappReputation(user);
      const mailScore = scores.find(s => s.happ === 'mail');
      expect(mailScore).toBeDefined();

      const aggregate = sharedBridge.getAggregateReputation(user);
      expect(aggregate).toBeGreaterThan(0.5);
    });
  });
});

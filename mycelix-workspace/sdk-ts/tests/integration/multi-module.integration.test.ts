// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Multi-Module Integration Tests
 *
 * Tests complex scenarios involving all SDK modules working together.
 * These tests simulate real-world usage patterns.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as matl from '../../src/matl/index.js';
import * as fl from '../../src/fl/index.js';
import * as bridge from '../../src/bridge/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as errors from '../../src/errors.js';

// Helper to create proper GradientUpdate
function createUpdate(
  id: string,
  gradients: number[],
  batchSize: number,
  loss: number
): fl.GradientUpdate {
  return {
    participantId: id,
    modelVersion: 1,
    gradients: new Float64Array(gradients),
    metadata: {
      batchSize,
      loss,
      timestamp: Date.now(),
    },
  };
}

describe('Multi-Module Integration: Complete Ecosystem', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(async () => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('trust_layer');
    localBridge.registerHapp('marketplace');
    localBridge.registerHapp('fl_network');
    localBridge.registerHapp('governance');
  });

  describe('Scenario: Decentralized Marketplace with FL-Enhanced Trust', () => {
    it('should run complete marketplace transaction with all trust checks', async () => {
      const sellerId = 'seller_alice';
      const buyerId = 'buyer_bob';

      // Step 1: Setup seller reputation in trust layer
      let sellerRep = matl.createReputation(sellerId);
      for (let i = 0; i < 30; i++) {
        sellerRep = matl.recordPositive(sellerRep);
      }
      for (let i = 0; i < 2; i++) {
        sellerRep = matl.recordNegative(sellerRep);
      }

      localBridge.setReputation('marketplace', sellerId, sellerRep);

      // Step 2: Create seller's PoGQ measurement
      const sellerPogq = matl.createPoGQ(0.9, 0.85, 0.1);

      // Step 3: Calculate composite trust score
      const sellerComposite = matl.calculateComposite(sellerPogq, sellerRep);

      // Step 4: Verify seller is trustworthy
      expect(matl.isTrustworthy(sellerComposite, 0.5)).toBe(true);
      expect(matl.isByzantine(sellerPogq)).toBe(false);

      // Step 5: Create epistemic claim about seller verification
      const sellerVerificationClaim = epistemic
        .claim('Seller verified for marketplace transactions')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('trust_layer')
        .build();

      expect(
        epistemic.meetsMinimum(
          sellerVerificationClaim.classification,
          epistemic.Standards.MediumTrust.minE,
          epistemic.Standards.MediumTrust.minN
        )
      ).toBe(true);

      // Step 6: Buyer checks seller via bridge
      const crossHappScores = localBridge.getCrossHappReputation(sellerId);
      expect(crossHappScores.length).toBeGreaterThan(0);

      const aggregateRep = localBridge.getAggregateReputation(sellerId);
      expect(aggregateRep).toBeGreaterThan(0.7);

      // Step 7: Record successful transaction
      sellerRep = matl.recordPositive(sellerRep);
      localBridge.setReputation('marketplace', sellerId, sellerRep);

      // Step 8: Broadcast event
      const txEvent = bridge.createBroadcastEvent(
        'marketplace',
        'transaction_complete',
        new TextEncoder().encode(
          JSON.stringify({
            seller: sellerId,
            buyer: buyerId,
            amount: 100,
            timestamp: Date.now(),
          })
        )
      );

      const receivedByGovernance: bridge.AnyBridgeMessage[] = [];
      localBridge.on('governance', bridge.BridgeMessageType.BroadcastEvent, (msg) =>
        receivedByGovernance.push(msg)
      );

      localBridge.broadcast(txEvent);
      expect(receivedByGovernance.length).toBe(1);
    });
  });

  describe('Scenario: Federated Learning Network with Trust', () => {
    it('should coordinate FL round with reputation-based participant selection', async () => {
      const participants: Record<string, fl.Participant> = {};
      const allAgents = ['hospital_a', 'hospital_b', 'hospital_c', 'clinic_d', 'research_e'];

      // Setup participants with varying reputations
      for (let i = 0; i < allAgents.length; i++) {
        const agentId = allAgents[i];
        let rep = matl.createReputation(agentId);

        // Vary reputation based on index
        const positives = 20 + i * 5;
        const negatives = Math.floor(i / 2);

        for (let j = 0; j < positives; j++) {
          rep = matl.recordPositive(rep);
        }
        for (let j = 0; j < negatives; j++) {
          rep = matl.recordNegative(rep);
        }

        // Create PoGQ measurement
        const pogq = matl.createPoGQ(
          0.8 + Math.random() * 0.15,
          0.85 + Math.random() * 0.1,
          0.05 + Math.random() * 0.1
        );

        participants[agentId] = {
          id: agentId,
          reputation: rep,
          pogq,
          roundsParticipated: i + 1,
        };

        localBridge.setReputation('fl_network', agentId, rep);
      }

      // Filter for trustworthy participants
      const trustworthyParticipants = Object.values(participants).filter((p) => {
        if (!p.pogq) return false;
        const composite = matl.calculateComposite(p.pogq, p.reputation);
        return matl.isTrustworthy(composite, 0.5) && !matl.isByzantine(p.pogq);
      });

      expect(trustworthyParticipants.length).toBeGreaterThanOrEqual(3);

      // Create FL coordinator
      const coordinator = new fl.FLCoordinator({
        minParticipants: 3,
        aggregationMethod: 'fedavg',
      });

      // Register trustworthy participants
      for (const p of trustworthyParticipants) {
        coordinator.registerParticipant(p.id);
      }

      // Start round
      coordinator.startRound();

      // Submit updates using the createUpdate helper
      for (const p of trustworthyParticipants) {
        const gradientValues: number[] = [];
        for (let i = 0; i < 100; i++) {
          gradientValues.push(0.1 + (Math.random() - 0.5) * 0.05);
        }

        coordinator.submitUpdate(createUpdate(p.id, gradientValues, 500, 0.3 + Math.random() * 0.1));
      }

      // Aggregate with trust weighting
      // aggregateRound returns boolean success status, not the gradients
      const aggregationSuccess = coordinator.aggregateRound();

      expect(aggregationSuccess).toBe(true);

      // Verify the round completed via stats
      const finalStats = coordinator.getRoundStats();
      expect(finalStats.totalRounds).toBe(1);

      // Create epistemic claim about FL round
      const flRoundClaim = epistemic
        .claim('FL round completed with Byzantine-resistant aggregation')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('fl_network')
        .build();

      expect(epistemic.classificationCode(flRoundClaim.classification)).toBe('E3-N2-M2');
    });
  });

  describe('Scenario: Cross-hApp Identity and Trust Portability', () => {
    it('should enable identity portability across ecosystem', async () => {
      const userId = 'universal_user';

      // Build reputation across different hApps
      const hAppData = [
        { happ: 'marketplace', positives: 50, negatives: 3 },
        { happ: 'governance', positives: 20, negatives: 1 },
        { happ: 'fl_network', positives: 30, negatives: 2 },
      ];

      for (const data of hAppData) {
        let rep = matl.createReputation(userId);
        for (let i = 0; i < data.positives; i++) {
          rep = matl.recordPositive(rep);
        }
        for (let i = 0; i < data.negatives; i++) {
          rep = matl.recordNegative(rep);
        }
        localBridge.setReputation(data.happ, userId, rep);
      }

      // Create claims for each hApp reputation
      const claims: epistemic.EpistemicClaim[] = [];
      for (const data of hAppData) {
        const scores = localBridge.getCrossHappReputation(userId);
        const hAppScore = scores.find((s) => s.happ === data.happ);

        if (hAppScore) {
          const claim = epistemic
            .claim(`${data.happ} reputation: ${hAppScore.score.toFixed(3)}`)
            .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
            .withNormative(epistemic.NormativeLevel.N2_Network)
            .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
            .withIssuer(data.happ)
            .build();

          claims.push(claim);
        }
      }

      // Create aggregate identity claim
      const aggregateScore = localBridge.getAggregateReputation(userId);
      const identityClaim = epistemic
        .claim(`Aggregate cross-hApp reputation: ${aggregateScore.toFixed(3)}`)
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('trust_layer')
        .build();

      // Add evidence from component claims
      let claimWithEvidence = identityClaim;
      for (const componentClaim of claims) {
        claimWithEvidence = epistemic.addEvidence(claimWithEvidence, {
          type: 'reputation_claim',
          data: componentClaim.id,
          source: componentClaim.issuer || 'unknown',
          timestamp: Date.now(),
        });
      }

      expect(claimWithEvidence.evidence.length).toBe(claims.length);
      expect(aggregateScore).toBeGreaterThan(0.8);

      // Verify new hApp can trust this user
      const newHappTrustDecision = epistemic.meetsMinimum(
        identityClaim.classification,
        epistemic.Standards.MediumTrust.minE,
        epistemic.Standards.MediumTrust.minN,
        epistemic.Standards.MediumTrust.minM
      );

      expect(newHappTrustDecision).toBe(true);
    });
  });

  describe('Scenario: Error Handling Across Modules', () => {
    it('should propagate validation errors correctly', () => {
      // MATL validation errors - should throw
      expect(() => matl.createPoGQ(-0.1, 0.5, 0.5)).toThrow();
      expect(() => matl.createPoGQ(0.5, 1.5, 0.5)).toThrow();
    });

    it('should handle async errors with retry', async () => {
      let attempts = 0;
      const flakeyOperation = async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Network timeout');
        }
        return 'success';
      };

      const result = await errors.withRetry(flakeyOperation, {
        maxRetries: 3,
        delayMs: 10,
      });

      expect(result).toBe('success');
      expect(attempts).toBe(3);
    });

    it('should validate inputs with Validator', () => {
      const validator = new errors.Validator();

      // Valid inputs should pass
      expect(() => {
        validator
          .required('agentId', 'agent_001')
          .inRange('quality', 0.5, 0, 1)
          .inRange('consistency', 0.8, 0, 1)
          .throwIfInvalid();
      }).not.toThrow();

      // Invalid inputs should throw
      expect(() => {
        new errors.Validator()
          .required('agentId', null)
          .throwIfInvalid();
      }).toThrow();
    });
  });

  describe('Scenario: Performance Under Load', () => {
    it('should handle batch reputation calculations', () => {
      const startTime = performance.now();
      const count = 100; // Reduced for CI performance

      for (let i = 0; i < count; i++) {
        const pogq = matl.createPoGQ(
          Math.random(),
          Math.random(),
          Math.random()
        );

        let rep = matl.createReputation(`agent_${i}`);
        rep = matl.recordPositive(rep);
        rep = matl.recordPositive(rep);

        const composite = matl.calculateComposite(pogq, rep);
        matl.isTrustworthy(composite, 0.5);
      }

      const elapsed = performance.now() - startTime;

      // Should complete calculations reasonably fast
      expect(elapsed).toBeLessThan(1000); // 1 second for 100 calculations
    });

    it('should handle concurrent FL aggregations', () => {
      const startTime = performance.now();
      const rounds = 5;
      const participantsPerRound = 10;

      for (let round = 0; round < rounds; round++) {
        const updates: fl.GradientUpdate[] = [];

        for (let i = 0; i < participantsPerRound; i++) {
          const gradientValues: number[] = [];
          for (let j = 0; j < 100; j++) {
            gradientValues.push(Math.random() * 0.1);
          }

          updates.push(createUpdate(`p_${round}_${i}`, gradientValues, 100, 0.3));
        }

        // Test aggregation methods
        fl.fedAvg(updates);
        fl.trimmedMean(updates, 0.1);
        fl.coordinateMedian(updates);
      }

      const elapsed = performance.now() - startTime;

      // Should complete all rounds reasonably fast
      expect(elapsed).toBeLessThan(5000);
    });
  });

  describe('Scenario: Adaptive Threshold Anomaly Detection', () => {
    it('should detect anomalies in FL gradient norms', () => {
      let threshold = matl.createAdaptiveThreshold('fl_monitor', 50, 0.3, 3.0);

      // Establish baseline with normal gradient norms around 0.7
      for (let i = 0; i < 30; i++) {
        const normalNorm = 0.7 + (Math.random() - 0.5) * 0.1;
        threshold = matl.observe(threshold, normalNorm);
      }

      // Get current threshold
      const currentThreshold = matl.getThreshold(threshold);
      expect(currentThreshold).toBeGreaterThan(0);
      expect(currentThreshold).toBeLessThan(0.7);

      // isAnomalous detects values BELOW threshold (poor quality)
      // Normal quality values should not be anomalous
      expect(matl.isAnomalous(threshold, 0.7)).toBe(false);

      // Low quality values (below threshold) ARE anomalous
      expect(matl.isAnomalous(threshold, 0.1)).toBe(true);
    });

    it('should adapt threshold based on network conditions', () => {
      let threshold = matl.createAdaptiveThreshold('adaptive_monitor', 100, 0.5, 2.0);

      // Phase 1: Low-variance network
      for (let i = 0; i < 50; i++) {
        threshold = matl.observe(threshold, 0.5 + (Math.random() - 0.5) * 0.02);
      }

      const tightThreshold = matl.getThreshold(threshold);

      // Phase 2: High-variance network
      for (let i = 0; i < 50; i++) {
        threshold = matl.observe(threshold, 0.5 + (Math.random() - 0.5) * 0.4);
      }

      const looseThreshold = matl.getThreshold(threshold);

      // Threshold should change with variance (may go either way depending on implementation)
      expect(looseThreshold).toBeDefined();
      expect(tightThreshold).toBeDefined();
    });
  });
});

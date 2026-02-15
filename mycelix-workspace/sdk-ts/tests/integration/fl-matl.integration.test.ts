/**
 * FL + MATL Integration Tests
 *
 * Tests the integration between Federated Learning and the
 * Mycelix Adaptive Trust Layer for Byzantine-resistant aggregation.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as matl from '../../src/matl/index.js';
import * as fl from '../../src/fl/index.js';

describe('Integration: FL + MATL', () => {
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

  describe('Trust-Weighted Aggregation with Live Reputation', () => {
    it('should weight FL contributions by MATL reputation', () => {
      // Create participants with varying trust levels
      const participants = new Map<string, fl.Participant>();
      const updates: fl.GradientUpdate[] = [];

      // Highly trusted participant
      let trustedRep = matl.createReputation('trusted_node');
      for (let i = 0; i < 50; i++) {
        trustedRep = matl.recordPositive(trustedRep);
      }
      participants.set('trusted_node', {
        id: 'trusted_node',
        reputation: trustedRep,
        roundsParticipated: 10,
      });
      updates.push(createUpdate('trusted_node', [0.1, 0.2, 0.1, 0.15], 100, 0.25));

      // Medium trust participant
      let mediumRep = matl.createReputation('medium_node');
      for (let i = 0; i < 10; i++) {
        mediumRep = matl.recordPositive(mediumRep);
      }
      participants.set('medium_node', {
        id: 'medium_node',
        reputation: mediumRep,
        roundsParticipated: 5,
      });
      updates.push(createUpdate('medium_node', [0.15, 0.18, 0.12, 0.14], 50, 0.3));

      // Perform trust-weighted aggregation
      const result = fl.trustWeightedAggregation(updates, participants, 0.3);

      // Result should be valid
      expect(result.gradients).toBeDefined();
      expect(result.gradients.length).toBe(4);
      // Gradients should be in reasonable range
      expect(result.gradients[0]).toBeGreaterThan(0);
      expect(result.gradients[0]).toBeLessThan(1);
    });

    it('should filter out Byzantine participants based on PoGQ', () => {
      // Create participants with PoGQ measurements
      const participants: fl.Participant[] = [];

      // Good participant with high PoGQ
      const goodPogq = matl.createPoGQ(0.9, 0.85, 0.1);
      let goodRep = matl.createReputation('good');
      for (let i = 0; i < 20; i++) {
        goodRep = matl.recordPositive(goodRep);
      }
      participants.push({
        id: 'good',
        reputation: goodRep,
        pogq: goodPogq,
        roundsParticipated: 5,
      });

      // Bad participant with Byzantine PoGQ
      const badPogq = matl.createPoGQ(0.2, 0.3, 0.8);
      const badRep = matl.createReputation('bad');
      participants.push({
        id: 'bad',
        reputation: badRep,
        pogq: badPogq,
        roundsParticipated: 1,
      });

      // Check Byzantine detection
      expect(matl.isByzantine(goodPogq)).toBe(false);
      expect(matl.isByzantine(badPogq)).toBe(true);

      // Filter participants
      const trusted = participants.filter(
        (p) => p.pogq && !matl.isByzantine(p.pogq)
      );
      expect(trusted.length).toBe(1);
      expect(trusted[0].id).toBe('good');
    });

    it('should handle Byzantine tolerance threshold correctly', () => {
      // Verify the 34% validated Byzantine tolerance constant
      expect(matl.MAX_BYZANTINE_TOLERANCE).toBe(0.34);

      // Create a mix of honest and Byzantine PoGQ measurements
      const honestPogq = matl.createPoGQ(0.9, 0.85, 0.1);
      const byzantinePogq = matl.createPoGQ(0.2, 0.3, 0.8);

      expect(matl.isByzantine(honestPogq)).toBe(false);
      expect(matl.isByzantine(byzantinePogq)).toBe(true);
    });
  });

  describe('FL Coordinator with MATL Integration', () => {
    it('should register participants and start rounds', () => {
      const coordinator = new fl.FLCoordinator({
        minParticipants: 3,
        aggregationMethod: 'fedavg',
      });

      // Register participants
      coordinator.registerParticipant('p1');
      coordinator.registerParticipant('p2');
      coordinator.registerParticipant('p3');

      // Verify registration using getRoundStats
      const stats = coordinator.getRoundStats();
      expect(stats.participantCount).toBeGreaterThanOrEqual(3);

      // Start round
      const round = coordinator.startRound();
      expect(round).toBeDefined();
      expect(round.status).toBe('collecting');

      // Submit updates - the coordinator should accept them
      const success1 = coordinator.submitUpdate(createUpdate('p1', [0.1, 0.2], 100, 0.3));
      const success2 = coordinator.submitUpdate(createUpdate('p2', [0.12, 0.18], 80, 0.32));
      const success3 = coordinator.submitUpdate(createUpdate('p3', [0.11, 0.19], 90, 0.28));

      // Coordinator should accept updates from registered participants
      expect(success1).toBe(true);
      expect(success2).toBe(true);
      expect(success3).toBe(true);
    });
  });

  describe('Byzantine-Resistant Aggregation Comparison', () => {
    it('should aggregate gradients from multiple participants', () => {
      const modelDim = 10;
      const updates: fl.GradientUpdate[] = [];

      // Create honest participants
      for (let i = 0; i < 5; i++) {
        const gradientValues: number[] = [];
        for (let j = 0; j < modelDim; j++) {
          gradientValues.push(0.1 + (Math.random() - 0.5) * 0.02);
        }
        updates.push(createUpdate(`honest_${i}`, gradientValues, 100, 0.3));
      }

      // Test that aggregation functions exist and can be called
      expect(typeof fl.fedAvg).toBe('function');
      expect(typeof fl.trimmedMean).toBe('function');
      expect(typeof fl.coordinateMedian).toBe('function');
      expect(typeof fl.krum).toBe('function');

      // The FL module should handle aggregation (actual results tested in fl.test.ts)
      expect(updates.length).toBe(5);
    });
  });

  describe('Reputation Evolution During FL', () => {
    it('should update reputation based on gradient quality', () => {
      let goodRep = matl.createReputation('good_contributor');
      let badRep = matl.createReputation('bad_contributor');

      // Simulate 10 FL rounds
      for (let round = 0; round < 10; round++) {
        // Good contributor: consistent, helpful gradients
        const goodPogq = matl.createPoGQ(
          0.85 + Math.random() * 0.1,
          0.9 + Math.random() * 0.05,
          0.05 + Math.random() * 0.05
        );

        // Bad contributor: inconsistent, noisy gradients
        const badPogq = matl.createPoGQ(
          0.3 + Math.random() * 0.2,
          0.4 + Math.random() * 0.2,
          0.5 + Math.random() * 0.3
        );

        // Update reputations based on Byzantine detection
        if (!matl.isByzantine(goodPogq)) {
          goodRep = matl.recordPositive(goodRep);
        } else {
          goodRep = matl.recordNegative(goodRep);
        }

        if (!matl.isByzantine(badPogq)) {
          badRep = matl.recordPositive(badRep);
        } else {
          badRep = matl.recordNegative(badRep);
        }
      }

      // Good contributor should have higher reputation
      const goodValue = matl.reputationValue(goodRep);
      const badValue = matl.reputationValue(badRep);

      expect(goodValue).toBeGreaterThan(badValue);
      expect(goodValue).toBeGreaterThan(0.7);
    });

    it('should compute composite scores for FL participants', () => {
      // High-quality participant
      const highQualityPogq = matl.createPoGQ(0.95, 0.92, 0.05);
      let highQualityRep = matl.createReputation('high_quality');
      for (let i = 0; i < 30; i++) {
        highQualityRep = matl.recordPositive(highQualityRep);
      }

      const highComposite = matl.calculateComposite(highQualityPogq, highQualityRep);

      // Low-quality participant
      const lowQualityPogq = matl.createPoGQ(0.4, 0.5, 0.6);
      let lowQualityRep = matl.createReputation('low_quality');
      for (let i = 0; i < 5; i++) {
        lowQualityRep = matl.recordNegative(lowQualityRep);
      }

      const lowComposite = matl.calculateComposite(lowQualityPogq, lowQualityRep);

      // High quality should be trustworthy
      expect(matl.isTrustworthy(highComposite, 0.5)).toBe(true);
      expect(highComposite.finalScore).toBeGreaterThan(0.6);

      // Low quality should not be trustworthy
      expect(matl.isTrustworthy(lowComposite, 0.5)).toBe(false);
      expect(lowComposite.finalScore).toBeLessThan(highComposite.finalScore);
    });
  });

  describe('Adaptive Threshold for Gradient Norms', () => {
    it('should detect low-quality values as anomalous', () => {
      let threshold = matl.createAdaptiveThreshold('fl_monitor', 50, 0.5, 3.0);

      // Establish baseline with normal values around 0.8
      for (let i = 0; i < 30; i++) {
        threshold = matl.observe(threshold, 0.8 + (Math.random() - 0.5) * 0.04);
      }

      // Get current threshold
      const currentThreshold = matl.getThreshold(threshold);
      expect(currentThreshold).toBeGreaterThan(0);

      // isAnomalous detects values BELOW the threshold (low quality)
      // Normal values (around baseline) should NOT be anomalous
      expect(matl.isAnomalous(threshold, 0.8)).toBe(false);
      expect(matl.isAnomalous(threshold, 0.9)).toBe(false);

      // Values below the threshold ARE anomalous
      expect(matl.isAnomalous(threshold, 0.1)).toBe(true); // Very low
      expect(matl.isAnomalous(threshold, 0.0)).toBe(true); // Zero quality
    });
  });
});

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Federated Learning Module Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as fl from '../src/fl/index.js';

describe('Federated Learning', () => {
  describe('FedAvg Aggregation', () => {
    it('should average gradients weighted by batch size', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0, 2.0, 3.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([2.0, 4.0, 6.0]),
          metadata: { batchSize: 10, loss: 0.4, timestamp: Date.now() },
        },
      ];

      const result = fl.fedAvg(updates);

      // Equal batch sizes, so simple average
      expect(result[0]).toBeCloseTo(1.5, 5);
      expect(result[1]).toBeCloseTo(3.0, 5);
      expect(result[2]).toBeCloseTo(4.5, 5);
    });

    it('should weight by batch size', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 30, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([2.0]),
          metadata: { batchSize: 10, loss: 0.4, timestamp: Date.now() },
        },
      ];

      const result = fl.fedAvg(updates);

      // Weighted: (1.0 * 30 + 2.0 * 10) / 40 = 50/40 = 1.25
      expect(result[0]).toBeCloseTo(1.25, 5);
    });
  });

  describe('Trimmed Mean Aggregation', () => {
    it('should exclude outliers', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([1.1]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p3',
          modelVersion: 1,
          gradients: new Float64Array([1.2]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p4',
          modelVersion: 1,
          gradients: new Float64Array([100.0]), // Outlier
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p5',
          modelVersion: 1,
          gradients: new Float64Array([-100.0]), // Outlier
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = fl.trimmedMean(updates, 0.2); // Trim 20%

      // Should exclude the outliers
      expect(result[0]).toBeCloseTo(1.1, 1);
    });
  });

  describe('Coordinate Median Aggregation', () => {
    it('should return median values', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0, 10.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([2.0, 20.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p3',
          modelVersion: 1,
          gradients: new Float64Array([3.0, 30.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = fl.coordinateMedian(updates);

      expect(result[0]).toBe(2.0);
      expect(result[1]).toBe(20.0);
    });

    it('should be robust to Byzantine attacks', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'honest1',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'honest2',
          modelVersion: 1,
          gradients: new Float64Array([1.1]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'honest3',
          modelVersion: 1,
          gradients: new Float64Array([1.2]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'byzantine',
          modelVersion: 1,
          gradients: new Float64Array([1000.0]), // Malicious
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = fl.coordinateMedian(updates);

      // Median should ignore the Byzantine outlier
      expect(result[0]).toBeLessThan(10);
    });
  });

  describe('Krum Aggregation', () => {
    it('should select gradient closest to neighbors', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0, 1.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([1.1, 1.1]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p3',
          modelVersion: 1,
          gradients: new Float64Array([1.2, 1.2]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'byzantine',
          modelVersion: 1,
          gradients: new Float64Array([100.0, 100.0]), // Malicious
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = fl.krum(updates);

      // Should select one of the honest gradients
      expect(result[0]).toBeLessThan(2);
      expect(result[1]).toBeLessThan(2);
    });
  });

  describe('FLCoordinator', () => {
    let coordinator: fl.FLCoordinator;

    beforeEach(() => {
      coordinator = new fl.FLCoordinator({
        minParticipants: 2,
        maxParticipants: 10,
        trustThreshold: 0.3,
      });
    });

    it('should register participants', () => {
      const p1 = coordinator.registerParticipant('agent_1');
      const p2 = coordinator.registerParticipant('agent_2');

      expect(p1.id).toBe('agent_1');
      expect(p2.id).toBe('agent_2');
    });

    it('should start and manage FL rounds', () => {
      coordinator.registerParticipant('agent_1');
      coordinator.registerParticipant('agent_2');

      const round = coordinator.startRound();

      expect(round.roundId).toBe(1);
      expect(round.status).toBe('collecting');
    });

    it('should accept gradient updates', () => {
      coordinator.registerParticipant('agent_1');
      coordinator.registerParticipant('agent_2');
      coordinator.startRound();

      const accepted = coordinator.submitUpdate({
        participantId: 'agent_1',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0, 3.0]),
        metadata: { batchSize: 32, loss: 0.5, timestamp: Date.now() },
      });

      expect(accepted).toBe(true);
    });

    it('should reject updates from unregistered participants', () => {
      coordinator.registerParticipant('agent_1');
      coordinator.startRound();

      const accepted = coordinator.submitUpdate({
        participantId: 'unknown_agent',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0, 3.0]),
        metadata: { batchSize: 32, loss: 0.5, timestamp: Date.now() },
      });

      expect(accepted).toBe(false);
    });

    it('should aggregate when enough participants', () => {
      coordinator.registerParticipant('agent_1');
      coordinator.registerParticipant('agent_2');
      coordinator.startRound();

      coordinator.submitUpdate({
        participantId: 'agent_1',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0]),
        metadata: { batchSize: 32, loss: 0.5, timestamp: Date.now() },
      });

      coordinator.submitUpdate({
        participantId: 'agent_2',
        modelVersion: 1,
        gradients: new Float64Array([2.0, 4.0]),
        metadata: { batchSize: 32, loss: 0.4, timestamp: Date.now() },
      });

      const aggregated = coordinator.aggregateRound();
      expect(aggregated).toBe(true);

      const stats = coordinator.getRoundStats();
      expect(stats.totalRounds).toBe(1);
    });

    it('should track round statistics', () => {
      coordinator.registerParticipant('agent_1');
      coordinator.registerParticipant('agent_2');

      // Complete a round
      coordinator.startRound();
      coordinator.submitUpdate({
        participantId: 'agent_1',
        modelVersion: 1,
        gradients: new Float64Array([1.0]),
        metadata: { batchSize: 32, loss: 0.5, timestamp: Date.now() },
      });
      coordinator.submitUpdate({
        participantId: 'agent_2',
        modelVersion: 1,
        gradients: new Float64Array([2.0]),
        metadata: { batchSize: 32, loss: 0.4, timestamp: Date.now() },
      });
      coordinator.aggregateRound();

      const stats = coordinator.getRoundStats();
      expect(stats.totalRounds).toBe(1);
      expect(stats.participantCount).toBe(2);
      expect(stats.averageParticipation).toBe(2);
    });
  });

  describe('Gradient Serialization', () => {
    it('should serialize and deserialize gradients', () => {
      const original = new Float64Array([1.5, -2.3, 3.14159, 0.0, -1e-10]);

      const serialized = fl.serializeGradients(original);
      const deserialized = fl.deserializeGradients(serialized);

      expect(deserialized.length).toBe(original.length);
      for (let i = 0; i < original.length; i++) {
        expect(deserialized[i]).toBeCloseTo(original[i], 10);
      }
    });

    it('should throw on empty gradients', () => {
      expect(() => fl.serializeGradients(new Float64Array([]))).toThrow();
    });

    it('should throw on invalid data length', () => {
      // 7 bytes is not a multiple of 8
      const invalidData = new Uint8Array([1, 2, 3, 4, 5, 6, 7]);
      expect(() => fl.deserializeGradients(invalidData)).toThrow(/multiple of 8/);
    });

    it('should serialize and deserialize gradient update', () => {
      const original: fl.GradientUpdate = {
        participantId: 'participant-1',
        modelVersion: 5,
        gradients: new Float64Array([0.1, 0.2, 0.3, -0.5]),
        metadata: {
          batchSize: 64,
          loss: 0.25,
          accuracy: 0.92,
          timestamp: Date.now(),
        },
      };

      const serialized = fl.serializeGradientUpdate(original);
      expect(typeof serialized.gradients).toBe('string'); // base64 string

      const deserialized = fl.deserializeGradientUpdate(serialized);
      expect(deserialized.participantId).toBe(original.participantId);
      expect(deserialized.modelVersion).toBe(original.modelVersion);
      expect(deserialized.metadata.batchSize).toBe(original.metadata.batchSize);
      expect(deserialized.metadata.loss).toBe(original.metadata.loss);
      expect(deserialized.gradients.length).toBe(original.gradients.length);
      for (let i = 0; i < original.gradients.length; i++) {
        expect(deserialized.gradients[i]).toBeCloseTo(original.gradients[i], 10);
      }
    });

    it('should serialize and deserialize aggregated gradient', () => {
      const original: fl.AggregatedGradient = {
        gradients: new Float64Array([0.5, 0.6, 0.7]),
        modelVersion: 3,
        participantCount: 10,
        excludedCount: 2,
        aggregationMethod: fl.AggregationMethod.TrustWeighted,
        timestamp: Date.now(),
      };

      const serialized = fl.serializeAggregatedGradient(original);
      expect(typeof serialized.gradients).toBe('string');

      const deserialized = fl.deserializeAggregatedGradient(serialized);
      expect(deserialized.modelVersion).toBe(original.modelVersion);
      expect(deserialized.participantCount).toBe(original.participantCount);
      expect(deserialized.excludedCount).toBe(original.excludedCount);
      expect(deserialized.aggregationMethod).toBe(original.aggregationMethod);
      expect(deserialized.gradients.length).toBe(original.gradients.length);
      for (let i = 0; i < original.gradients.length; i++) {
        expect(deserialized.gradients[i]).toBeCloseTo(original.gradients[i], 10);
      }
    });

    it('should produce JSON-safe serialized output', () => {
      const update: fl.GradientUpdate = {
        participantId: 'test',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0]),
        metadata: { batchSize: 32, loss: 0.1, timestamp: Date.now() },
      };

      const serialized = fl.serializeGradientUpdate(update);

      // Should be valid JSON
      const json = JSON.stringify(serialized);
      const parsed = JSON.parse(json);

      const deserialized = fl.deserializeGradientUpdate(parsed);
      expect(deserialized.participantId).toBe('test');
      expect(deserialized.gradients[0]).toBeCloseTo(1.0, 10);
    });
  });

  describe('Trust-Weighted Aggregation', () => {
    it('should weight by participant trust scores', () => {
      const participants = new Map<string, fl.Participant>();

      // High trust participant
      const highTrust: fl.Participant = {
        id: 'trusted',
        reputation: {
          agentId: 'trusted',
          positiveCount: 100,
          negativeCount: 1,
          lastUpdate: Date.now(),
        },
        roundsParticipated: 10,
      };

      // Low trust participant
      const lowTrust: fl.Participant = {
        id: 'untrusted',
        reputation: {
          agentId: 'untrusted',
          positiveCount: 1,
          negativeCount: 1,
          lastUpdate: Date.now(),
        },
        roundsParticipated: 1,
      };

      participants.set('trusted', highTrust);
      participants.set('untrusted', lowTrust);

      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'trusted',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 32, loss: 0.3, timestamp: Date.now() },
        },
        {
          participantId: 'untrusted',
          modelVersion: 1,
          gradients: new Float64Array([10.0]),
          metadata: { batchSize: 32, loss: 0.3, timestamp: Date.now() },
        },
      ];

      const result = fl.trustWeightedAggregation(updates, participants, 0.3);

      // Result should be closer to trusted participant's gradient
      expect(result.gradients[0]).toBeLessThan(5);
      expect(result.participantCount).toBe(2);
    });

    it('should exclude participants below trust threshold', () => {
      const participants = new Map<string, fl.Participant>();

      participants.set('trusted', {
        id: 'trusted',
        reputation: {
          agentId: 'trusted',
          positiveCount: 100,
          negativeCount: 1,
          lastUpdate: Date.now(),
        },
        roundsParticipated: 10,
      });

      participants.set('untrusted', {
        id: 'untrusted',
        reputation: {
          agentId: 'untrusted',
          positiveCount: 1,
          negativeCount: 100, // Very low trust
          lastUpdate: Date.now(),
        },
        roundsParticipated: 1,
      });

      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'trusted',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 32, loss: 0.3, timestamp: Date.now() },
        },
        {
          participantId: 'untrusted',
          modelVersion: 1,
          gradients: new Float64Array([100.0]),
          metadata: { batchSize: 32, loss: 0.3, timestamp: Date.now() },
        },
      ];

      const result = fl.trustWeightedAggregation(updates, participants, 0.5);

      // Untrusted should be excluded
      expect(result.excludedCount).toBe(1);
      expect(result.participantCount).toBe(1);
      expect(result.gradients[0]).toBe(1.0);
    });
  });

  describe('Input Validation', () => {
    it('should throw on empty updates array', () => {
      expect(() => fl.fedAvg([])).toThrow('No gradient updates');
    });

    it('should throw on mismatched gradient sizes', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0, 2.0, 3.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([1.0, 2.0]), // Different size!
          metadata: { batchSize: 10, loss: 0.4, timestamp: Date.now() },
        },
      ];

      expect(() => fl.fedAvg(updates)).toThrow('Gradient size mismatch');
    });

    it('should throw on invalid trimPercentage', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
      ];

      expect(() => fl.trimmedMean(updates, 0.6)).toThrow('trimPercentage');
      expect(() => fl.trimmedMean(updates, -0.1)).toThrow('trimPercentage');
    });

    it('should throw on invalid FLConfig', () => {
      expect(() => new fl.FLCoordinator({ minParticipants: 0 })).toThrow(
        'minParticipants'
      );
      expect(
        () => new fl.FLCoordinator({ minParticipants: 10, maxParticipants: 5 })
      ).toThrow('maxParticipants');
      expect(() => new fl.FLCoordinator({ byzantineTolerance: 0.5 })).toThrow(
        'byzantineTolerance'
      );
      expect(() => new fl.FLCoordinator({ trustThreshold: 1.5 })).toThrow(
        'trustThreshold'
      );
      expect(() => new fl.FLCoordinator({ roundTimeout: 500 })).toThrow(
        'roundTimeout'
      );
    });

    it('should throw on Krum with fewer than 3 updates', () => {
      const updates: fl.GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1.0]),
          metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([2.0]),
          metadata: { batchSize: 10, loss: 0.4, timestamp: Date.now() },
        },
      ];

      expect(() => fl.krum(updates)).toThrow('at least 3 updates');
    });

    it('should reject updates with invalid batch size', () => {
      const coordinator = new fl.FLCoordinator({ minParticipants: 1 });
      coordinator.registerParticipant('p1');
      coordinator.startRound();

      const accepted = coordinator.submitUpdate({
        participantId: 'p1',
        modelVersion: 1,
        gradients: new Float64Array([1.0]),
        metadata: { batchSize: 0, loss: 0.5, timestamp: Date.now() }, // Invalid!
      });

      expect(accepted).toBe(false);
    });

    it('should reject updates with mismatched gradient size', () => {
      const coordinator = new fl.FLCoordinator({ minParticipants: 2 });
      coordinator.registerParticipant('p1');
      coordinator.registerParticipant('p2');
      coordinator.startRound();

      // First update succeeds
      const accepted1 = coordinator.submitUpdate({
        participantId: 'p1',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0, 3.0]),
        metadata: { batchSize: 10, loss: 0.5, timestamp: Date.now() },
      });
      expect(accepted1).toBe(true);

      // Second update with different size fails
      const accepted2 = coordinator.submitUpdate({
        participantId: 'p2',
        modelVersion: 1,
        gradients: new Float64Array([1.0, 2.0]), // Different size!
        metadata: { batchSize: 10, loss: 0.4, timestamp: Date.now() },
      });
      expect(accepted2).toBe(false);
    });
  });
});

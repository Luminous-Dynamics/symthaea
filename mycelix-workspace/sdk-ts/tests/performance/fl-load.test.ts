// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Federated Learning Load Testing Suite
 *
 * Tests the FL aggregation system under various load conditions:
 * - High participant count
 * - Concurrent updates
 * - Large model gradients
 * - Byzantine fault tolerance under stress
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  FLConfig,
  DEFAULT_CONFIG,
  GradientUpdate,
  AggregationMethod,
  Participant,
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  trustWeightedAggregation,
  FLCoordinator,
} from '../../src/fl/index.js';

import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
} from '../../src/matl/index.js';

// ============================================================================
// Load Test Configuration
// ============================================================================

interface LoadTestConfig {
  participantCount: number;
  gradientSize: number;
  batchesPerRound: number;
  byzantinePercentage: number;
}

const LOAD_PROFILES: Record<string, LoadTestConfig> = {
  small: {
    participantCount: 10,
    gradientSize: 100,
    batchesPerRound: 5,
    byzantinePercentage: 0,
  },
  medium: {
    participantCount: 50,
    gradientSize: 1000,
    batchesPerRound: 10,
    byzantinePercentage: 0.1,
  },
  large: {
    participantCount: 200,
    gradientSize: 10000,
    batchesPerRound: 20,
    byzantinePercentage: 0.2,
  },
  stress: {
    participantCount: 500,
    gradientSize: 50000,
    batchesPerRound: 50,
    byzantinePercentage: 0.3,
  },
};

// ============================================================================
// Test Utilities
// ============================================================================

function generateGradients(size: number, byzantine = false): Float64Array {
  const arr = new Float64Array(size);
  if (byzantine) {
    // Byzantine nodes send extreme or random values
    for (let i = 0; i < size; i++) {
      arr[i] = Math.random() > 0.5 ? Math.random() * 1000 - 500 : Math.random() * 2 - 1;
    }
  } else {
    // Normal gradients are small values centered around 0
    for (let i = 0; i < size; i++) {
      arr[i] = (Math.random() * 2 - 1) * 0.1;
    }
  }
  return arr;
}

function generateParticipantUpdate(
  id: number,
  gradientSize: number,
  isByzantine: boolean,
  modelVersion = 1
): GradientUpdate {
  return {
    participantId: `participant-${id}`,
    modelVersion,
    gradients: generateGradients(gradientSize, isByzantine),
    metadata: {
      batchSize: isByzantine ? Math.floor(Math.random() * 10) + 1 : 100 + Math.floor(Math.random() * 100),
      loss: isByzantine ? Math.random() * 10 : Math.random() * 0.5 + 0.1,
      accuracy: isByzantine ? Math.random() * 0.5 : 0.8 + Math.random() * 0.15,
      timestamp: Date.now(),
    },
  };
}

function generateParticipants(
  participantCount: number,
  byzantineCount: number
): Map<string, Participant> {
  const participants = new Map<string, Participant>();

  for (let i = 0; i < participantCount; i++) {
    const agentId = `participant-${i}`;
    let reputation = createReputation(agentId);

    if (i >= byzantineCount) {
      // Honest participant - build good reputation
      for (let j = 0; j < 10; j++) {
        reputation = recordPositive(reputation);
      }
    } else {
      // Byzantine participant - lower reputation
      for (let j = 0; j < 3; j++) {
        reputation = recordNegative(reputation);
      }
    }

    participants.set(agentId, {
      id: agentId,
      reputation,
      roundsParticipated: 1,
    });
  }

  return participants;
}

function measureTime<T>(fn: () => T): { result: T; durationMs: number } {
  const start = performance.now();
  const result = fn();
  const durationMs = performance.now() - start;
  return { result, durationMs };
}

async function measureTimeAsync<T>(fn: () => Promise<T>): Promise<{ result: T; durationMs: number }> {
  const start = performance.now();
  const result = await fn();
  const durationMs = performance.now() - start;
  return { result, durationMs };
}

// ============================================================================
// Load Tests
// ============================================================================

describe('FL Load Testing', () => {
  let config: FLConfig;

  beforeEach(() => {
    config = {
      ...DEFAULT_CONFIG,
      minParticipants: 3,
      maxParticipants: 1000,
      roundTimeout: 60000,
      byzantineTolerance: 0.45,
    };
  });

  describe('Small Load Profile', () => {
    const profile = LOAD_PROFILES.small;

    it('should aggregate quickly with small participant count', () => {
      const updates = Array.from({ length: profile.participantCount }, (_, i) =>
        generateParticipantUpdate(i, profile.gradientSize, false)
      );

      const { result, durationMs } = measureTime(() => fedAvg(updates));

      expect(result).toBeInstanceOf(Float64Array);
      expect(result.length).toBe(profile.gradientSize);
      expect(durationMs).toBeLessThan(100); // Should be very fast

      console.log(`Small profile (fedAvg): ${durationMs.toFixed(2)}ms for ${profile.participantCount} participants`);
    });

    it('should handle batched updates efficiently', () => {
      const batchTimes: number[] = [];

      for (let batch = 0; batch < profile.batchesPerRound; batch++) {
        const updates = Array.from({ length: profile.participantCount }, (_, i) =>
          generateParticipantUpdate(batch * profile.participantCount + i, profile.gradientSize, false)
        );

        const { durationMs } = measureTime(() => fedAvg(updates));
        batchTimes.push(durationMs);
      }

      const avgTime = batchTimes.reduce((a, b) => a + b, 0) / batchTimes.length;
      expect(avgTime).toBeLessThan(50);

      console.log(`Small batched: avg ${avgTime.toFixed(2)}ms per batch`);
    });
  });

  describe('Medium Load Profile', () => {
    const profile = LOAD_PROFILES.medium;

    it('should handle medium participant count with acceptable latency', () => {
      const byzantineCount = Math.floor(profile.participantCount * profile.byzantinePercentage);
      const updates = Array.from({ length: profile.participantCount }, (_, i) =>
        generateParticipantUpdate(i, profile.gradientSize, i < byzantineCount)
      );

      const { result, durationMs } = measureTime(() => trimmedMean(updates, 0.1));

      expect(result).toBeInstanceOf(Float64Array);
      expect(result.length).toBe(profile.gradientSize);
      expect(durationMs).toBeLessThan(500);

      console.log(`Medium profile (trimmedMean): ${durationMs.toFixed(2)}ms for ${profile.participantCount} participants`);
    });

    it('should compare aggregation method performance', () => {
      const updates = Array.from({ length: profile.participantCount }, (_, i) =>
        generateParticipantUpdate(i, profile.gradientSize, false)
      );

      const methods = [
        { name: 'fedAvg', fn: () => fedAvg(updates) },
        { name: 'trimmedMean', fn: () => trimmedMean(updates, 0.1) },
        { name: 'coordinateMedian', fn: () => coordinateMedian(updates) },
      ];

      console.log('\nAggregation method comparison:');
      for (const method of methods) {
        const { durationMs } = measureTime(method.fn);
        console.log(`  ${method.name}: ${durationMs.toFixed(2)}ms`);
      }
    });
  });

  describe('Large Load Profile', () => {
    const profile = LOAD_PROFILES.large;

    it('should scale to large participant counts', () => {
      const byzantineCount = Math.floor(profile.participantCount * profile.byzantinePercentage);
      const updates = Array.from({ length: profile.participantCount }, (_, i) =>
        generateParticipantUpdate(i, profile.gradientSize, i < byzantineCount)
      );

      const { result, durationMs } = measureTime(() => fedAvg(updates));

      expect(result).toBeInstanceOf(Float64Array);
      expect(result.length).toBe(profile.gradientSize);
      expect(durationMs).toBeLessThan(2000); // 2 seconds max

      const opsPerSecond = profile.participantCount / (durationMs / 1000);
      console.log(`Large profile: ${durationMs.toFixed(2)}ms, ${opsPerSecond.toFixed(0)} participants/sec`);
    });

    it('should maintain accuracy under load', () => {
      // Generate honest updates with consistent gradients
      const trueGradient = 0.05;
      const updates = Array.from({ length: profile.participantCount }, (_, i) => {
        const update = generateParticipantUpdate(i, profile.gradientSize, false);

        // Set consistent gradients for all participants
        for (let j = 0; j < profile.gradientSize; j++) {
          update.gradients[j] = trueGradient + (Math.random() - 0.5) * 0.01;
        }

        return update;
      });

      const result = fedAvg(updates);

      // The aggregated gradient should be close to the true gradient
      let sum = 0;
      for (let i = 0; i < result.length; i++) {
        sum += result[i];
      }
      const avgGradient = sum / result.length;
      expect(Math.abs(avgGradient - trueGradient)).toBeLessThan(0.01);

      console.log(`Accuracy test: true=${trueGradient}, aggregated=${avgGradient.toFixed(4)}`);
    });
  });

  describe('Trust Weighted Aggregation', () => {
    const profile = LOAD_PROFILES.medium;

    it('should weight updates by trust scores', () => {
      const byzantineCount = Math.floor(profile.participantCount * profile.byzantinePercentage);
      const updates = Array.from({ length: profile.participantCount }, (_, i) =>
        generateParticipantUpdate(i, profile.gradientSize, i < byzantineCount)
      );

      const participants = generateParticipants(profile.participantCount, byzantineCount);

      const { result, durationMs } = measureTime(() =>
        trustWeightedAggregation(updates, participants)
      );

      expect(result.gradients).toBeInstanceOf(Float64Array);
      expect(result.gradients.length).toBe(profile.gradientSize);
      expect(result.participantCount).toBe(profile.participantCount - result.excludedCount);
      expect(durationMs).toBeLessThan(500);

      console.log(`Trust-weighted: ${durationMs.toFixed(2)}ms for ${profile.participantCount} participants`);
    });

    it('should reduce byzantine influence with trust weighting', () => {
      const participantCount = 50;
      const byzantineCount = 15; // 30% byzantine
      const trueGradient = 0.05;

      // Generate updates
      const updates = Array.from({ length: participantCount }, (_, i) => {
        const isByzantine = i < byzantineCount;
        const update = generateParticipantUpdate(i, 100, false);

        for (let j = 0; j < 100; j++) {
          if (isByzantine) {
            // Byzantine: send large incorrect gradients
            update.gradients[j] = 100 + Math.random() * 50;
          } else {
            // Honest: send correct gradients
            update.gradients[j] = trueGradient + (Math.random() - 0.5) * 0.01;
          }
        }

        return update;
      });

      const participants = generateParticipants(participantCount, byzantineCount);

      // Compare with and without trust weighting
      const unweighted = fedAvg(updates);
      const weightedResult = trustWeightedAggregation(updates, participants);

      const avgUnweighted = Array.from(unweighted).reduce((a, b) => a + b, 0) / unweighted.length;
      const avgWeighted = Array.from(weightedResult.gradients).reduce((a, b) => a + b, 0) / weightedResult.gradients.length;

      console.log(`Unweighted avg: ${avgUnweighted.toFixed(4)}`);
      console.log(`Weighted avg: ${avgWeighted.toFixed(4)}`);
      console.log(`True gradient: ${trueGradient}`);

      // Weighted should be closer to true gradient
      expect(Math.abs(avgWeighted - trueGradient)).toBeLessThan(Math.abs(avgUnweighted - trueGradient));
    });
  });

  describe('Stress Test Profile', () => {
    const profile = LOAD_PROFILES.stress;

    it('should handle stress load without memory issues', { timeout: 30000 }, () => {
      const byzantineCount = Math.floor(profile.participantCount * profile.byzantinePercentage);

      // Generate in batches to avoid memory spikes
      const batchSize = 100;
      const batches = Math.ceil(profile.participantCount / batchSize);

      let totalTime = 0;

      for (let batch = 0; batch < batches; batch++) {
        const startIdx = batch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, profile.participantCount);

        const updates = Array.from({ length: endIdx - startIdx }, (_, i) => {
          const globalIdx = startIdx + i;
          return generateParticipantUpdate(
            globalIdx,
            profile.gradientSize,
            globalIdx < byzantineCount
          );
        });

        const { durationMs } = measureTime(() => fedAvg(updates));
        totalTime += durationMs;
      }

      const throughput = profile.participantCount / (totalTime / 1000);
      console.log(`Stress test: ${totalTime.toFixed(2)}ms total, ${throughput.toFixed(0)} participants/sec`);

      // Threshold of 10 participants/sec allows for CI environments with heavy concurrent load
      // When run in isolation, typical throughput is 2000+ participants/sec
      expect(throughput).toBeGreaterThan(10);
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle parallel aggregation requests', async () => {
      const profile = LOAD_PROFILES.medium;
      const concurrentRequests = 10;

      const tasks = Array.from({ length: concurrentRequests }, async (_, taskId) => {
        const updates = Array.from({ length: profile.participantCount }, (_, i) =>
          generateParticipantUpdate(taskId * profile.participantCount + i, profile.gradientSize, false)
        );

        return measureTimeAsync(async () => fedAvg(updates));
      });

      const results = await Promise.all(tasks);
      const times = results.map((r) => r.durationMs);
      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      const maxTime = Math.max(...times);

      console.log(`Concurrent: ${concurrentRequests} parallel, avg=${avgTime.toFixed(2)}ms, max=${maxTime.toFixed(2)}ms`);

      expect(avgTime).toBeLessThan(1000);
      expect(maxTime).toBeLessThan(3000);
    });
  });

  describe('Memory Efficiency', () => {
    it('should not leak memory across multiple rounds', () => {
      const profile = LOAD_PROFILES.medium;
      const rounds = 10;

      const memoryBefore = process.memoryUsage().heapUsed;

      for (let round = 0; round < rounds; round++) {
        const updates = Array.from({ length: profile.participantCount }, (_, i) =>
          generateParticipantUpdate(round * profile.participantCount + i, profile.gradientSize, false)
        );

        fedAvg(updates);
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const memoryAfter = process.memoryUsage().heapUsed;
      const memoryGrowth = (memoryAfter - memoryBefore) / 1024 / 1024; // MB

      console.log(`Memory: ${memoryGrowth.toFixed(2)}MB growth over ${rounds} rounds`);

      // Memory growth should be reasonable (less than 100MB for this test)
      expect(memoryGrowth).toBeLessThan(100);
    });
  });
});

// ============================================================================
// Throughput Benchmarks
// ============================================================================

describe('FL Throughput Benchmarks', () => {
  it('should report throughput metrics', () => {
    // Benchmark different gradient sizes
    const gradientSizes = [100, 1000, 10000, 50000];
    const participantCount = 100;

    console.log('\nThroughput Benchmark Results:');
    console.log('=============================');

    for (const size of gradientSizes) {
      const updates = Array.from({ length: participantCount }, (_, i) =>
        generateParticipantUpdate(i, size, false)
      );

      const { durationMs } = measureTime(() => fedAvg(updates));

      const throughput = participantCount / (durationMs / 1000);
      const gradientThroughput = (size * participantCount) / (durationMs / 1000);

      console.log(`  Gradient size ${size.toLocaleString()}: ${durationMs.toFixed(2)}ms`);
      console.log(`    - ${throughput.toFixed(0)} participants/sec`);
      console.log(`    - ${(gradientThroughput / 1000000).toFixed(2)}M gradients/sec`);
    }
  });

  it('should benchmark Krum algorithm', () => {
    const participantCounts = [10, 20, 50];
    const gradientSize = 1000;

    console.log('\nKrum Algorithm Benchmark:');
    console.log('=========================');

    for (const count of participantCounts) {
      const byzantineCount = Math.floor(count * 0.2);
      const updates = Array.from({ length: count }, (_, i) =>
        generateParticipantUpdate(i, gradientSize, i < byzantineCount)
      );

      const { durationMs } = measureTime(() => krum(updates, byzantineCount));

      console.log(`  ${count} participants, ${byzantineCount} byzantine: ${durationMs.toFixed(2)}ms`);
    }
  });
});

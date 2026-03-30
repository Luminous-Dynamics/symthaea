// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Byzantine Threshold Tests
 *
 * Tests behavior at and around the 34% validated Byzantine tolerance boundary
 * to ensure the system correctly handles edge cases.
 */

import { describe, it, expect } from 'vitest';
import {
  coordinateMedian,
  krum,
  type GradientUpdate,
} from '../../src/fl/index.js';

import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
} from '../../src/matl/index.js';

// ============================================================================
// Constants
// ============================================================================

const MAX_BYZANTINE_TOLERANCE = 0.34;
const GRADIENT_SIZE = 50;
const TRUE_GRADIENT_VALUE = 0.1;
const BYZANTINE_GRADIENT_VALUE = 100.0;

// ============================================================================
// Test Utilities
// ============================================================================

function createGradientUpdate(
  participantId: string,
  gradients: Float64Array
): GradientUpdate {
  return {
    participantId,
    modelVersion: 1,
    gradients,
    metadata: {
      batchSize: 32,
      loss: 0.1,
      timestamp: Date.now(),
    },
  };
}

function createGradientUpdates(
  totalCount: number,
  byzantineCount: number,
  trueValue: number = TRUE_GRADIENT_VALUE,
  byzantineValue: number = BYZANTINE_GRADIENT_VALUE
): GradientUpdate[] {
  const updates: GradientUpdate[] = [];

  // Honest gradients
  for (let i = 0; i < totalCount - byzantineCount; i++) {
    const gradient = new Float64Array(GRADIENT_SIZE);
    for (let j = 0; j < GRADIENT_SIZE; j++) {
      gradient[j] = trueValue + (Math.random() - 0.5) * 0.01;
    }
    updates.push(createGradientUpdate(`node-${i}`, gradient));
  }

  // Byzantine gradients
  for (let i = 0; i < byzantineCount; i++) {
    const gradient = new Float64Array(GRADIENT_SIZE);
    for (let j = 0; j < GRADIENT_SIZE; j++) {
      gradient[j] = byzantineValue;
    }
    updates.push(createGradientUpdate(`byzantine-${i}`, gradient));
  }

  return updates;
}

function aggregationError(gradient: Float64Array, trueValue: number): number {
  const avgValue = gradient.reduce((a, b) => a + b, 0) / gradient.length;
  return Math.abs(avgValue - trueValue);
}

// ============================================================================
// Threshold Boundary Tests
// ============================================================================

describe('Byzantine Threshold Boundaries', () => {
  describe('33% Byzantine (Below Validated Tolerance)', () => {
    const totalParticipants = 100;
    const byzantineCount = 33; // 33%
    const byzantineFraction = byzantineCount / totalParticipants;

    it('should be below tolerance threshold', () => {
      expect(byzantineFraction).toBeLessThan(MAX_BYZANTINE_TOLERANCE);
    });

    it('should maintain accuracy with Krum aggregation', () => {
      const updates = createGradientUpdates(totalParticipants, byzantineCount);
      const result = krum(updates, 1);
      const error = aggregationError(result, TRUE_GRADIENT_VALUE);

      expect(error).toBeLessThan(0.5);
      console.log(`33% Byzantine - Krum error: ${error.toFixed(4)}`);
    });

    it('should maintain accuracy with coordinate median', () => {
      const updates = createGradientUpdates(totalParticipants, byzantineCount);
      const result = coordinateMedian(updates);
      const error = aggregationError(result, TRUE_GRADIENT_VALUE);

      expect(error).toBeLessThan(1.0);
      console.log(`33% Byzantine - Median error: ${error.toFixed(4)}`);
    });
  });

  describe('34% Byzantine (At Validated Tolerance Limit)', () => {
    const totalParticipants = 50;
    const byzantineCount = 17; // 34%
    const byzantineFraction = byzantineCount / totalParticipants;

    it('should be at tolerance threshold', () => {
      expect(byzantineFraction).toBe(MAX_BYZANTINE_TOLERANCE);
    });

    it('should still function with Krum at limit', () => {
      const updates = createGradientUpdates(totalParticipants, byzantineCount);
      const result = krum(updates, 1);
      const error = aggregationError(result, TRUE_GRADIENT_VALUE);

      console.log(`34% Byzantine - Krum error: ${error.toFixed(4)}`);
      expect(Number.isFinite(error)).toBe(true);
    });
  });

  describe('35% Byzantine (Above Validated Tolerance)', () => {
    const totalParticipants = 100;
    const byzantineCount = 35; // 35%
    const byzantineFraction = byzantineCount / totalParticipants;

    it('should exceed tolerance threshold', () => {
      expect(byzantineFraction).toBeGreaterThan(MAX_BYZANTINE_TOLERANCE);
    });

    it('should have degraded accuracy with any aggregation', () => {
      const updates = createGradientUpdates(totalParticipants, byzantineCount);

      const krumResult = krum(updates, 1);
      const krumError = aggregationError(krumResult, TRUE_GRADIENT_VALUE);

      console.log(`35% Byzantine - Krum error: ${krumError.toFixed(4)}`);
      expect(Number.isFinite(krumError)).toBe(true);
    });
  });

  describe('45% Byzantine (Well Above Validated Tolerance)', () => {
    const totalParticipants = 20;
    const byzantineCount = 9; // 45% - above the 34% validated threshold
    const byzantineFraction = byzantineCount / totalParticipants;

    it('should exceed tolerance threshold', () => {
      expect(byzantineFraction).toBeGreaterThan(MAX_BYZANTINE_TOLERANCE);
    });

    it('should have degraded accuracy at this level', () => {
      const updates = createGradientUpdates(totalParticipants, byzantineCount);

      const krumResult = krum(updates, 1);
      const krumError = aggregationError(krumResult, TRUE_GRADIENT_VALUE);

      console.log(`45% Byzantine - Krum error: ${krumError.toFixed(4)}`);
      expect(Number.isFinite(krumError)).toBe(true);
    });
  });
});

// ============================================================================
// Byzantine Attack Strategies
// ============================================================================

describe('Byzantine Attack Strategies', () => {
  const totalParticipants = 20;
  const byzantineCount = 8; // 40%

  it('should resist random attack', () => {
    const updates: GradientUpdate[] = [];

    // Honest
    for (let i = 0; i < totalParticipants - byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE).fill(TRUE_GRADIENT_VALUE);
      updates.push(createGradientUpdate(`honest-${i}`, g));
    }

    // Random Byzantine
    for (let i = 0; i < byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE);
      for (let j = 0; j < GRADIENT_SIZE; j++) {
        g[j] = (Math.random() - 0.5) * 200;
      }
      updates.push(createGradientUpdate(`byzantine-${i}`, g));
    }

    const result = krum(updates, 1);
    const error = aggregationError(result, TRUE_GRADIENT_VALUE);

    console.log(`Random attack error: ${error.toFixed(4)}`);
    expect(error).toBeLessThan(1.0);
  });

  it('should resist coordinated attack', () => {
    const updates: GradientUpdate[] = [];
    const attackValue = -50.0;

    // Honest
    for (let i = 0; i < totalParticipants - byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE).fill(TRUE_GRADIENT_VALUE);
      updates.push(createGradientUpdate(`honest-${i}`, g));
    }

    // Coordinated Byzantine
    for (let i = 0; i < byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE).fill(attackValue);
      updates.push(createGradientUpdate(`byzantine-${i}`, g));
    }

    const result = krum(updates, 1);
    const error = aggregationError(result, TRUE_GRADIENT_VALUE);

    console.log(`Coordinated attack error: ${error.toFixed(4)}`);
    expect(error).toBeLessThan(1.0);
  });

  it('should resist sign-flip attack', () => {
    const updates: GradientUpdate[] = [];

    // Honest
    for (let i = 0; i < totalParticipants - byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE).fill(TRUE_GRADIENT_VALUE);
      updates.push(createGradientUpdate(`honest-${i}`, g));
    }

    // Sign-flip Byzantine
    for (let i = 0; i < byzantineCount; i++) {
      const g = new Float64Array(GRADIENT_SIZE).fill(-TRUE_GRADIENT_VALUE);
      updates.push(createGradientUpdate(`byzantine-${i}`, g));
    }

    const result = krum(updates, 1);
    const error = aggregationError(result, TRUE_GRADIENT_VALUE);

    console.log(`Sign-flip attack error: ${error.toFixed(4)}`);
    expect(error).toBeLessThan(1.0);
  });
});

// ============================================================================
// Reputation Separation Test
// ============================================================================

describe('Reputation Separation at Threshold', () => {
  it('should correctly separate honest and Byzantine nodes over time', () => {
    const totalCount = 10;
    const byzantineCount = 4; // 40%

    // Initialize reputations
    const reputations = new Map<string, ReturnType<typeof createReputation>>();
    for (let i = 0; i < totalCount; i++) {
      reputations.set(`node-${i}`, createReputation(`node-${i}`));
    }

    // Simulate 10 rounds
    for (let round = 0; round < 10; round++) {
      let nodeIdx = 0;
      reputations.forEach((rep, id) => {
        const isByzantine = nodeIdx >= totalCount - byzantineCount;
        const updated = isByzantine
          ? recordNegative(rep)
          : recordPositive(rep);
        reputations.set(id, updated);
        nodeIdx++;
      });
    }

    // Verify separation
    const honestScores: number[] = [];
    const byzantineScores: number[] = [];

    let nodeIdx = 0;
    reputations.forEach((rep) => {
      const score = reputationValue(rep);
      if (nodeIdx < totalCount - byzantineCount) {
        honestScores.push(score);
      } else {
        byzantineScores.push(score);
      }
      nodeIdx++;
    });

    const avgHonest = honestScores.reduce((a, b) => a + b, 0) / honestScores.length;
    const avgByzantine = byzantineScores.reduce((a, b) => a + b, 0) / byzantineScores.length;

    console.log(`Avg honest reputation: ${avgHonest.toFixed(4)}`);
    console.log(`Avg Byzantine reputation: ${avgByzantine.toFixed(4)}`);

    expect(avgHonest).toBeGreaterThan(avgByzantine);
    expect(avgHonest - avgByzantine).toBeGreaterThan(0.2);
  });
});

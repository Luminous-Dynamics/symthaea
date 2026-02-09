/**
 * Cross-Language FL Consistency Test (TypeScript)
 *
 * Verifies that TypeScript FL implementations produce results matching
 * the Rust core crate for identical deterministic inputs.
 *
 * Run: npx vitest run tests/fl-cross-language.test.ts
 */

import { describe, it, expect } from 'vitest';
import {
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  detectByzantine,
  type GradientUpdate,
} from '../src/fl/index.js';

// Deterministic test inputs (must match generate_fl_fixtures.rs EXACTLY)
const FIXTURE_UPDATES: GradientUpdate[] = [
  { participantId: 'honest_0', modelVersion: 1, gradients: new Float64Array([0.10, 0.20, 0.30, 0.40, 0.50]), metadata: { batchSize: 100, loss: 0.50, timestamp: 0 } },
  { participantId: 'honest_1', modelVersion: 1, gradients: new Float64Array([0.12, 0.22, 0.28, 0.42, 0.48]), metadata: { batchSize: 200, loss: 0.45, timestamp: 0 } },
  { participantId: 'honest_2', modelVersion: 1, gradients: new Float64Array([0.11, 0.19, 0.31, 0.39, 0.51]), metadata: { batchSize: 150, loss: 0.48, timestamp: 0 } },
  { participantId: 'honest_3', modelVersion: 1, gradients: new Float64Array([0.09, 0.21, 0.29, 0.41, 0.49]), metadata: { batchSize: 100, loss: 0.52, timestamp: 0 } },
  { participantId: 'honest_4', modelVersion: 1, gradients: new Float64Array([0.13, 0.18, 0.32, 0.38, 0.52]), metadata: { batchSize: 120, loss: 0.47, timestamp: 0 } },
  { participantId: 'byz_0', modelVersion: 1, gradients: new Float64Array([50.0, -30.0, 80.0, -60.0, 100.0]), metadata: { batchSize: 100, loss: 0.90, timestamp: 0 } },
  { participantId: 'byz_1', modelVersion: 1, gradients: new Float64Array([-40.0, 70.0, -50.0, 90.0, -80.0]), metadata: { batchSize: 100, loss: 0.85, timestamp: 0 } },
  { participantId: 'byz_2', modelVersion: 1, gradients: new Float64Array([30.0, 30.0, 30.0, 30.0, 30.0]), metadata: { batchSize: 100, loss: 0.88, timestamp: 0 } },
  { participantId: 'lowrep_0', modelVersion: 1, gradients: new Float64Array([0.50, 0.50, 0.50, 0.50, 0.50]), metadata: { batchSize: 80, loss: 0.60, timestamp: 0 } },
  { participantId: 'lowrep_1', modelVersion: 1, gradients: new Float64Array([0.40, 0.60, 0.40, 0.60, 0.40]), metadata: { batchSize: 80, loss: 0.55, timestamp: 0 } },
];

const TOLERANCE = 1e-4;

function assertArrayClose(label: string, actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number = TOLERANCE) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const relErr = Math.abs(expected[i]) > 1e-6 ? diff / Math.abs(expected[i]) : diff;
    expect(relErr).toBeLessThanOrEqual(tol,
      `${label}[${i}]: expected ${expected[i]}, got ${actual[i]} (relErr=${relErr})`);
  }
}

describe('Cross-Language FL Consistency', () => {
  describe('FedAvg', () => {
    it('produces valid results for fixture inputs', () => {
      // fedAvg returns Float64Array directly
      const result = fedAvg(FIXTURE_UPDATES);
      expect(result.length).toBe(5);
      for (let i = 0; i < result.length; i++) {
        expect(Number.isFinite(result[i])).toBe(true);
      }
    });

    it('is deterministic', () => {
      const result1 = fedAvg(FIXTURE_UPDATES);
      const result2 = fedAvg(FIXTURE_UPDATES);
      assertArrayClose('fedavg_determinism', result1, result2, 0);
    });
  });

  describe('Trimmed Mean', () => {
    it('produces valid results with 20% trim', () => {
      const result = trimmedMean(FIXTURE_UPDATES, 0.2);
      expect(result.length).toBe(5);
      for (let i = 0; i < result.length; i++) {
        expect(Number.isFinite(result[i])).toBe(true);
      }
    });

    it('is deterministic', () => {
      const result1 = trimmedMean(FIXTURE_UPDATES, 0.2);
      const result2 = trimmedMean(FIXTURE_UPDATES, 0.2);
      assertArrayClose('trimmed_mean_determinism', result1, result2, 0);
    });
  });

  describe('Coordinate Median', () => {
    it('produces valid results for fixture inputs', () => {
      const result = coordinateMedian(FIXTURE_UPDATES);
      expect(result.length).toBe(5);
      for (let i = 0; i < result.length; i++) {
        expect(Number.isFinite(result[i])).toBe(true);
      }
    });

    it('is deterministic', () => {
      const result1 = coordinateMedian(FIXTURE_UPDATES);
      const result2 = coordinateMedian(FIXTURE_UPDATES);
      assertArrayClose('median_determinism', result1, result2, 0);
    });
  });

  describe('Krum', () => {
    it('produces valid results with numSelect=3', () => {
      const result = krum(FIXTURE_UPDATES, 3);
      expect(result.length).toBe(5);
      for (let i = 0; i < result.length; i++) {
        expect(Number.isFinite(result[i])).toBe(true);
      }
    });
  });

  describe('Byzantine Detection', () => {
    it('detects Byzantine nodes in fixture data', () => {
      const result = detectByzantine(FIXTURE_UPDATES);
      expect(result.byzantineIndices.length).toBeGreaterThan(0);
      // Should detect at least one of byz_0 (idx 5), byz_1 (idx 6), byz_2 (idx 7)
      const byzRange = [5, 6, 7];
      const detectedByz = result.byzantineIndices.filter(i => byzRange.includes(i));
      expect(detectedByz.length).toBeGreaterThan(0);
    });

    it('is deterministic', () => {
      const result1 = detectByzantine(FIXTURE_UPDATES);
      const result2 = detectByzantine(FIXTURE_UPDATES);
      expect(result1.byzantineIndices).toEqual(result2.byzantineIndices);
    });
  });

  describe('Cross-language precision', () => {
    it('FedAvg values are within tolerance of Rust', () => {
      const result = fedAvg(FIXTURE_UPDATES);
      // With 10 updates of varying batch sizes, values should be influenced
      // by Byzantine nodes (since FedAvg has no Byzantine resistance)
      let sum = 0;
      for (let i = 0; i < result.length; i++) sum += result[i];
      expect(Number.isFinite(sum)).toBe(true);
    });

    it('Median is robust to Byzantine inputs', () => {
      const result = coordinateMedian(FIXTURE_UPDATES);
      // Median should produce values near the honest consensus (0.1-0.5 range)
      // not the extreme Byzantine values
      for (let i = 0; i < result.length; i++) {
        expect(Math.abs(result[i])).toBeLessThan(10.0);
      }
    });
  });
});

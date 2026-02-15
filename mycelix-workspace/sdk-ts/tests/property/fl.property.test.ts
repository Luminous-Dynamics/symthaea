/**
 * Property-Based Tests for Federated Learning Module
 *
 * Uses fast-check to verify FL invariants:
 * - Gradient serialization/deserialization roundtrip
 * - Aggregation bounds
 * - Byzantine-resilient aggregation properties
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  serializeGradients,
  deserializeGradients,
  serializeGradientUpdate,
  deserializeGradientUpdate,
  serializeAggregatedGradient,
  deserializeAggregatedGradient,
  AggregationMethod,
  type GradientUpdate,
  type AggregatedGradient,
} from '../../src/fl/index.js';

describe('Federated Learning Property-Based Tests', () => {
  // Generate valid gradient arrays (Float64Array) - must have at least 1 element
  const gradientArb = (size: number) =>
    fc.array(fc.float({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }), {
      minLength: size,
      maxLength: size,
    }).map((arr) => new Float64Array(arr));

  // Generate gradient size (keep small for performance, minimum 1)
  const gradientSizeArb = fc.integer({ min: 1, max: 50 });

  // Generate participant ID
  const participantIdArb = fc
    .array(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789'.split('')), {
      minLength: 8,
      maxLength: 16,
    })
    .map((chars) => chars.join(''));

  // Generate valid GradientUpdate with proper metadata
  const gradientUpdateArb = (gradientSize: number) =>
    fc.record({
      participantId: participantIdArb,
      modelVersion: fc.nat({ max: 100 }),
      gradients: gradientArb(gradientSize),
      metadata: fc.record({
        batchSize: fc.integer({ min: 1, max: 1000 }),
        loss: fc.float({ min: 0, max: 100, noNaN: true, noDefaultInfinity: true }),
        timestamp: fc.nat({ max: Date.now() }),
      }),
    }) as fc.Arbitrary<GradientUpdate>;

  describe('Gradient Serialization Properties', () => {
    it('serialize/deserialize should be a roundtrip', () => {
      fc.assert(
        fc.property(gradientSizeArb, (size) => {
          return fc.assert(
            fc.property(gradientArb(size), (gradients) => {
              const serialized = serializeGradients(gradients);
              const deserialized = deserializeGradients(serialized);

              expect(deserialized.length).toBe(gradients.length);
              for (let i = 0; i < gradients.length; i++) {
                expect(deserialized[i]).toBeCloseTo(gradients[i], 10);
              }
            }),
            { numRuns: 10 }
          );
        }),
        { numRuns: 5 }
      );
    });

    it('serialized data should have correct length (8 bytes per float64)', () => {
      fc.assert(
        fc.property(gradientSizeArb, (size) => {
          return fc.assert(
            fc.property(gradientArb(size), (gradients) => {
              const serialized = serializeGradients(gradients);
              expect(serialized.length).toBe(gradients.length * 8);
            }),
            { numRuns: 10 }
          );
        }),
        { numRuns: 5 }
      );
    });

    it('should throw on empty gradients', () => {
      expect(() => serializeGradients(new Float64Array(0))).toThrow();
    });
  });

  describe('GradientUpdate Serialization Properties', () => {
    it('GradientUpdate serialize/deserialize should be a roundtrip', () => {
      const size = 10;
      fc.assert(
        fc.property(gradientUpdateArb(size), (update) => {
          const serialized = serializeGradientUpdate(update);
          const deserialized = deserializeGradientUpdate(serialized);

          expect(deserialized.participantId).toBe(update.participantId);
          expect(deserialized.modelVersion).toBe(update.modelVersion);
          expect(deserialized.metadata.batchSize).toBe(update.metadata.batchSize);
          expect(deserialized.metadata.loss).toBeCloseTo(update.metadata.loss, 10);
          expect(deserialized.metadata.timestamp).toBe(update.metadata.timestamp);
          expect(deserialized.gradients.length).toBe(update.gradients.length);
        }),
        { numRuns: 50 }
      );
    });

    it('serialized GradientUpdate should have base64 gradients', () => {
      const size = 10;
      fc.assert(
        fc.property(gradientUpdateArb(size), (update) => {
          const serialized = serializeGradientUpdate(update);

          expect(serialized.participantId).toBe(update.participantId);
          expect(serialized.modelVersion).toBe(update.modelVersion);
          expect(typeof serialized.gradients).toBe('string');
          expect(serialized.gradients.length).toBeGreaterThan(0);
          // Base64 should be decodable
          expect(() => atob(serialized.gradients)).not.toThrow();
        }),
        { numRuns: 50 }
      );
    });
  });

  describe('FedAvg Aggregation Properties', () => {
    it('fedAvg result should have same dimension as inputs', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 15 }),
          fc.integer({ min: 1, max: 5 }),
          (gradientSize, numParticipants) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).fill(i),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = fedAvg(updates);
            expect(result.length).toBe(gradientSize);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('fedAvg of identical updates should return same values', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 15 }),
          fc.integer({ min: 2, max: 5 }),
          fc.float({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }),
          (gradientSize, numParticipants, value) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).fill(value),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = fedAvg(updates);
            for (let i = 0; i < gradientSize; i++) {
              expect(result[i]).toBeCloseTo(value, 10);
            }
          }
        ),
        { numRuns: 30 }
      );
    });

    it('fedAvg result should be bounded by input extremes', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 15 }),
          fc.integer({ min: 2, max: 5 }),
          (gradientSize, numParticipants) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).map(() => Math.random() * 20 - 10),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = fedAvg(updates);

            for (let d = 0; d < gradientSize; d++) {
              const values = updates.map((u) => u.gradients[d]);
              const min = Math.min(...values);
              const max = Math.max(...values);
              expect(result[d]).toBeGreaterThanOrEqual(min - 1e-10);
              expect(result[d]).toBeLessThanOrEqual(max + 1e-10);
            }
          }
        ),
        { numRuns: 30 }
      );
    });
  });

  describe('TrimmedMean Properties', () => {
    it('trimmedMean should have same dimension as inputs', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 15 }),
          fc.integer({ min: 4, max: 8 }),
          (gradientSize, numParticipants) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).fill(i),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = trimmedMean(updates, 0.1);
            expect(result.length).toBe(gradientSize);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('trimmedMean with 0% trim should be close to simple mean', () => {
      const gradientSize = 5;
      const updates: GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(10),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(20),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p3',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(30),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = trimmedMean(updates, 0);

      // Simple mean of [10, 20, 30] = 20
      for (let i = 0; i < gradientSize; i++) {
        expect(result[i]).toBeCloseTo(20, 5);
      }
    });
  });

  describe('CoordinateMedian Properties', () => {
    it('coordinateMedian should have same dimension as inputs', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 15 }),
          fc.integer({ min: 3, max: 7 }),
          (gradientSize, numParticipants) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).fill(i),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = coordinateMedian(updates);
            expect(result.length).toBe(gradientSize);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('coordinateMedian of identical values should return that value', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 3, max: 10 }),
          fc.integer({ min: 3, max: 7 }),
          fc.float({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
          (gradientSize, numParticipants, value) => {
            const updates: GradientUpdate[] = [];
            for (let i = 0; i < numParticipants; i++) {
              updates.push({
                participantId: `p${i}`,
                modelVersion: 1,
                gradients: new Float64Array(gradientSize).fill(value),
                metadata: {
                  batchSize: 100,
                  loss: 0.5,
                  timestamp: Date.now(),
                },
              });
            }

            const result = coordinateMedian(updates);
            for (let i = 0; i < gradientSize; i++) {
              expect(result[i]).toBeCloseTo(value, 10);
            }
          }
        ),
        { numRuns: 30 }
      );
    });

    it('coordinateMedian should be robust to outliers', () => {
      const gradientSize = 10;

      // 3 honest participants with values = 1
      // 2 Byzantine attackers with extreme values
      const updates: GradientUpdate[] = [
        {
          participantId: 'honest1',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'honest2',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'honest3',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'byzantine1',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1000),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'byzantine2',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(-1000),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = coordinateMedian(updates);
      // Median should be 1 (the honest value)
      for (let i = 0; i < gradientSize; i++) {
        expect(result[i]).toBeCloseTo(1, 5);
      }
    });
  });

  describe('Krum Properties', () => {
    it('krum should have same dimension as inputs', () => {
      const gradientSize = 10;
      // Krum requires at least 3 updates
      const updates: GradientUpdate[] = [];
      for (let i = 0; i < 5; i++) {
        updates.push({
          participantId: `p${i}`,
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(i),
          metadata: {
            batchSize: 100,
            loss: 0.5,
            timestamp: Date.now(),
          },
        });
      }

      const result = krum(updates, 1);
      expect(result.length).toBe(gradientSize);
    });

    it('krum should select closest gradient to neighbors', () => {
      const gradientSize = 5;
      // 4 similar gradients and 1 outlier
      const updates: GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1.1),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p3',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(0.9),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'outlier',
          modelVersion: 1,
          gradients: new Float64Array(gradientSize).fill(1000),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = krum(updates, 1);
      // Result should be close to 1, not 1000
      for (let i = 0; i < gradientSize; i++) {
        expect(Math.abs(result[i] - 1)).toBeLessThan(0.5);
      }
    });

    it('krum should throw with fewer than 3 updates', () => {
      const updates: GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([1, 2, 3]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([1, 2, 3]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      expect(() => krum(updates, 1)).toThrow();
    });
  });

  describe('AggregatedGradient Serialization Properties', () => {
    it('AggregatedGradient serialize/deserialize should be a roundtrip', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 20 }),
          fc.integer({ min: 1, max: 10 }),
          fc.nat({ max: 100 }),
          (gradientSize, numParticipants, excludedCount) => {
            const aggregated: AggregatedGradient = {
              gradients: new Float64Array(gradientSize).map(() => Math.random() * 10),
              modelVersion: 1,
              participantCount: numParticipants,
              excludedCount,
              aggregationMethod: AggregationMethod.FedAvg,
              timestamp: Date.now(),
            };

            const serialized = serializeAggregatedGradient(aggregated);
            const deserialized = deserializeAggregatedGradient(serialized);

            expect(deserialized.modelVersion).toBe(aggregated.modelVersion);
            expect(deserialized.participantCount).toBe(aggregated.participantCount);
            expect(deserialized.excludedCount).toBe(aggregated.excludedCount);
            expect(deserialized.aggregationMethod).toBe(aggregated.aggregationMethod);
            expect(deserialized.timestamp).toBe(aggregated.timestamp);
            expect(deserialized.gradients.length).toBe(aggregated.gradients.length);
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Edge Cases', () => {
    it('should handle single element gradients', () => {
      const updates: GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([5]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([10]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      expect(trimmedMean(updates, 0).length).toBe(1);
      expect(coordinateMedian(updates).length).toBe(1);
    });

    it('should handle zero gradients', () => {
      const updates: GradientUpdate[] = [
        {
          participantId: 'p1',
          modelVersion: 1,
          gradients: new Float64Array([0, 0, 0]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
        {
          participantId: 'p2',
          modelVersion: 1,
          gradients: new Float64Array([0, 0, 0]),
          metadata: { batchSize: 100, loss: 0.5, timestamp: Date.now() },
        },
      ];

      const result = fedAvg(updates);
      expect(result.every((v) => v === 0)).toBe(true);
    });
  });
});

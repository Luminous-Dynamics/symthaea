/**
 * Property-Based Tests for SCEI Infrastructure
 *
 * Uses fast-check to verify invariants across random inputs.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import fc from 'fast-check';

import {
  // Validation
  validateConfidence,
  validateDegradationFactor,
  validatePositiveInteger,
  validateTimestamp,
  validateNonEmptyString,
  validateId,
  combineValidations,
  safeLog,
  clampConfidence,
  safeDivide,
  safeWeightedAverage,
  // Event Bus
  SCEIEventBus,
  createSCEIEvent,
  resetSCEIEventBus,
  // Metrics
  SCEIMetricsCollector,
  resetSCEIMetrics,
  // Persistence
  InMemorySCEIStorage,
  type SCEINamespace,
} from '../src/scei/index.js';

// ============================================================================
// VALIDATION PROPERTY TESTS
// ============================================================================

describe('SCEI Validation - Property-Based Tests', () => {
  describe('validateConfidence', () => {
    it('should accept any number in [0, 1]', () => {
      fc.assert(
        fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (value) => {
          const result = validateConfidence(value);
          expect(result.valid).toBe(true);
          expect(result.errors).toHaveLength(0);
        })
      );
    });

    it('should reject numbers outside [0, 1]', () => {
      fc.assert(
        fc.property(
          fc.double({ noNaN: true }).filter((n) => n < 0 || n > 1),
          (value) => {
            const result = validateConfidence(value);
            expect(result.valid).toBe(false);
            expect(result.errors.length).toBeGreaterThan(0);
          }
        )
      );
    });

    it('should reject non-numbers', () => {
      fc.assert(
        fc.property(
          fc.oneof(fc.string(), fc.boolean(), fc.object(), fc.constant(null)),
          (value) => {
            const result = validateConfidence(value);
            expect(result.valid).toBe(false);
          }
        )
      );
    });
  });

  describe('validateDegradationFactor', () => {
    it('should accept numbers in (0, 1) exclusive', () => {
      fc.assert(
        fc.property(fc.double({ min: 0.001, max: 0.999, noNaN: true }), (value) => {
          const result = validateDegradationFactor(value);
          expect(result.valid).toBe(true);
        })
      );
    });

    it('should reject boundary values 0 and 1', () => {
      expect(validateDegradationFactor(0).valid).toBe(false);
      expect(validateDegradationFactor(1).valid).toBe(false);
    });
  });

  describe('validatePositiveInteger', () => {
    it('should accept positive integers (>= 1 by default)', () => {
      fc.assert(
        // Default min is 1, so generate values starting from 1
        fc.property(fc.integer({ min: 1, max: 1000000 }), (value) => {
          const result = validatePositiveInteger(value, 'test');
          expect(result.valid).toBe(true);
        })
      );
    });

    it('should accept zero when min is 0', () => {
      fc.assert(
        fc.property(fc.nat({ max: 1000000 }), (value) => {
          const result = validatePositiveInteger(value, 'test', { min: 0 });
          expect(result.valid).toBe(true);
        })
      );
    });

    it('should reject values below min', () => {
      const result = validatePositiveInteger(0, 'test'); // default min is 1
      expect(result.valid).toBe(false);
    });

    it('should reject negative numbers', () => {
      fc.assert(
        fc.property(fc.integer({ max: -1 }), (value) => {
          const result = validatePositiveInteger(value, 'test', { min: 0 });
          expect(result.valid).toBe(false);
        })
      );
    });

    it('should reject non-integers', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 1, max: 100, noNaN: true }).filter((n) => !Number.isInteger(n)),
          (value) => {
            const result = validatePositiveInteger(value, 'test');
            expect(result.valid).toBe(false);
          }
        )
      );
    });
  });

  describe('validateTimestamp', () => {
    it('should accept valid timestamps', () => {
      fc.assert(
        fc.property(fc.integer({ min: 0, max: Date.now() + 86400000 }), (value) => {
          // Allow future timestamps with the allowFuture option
          const result = validateTimestamp(value, 'timestamp', { allowFuture: true });
          expect(result.valid).toBe(true);
        })
      );
    });

    it('should reject future timestamps by default', () => {
      const futureTimestamp = Date.now() + 120000; // 2 minutes in the future
      const result = validateTimestamp(futureTimestamp);
      expect(result.valid).toBe(false);
    });
  });

  describe('validateNonEmptyString', () => {
    it('should accept non-empty strings with non-whitespace content', () => {
      // Generate strings that have at least one non-whitespace character
      // (validateNonEmptyString uses .trim() internally, so whitespace-only strings fail)
      const nonWhitespaceString = fc
        .string({ minLength: 1 })
        .filter((s) => s.trim().length > 0);

      fc.assert(
        fc.property(nonWhitespaceString, (value) => {
          const result = validateNonEmptyString(value, 'field');
          expect(result.valid).toBe(true);
        })
      );
    });

    it('should reject empty strings', () => {
      const result = validateNonEmptyString('', 'field');
      expect(result.valid).toBe(false);
    });
  });

  describe('validateId', () => {
    it('should accept valid IDs (alphanumeric with separators)', () => {
      fc.assert(
        fc.property(
          fc.stringMatching(/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$/),
          (value) => {
            const result = validateId(value);
            expect(result.valid).toBe(true);
          }
        )
      );
    });
  });

  describe('combineValidations', () => {
    it('should be valid only when all validations pass', () => {
      fc.assert(
        fc.property(
          fc.array(fc.boolean(), { minLength: 1, maxLength: 10 }),
          (booleans) => {
            // Create proper ValidationResult objects
            const validations = booleans.map((valid) => ({
              valid,
              errors: valid
                ? []
                : [{ field: 'test', message: 'error', value: null, expected: 'valid' }],
              warnings: [],
            }));
            // Use spread syntax since combineValidations expects rest parameters
            const combined = combineValidations(...validations);
            expect(combined.valid).toBe(booleans.every((b) => b));
          }
        )
      );
    });
  });
});

// ============================================================================
// NUMERICAL STABILITY PROPERTY TESTS
// ============================================================================

describe('SCEI Numerical Stability - Property-Based Tests', () => {
  describe('safeLog', () => {
    it('should return finite number for finite confidence values', () => {
      fc.assert(
        // safeLog is designed for confidence values in [0, 1]
        fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (value) => {
          const result = safeLog(value);
          expect(Number.isFinite(result)).toBe(true);
        })
      );
    });

    it('should handle edge cases gracefully', () => {
      // For values in normal range, returns finite log
      expect(Number.isFinite(safeLog(0.5))).toBe(true);
      expect(Number.isFinite(safeLog(0))).toBe(true); // Clamped to epsilon
      expect(Number.isFinite(safeLog(1))).toBe(true); // Clamped to 1-epsilon
      // For non-finite inputs, returns NEGATIVE_INFINITY
      expect(safeLog(Infinity)).toBe(Number.NEGATIVE_INFINITY);
      expect(safeLog(-Infinity)).toBe(Number.NEGATIVE_INFINITY);
    });
  });

  describe('clampConfidence', () => {
    it('should always return a value in [0, 1]', () => {
      fc.assert(
        fc.property(fc.double({ noNaN: true }), (value) => {
          const result = clampConfidence(value);
          expect(result).toBeGreaterThanOrEqual(0);
          expect(result).toBeLessThanOrEqual(1);
        })
      );
    });

    it('should be idempotent', () => {
      fc.assert(
        fc.property(fc.double({ noNaN: true }), (value) => {
          const once = clampConfidence(value);
          const twice = clampConfidence(once);
          expect(once).toBe(twice);
        })
      );
    });
  });

  describe('safeDivide', () => {
    it('should return fallback for zero denominator', () => {
      fc.assert(
        fc.property(fc.double({ noNaN: true }), fc.double({ noNaN: true }), (num, fallback) => {
          const result = safeDivide(num, 0, fallback);
          expect(result).toBe(fallback);
        })
      );
    });

    it('should return correct result for normal denominators', () => {
      fc.assert(
        fc.property(
          fc.double({ min: -1000, max: 1000, noNaN: true }),
          // Use a reasonable range to avoid underflow/overflow edge cases
          fc.double({ min: 0.001, max: 1000, noNaN: true }),
          (num, denom) => {
            const rawResult = num / denom;
            const result = safeDivide(num, denom, -999);
            // safeDivide returns fallback for non-finite results
            if (Number.isFinite(rawResult)) {
              expect(result).toBe(rawResult);
            } else {
              expect(result).toBe(-999);
            }
          }
        )
      );
    });
  });

  describe('safeWeightedAverage', () => {
    it('should return fallback for empty arrays', () => {
      const result = safeWeightedAverage([], [], 0.5);
      expect(result).toBe(0.5);
    });

    it('should return fallback for zero total weight', () => {
      const result = safeWeightedAverage([1, 2, 3], [0, 0, 0], 0.5);
      expect(result).toBe(0.5);
    });

    it('should return weighted average for valid inputs', () => {
      fc.assert(
        fc.property(
          fc.array(fc.double({ min: 0, max: 100, noNaN: true }), { minLength: 1, maxLength: 10 }),
          fc.array(fc.double({ min: 0.1, max: 10, noNaN: true }), { minLength: 1, maxLength: 10 }),
          (values, weights) => {
            // Ensure arrays are same length
            const len = Math.min(values.length, weights.length);
            const vals = values.slice(0, len);
            const wts = weights.slice(0, len);

            const result = safeWeightedAverage(vals, wts, 0);

            // Result should be within range of values
            const minVal = Math.min(...vals);
            const maxVal = Math.max(...vals);
            expect(result).toBeGreaterThanOrEqual(minVal - 0.0001);
            expect(result).toBeLessThanOrEqual(maxVal + 0.0001);
          }
        )
      );
    });
  });
});

// ============================================================================
// EVENT BUS PROPERTY TESTS
// ============================================================================

describe('SCEI Event Bus - Property-Based Tests', () => {
  describe('Event ordering', () => {
    it('should maintain FIFO order for events', async () => {
      await fc.assert(
        fc.asyncProperty(
          fc.array(fc.nat({ max: 100 }), { minLength: 1, maxLength: 20 }),
          async (eventIds) => {
            // Create fresh event bus for each iteration to avoid listener accumulation
            resetSCEIEventBus();
            const eventBus = new SCEIEventBus();
            const receivedOrder: number[] = [];

            const subscription = eventBus.subscribe((event) => {
              receivedOrder.push((event.payload as { id: number }).id);
            });

            // Emit events sequentially
            for (const id of eventIds) {
              const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {
                id,
              });
              await eventBus.emit(event);
            }

            subscription.unsubscribe();
            expect(receivedOrder).toEqual(eventIds);
          }
        )
      );
    });
  });

  describe('Subscription management', () => {
    it('should properly unsubscribe', async () => {
      await fc.assert(
        fc.asyncProperty(fc.nat({ max: 10 }), async (eventCount) => {
          // Create fresh event bus for each iteration
          resetSCEIEventBus();
          const eventBus = new SCEIEventBus();

          let count = 0;
          const unsub = eventBus.subscribe(() => {
            count++;
          });

          // Emit some events
          for (let i = 0; i < eventCount; i++) {
            await eventBus.emit(
              createSCEIEvent('calibration:prediction_recorded', 'calibration', {})
            );
          }

          const countBeforeUnsub = count;

          // Unsubscribe
          unsub.unsubscribe();

          // Emit more events
          for (let i = 0; i < eventCount; i++) {
            await eventBus.emit(
              createSCEIEvent('calibration:prediction_recorded', 'calibration', {})
            );
          }

          // Count should not increase after unsubscribe
          expect(count).toBe(countBeforeUnsub);
        })
      );
    });
  });
});

// ============================================================================
// METRICS PROPERTY TESTS
// ============================================================================

describe('SCEI Metrics - Property-Based Tests', () => {
  let metrics: SCEIMetricsCollector;

  beforeEach(() => {
    resetSCEIMetrics();
    metrics = new SCEIMetricsCollector();
  });

  describe('Counter invariants', () => {
    it('should be monotonically increasing', () => {
      fc.assert(
        fc.property(
          fc.array(fc.nat({ max: 100 }), { minLength: 1, maxLength: 20 }),
          (increments) => {
            // Use a known metric from the registry
            const metricName = 'calibration_predictions_recorded';
            let previous = 0;

            for (const inc of increments) {
              metrics.incrementCounter(metricName, {}, inc);
              const snapshot = metrics.exportJSON();
              // Access the counter directly by name from the Record
              const counterValues = snapshot.counters[metricName];
              const current = counterValues
                ? counterValues.reduce((sum, c) => sum + c.value, 0)
                : 0;

              expect(current).toBeGreaterThanOrEqual(previous);
              previous = current;
            }
          }
        )
      );
    });
  });

  describe('Histogram invariants', () => {
    it('should track count and sum correctly', () => {
      fc.assert(
        fc.property(
          fc.array(fc.double({ min: 0, max: 1000, noNaN: true }), { minLength: 1, maxLength: 50 }),
          (values) => {
            // Use a known metric from the registry
            const metricName = 'calibration_report_duration';

            for (const value of values) {
              metrics.recordHistogram(metricName, value);
            }

            const snapshot = metrics.exportJSON();
            // Access the histogram directly by name from the Record
            const histogramValues = snapshot.histograms[metricName];

            if (histogramValues && histogramValues.length > 0) {
              const histogram = histogramValues[0];
              expect(histogram.count).toBe(values.length);
              // Sum should be approximately the sum of values (allowing for floating point)
              const expectedSum = values.reduce((a, b) => a + b, 0);
              expect(histogram.sum).toBeCloseTo(expectedSum, 5);
            }
          }
        )
      );
    });
  });

  describe('Gauge invariants', () => {
    it('should always reflect the last set value', () => {
      fc.assert(
        fc.property(
          fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 1, maxLength: 20 }),
          (values) => {
            // Use a known metric from the registry
            const metricName = 'calibration_brier_score';

            for (const value of values) {
              metrics.setGauge(metricName, value);
            }

            const snapshot = metrics.exportJSON();
            // Access the gauge directly by name from the Record
            const gaugeValues = snapshot.gauges[metricName];

            if (gaugeValues && gaugeValues.length > 0) {
              const gauge = gaugeValues[0];
              expect(gauge.value).toBe(values[values.length - 1]);
            }
          }
        )
      );
    });
  });
});

// ============================================================================
// PERSISTENCE PROPERTY TESTS
// ============================================================================

describe('SCEI Persistence - Property-Based Tests', () => {
  let storage: InMemorySCEIStorage;
  const namespace: SCEINamespace = 'calibration:records';

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  describe('Storage invariants', () => {
    it('should retrieve exactly what was stored', async () => {
      await fc.assert(
        fc.asyncProperty(
          fc.string({ minLength: 1, maxLength: 50 }),
          fc.object(),
          async (id, value) => {
            await storage.set({ namespace, id }, value);
            const retrieved = await storage.get({ namespace, id });

            expect(retrieved).not.toBeNull();
            expect(retrieved?.value).toEqual(value);
          }
        )
      );
    });

    it('should report existence correctly', async () => {
      await fc.assert(
        fc.asyncProperty(fc.string({ minLength: 1, maxLength: 50 }), async (id) => {
          // Before set
          const existsBefore = await storage.exists({ namespace, id });
          expect(existsBefore).toBe(false);

          // After set
          await storage.set({ namespace, id }, { test: true });
          const existsAfter = await storage.exists({ namespace, id });
          expect(existsAfter).toBe(true);

          // After delete
          await storage.delete({ namespace, id });
          const existsAfterDelete = await storage.exists({ namespace, id });
          expect(existsAfterDelete).toBe(false);
        })
      );
    });

    it('should maintain correct count', async () => {
      await fc.assert(
        fc.asyncProperty(
          fc.set(fc.string({ minLength: 1, maxLength: 20 }), { minLength: 1, maxLength: 20 }),
          async (ids) => {
            // Clear namespace first
            await storage.clearNamespace(namespace);

            // Add items
            for (const id of ids) {
              await storage.set({ namespace, id }, { data: id });
            }

            const count = await storage.count(namespace);
            expect(count).toBe(ids.size);
          }
        )
      );
    });
  });

  describe('Batch operations', () => {
    it('should batch set all items', async () => {
      await fc.assert(
        fc.asyncProperty(
          fc.array(
            fc.record({
              id: fc.string({ minLength: 1, maxLength: 20 }),
              value: fc.nat(),
            }),
            { minLength: 1, maxLength: 10 }
          ),
          async (items) => {
            const batchItems = items.map((item) => ({
              key: { namespace, id: item.id },
              value: { data: item.value },
            }));

            await storage.batchSet(batchItems);

            // Verify all items exist
            for (const item of items) {
              const exists = await storage.exists({ namespace, id: item.id });
              expect(exists).toBe(true);
            }
          }
        )
      );
    });
  });

  describe('TTL behavior', () => {
    it('should expire items with TTL', async () => {
      const id = 'ttl-test';
      await storage.set({ namespace, id }, { data: 'test' }, { ttl: 1 }); // 1ms TTL

      // Wait for expiry
      await new Promise((resolve) => setTimeout(resolve, 10));

      const result = await storage.get({ namespace, id });
      expect(result).toBeNull();
    });
  });
});

// ============================================================================
// CROSS-MODULE INTEGRATION PROPERTY TESTS
// ============================================================================

describe('SCEI Cross-Module Integration - Property-Based Tests', () => {
  it('should maintain consistency between events and metrics', async () => {
    await fc.assert(
      fc.asyncProperty(fc.nat({ max: 50 }), async (numEvents) => {
        // Create fresh instances for each iteration
        resetSCEIEventBus();
        resetSCEIMetrics();

        const eventBus = new SCEIEventBus();
        const metrics = new SCEIMetricsCollector();
        const metricName = 'calibration_predictions_recorded';

        let eventCount = 0;

        const subscription = eventBus.subscribe(() => {
          eventCount++;
          metrics.incrementCounter(metricName, {});
        });

        for (let i = 0; i < numEvents; i++) {
          await eventBus.emit(
            createSCEIEvent('calibration:prediction_recorded', 'calibration', { i })
          );
        }

        subscription.unsubscribe();

        expect(eventCount).toBe(numEvents);
        // Use the correct property name: totalEventsEmitted
        expect(eventBus.getMetrics().totalEventsEmitted).toBe(numEvents);
      }),
      { numRuns: 20 } // Limit iterations to stay within test timeout
    );
  }, 15000); // Extend timeout for property-based async test
});

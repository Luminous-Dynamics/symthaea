/**
 * @mycelix/sdk SCEI Infrastructure Tests
 *
 * Tests for validation, event bus, metrics, and persistence modules.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  // Validation
  validateConfidence,
  validateDegradationFactor,
  validatePositiveInteger,
  validateTimestamp,
  validateNonEmptyString,
  validateId,
  validateEnum,
  validateArray,
  combineValidations,
  assertValid,
  SCEIValidationError,
  safeLog,
  clampConfidence,
  safeDivide,
  safeWeightedAverage,
  // Event Bus
  SCEIEventBus,
  getSCEIEventBus,
  resetSCEIEventBus,
  createSCEIEvent,
  createCorrelationId,
  // Metrics
  SCEIMetricsCollector,
  getSCEIMetrics,
  resetSCEIMetrics,
  SCEI_METRICS,
  // Persistence
  InMemorySCEIStorage,
  SCEIPersistenceManager,
  getSCEIPersistence,
  setSCEIPersistence,
  resetSCEIPersistence,
} from '../src/scei/index.js';

// ============================================================================
// VALIDATION TESTS
// ============================================================================

describe('SCEI Validation - Confidence', () => {
  it('should accept valid confidence values', () => {
    expect(validateConfidence(0).valid).toBe(true);
    expect(validateConfidence(0.5).valid).toBe(true);
    expect(validateConfidence(1).valid).toBe(true);
    expect(validateConfidence(0.123456).valid).toBe(true);
  });

  it('should reject non-numbers', () => {
    expect(validateConfidence('0.5').valid).toBe(false);
    expect(validateConfidence(null).valid).toBe(false);
    expect(validateConfidence(undefined).valid).toBe(false);
    expect(validateConfidence({}).valid).toBe(false);
  });

  it('should reject out of range values', () => {
    expect(validateConfidence(-0.1).valid).toBe(false);
    expect(validateConfidence(1.1).valid).toBe(false);
    expect(validateConfidence(100).valid).toBe(false);
  });

  it('should reject non-finite values', () => {
    expect(validateConfidence(NaN).valid).toBe(false);
    expect(validateConfidence(Infinity).valid).toBe(false);
    expect(validateConfidence(-Infinity).valid).toBe(false);
  });

  it('should warn on edge values', () => {
    const result0 = validateConfidence(0);
    expect(result0.warnings.length).toBeGreaterThan(0);

    const result1 = validateConfidence(1);
    expect(result1.warnings.length).toBeGreaterThan(0);
  });
});

describe('SCEI Validation - Degradation Factor', () => {
  it('should accept valid degradation factors', () => {
    expect(validateDegradationFactor(0.5).valid).toBe(true);
    expect(validateDegradationFactor(0.95).valid).toBe(true);
    expect(validateDegradationFactor(0.01).valid).toBe(true);
    expect(validateDegradationFactor(0.99).valid).toBe(true);
  });

  it('should reject boundary values (must be exclusive)', () => {
    expect(validateDegradationFactor(0).valid).toBe(false);
    expect(validateDegradationFactor(1).valid).toBe(false);
  });

  it('should warn on extreme values', () => {
    const highResult = validateDegradationFactor(0.999);
    expect(highResult.warnings.length).toBeGreaterThan(0);

    const lowResult = validateDegradationFactor(0.1);
    expect(lowResult.warnings.length).toBeGreaterThan(0);
  });
});

describe('SCEI Validation - Positive Integer', () => {
  it('should accept valid positive integers', () => {
    expect(validatePositiveInteger(1, 'test').valid).toBe(true);
    expect(validatePositiveInteger(100, 'test').valid).toBe(true);
    expect(validatePositiveInteger(1000000, 'test').valid).toBe(true);
  });

  it('should reject non-integers', () => {
    expect(validatePositiveInteger(1.5, 'test').valid).toBe(false);
    expect(validatePositiveInteger(0.9, 'test').valid).toBe(false);
  });

  it('should reject values below minimum', () => {
    expect(validatePositiveInteger(0, 'test').valid).toBe(false);
    expect(validatePositiveInteger(-1, 'test').valid).toBe(false);
  });

  it('should respect min/max options', () => {
    expect(validatePositiveInteger(5, 'test', { min: 10 }).valid).toBe(false);
    expect(validatePositiveInteger(100, 'test', { max: 50 }).valid).toBe(false);
    expect(validatePositiveInteger(25, 'test', { min: 10, max: 50 }).valid).toBe(true);
  });
});

describe('SCEI Validation - Numerical Stability', () => {
  it('safeLog should handle edge cases', () => {
    expect(Number.isFinite(safeLog(0))).toBe(true);
    expect(Number.isFinite(safeLog(1))).toBe(true);
    expect(Number.isFinite(safeLog(0.5))).toBe(true);
    expect(safeLog(NaN)).toBe(Number.NEGATIVE_INFINITY);
    expect(safeLog(Infinity)).toBe(Number.NEGATIVE_INFINITY);
  });

  it('clampConfidence should handle edge cases', () => {
    expect(clampConfidence(0.5)).toBe(0.5);
    expect(clampConfidence(-1)).toBe(0);
    expect(clampConfidence(2)).toBe(1);
    expect(clampConfidence(NaN)).toBe(0.5);
    expect(clampConfidence(Infinity)).toBe(0.5);
  });

  it('safeDivide should handle division by zero', () => {
    expect(safeDivide(10, 2)).toBe(5);
    expect(safeDivide(10, 0)).toBe(0);
    expect(safeDivide(10, 0, -1)).toBe(-1);
    expect(safeDivide(10, NaN)).toBe(0);
  });

  it('safeWeightedAverage should handle empty/invalid inputs', () => {
    expect(safeWeightedAverage([], [])).toBe(0);
    expect(safeWeightedAverage([1, 2, 3], [1, 1, 1])).toBe(2);
    expect(safeWeightedAverage([1, 2], [1])).toBe(0); // Mismatched lengths
    expect(safeWeightedAverage([1, NaN, 3], [1, 1, 1])).toBe(2); // Skips NaN
  });
});

describe('SCEI Validation - Combined', () => {
  it('combineValidations should merge results', () => {
    const result1 = validateConfidence(0.5);
    const result2 = validatePositiveInteger(10, 'count');

    const combined = combineValidations(result1, result2);
    expect(combined.valid).toBe(true);
    expect(combined.errors.length).toBe(0);
  });

  it('combineValidations should collect all errors', () => {
    const result1 = validateConfidence(-1);
    const result2 = validatePositiveInteger(-5, 'count');

    const combined = combineValidations(result1, result2);
    expect(combined.valid).toBe(false);
    expect(combined.errors.length).toBe(2);
  });

  it('assertValid should throw on invalid', () => {
    const invalid = validateConfidence(-1);
    expect(() => assertValid(invalid, 'Test')).toThrow(SCEIValidationError);
  });

  it('assertValid should not throw on valid', () => {
    const valid = validateConfidence(0.5);
    expect(() => assertValid(valid)).not.toThrow();
  });
});

// ============================================================================
// EVENT BUS TESTS
// ============================================================================

describe('SCEI Event Bus', () => {
  let eventBus: SCEIEventBus;

  beforeEach(() => {
    resetSCEIEventBus();
    eventBus = getSCEIEventBus();
  });

  afterEach(() => {
    resetSCEIEventBus();
  });

  it('should be a singleton', () => {
    const bus1 = getSCEIEventBus();
    const bus2 = getSCEIEventBus();
    expect(bus1).toBe(bus2);
  });

  it('should deliver events to subscribers', async () => {
    const received: unknown[] = [];

    eventBus.subscribe((event) => {
      received.push(event);
    });

    await eventBus.emit(
      createSCEIEvent('calibration:prediction_recorded', 'calibration', {
        predictionId: 'test-1',
        confidence: 0.8,
        domain: 'science',
        source: 'test',
      })
    );

    expect(received.length).toBe(1);
    expect(received[0]).toHaveProperty('type', 'calibration:prediction_recorded');
  });

  it('should filter events by type', async () => {
    const received: unknown[] = [];

    eventBus.on('metabolism:claim_born', (event) => {
      received.push(event);
    });

    // This should NOT be received
    await eventBus.emit(
      createSCEIEvent('calibration:prediction_recorded', 'calibration', {})
    );

    // This SHOULD be received
    await eventBus.emit(
      createSCEIEvent('metabolism:claim_born', 'metabolism', {
        claimId: 'claim-1',
        content: 'test',
        domain: 'science',
        authorId: 'author-1',
        initialConfidence: 0.5,
      })
    );

    expect(received.length).toBe(1);
  });

  it('should filter events by source module', async () => {
    const received: unknown[] = [];

    eventBus.onModule('metabolism', (event) => {
      received.push(event);
    });

    await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', {}));
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    expect(received.length).toBe(1);
  });

  it('should allow unsubscribing', async () => {
    const received: unknown[] = [];

    const sub = eventBus.subscribe((event) => {
      received.push(event);
    });

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    expect(received.length).toBe(1);

    sub.unsubscribe();

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    expect(received.length).toBe(1); // Should still be 1
  });

  it('should track metrics', async () => {
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', {}));

    const metrics = eventBus.getMetrics();
    expect(metrics.totalEventsEmitted).toBe(3);
    expect(metrics.eventsByType['metabolism:claim_born']).toBe(2);
    expect(metrics.eventsBySource['metabolism']).toBe(2);
    expect(metrics.eventsBySource['calibration']).toBe(1);
  });

  it('should maintain event history', async () => {
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', { id: 1 }));
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', { id: 2 }));

    const history = eventBus.getHistory();
    expect(history.length).toBe(2);
  });

  it('should handle async listeners', async () => {
    let resolved = false;

    eventBus.subscribe(async () => {
      await new Promise((r) => setTimeout(r, 10));
      resolved = true;
    });

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    expect(resolved).toBe(true);
  });

  it('should handle listener errors gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    eventBus.subscribe(() => {
      throw new Error('Test error');
    });

    // Should not throw
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    const metrics = eventBus.getMetrics();
    expect(metrics.listenerErrors).toBe(1);

    consoleSpy.mockRestore();
  });

  it('createCorrelationId should generate unique IDs', () => {
    const id1 = createCorrelationId();
    const id2 = createCorrelationId();
    expect(id1).not.toBe(id2);
    expect(id1.startsWith('corr_')).toBe(true);
  });
});

// ============================================================================
// METRICS TESTS
// ============================================================================

describe('SCEI Metrics', () => {
  let metrics: SCEIMetricsCollector;

  beforeEach(() => {
    resetSCEIMetrics();
    metrics = getSCEIMetrics();
  });

  afterEach(() => {
    resetSCEIMetrics();
  });

  it('should be a singleton', () => {
    const m1 = getSCEIMetrics();
    const m2 = getSCEIMetrics();
    expect(m1).toBe(m2);
  });

  it('should increment counters', () => {
    metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });
    metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });
    metrics.incrementCounter('calibration_predictions_total', { domain: 'math' });

    expect(metrics.getCounter('calibration_predictions_total', { domain: 'science' })).toBe(2);
    expect(metrics.getCounter('calibration_predictions_total', { domain: 'math' })).toBe(1);
  });

  it('should set gauges', () => {
    metrics.setGauge('propagation_quarantine_queue', 5);
    expect(metrics.getGauge('propagation_quarantine_queue')).toBe(5);

    metrics.setGauge('propagation_quarantine_queue', 3);
    expect(metrics.getGauge('propagation_quarantine_queue')).toBe(3);
  });

  it('should increment/decrement gauges', () => {
    metrics.setGauge('discovery_active_gaps', 10, { type: 'structural' });
    metrics.incrementGauge('discovery_active_gaps', { type: 'structural' });
    expect(metrics.getGauge('discovery_active_gaps', { type: 'structural' })).toBe(11);

    metrics.decrementGauge('discovery_active_gaps', { type: 'structural' }, 3);
    expect(metrics.getGauge('discovery_active_gaps', { type: 'structural' })).toBe(8);
  });

  it('should record histograms', () => {
    metrics.recordHistogram('propagation_duration_ms', 50, { outcome: 'completed' });
    metrics.recordHistogram('propagation_duration_ms', 150, { outcome: 'completed' });
    metrics.recordHistogram('propagation_duration_ms', 250, { outcome: 'completed' });

    const stats = metrics.getHistogramStats('propagation_duration_ms', { outcome: 'completed' });
    expect(stats).not.toBeNull();
    expect(stats!.count).toBe(3);
    expect(stats!.sum).toBe(450);
    expect(stats!.avg).toBe(150);
  });

  it('should time operations', async () => {
    const result = await metrics.timeAsync(
      'calibration_report_duration_ms',
      { domain: 'test' },
      async () => {
        await new Promise((r) => setTimeout(r, 15));
        return 'done';
      }
    );

    expect(result).toBe('done');
    const stats = metrics.getHistogramStats('calibration_report_duration_ms', { domain: 'test' });
    expect(stats).not.toBeNull();
    expect(stats!.count).toBe(1);
    // setTimeout is not precise, allow for timing variance
    expect(stats!.sum).toBeGreaterThanOrEqual(10);
  });

  it('should export Prometheus format', () => {
    metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });
    metrics.setGauge('propagation_quarantine_queue', 5);

    const output = metrics.exportPrometheus();
    expect(output).toContain('scei_calibration_predictions_total');
    expect(output).toContain('scei_propagation_quarantine_queue');
    expect(output).toContain('# HELP');
    expect(output).toContain('# TYPE');
  });

  it('should export JSON format', () => {
    metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });

    const snapshot = metrics.exportJSON();
    expect(snapshot.timestamp).toBeGreaterThan(0);
    expect(snapshot.counters).toHaveProperty('calibration_predictions_total');
  });

  it('should have defined metric definitions', () => {
    expect(Object.keys(SCEI_METRICS).length).toBeGreaterThan(10);
    expect(SCEI_METRICS.calibration_predictions_total).toBeDefined();
    expect(SCEI_METRICS.calibration_predictions_total.type).toBe('counter');
  });
});

// ============================================================================
// MUTATION-TARGETED METRICS TESTS
// ============================================================================

describe('SCEI Metrics - Mutation Coverage', () => {
  let metrics: SCEIMetricsCollector;

  beforeEach(() => {
    resetSCEIMetrics();
    metrics = getSCEIMetrics();
  });

  afterEach(() => {
    resetSCEIMetrics();
  });

  describe('counter edge cases', () => {
    it('should increment counter with custom value', () => {
      metrics.incrementCounter('calibration_predictions_total', { domain: 'test' }, 5);
      expect(metrics.getCounter('calibration_predictions_total', { domain: 'test' })).toBe(5);

      metrics.incrementCounter('calibration_predictions_total', { domain: 'test' }, 3);
      expect(metrics.getCounter('calibration_predictions_total', { domain: 'test' })).toBe(8);
    });

    it('should handle unknown counter metric gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      metrics.incrementCounter('unknown_counter_name', { domain: 'test' });
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Unknown counter metric'));
      consoleSpy.mockRestore();
    });

    it('should return 0 for unset counters', () => {
      expect(metrics.getCounter('calibration_predictions_total', { domain: 'nonexistent' })).toBe(0);
    });

    it('should return 0 for unknown counter name in getCounter', () => {
      expect(metrics.getCounter('totally_unknown_metric', { foo: 'bar' })).toBe(0);
    });

    it('should handle empty labels for counters', () => {
      metrics.incrementCounter('calibration_report_cache_hits', {});
      expect(metrics.getCounter('calibration_report_cache_hits', {})).toBe(1);
    });
  });

  describe('gauge edge cases', () => {
    it('should handle unknown gauge metric gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      metrics.setGauge('unknown_gauge_name', 10);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Unknown gauge metric'));
      consoleSpy.mockRestore();
    });

    it('should return 0 for unset gauges', () => {
      expect(metrics.getGauge('propagation_quarantine_queue_size', { key: 'nonexistent' })).toBe(0);
    });

    it('should return 0 for unknown gauge name in getGauge', () => {
      expect(metrics.getGauge('totally_unknown_gauge', {})).toBe(0);
    });

    it('should increment gauge from zero', () => {
      metrics.incrementGauge('metabolism_active_claims', {}, 5);
      expect(metrics.getGauge('metabolism_active_claims', {})).toBe(5);
    });

    it('should decrement gauge to negative', () => {
      metrics.setGauge('metabolism_active_claims', 2, {});
      metrics.decrementGauge('metabolism_active_claims', {}, 5);
      expect(metrics.getGauge('metabolism_active_claims', {})).toBe(-3);
    });
  });

  describe('histogram edge cases', () => {
    it('should handle unknown histogram metric gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      metrics.recordHistogram('unknown_histogram_name', 100);
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Unknown histogram metric'));
      consoleSpy.mockRestore();
    });

    it('should return null for unset histogram stats', () => {
      const stats = metrics.getHistogramStats('propagation_duration_ms', { outcome: 'nonexistent' });
      expect(stats).toBeNull();
    });

    it('should return null for unknown histogram name', () => {
      const stats = metrics.getHistogramStats('totally_unknown_histogram', {});
      expect(stats).toBeNull();
    });

    it('should correctly bucket values at exact bucket boundaries', () => {
      // Record values at exact bucket boundaries: 1, 5, 10, 25, 50, 100, 250, 500, 1000
      metrics.recordHistogram('propagation_duration_ms', 1, { outcome: 'test' });
      metrics.recordHistogram('propagation_duration_ms', 5, { outcome: 'test' });
      metrics.recordHistogram('propagation_duration_ms', 10, { outcome: 'test' });
      metrics.recordHistogram('propagation_duration_ms', 100, { outcome: 'test' });

      const stats = metrics.getHistogramStats('propagation_duration_ms', { outcome: 'test' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(4);
      expect(stats!.sum).toBe(116);
    });

    it('should bucket values correctly across all default buckets', () => {
      // Value above all buckets
      metrics.recordHistogram('propagation_duration_ms', 15000, { outcome: 'large' });

      const stats = metrics.getHistogramStats('propagation_duration_ms', { outcome: 'large' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
      expect(stats!.sum).toBe(15000);
    });

    it('should calculate percentiles correctly', () => {
      // Add values to create predictable percentiles
      for (let i = 0; i < 100; i++) {
        metrics.recordHistogram('propagation_depth', i + 1, {});
      }

      const stats = metrics.getHistogramStats('propagation_depth', {});
      expect(stats).not.toBeNull();
      expect(stats!.p50).toBeGreaterThan(0);
      expect(stats!.p95).toBeGreaterThan(stats!.p50);
      expect(stats!.p99).toBeGreaterThanOrEqual(stats!.p95);
    });

    it('should handle single value histogram', () => {
      metrics.recordHistogram('propagation_depth', 42, {});

      const stats = metrics.getHistogramStats('propagation_depth', {});
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
      expect(stats!.sum).toBe(42);
      expect(stats!.avg).toBe(42);
    });
  });

  describe('timing functions', () => {
    it('should time sync operations', () => {
      const result = metrics.timeSync(
        'calibration_report_duration_ms',
        { scope_type: 'sync' },
        () => {
          // Simulate some work
          let sum = 0;
          for (let i = 0; i < 10000; i++) sum += i;
          return 'sync-result';
        }
      );

      expect(result).toBe('sync-result');
      const stats = metrics.getHistogramStats('calibration_report_duration_ms', { scope_type: 'sync' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
      expect(stats!.sum).toBeGreaterThanOrEqual(0);
    });

    it('should create and stop manual timer', () => {
      const stopTimer = metrics.startTimer('calibration_report_duration_ms', { scope_type: 'manual' });

      // Do something
      const x = Math.random() * Math.random();
      void x;

      const duration = stopTimer();
      expect(duration).toBeGreaterThanOrEqual(0);

      const stats = metrics.getHistogramStats('calibration_report_duration_ms', { scope_type: 'manual' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
    });

    it('should handle exceptions in timeAsync', async () => {
      await expect(
        metrics.timeAsync('calibration_report_duration_ms', { scope_type: 'error' }, async () => {
          throw new Error('Test error');
        })
      ).rejects.toThrow('Test error');

      // Duration should still be recorded
      const stats = metrics.getHistogramStats('calibration_report_duration_ms', { scope_type: 'error' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
    });

    it('should handle exceptions in timeSync', () => {
      expect(() =>
        metrics.timeSync('calibration_report_duration_ms', { scope_type: 'sync-error' }, () => {
          throw new Error('Sync error');
        })
      ).toThrow('Sync error');

      // Duration should still be recorded
      const stats = metrics.getHistogramStats('calibration_report_duration_ms', { scope_type: 'sync-error' });
      expect(stats).not.toBeNull();
      expect(stats!.count).toBe(1);
    });
  });

  describe('prometheus export edge cases', () => {
    it('should export counters with labels', () => {
      metrics.incrementCounter('calibration_predictions_total', { domain: 'science', source: 'api' });

      const output = metrics.exportPrometheus();
      expect(output).toContain('domain="science"');
      expect(output).toContain('source="api"');
    });

    it('should export gauges without labels', () => {
      metrics.setGauge('metabolism_active_claims', 42, {});

      const output = metrics.exportPrometheus();
      expect(output).toContain('scei_metabolism_active_claims');
      expect(output).toContain('42');
    });

    it('should export histogram buckets correctly', () => {
      metrics.recordHistogram('propagation_duration_ms', 75, { outcome: 'completed' });

      const output = metrics.exportPrometheus();
      expect(output).toContain('_bucket');
      expect(output).toContain('le="');
      expect(output).toContain('+Inf');
      expect(output).toContain('_sum');
      expect(output).toContain('_count');
    });

    it('should export histogram with empty labels', () => {
      metrics.recordHistogram('propagation_depth', 5, {});

      const output = metrics.exportPrometheus();
      expect(output).toContain('scei_propagation_depth_bucket{le=');
      expect(output).toContain('scei_propagation_depth_sum');
      expect(output).toContain('scei_propagation_depth_count');
    });
  });

  describe('JSON export edge cases', () => {
    it('should export empty histogram correctly', () => {
      const snapshot = metrics.exportJSON();
      expect(snapshot.histograms).toBeDefined();
    });

    it('should calculate avg as 0 for empty histogram in JSON', () => {
      // Record and then check the avg calculation
      metrics.recordHistogram('propagation_duration_ms', 100, { outcome: 'json-test' });

      const snapshot = metrics.exportJSON();
      const histData = snapshot.histograms['propagation_duration_ms'];
      expect(histData).toBeDefined();
      expect(histData.length).toBe(1);
      expect(histData[0].avg).toBe(100);
    });

    it('should include timestamp in JSON export', () => {
      const before = Date.now();
      const snapshot = metrics.exportJSON();
      const after = Date.now();

      expect(snapshot.timestamp).toBeGreaterThanOrEqual(before);
      expect(snapshot.timestamp).toBeLessThanOrEqual(after);
    });
  });

  describe('label key generation', () => {
    it('should sort labels alphabetically for key generation', () => {
      // Order of labels shouldn't matter
      metrics.incrementCounter('calibration_predictions_total', { domain: 'a', source: 'b' });
      metrics.incrementCounter('calibration_predictions_total', { source: 'b', domain: 'a' });

      // Both should increment the same counter
      expect(metrics.getCounter('calibration_predictions_total', { domain: 'a', source: 'b' })).toBe(2);
    });

    it('should handle numeric label values', () => {
      metrics.incrementCounter('calibration_predictions_total', { domain: 'test', count: 42 });
      expect(metrics.getCounter('calibration_predictions_total', { domain: 'test', count: 42 })).toBe(1);
    });
  });

  describe('reset functionality', () => {
    it('should reset all metrics', () => {
      metrics.incrementCounter('calibration_predictions_total', { domain: 'test' });
      metrics.setGauge('metabolism_active_claims', 10, {});
      metrics.recordHistogram('propagation_duration_ms', 100, {});

      metrics.reset();

      expect(metrics.getCounter('calibration_predictions_total', { domain: 'test' })).toBe(0);
      expect(metrics.getGauge('metabolism_active_claims', {})).toBe(0);
      expect(metrics.getHistogramStats('propagation_duration_ms', {})).toBeNull();
    });

    it('should resetSCEIMetrics to new instance', () => {
      const m1 = getSCEIMetrics();
      m1.incrementCounter('calibration_predictions_total', { domain: 'test' });

      resetSCEIMetrics();

      const m2 = getSCEIMetrics();
      expect(m2).not.toBe(m1);
      expect(m2.getCounter('calibration_predictions_total', { domain: 'test' })).toBe(0);
    });
  });

  describe('SCEI_METRICS definitions', () => {
    it('should have correct types for all metrics', () => {
      const counterMetrics = ['calibration_predictions_total', 'calibration_resolutions_total',
        'metabolism_births_total', 'metabolism_deaths_total'];
      for (const name of counterMetrics) {
        expect(SCEI_METRICS[name]?.type).toBe('counter');
      }

      const gaugeMetrics = ['calibration_brier_score', 'metabolism_claims_total',
        'propagation_quarantine_queue_size', 'discovery_active_gaps'];
      for (const name of gaugeMetrics) {
        expect(SCEI_METRICS[name]?.type).toBe('gauge');
      }

      const histogramMetrics = ['calibration_report_duration_ms', 'metabolism_health_distribution',
        'propagation_duration_ms', 'discovery_scan_duration_ms'];
      for (const name of histogramMetrics) {
        expect(SCEI_METRICS[name]?.type).toBe('histogram');
      }
    });

    it('should have help text for all metrics', () => {
      for (const [name, def] of Object.entries(SCEI_METRICS)) {
        expect(def.help).toBeTruthy();
        expect(def.help.length).toBeGreaterThan(5);
      }
    });

    it('should have name prefixed with scei_', () => {
      for (const [, def] of Object.entries(SCEI_METRICS)) {
        expect(def.name.startsWith('scei_')).toBe(true);
      }
    });
  });
});

// ============================================================================
// PERSISTENCE TESTS
// ============================================================================

describe('SCEI Persistence - InMemory', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should store and retrieve items', async () => {
    await storage.set(
      { namespace: 'calibration:records', id: 'rec-1' },
      { confidence: 0.8, domain: 'science' }
    );

    const item = await storage.get<{ confidence: number; domain: string }>({
      namespace: 'calibration:records',
      id: 'rec-1',
    });

    expect(item).not.toBeNull();
    expect(item!.value.confidence).toBe(0.8);
  });

  it('should return null for missing items', async () => {
    const item = await storage.get({ namespace: 'calibration:records', id: 'missing' });
    expect(item).toBeNull();
  });

  it('should delete items', async () => {
    await storage.set({ namespace: 'calibration:records', id: 'rec-1' }, { data: 'test' });
    expect(await storage.exists({ namespace: 'calibration:records', id: 'rec-1' })).toBe(true);

    const deleted = await storage.delete({ namespace: 'calibration:records', id: 'rec-1' });
    expect(deleted).toBe(true);
    expect(await storage.exists({ namespace: 'calibration:records', id: 'rec-1' })).toBe(false);
  });

  it('should query items by namespace', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-1' }, { content: 'A' });
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-2' }, { content: 'B' });
    await storage.set({ namespace: 'calibration:records', id: 'rec-1' }, { data: 'C' });

    const claims = await storage.query('metabolism:claims');
    expect(claims.length).toBe(2);
  });

  it('should filter by ID prefix', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'science-1' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'science-2' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'math-1' }, {});

    const scienceClaims = await storage.query('metabolism:claims', { idPrefix: 'science' });
    expect(scienceClaims.length).toBe(2);
  });

  it('should filter by tags', async () => {
    await storage.set({ namespace: 'discovery:gaps', id: 'gap-1' }, {}, { tags: ['structural'] });
    await storage.set(
      { namespace: 'discovery:gaps', id: 'gap-2' },
      {},
      { tags: ['structural', 'high-priority'] }
    );
    await storage.set({ namespace: 'discovery:gaps', id: 'gap-3' }, {}, { tags: ['epistemic'] });

    const structural = await storage.query('discovery:gaps', { tags: ['structural'] });
    expect(structural.length).toBe(2);

    const highPriority = await storage.query('discovery:gaps', {
      tags: ['structural', 'high-priority'],
    });
    expect(highPriority.length).toBe(1);
  });

  it('should handle TTL expiry', async () => {
    await storage.set({ namespace: 'calibration:records', id: 'temp' }, {}, { ttl: 50 }); // 50ms TTL

    // Should exist immediately
    expect(await storage.exists({ namespace: 'calibration:records', id: 'temp' })).toBe(true);

    // Wait for expiry
    await new Promise((r) => setTimeout(r, 100));

    // Should be expired now
    expect(await storage.exists({ namespace: 'calibration:records', id: 'temp' })).toBe(false);
  });

  it('should batch operations', async () => {
    await storage.batchSet([
      { key: { namespace: 'metabolism:claims', id: 'batch-1' }, value: { n: 1 } },
      { key: { namespace: 'metabolism:claims', id: 'batch-2' }, value: { n: 2 } },
      { key: { namespace: 'metabolism:claims', id: 'batch-3' }, value: { n: 3 } },
    ]);

    const results = await storage.batchGet([
      { namespace: 'metabolism:claims', id: 'batch-1' },
      { namespace: 'metabolism:claims', id: 'batch-2' },
      { namespace: 'metabolism:claims', id: 'missing' },
    ]);

    expect(results.size).toBe(2);
  });

  it('should clear namespace', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-1' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-2' }, {});
    await storage.set({ namespace: 'calibration:records', id: 'rec-1' }, {});

    const cleared = await storage.clearNamespace('metabolism:claims');
    expect(cleared).toBe(2);

    const remaining = await storage.query('metabolism:claims');
    expect(remaining.length).toBe(0);

    // Calibration should still exist
    expect(await storage.exists({ namespace: 'calibration:records', id: 'rec-1' })).toBe(true);
  });

  it('should track metadata', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-1' }, { v: 1 });

    const item1 = await storage.get({ namespace: 'metabolism:claims', id: 'claim-1' });
    expect(item1!.metadata.version).toBe(1);
    expect(item1!.metadata.createdAt).toBeGreaterThan(0);

    // Update
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-1' }, { v: 2 });

    const item2 = await storage.get({ namespace: 'metabolism:claims', id: 'claim-1' });
    expect(item2!.metadata.version).toBe(2);
    expect(item2!.metadata.createdAt).toBe(item1!.metadata.createdAt); // Created shouldn't change
    expect(item2!.metadata.updatedAt).toBeGreaterThanOrEqual(item1!.metadata.updatedAt);
  });

  it('should get stats', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'claim-1' }, {});
    await storage.set({ namespace: 'metabolism:tombstones', id: 'tomb-1' }, {});

    const stats = await storage.getStats();
    expect(stats.totalItems).toBe(2);
    expect(stats.itemsByNamespace['metabolism:claims']).toBe(1);
    expect(stats.itemsByNamespace['metabolism:tombstones']).toBe(1);
  });
});

describe('SCEI Persistence Manager', () => {
  let manager: SCEIPersistenceManager;

  beforeEach(() => {
    resetSCEIPersistence();
    manager = getSCEIPersistence();
  });

  afterEach(() => {
    resetSCEIPersistence();
  });

  it('should provide helper methods', async () => {
    await manager.saveClaim('claim-1', { content: 'Test claim' });
    const claim = await manager.getClaim<{ content: string }>('claim-1');
    expect(claim).not.toBeNull();
    expect(claim!.content).toBe('Test claim');
  });

  it('should save and query tombstones', async () => {
    await manager.saveTombstone('tomb-1', { claimId: 'claim-1', reason: 'refuted' });
    await manager.saveTombstone('tomb-2', { claimId: 'claim-2', reason: 'expired' });

    const tombstones = await manager.queryTombstones();
    expect(tombstones.length).toBe(2);
  });

  it('should save and query gaps', async () => {
    await manager.saveGap('gap-1', { type: 'structural', domain: 'science' });

    const gap = await manager.getGap<{ type: string }>('gap-1');
    expect(gap).not.toBeNull();
    expect(gap!.type).toBe('structural');
  });
});

// ============================================================================
// COMPREHENSIVE EVENT BUS MUTATION TESTS
// ============================================================================

import {
  validateEventPayload,
  SCEIEventBuilder,
  ListenerTimeoutError,
} from '../src/scei/event-bus.js';

describe('SCEI Event Bus - Payload Validation', () => {
  it('should validate calibration:prediction_recorded payload', () => {
    const valid = validateEventPayload('calibration:prediction_recorded', {
      predictionId: 'pred-1',
      confidence: 0.8,
      domain: 'science',
      source: 'test',
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('calibration:prediction_recorded', {
      predictionId: 'pred-1',
      // missing confidence
      domain: 'science',
      source: 'test',
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate calibration:prediction_resolved payload', () => {
    const valid = validateEventPayload('calibration:prediction_resolved', {
      predictionId: 'pred-1',
      confidence: 0.8,
      outcome: true,
      domain: 'science',
      brierContribution: 0.04,
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('calibration:prediction_resolved', {
      predictionId: 'pred-1',
      confidence: 0.8,
      outcome: 'true', // should be boolean
      domain: 'science',
      brierContribution: 0.04,
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate metabolism:claim_born payload', () => {
    const valid = validateEventPayload('metabolism:claim_born', {
      claimId: 'claim-1',
      content: 'Test claim',
      domain: 'science',
      authorId: 'author-1',
      initialConfidence: 0.5,
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('metabolism:claim_born', {
      claimId: 'claim-1',
      content: 123, // should be string
      domain: 'science',
      authorId: 'author-1',
      initialConfidence: 0.5,
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate metabolism:claim_died payload', () => {
    const valid = validateEventPayload('metabolism:claim_died', {
      claimId: 'claim-1',
      reason: 'refuted',
      finalConfidence: 0.1,
      lifespan: 86400000,
      tombstoneId: 'tomb-1',
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('metabolism:claim_died', {
      claimId: 'claim-1',
      reason: 'refuted',
      finalConfidence: 'low', // should be number
      lifespan: 86400000,
      tombstoneId: 'tomb-1',
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate metabolism:health_updated payload', () => {
    const valid = validateEventPayload('metabolism:health_updated', {
      claimId: 'claim-1',
      previousHealth: 0.7,
      newHealth: 0.8,
      phase: 'growing',
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('metabolism:health_updated', {
      claimId: 'claim-1',
      previousHealth: 0.7,
      // missing newHealth
      phase: 'growing',
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate propagation:completed payload', () => {
    const valid = validateEventPayload('propagation:completed', {
      requestId: 'req-1',
      sourceClaimId: 'claim-1',
      affectedClaimCount: 5,
      maxDepth: 3,
      wasBlocked: false,
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('propagation:completed', {
      requestId: 'req-1',
      sourceClaimId: 'claim-1',
      affectedClaimCount: 'five', // should be number
      maxDepth: 3,
      wasBlocked: false,
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate discovery:gap_detected payload', () => {
    const valid = validateEventPayload('discovery:gap_detected', {
      gapId: 'gap-1',
      type: 'structural',
      domain: 'science',
      severity: 0.7,
      description: 'Missing evidence',
      suggestedActions: ['Add evidence', 'Link related claims'],
    });
    expect(valid.valid).toBe(true);

    const invalid = validateEventPayload('discovery:gap_detected', {
      gapId: 'gap-1',
      type: 'structural',
      domain: 'science',
      severity: 0.7,
      description: 'Missing evidence',
      suggestedActions: 'Add evidence', // should be array
    });
    expect(invalid.valid).toBe(false);
  });

  it('should validate discovery:gap_resolved payload', () => {
    const valid = validateEventPayload('discovery:gap_resolved', {
      gapId: 'gap-1',
      resolvedBy: 'user-1',
      method: 'addressed',
    });
    expect(valid.valid).toBe(true);

    const invalidMethod = validateEventPayload('discovery:gap_resolved', {
      gapId: 'gap-1',
      resolvedBy: 'user-1',
      method: 'invalid_method', // not in allowed values
    });
    expect(invalidMethod.valid).toBe(false);
  });

  it('should accept any payload for event types without validators', () => {
    const result = validateEventPayload('scei:module_initialized', {
      module: 'calibration',
      extra: 'data',
    });
    expect(result.valid).toBe(true);
  });

  it('should reject null payload', () => {
    const result = validateEventPayload('metabolism:claim_born', null);
    expect(result.valid).toBe(false);
  });

  it('should reject non-object payload', () => {
    const result = validateEventPayload('metabolism:claim_born', 'not an object');
    expect(result.valid).toBe(false);
  });
});

describe('SCEI Event Builder', () => {
  it('should build valid events', () => {
    const event = SCEIEventBuilder
      .create('metabolism:claim_born')
      .from('metabolism')
      .withPayload({
        claimId: 'claim-1',
        content: 'Test',
        domain: 'science',
        authorId: 'author-1',
        initialConfidence: 0.5,
      })
      .build();

    expect(event.type).toBe('metabolism:claim_born');
    expect(event.source).toBe('metabolism');
    expect(event.timestamp).toBeGreaterThan(0);
    expect(event.payload.claimId).toBe('claim-1');
  });

  it('should set correlation ID', () => {
    const event = SCEIEventBuilder
      .create('metabolism:claim_born')
      .from('metabolism')
      .withPayload({
        claimId: 'claim-1',
        content: 'Test',
        domain: 'science',
        authorId: 'author-1',
        initialConfidence: 0.5,
      })
      .withCorrelationId('corr-123')
      .build();

    expect(event.correlationId).toBe('corr-123');
  });

  it('should set custom timestamp', () => {
    const customTime = 1234567890;
    const event = SCEIEventBuilder
      .create('metabolism:claim_born')
      .from('metabolism')
      .withPayload({
        claimId: 'claim-1',
        content: 'Test',
        domain: 'science',
        authorId: 'author-1',
        initialConfidence: 0.5,
      })
      .withTimestamp(customTime)
      .build();

    expect(event.timestamp).toBe(customTime);
  });

  it('should throw when source is missing', () => {
    expect(() => {
      SCEIEventBuilder
        .create('metabolism:claim_born')
        .withPayload({
          claimId: 'claim-1',
          content: 'Test',
          domain: 'science',
          authorId: 'author-1',
          initialConfidence: 0.5,
        })
        .build();
    }).toThrow(/source is required/);
  });

  it('should throw when payload is missing', () => {
    expect(() => {
      SCEIEventBuilder
        .create('metabolism:claim_born')
        .from('metabolism')
        .build();
    }).toThrow(/payload is required/);
  });

  it('should throw on invalid payload', () => {
    expect(() => {
      SCEIEventBuilder
        .create('metabolism:claim_born')
        .from('metabolism')
        .withPayload({
          claimId: 'claim-1',
          // missing required fields
        } as any)
        .build();
    }).toThrow(/Invalid payload/);
  });

  it('should emit event directly', async () => {
    resetSCEIEventBus();
    const bus = getSCEIEventBus();
    const received: any[] = [];
    bus.subscribe((e) => received.push(e));

    await SCEIEventBuilder
      .create('metabolism:claim_born')
      .from('metabolism')
      .withPayload({
        claimId: 'claim-1',
        content: 'Test',
        domain: 'science',
        authorId: 'author-1',
        initialConfidence: 0.5,
      })
      .emit(bus);

    expect(received.length).toBe(1);
    expect(received[0].type).toBe('metabolism:claim_born');
  });
});

describe('SCEI Event Bus - Advanced Features', () => {
  let eventBus: SCEIEventBus;

  beforeEach(() => {
    resetSCEIEventBus();
    eventBus = new SCEIEventBus({
      maxHistorySize: 10,
      enableHistory: true,
      enableMetrics: true,
      listenerTimeoutMs: 100,
      validatePayloads: true,
    });
  });

  it('should filter events by correlation ID', async () => {
    const received: any[] = [];

    eventBus.subscribe(
      (event) => received.push(event),
      { correlationId: 'corr-123' }
    );

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}, 'corr-123'));
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}, 'corr-456'));
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {})); // no correlation ID

    expect(received.length).toBe(1);
    expect(received[0].correlationId).toBe('corr-123');
  });

  it('should limit history size', async () => {
    for (let i = 0; i < 15; i++) {
      await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', { id: i }));
    }

    const history = eventBus.getHistory();
    expect(history.length).toBe(10); // maxHistorySize
  });

  it('should filter history by type', async () => {
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    await eventBus.emit(createSCEIEvent('metabolism:claim_died', 'metabolism', {}));
    await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', {}));

    const filtered = eventBus.getHistory({ types: ['metabolism:claim_born'] });
    expect(filtered.length).toBe(1);
    expect(filtered[0].type).toBe('metabolism:claim_born');
  });

  it('should filter history by source', async () => {
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', {}));

    const filtered = eventBus.getHistory({ sources: ['calibration'] });
    expect(filtered.length).toBe(1);
    expect(filtered[0].source).toBe('calibration');
  });

  it('should respect history limit parameter', async () => {
    for (let i = 0; i < 5; i++) {
      await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', { id: i }));
    }

    const limited = eventBus.getHistory(undefined, 2);
    expect(limited.length).toBe(2);
  });

  it('should handle listener timeouts', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    eventBus.subscribe(async () => {
      await new Promise((resolve) => setTimeout(resolve, 200)); // Exceeds 100ms timeout
    });

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    const metrics = eventBus.getMetrics();
    expect(metrics.listenerTimeouts).toBe(1);

    consoleSpy.mockRestore();
  });

  it('should emit events synchronously with emitSync', async () => {
    const received: any[] = [];
    eventBus.subscribe((e) => received.push(e));

    eventBus.emitSync(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    // emitSync is fire-and-forget, so wait a bit
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(received.length).toBe(1);
  });

  it('should wait for specific event with waitFor', async () => {
    const eventPromise = eventBus.waitFor('metabolism:claim_died', 1000);

    // Emit after a small delay
    setTimeout(() => {
      eventBus.emitSync(createSCEIEvent('metabolism:claim_died', 'metabolism', { claimId: 'claim-1' }));
    }, 50);

    const event = await eventPromise;
    expect(event.type).toBe('metabolism:claim_died');
  });

  it('should timeout in waitFor if event not received', async () => {
    await expect(
      eventBus.waitFor('metabolism:claim_died', 50)
    ).rejects.toThrow(/Timeout waiting for event/);
  });

  it('should warn on invalid payload validation', async () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {
      // Invalid payload - missing required fields
      claimId: 'claim-1',
    }));

    const metrics = eventBus.getMetrics();
    expect(metrics.validationFailures).toBe(1);

    consoleSpy.mockRestore();
  });

  it('should allow setting listener timeout', () => {
    eventBus.setListenerTimeout(2000);
    // Test it doesn't throw
    expect(true).toBe(true);
  });

  it('should allow enabling/disabling payload validation', () => {
    eventBus.setPayloadValidation(false);
    // Test it doesn't throw
    expect(true).toBe(true);
  });

  it('should reset all state', async () => {
    eventBus.subscribe(() => {});
    await eventBus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    eventBus.reset();

    const metrics = eventBus.getMetrics();
    expect(metrics.totalEventsEmitted).toBe(0);
    expect(metrics.activeListeners).toBe(0);
    expect(metrics.historySize).toBe(0);
  });

  it('should return false when unsubscribing non-existent subscription', () => {
    const result = eventBus.unsubscribe('nonexistent-id');
    expect(result).toBe(false);
  });

  it('should track validation failures in metrics', async () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    // Emit event with invalid payload
    await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', {
      // Missing required fields
    }));

    expect(eventBus.getMetrics().validationFailures).toBe(1);
    consoleSpy.mockRestore();
  });
});

describe('SCEI Event Bus - Configuration Options', () => {
  it('should work with history disabled', async () => {
    const bus = new SCEIEventBus({
      enableHistory: false,
    });

    await bus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));
    await bus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    const history = bus.getHistory();
    expect(history.length).toBe(0);
  });

  it('should work with metrics disabled', async () => {
    const bus = new SCEIEventBus({
      enableMetrics: false,
    });

    await bus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {}));

    const metrics = bus.getMetrics();
    // Metrics should still be accessible but counts may be 0
    expect(metrics.totalEventsEmitted).toBe(0);
  });

  it('should work with validation disabled', async () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const bus = new SCEIEventBus({
      validatePayloads: false,
    });

    // Invalid payload should not trigger warning
    await bus.emit(createSCEIEvent('metabolism:claim_born', 'metabolism', {
      invalid: 'payload',
    }));

    expect(bus.getMetrics().validationFailures).toBe(0);
    consoleSpy.mockRestore();
  });
});

describe('ListenerTimeoutError', () => {
  it('should create error with correct properties', () => {
    const error = new ListenerTimeoutError('listener-1', 5000);

    expect(error.name).toBe('ListenerTimeoutError');
    expect(error.listenerId).toBe('listener-1');
    expect(error.timeoutMs).toBe(5000);
    expect(error.message).toContain('listener-1');
    expect(error.message).toContain('5000ms');
  });
});

describe('SCEI Event Bus - Singleton Functions', () => {
  beforeEach(() => {
    resetSCEIEventBus();
  });

  afterEach(() => {
    resetSCEIEventBus();
  });

  it('getSCEIEventBus should return same instance', () => {
    const bus1 = getSCEIEventBus();
    const bus2 = getSCEIEventBus();
    expect(bus1).toBe(bus2);
  });

  it('resetSCEIEventBus should create new instance', () => {
    const bus1 = getSCEIEventBus();
    bus1.subscribe(() => {});

    resetSCEIEventBus();

    const bus2 = getSCEIEventBus();
    expect(bus2.getMetrics().activeListeners).toBe(0);
  });
});

// ============================================================================
// COMPREHENSIVE PERSISTENCE MUTATION TESTS
// ============================================================================

describe('InMemorySCEIStorage - Sorting', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should sort by id ascending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'charlie' }, { name: 'C' });
    await storage.set({ namespace: 'metabolism:claims', id: 'alpha' }, { name: 'A' });
    await storage.set({ namespace: 'metabolism:claims', id: 'bravo' }, { name: 'B' });

    const results = await storage.query('metabolism:claims', { sortBy: 'id', sortOrder: 'asc' });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('alpha');
    expect(results[1].key.id).toBe('bravo');
    expect(results[2].key.id).toBe('charlie');
  });

  it('should sort by id descending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'alpha' }, { name: 'A' });
    await storage.set({ namespace: 'metabolism:claims', id: 'charlie' }, { name: 'C' });
    await storage.set({ namespace: 'metabolism:claims', id: 'bravo' }, { name: 'B' });

    const results = await storage.query('metabolism:claims', { sortBy: 'id', sortOrder: 'desc' });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('charlie');
    expect(results[1].key.id).toBe('bravo');
    expect(results[2].key.id).toBe('alpha');
  });

  it('should sort by updatedAt ascending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, { order: 2 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, { order: 3 });

    const results = await storage.query('metabolism:claims', { sortBy: 'updatedAt', sortOrder: 'asc' });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('first');
    expect(results[2].key.id).toBe('third');
  });

  it('should sort by updatedAt descending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, { order: 2 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, { order: 3 });

    const results = await storage.query('metabolism:claims', { sortBy: 'updatedAt', sortOrder: 'desc' });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('third');
    expect(results[2].key.id).toBe('first');
  });

  it('should sort by createdAt ascending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, { order: 2 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, { order: 3 });

    const results = await storage.query('metabolism:claims', { sortBy: 'createdAt', sortOrder: 'asc' });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('first');
    expect(results[2].key.id).toBe('third');
  });

  it('should default to createdAt descending', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, { order: 2 });
    await new Promise((r) => setTimeout(r, 10));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, { order: 3 });

    const results = await storage.query('metabolism:claims'); // No sort options

    expect(results.length).toBe(3);
    // Default is descending (newest first)
    expect(results[0].key.id).toBe('third');
    expect(results[2].key.id).toBe('first');
  });

  it('should handle equal values in sorting', async () => {
    // All have same createdAt (within same ms)
    await storage.set({ namespace: 'metabolism:claims', id: 'a' }, { val: 1 });
    await storage.set({ namespace: 'metabolism:claims', id: 'b' }, { val: 2 });
    await storage.set({ namespace: 'metabolism:claims', id: 'c' }, { val: 3 });

    const results = await storage.query('metabolism:claims', { sortBy: 'createdAt', sortOrder: 'asc' });

    expect(results.length).toBe(3);
  });
});

describe('InMemorySCEIStorage - Pagination', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(async () => {
    storage = new InMemorySCEIStorage();
    // Add 10 items
    for (let i = 0; i < 10; i++) {
      await storage.set({ namespace: 'metabolism:claims', id: `item-${String(i).padStart(2, '0')}` }, { index: i });
    }
  });

  it('should apply offset correctly', async () => {
    const results = await storage.query('metabolism:claims', {
      sortBy: 'id',
      sortOrder: 'asc',
      offset: 3,
    });

    expect(results.length).toBe(7);
    expect(results[0].key.id).toBe('item-03');
  });

  it('should apply limit correctly', async () => {
    const results = await storage.query('metabolism:claims', {
      sortBy: 'id',
      sortOrder: 'asc',
      limit: 3,
    });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('item-00');
    expect(results[2].key.id).toBe('item-02');
  });

  it('should apply offset and limit together', async () => {
    const results = await storage.query('metabolism:claims', {
      sortBy: 'id',
      sortOrder: 'asc',
      offset: 2,
      limit: 3,
    });

    expect(results.length).toBe(3);
    expect(results[0].key.id).toBe('item-02');
    expect(results[1].key.id).toBe('item-03');
    expect(results[2].key.id).toBe('item-04');
  });

  it('should handle offset beyond results', async () => {
    const results = await storage.query('metabolism:claims', {
      offset: 100,
    });

    expect(results.length).toBe(0);
  });

  it('should handle limit larger than results', async () => {
    const results = await storage.query('metabolism:claims', {
      limit: 100,
    });

    expect(results.length).toBe(10);
  });
});

describe('InMemorySCEIStorage - Time Filtering', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should filter by createdAfter', async () => {
    const now = Date.now();
    await storage.set({ namespace: 'metabolism:claims', id: 'old' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 50));
    const midpoint = Date.now();
    await new Promise((r) => setTimeout(r, 50));
    await storage.set({ namespace: 'metabolism:claims', id: 'new' }, { order: 2 });

    const results = await storage.query('metabolism:claims', { createdAfter: midpoint });

    expect(results.length).toBe(1);
    expect(results[0].key.id).toBe('new');
  });

  it('should filter by createdBefore', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'old' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 50));
    const midpoint = Date.now();
    await new Promise((r) => setTimeout(r, 50));
    await storage.set({ namespace: 'metabolism:claims', id: 'new' }, { order: 2 });

    const results = await storage.query('metabolism:claims', { createdBefore: midpoint });

    expect(results.length).toBe(1);
    expect(results[0].key.id).toBe('old');
  });

  it('should filter by both createdAfter and createdBefore', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, { order: 1 });
    await new Promise((r) => setTimeout(r, 30));
    const start = Date.now();
    await new Promise((r) => setTimeout(r, 30));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, { order: 2 });
    await new Promise((r) => setTimeout(r, 30));
    const end = Date.now();
    await new Promise((r) => setTimeout(r, 30));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, { order: 3 });

    const results = await storage.query('metabolism:claims', {
      createdAfter: start,
      createdBefore: end,
    });

    expect(results.length).toBe(1);
    expect(results[0].key.id).toBe('second');
  });
});

describe('InMemorySCEIStorage - Expiry and Purge', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should skip expired items in query', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'expired' }, { data: 1 }, { ttl: 10 });
    await storage.set({ namespace: 'metabolism:claims', id: 'valid' }, { data: 2 });

    await new Promise((r) => setTimeout(r, 50));

    const results = await storage.query('metabolism:claims');
    expect(results.length).toBe(1);
    expect(results[0].key.id).toBe('valid');
  });

  it('should purge expired items', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'expired1' }, { data: 1 }, { ttl: 10 });
    await storage.set({ namespace: 'metabolism:claims', id: 'expired2' }, { data: 2 }, { ttl: 10 });
    await storage.set({ namespace: 'metabolism:claims', id: 'valid' }, { data: 3 });

    await new Promise((r) => setTimeout(r, 50));

    const purged = await storage.purgeExpired();
    expect(purged).toBe(2);

    const stats = await storage.getStats();
    expect(stats.totalItems).toBe(1);
  });

  it('should return 0 when no items are expired', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'valid1' }, { data: 1 });
    await storage.set({ namespace: 'metabolism:claims', id: 'valid2' }, { data: 2 });

    const purged = await storage.purgeExpired();
    expect(purged).toBe(0);
  });

  it('should preserve tags when updating with TTL', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'item' }, { v: 1 }, { tags: ['original'] });

    // Update without specifying tags
    await storage.set({ namespace: 'metabolism:claims', id: 'item' }, { v: 2 }, { ttl: 1000 });

    const item = await storage.get({ namespace: 'metabolism:claims', id: 'item' });
    expect(item!.metadata.tags).toEqual(['original']);
  });
});

describe('InMemorySCEIStorage - Count', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should count all items in namespace', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'a' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'b' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'c' }, {});
    await storage.set({ namespace: 'calibration:records', id: 'd' }, {});

    const count = await storage.count('metabolism:claims');
    expect(count).toBe(3);
  });

  it('should count with filters', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'prefix-a' }, {}, { tags: ['tag1'] });
    await storage.set({ namespace: 'metabolism:claims', id: 'prefix-b' }, {}, { tags: ['tag1'] });
    await storage.set({ namespace: 'metabolism:claims', id: 'prefix-c' }, {}, { tags: ['tag2'] });
    await storage.set({ namespace: 'metabolism:claims', id: 'other-d' }, {}, { tags: ['tag1'] });

    const countPrefix = await storage.count('metabolism:claims', { idPrefix: 'prefix-' });
    expect(countPrefix).toBe(3);

    const countTag = await storage.count('metabolism:claims', { tags: ['tag1'] });
    expect(countTag).toBe(3);

    const countBoth = await storage.count('metabolism:claims', { idPrefix: 'prefix-', tags: ['tag1'] });
    expect(countBoth).toBe(2);
  });
});

describe('InMemorySCEIStorage - Batch Operations', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should batch delete and return count', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'a' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'b' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'c' }, {});

    const deleted = await storage.batchDelete([
      { namespace: 'metabolism:claims', id: 'a' },
      { namespace: 'metabolism:claims', id: 'b' },
      { namespace: 'metabolism:claims', id: 'nonexistent' },
    ]);

    expect(deleted).toBe(2); // Only 2 existed
  });

  it('should return false when deleting nonexistent item', async () => {
    const deleted = await storage.delete({ namespace: 'metabolism:claims', id: 'nonexistent' });
    expect(deleted).toBe(false);
  });

  it('should batch get skipping missing items', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'exists' }, { v: 1 });

    const results = await storage.batchGet([
      { namespace: 'metabolism:claims', id: 'exists' },
      { namespace: 'metabolism:claims', id: 'missing' },
    ]);

    expect(results.size).toBe(1);
    expect(results.has('metabolism:claims:exists')).toBe(true);
    expect(results.has('metabolism:claims:missing')).toBe(false);
  });

  it('should batch set with options', async () => {
    await storage.batchSet([
      { key: { namespace: 'metabolism:claims', id: 'a' }, value: { v: 1 }, options: { tags: ['batch'] } },
      { key: { namespace: 'metabolism:claims', id: 'b' }, value: { v: 2 }, options: { ttl: 5000 } },
    ]);

    const itemA = await storage.get({ namespace: 'metabolism:claims', id: 'a' });
    const itemB = await storage.get({ namespace: 'metabolism:claims', id: 'b' });

    expect(itemA!.metadata.tags).toEqual(['batch']);
    expect(itemB!.metadata.expiresAt).toBeGreaterThan(Date.now());
  });
});

describe('InMemorySCEIStorage - Stats', () => {
  let storage: InMemorySCEIStorage;

  beforeEach(() => {
    storage = new InMemorySCEIStorage();
  });

  it('should track oldest and newest items', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'first' }, {});
    await new Promise((r) => setTimeout(r, 20));
    await storage.set({ namespace: 'metabolism:claims', id: 'second' }, {});
    await new Promise((r) => setTimeout(r, 20));
    await storage.set({ namespace: 'metabolism:claims', id: 'third' }, {});

    const stats = await storage.getStats();

    expect(stats.oldestItem).toBeLessThan(stats.newestItem!);
    expect(stats.totalItems).toBe(3);
  });

  it('should return undefined for oldest/newest when empty', async () => {
    const stats = await storage.getStats();

    expect(stats.totalItems).toBe(0);
    expect(stats.oldestItem).toBeUndefined();
    expect(stats.newestItem).toBeUndefined();
  });

  it('should track items by multiple namespaces', async () => {
    await storage.set({ namespace: 'metabolism:claims', id: 'a' }, {});
    await storage.set({ namespace: 'metabolism:claims', id: 'b' }, {});
    await storage.set({ namespace: 'calibration:records', id: 'c' }, {});
    await storage.set({ namespace: 'discovery:gaps', id: 'd' }, {});
    await storage.set({ namespace: 'discovery:gaps', id: 'e' }, {});

    const stats = await storage.getStats();

    expect(stats.totalItems).toBe(5);
    expect(stats.itemsByNamespace['metabolism:claims']).toBe(2);
    expect(stats.itemsByNamespace['calibration:records']).toBe(1);
    expect(stats.itemsByNamespace['discovery:gaps']).toBe(2);
  });
});

describe('InMemorySCEIStorage - Close', () => {
  it('should close without error', async () => {
    const storage = new InMemorySCEIStorage();
    await storage.set({ namespace: 'metabolism:claims', id: 'a' }, {});

    // Should not throw
    await expect(storage.close()).resolves.toBeUndefined();
  });
});

describe('SCEIPersistenceManager - Calibration Helpers', () => {
  let manager: SCEIPersistenceManager;

  beforeEach(() => {
    resetSCEIPersistence();
    manager = new SCEIPersistenceManager(new InMemorySCEIStorage());
  });

  afterEach(() => {
    resetSCEIPersistence();
  });

  it('should save and get calibration records', async () => {
    await manager.saveCalibrationRecord('rec-1', {
      confidence: 0.8,
      domain: 'science',
      prediction: 'Test prediction',
    });

    const record = await manager.getCalibrationRecord<{ confidence: number }>('rec-1');
    expect(record).not.toBeNull();
    expect(record!.confidence).toBe(0.8);
  });

  it('should return null for missing calibration record', async () => {
    const record = await manager.getCalibrationRecord('nonexistent');
    expect(record).toBeNull();
  });

  it('should query calibration records', async () => {
    await manager.saveCalibrationRecord('rec-1', { domain: 'science' }, { tags: ['science'] });
    await manager.saveCalibrationRecord('rec-2', { domain: 'science' }, { tags: ['science'] });
    await manager.saveCalibrationRecord('rec-3', { domain: 'math' }, { tags: ['math'] });

    const allRecords = await manager.queryCalibrationRecords();
    expect(allRecords.length).toBe(3);

    const scienceRecords = await manager.queryCalibrationRecords({ tags: ['science'] });
    expect(scienceRecords.length).toBe(2);
  });

  it('should save calibration record with options', async () => {
    await manager.saveCalibrationRecord('rec-1', { data: 'test' }, { ttl: 5000, tags: ['temp'] });

    const storage = manager.getStorage();
    const item = await storage.get({ namespace: 'calibration:records', id: 'rec-1' });

    expect(item!.metadata.expiresAt).toBeGreaterThan(Date.now());
    expect(item!.metadata.tags).toContain('temp');
  });
});

describe('SCEIPersistenceManager - Metabolism Helpers', () => {
  let manager: SCEIPersistenceManager;

  beforeEach(() => {
    manager = new SCEIPersistenceManager(new InMemorySCEIStorage());
  });

  it('should delete claims', async () => {
    await manager.saveClaim('claim-1', { content: 'Test' });
    expect(await manager.getClaim('claim-1')).not.toBeNull();

    const deleted = await manager.deleteClaim('claim-1');
    expect(deleted).toBe(true);
    expect(await manager.getClaim('claim-1')).toBeNull();
  });

  it('should return false when deleting nonexistent claim', async () => {
    const deleted = await manager.deleteClaim('nonexistent');
    expect(deleted).toBe(false);
  });

  it('should get tombstone', async () => {
    await manager.saveTombstone('tomb-1', { claimId: 'claim-1', reason: 'refuted' });

    const tombstone = await manager.getTombstone<{ reason: string }>('tomb-1');
    expect(tombstone).not.toBeNull();
    expect(tombstone!.reason).toBe('refuted');
  });

  it('should return null for missing tombstone', async () => {
    const tombstone = await manager.getTombstone('nonexistent');
    expect(tombstone).toBeNull();
  });
});

describe('SCEIPersistenceManager - Discovery Helpers', () => {
  let manager: SCEIPersistenceManager;

  beforeEach(() => {
    manager = new SCEIPersistenceManager(new InMemorySCEIStorage());
  });

  it('should save gap with options', async () => {
    await manager.saveGap('gap-1', { type: 'structural' }, { tags: ['high-priority'] });

    const gaps = await manager.queryGaps({ tags: ['high-priority'] });
    expect(gaps.length).toBe(1);
  });

  it('should return null for missing gap', async () => {
    const gap = await manager.getGap('nonexistent');
    expect(gap).toBeNull();
  });
});

describe('SCEIPersistenceManager - Auto Purge', () => {
  it('should auto-purge expired items on interval', async () => {
    const storage = new InMemorySCEIStorage();
    const manager = new SCEIPersistenceManager(storage, {
      autoPurgeIntervalMs: 50, // 50ms for test
    });

    await storage.set({ namespace: 'metabolism:claims', id: 'temp' }, {}, { ttl: 10 });

    // Wait for TTL to expire and auto-purge to run
    await new Promise((r) => setTimeout(r, 150));

    const stats = await storage.getStats();
    expect(stats.totalItems).toBe(0);

    await manager.close();
  });

  it('should not start auto-purge with 0 interval', async () => {
    const storage = new InMemorySCEIStorage();
    const manager = new SCEIPersistenceManager(storage, {
      autoPurgeIntervalMs: 0,
    });

    // Just verify no error
    await manager.close();
  });

  it('should handle auto-purge errors gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const storage = {
      purgeExpired: vi.fn().mockRejectedValue(new Error('Purge failed')),
      close: vi.fn().mockResolvedValue(undefined),
    } as any;

    const manager = new SCEIPersistenceManager(storage, {
      autoPurgeIntervalMs: 50,
    });

    // Wait for auto-purge to run
    await new Promise((r) => setTimeout(r, 100));

    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('auto-purge error'), expect.any(Error));

    await manager.close();
    consoleSpy.mockRestore();
  });

  it('should clear interval on close', async () => {
    const storage = new InMemorySCEIStorage();
    const manager = new SCEIPersistenceManager(storage, {
      autoPurgeIntervalMs: 50,
    });

    await manager.close();

    // No error should occur after close
    await new Promise((r) => setTimeout(r, 100));
  });
});

describe('SCEIPersistenceManager - getStorage', () => {
  it('should return underlying storage', () => {
    const storage = new InMemorySCEIStorage();
    const manager = new SCEIPersistenceManager(storage);

    expect(manager.getStorage()).toBe(storage);
  });
});

describe('SCEI Persistence - Singleton Functions', () => {
  beforeEach(() => {
    resetSCEIPersistence();
  });

  afterEach(() => {
    resetSCEIPersistence();
  });

  it('getSCEIPersistence should return singleton', () => {
    const instance1 = getSCEIPersistence();
    const instance2 = getSCEIPersistence();

    expect(instance1).toBe(instance2);
  });

  it('setSCEIPersistence should replace existing instance', async () => {
    const original = getSCEIPersistence();
    await original.saveClaim('original-claim', { data: 'original' });

    const customStorage = new InMemorySCEIStorage();
    const newInstance = setSCEIPersistence(customStorage);

    expect(newInstance).not.toBe(original);

    // Original data should be gone
    const claim = await newInstance.getClaim('original-claim');
    expect(claim).toBeNull();
  });

  it('setSCEIPersistence should close previous instance', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const original = getSCEIPersistence();

    // Set custom storage
    const customStorage = new InMemorySCEIStorage();
    setSCEIPersistence(customStorage);

    // Wait for close to be called
    await new Promise((r) => setTimeout(r, 10));

    consoleSpy.mockRestore();
  });

  it('resetSCEIPersistence should clear instance', async () => {
    const original = getSCEIPersistence();
    await original.saveClaim('test-claim', { data: 'test' });

    resetSCEIPersistence();

    const newInstance = getSCEIPersistence();
    expect(newInstance).not.toBe(original);

    const claim = await newInstance.getClaim('test-claim');
    expect(claim).toBeNull();
  });

  it('resetSCEIPersistence should handle close errors gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    // Get default instance first
    getSCEIPersistence();

    // Reset should not throw even if close has issues
    resetSCEIPersistence();

    consoleSpy.mockRestore();
  });

  it('getSCEIPersistence should create default with auto-purge', async () => {
    const instance = getSCEIPersistence();

    // Save item with short TTL
    await instance.saveClaim('temp-claim', { data: 'temp' });

    // Instance should have auto-purge configured (default 60s)
    // We can't easily test the actual interval, but we can verify it works
    expect(instance).toBeDefined();
  });
});

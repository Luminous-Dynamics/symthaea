/**
 * SCEI (Self-Correcting Epistemic Infrastructure) Benchmarks
 *
 * Performance tests for:
 * - Event Bus (emit, subscribe, filter)
 * - Metrics Collection (counters, histograms, gauges)
 * - Persistence (CRUD, queries, batch operations)
 * - Validation functions
 * - Numerical stability helpers
 * - Calibration Engine
 * - Metabolism Engine
 * - Propagation Engine
 * - Gap Detector
 */

import { bench, describe, beforeAll, afterAll } from 'vitest';
import {
  // Event Bus
  SCEIEventBus,
  createSCEIEvent,
  resetSCEIEventBus,
  // Metrics
  SCEIMetricsCollector,
  resetSCEIMetrics,
  // Persistence
  InMemorySCEIStorage,
  // Validation
  validateConfidence,
  validateDegradationFactor,
  validatePositiveInteger,
  validateTimestamp,
  validateNonEmptyString,
  validateId,
  combineValidations,
  // Numerical helpers
  safeLog,
  clampConfidence,
  safeDivide,
  safeWeightedAverage,
} from '../src/scei/index.js';
import { CalibrationEngine } from '../src/calibration/engine.js';
import { MetabolismEngine } from '../src/metabolism/engine.js';
import { SafeBeliefPropagator } from '../src/propagation/engine.js';
import { GapDetector } from '../src/discovery/gap-detector.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../src/epistemic/index.js';

// ============================================================================
// EVENT BUS BENCHMARKS
// ============================================================================

describe('SCEI: Event Bus', () => {
  let eventBus: SCEIEventBus;

  beforeAll(() => {
    resetSCEIEventBus();
    eventBus = new SCEIEventBus();
  });

  afterAll(() => {
    resetSCEIEventBus();
  });

  bench('createSCEIEvent()', () => {
    createSCEIEvent('calibration:prediction_recorded', 'calibration', {
      predictionId: 'pred-123',
      confidence: 0.85,
    });
  });

  bench('emit() - single event', async () => {
    const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {
      predictionId: 'pred-123',
    });
    await eventBus.emit(event);
  });

  bench('emit() - with 1 subscriber', async () => {
    const bus = new SCEIEventBus();
    bus.subscribe(() => {
      /* noop */
    });
    const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {});
    await bus.emit(event);
  });

  bench('emit() - with 10 subscribers', async () => {
    const bus = new SCEIEventBus();
    for (let i = 0; i < 10; i++) {
      bus.subscribe(() => {
        /* noop */
      });
    }
    const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {});
    await bus.emit(event);
  });

  bench('emit() - with type filter', async () => {
    const bus = new SCEIEventBus();
    bus.subscribe(() => {}, { types: ['calibration:prediction_recorded'] });
    const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {});
    await bus.emit(event);
  });

  bench('emit() - with source filter', async () => {
    const bus = new SCEIEventBus();
    bus.subscribe(() => {}, { sources: ['calibration'] });
    const event = createSCEIEvent('calibration:prediction_recorded', 'calibration', {});
    await bus.emit(event);
  });

  bench('subscribe/unsubscribe cycle', () => {
    const bus = new SCEIEventBus();
    const sub = bus.subscribe(() => {});
    sub.unsubscribe();
  });

  bench('emit() - 100 events sequentially', async () => {
    const bus = new SCEIEventBus();
    bus.subscribe(() => {});
    for (let i = 0; i < 100; i++) {
      await bus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', { i }));
    }
  });

  bench('getHistory() - 100 events', () => {
    const bus = new SCEIEventBus({ enableHistory: true, maxHistorySize: 100 });
    bus.getHistory();
  });

  bench('getMetrics()', () => {
    eventBus.getMetrics();
  });
});

// ============================================================================
// METRICS COLLECTION BENCHMARKS
// ============================================================================

describe('SCEI: Metrics Collection', () => {
  let metrics: SCEIMetricsCollector;

  beforeAll(() => {
    resetSCEIMetrics();
    metrics = new SCEIMetricsCollector();
  });

  afterAll(() => {
    resetSCEIMetrics();
  });

  bench('incrementCounter() - single', () => {
    metrics.incrementCounter('calibration_predictions_recorded', {});
  });

  bench('incrementCounter() - with labels', () => {
    metrics.incrementCounter('calibration_predictions_recorded', { domain: 'science', region: 'us' });
  });

  bench('incrementCounter() - 100 increments', () => {
    for (let i = 0; i < 100; i++) {
      metrics.incrementCounter('calibration_predictions_recorded', {});
    }
  });

  bench('setGauge()', () => {
    metrics.setGauge('calibration_brier_score', 0.15);
  });

  bench('setGauge() - with labels', () => {
    metrics.setGauge('calibration_brier_score', 0.15, { domain: 'science' });
  });

  bench('recordHistogram()', () => {
    metrics.recordHistogram('calibration_report_duration', 125);
  });

  bench('recordHistogram() - with labels', () => {
    metrics.recordHistogram('calibration_report_duration', 125, { cached: 'false' });
  });

  bench('recordHistogram() - 100 observations', () => {
    for (let i = 0; i < 100; i++) {
      metrics.recordHistogram('calibration_report_duration', Math.random() * 500);
    }
  });

  bench('exportJSON()', () => {
    metrics.exportJSON();
  });

  bench('exportPrometheus()', () => {
    metrics.exportPrometheus();
  });

  bench('reset()', () => {
    const m = new SCEIMetricsCollector();
    m.incrementCounter('calibration_predictions_recorded', {}, 100);
    m.reset();
  });
});

// ============================================================================
// PERSISTENCE BENCHMARKS
// ============================================================================

describe('SCEI: Persistence', () => {
  let storage: InMemorySCEIStorage;
  const namespace = 'calibration:records' as const;

  beforeAll(async () => {
    storage = new InMemorySCEIStorage();
    // Pre-populate with some data
    for (let i = 0; i < 100; i++) {
      await storage.set({ namespace, id: `item-${i}` }, { value: i, data: `test-${i}` });
    }
  });

  bench('set() - single item', async () => {
    await storage.set({ namespace, id: 'bench-item' }, { data: 'test' });
  });

  bench('set() - with TTL', async () => {
    await storage.set({ namespace, id: 'bench-ttl' }, { data: 'test' }, { ttl: 60000 });
  });

  bench('set() - with tags', async () => {
    await storage.set({ namespace, id: 'bench-tags' }, { data: 'test' }, { tags: ['important', 'verified'] });
  });

  bench('get() - existing item', async () => {
    await storage.get({ namespace, id: 'item-50' });
  });

  bench('get() - non-existing item', async () => {
    await storage.get({ namespace, id: 'non-existent' });
  });

  bench('exists() - existing', async () => {
    await storage.exists({ namespace, id: 'item-50' });
  });

  bench('exists() - non-existing', async () => {
    await storage.exists({ namespace, id: 'non-existent' });
  });

  bench('delete()', async () => {
    await storage.set({ namespace, id: 'to-delete' }, { data: 'test' });
    await storage.delete({ namespace, id: 'to-delete' });
  });

  bench('query() - no filters', async () => {
    await storage.query(namespace, { limit: 10 });
  });

  bench('query() - with idPrefix', async () => {
    await storage.query(namespace, { idPrefix: 'item-5', limit: 10 });
  });

  bench('query() - with sorting', async () => {
    await storage.query(namespace, { sortBy: 'createdAt', sortOrder: 'desc', limit: 10 });
  });

  bench('batchSet() - 10 items', async () => {
    const items = Array.from({ length: 10 }, (_, i) => ({
      key: { namespace, id: `batch-${i}` },
      value: { data: `batch-data-${i}` },
    }));
    await storage.batchSet(items);
  });

  bench('batchGet() - 10 items', async () => {
    const keys = Array.from({ length: 10 }, (_, i) => ({ namespace, id: `item-${i}` }));
    await storage.batchGet(keys);
  });

  bench('count()', async () => {
    await storage.count(namespace);
  });
});

// ============================================================================
// VALIDATION BENCHMARKS
// ============================================================================

describe('SCEI: Validation Functions', () => {
  bench('validateConfidence() - valid', () => {
    validateConfidence(0.85);
  });

  bench('validateConfidence() - invalid', () => {
    validateConfidence(1.5);
  });

  bench('validateDegradationFactor() - valid', () => {
    validateDegradationFactor(0.95);
  });

  bench('validatePositiveInteger() - valid', () => {
    validatePositiveInteger(42, 'count');
  });

  bench('validateTimestamp() - valid', () => {
    validateTimestamp(Date.now());
  });

  bench('validateNonEmptyString() - valid', () => {
    validateNonEmptyString('test-value', 'field');
  });

  bench('validateId() - valid', () => {
    validateId('claim-abc-123', 'claimId');
  });

  bench('combineValidations() - 5 results', () => {
    const results = [
      validateConfidence(0.85),
      validateDegradationFactor(0.95),
      validatePositiveInteger(10, 'count'),
      validateTimestamp(Date.now()),
      validateNonEmptyString('test', 'field'),
    ];
    combineValidations(...results);
  });

  bench('validation chain - 10 validations', () => {
    for (let i = 0; i < 10; i++) {
      validateConfidence(0.5 + i * 0.05);
    }
  });
});

// ============================================================================
// NUMERICAL STABILITY BENCHMARKS
// ============================================================================

describe('SCEI: Numerical Stability Helpers', () => {
  bench('safeLog() - normal value', () => {
    safeLog(0.5);
  });

  bench('safeLog() - edge case (0)', () => {
    safeLog(0);
  });

  bench('safeLog() - edge case (1)', () => {
    safeLog(1);
  });

  bench('clampConfidence() - in range', () => {
    clampConfidence(0.5);
  });

  bench('clampConfidence() - out of range', () => {
    clampConfidence(1.5);
  });

  bench('safeDivide() - normal', () => {
    safeDivide(10, 3, 0);
  });

  bench('safeDivide() - zero denominator', () => {
    safeDivide(10, 0, 0);
  });

  bench('safeWeightedAverage() - 5 values', () => {
    safeWeightedAverage([0.8, 0.9, 0.7, 0.85, 0.75], [1, 2, 1, 2, 1], 0.5);
  });

  bench('safeWeightedAverage() - 100 values', () => {
    const values = Array.from({ length: 100 }, () => Math.random());
    const weights = Array.from({ length: 100 }, () => Math.random() * 10);
    safeWeightedAverage(values, weights, 0.5);
  });

  bench('numerical pipeline (typical SCEI operation)', () => {
    // Typical calculation: weighted average of clamped confidences with safe log
    const values = [0.8, 0.9, 0.95, 0.7, 0.85];
    const weights = [1, 2, 1.5, 1, 2];
    const clamped = values.map(clampConfidence);
    const avg = safeWeightedAverage(clamped, weights, 0.5);
    safeLog(avg);
    safeDivide(avg, values.length, 0);
  });
});

// ============================================================================
// CALIBRATION ENGINE BENCHMARKS
// ============================================================================

describe('SCEI: Calibration Engine', () => {
  let engine: CalibrationEngine;

  beforeAll(() => {
    engine = new CalibrationEngine();
    // Pre-populate with predictions
    for (let i = 0; i < 100; i++) {
      engine.recordPrediction({
        id: `pred-${i}`,
        domain: 'science',
        confidence: 0.5 + (Math.random() * 0.4),
        timestamp: Date.now() - (i * 1000),
      });
    }
    // Resolve some predictions
    for (let i = 0; i < 50; i++) {
      engine.recordResolution(`pred-${i}`, i % 2 === 0);
    }
  });

  bench('recordPrediction()', () => {
    engine.recordPrediction({
      id: `bench-pred-${Date.now()}`,
      domain: 'test',
      confidence: 0.75,
      timestamp: Date.now(),
    });
  });

  bench('recordResolution()', () => {
    const id = `resolve-${Date.now()}`;
    engine.recordPrediction({ id, domain: 'test', confidence: 0.8, timestamp: Date.now() });
    engine.recordResolution(id, true);
  });

  bench('generateReport()', () => {
    engine.generateReport({ domain: 'science' });
  });

  bench('generateReport() - cached', () => {
    // First call caches
    engine.generateReport({ domain: 'science' });
    // Second call uses cache
    engine.generateReport({ domain: 'science' });
  });

  bench('getBrierScore()', () => {
    engine.getBrierScore('science');
  });

  bench('getCalibrationCurve()', () => {
    engine.getCalibrationCurve('science');
  });
});

// ============================================================================
// METABOLISM ENGINE BENCHMARKS
// ============================================================================

describe('SCEI: Metabolism Engine', () => {
  let engine: MetabolismEngine;

  beforeAll(() => {
    engine = new MetabolismEngine();
    // Pre-populate with claims
    for (let i = 0; i < 50; i++) {
      engine.birthClaim({
        domain: 'science',
        subject: `Test claim ${i}`,
        confidence: 0.7 + (Math.random() * 0.2),
        classification: {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N1_Local,
          materiality: MaterialityLevel.M1_Temporal,
        },
      });
    }
  });

  bench('birthClaim()', () => {
    engine.birthClaim({
      domain: 'test',
      subject: 'Benchmark claim',
      confidence: 0.85,
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Local,
        materiality: MaterialityLevel.M1_Temporal,
      },
    });
  });

  bench('getClaim()', () => {
    const claims = engine.getActiveClaims('science');
    if (claims.length > 0) {
      engine.getClaim(claims[0].id);
    }
  });

  bench('addEvidence()', () => {
    const claims = engine.getActiveClaims('science');
    if (claims.length > 0) {
      engine.addEvidence(claims[0].id, {
        type: 'supporting',
        strength: 0.7,
        source: 'benchmark',
        timestamp: Date.now(),
      });
    }
  });

  bench('getActiveClaims()', () => {
    engine.getActiveClaims('science');
  });

  bench('runDecayCycle()', () => {
    engine.runDecayCycle();
  });

  bench('getHealthMetrics()', () => {
    engine.getHealthMetrics();
  });

  bench('killClaim()', () => {
    const id = engine.birthClaim({
      domain: 'temp',
      subject: 'To kill',
      confidence: 0.5,
      classification: {
        empirical: EmpiricalLevel.E1_Subjective,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
    });
    engine.killClaim(id, 'benchmark');
  });
});

// ============================================================================
// PROPAGATION ENGINE BENCHMARKS
// ============================================================================

describe('SCEI: Propagation Engine', () => {
  let propagator: SafeBeliefPropagator;

  beforeAll(() => {
    propagator = new SafeBeliefPropagator();
    // Set up some claims and dependencies
    for (let i = 0; i < 20; i++) {
      propagator.registerClaim(`claim-${i}`, { confidence: 0.7 + (Math.random() * 0.2) });
    }
    // Create dependency chains
    for (let i = 1; i < 20; i++) {
      propagator.addDependency(`claim-${i}`, `claim-${i - 1}`, 0.8);
    }
  });

  bench('registerClaim()', () => {
    propagator.registerClaim(`bench-${Date.now()}`, { confidence: 0.85 });
  });

  bench('addDependency()', () => {
    const id1 = `dep-src-${Date.now()}`;
    const id2 = `dep-tgt-${Date.now()}`;
    propagator.registerClaim(id1, { confidence: 0.8 });
    propagator.registerClaim(id2, { confidence: 0.75 });
    propagator.addDependency(id1, id2, 0.9);
  });

  bench('propagate() - single claim', () => {
    propagator.propagate({
      id: `prop-${Date.now()}`,
      sourceClaimId: 'claim-10',
      confidenceDelta: -0.1,
      reason: 'benchmark',
    });
  });

  bench('propagate() - chain depth 5', () => {
    propagator.propagate({
      id: `prop-chain-${Date.now()}`,
      sourceClaimId: 'claim-15',
      confidenceDelta: -0.05,
      reason: 'benchmark-chain',
      maxDepth: 5,
    });
  });

  bench('getDependencyGraph()', () => {
    propagator.getDependencyGraph('claim-10');
  });

  bench('getAffectedClaims()', () => {
    propagator.getAffectedClaims('claim-10', 0.1);
  });
});

// ============================================================================
// GAP DETECTOR BENCHMARKS
// ============================================================================

describe('SCEI: Gap Detector', () => {
  let detector: GapDetector;

  beforeAll(() => {
    detector = new GapDetector();
    // Register some domains and claims
    const domains = ['science', 'economics', 'technology', 'politics', 'environment'];
    for (const domain of domains) {
      detector.registerDomain(domain, {
        minCoverage: 0.7,
        minConfidence: 0.6,
        expectedClaimCount: 100,
      });
      // Add some claims
      for (let i = 0; i < 50; i++) {
        detector.registerClaim({
          id: `${domain}-claim-${i}`,
          domain,
          confidence: 0.5 + (Math.random() * 0.4),
          tags: i % 3 === 0 ? ['verified'] : [],
        });
      }
    }
  });

  bench('registerDomain()', () => {
    detector.registerDomain(`bench-domain-${Date.now()}`, {
      minCoverage: 0.7,
      minConfidence: 0.6,
      expectedClaimCount: 50,
    });
  });

  bench('registerClaim()', () => {
    detector.registerClaim({
      id: `bench-claim-${Date.now()}`,
      domain: 'science',
      confidence: 0.85,
      tags: ['benchmark'],
    });
  });

  bench('detectGaps() - single domain', () => {
    detector.detectGaps('science');
  });

  bench('detectGaps() - all domains', () => {
    detector.detectGaps();
  });

  bench('getGapReport()', () => {
    detector.getGapReport();
  });

  bench('resolveGap()', () => {
    const gaps = detector.detectGaps('science');
    if (gaps.length > 0) {
      detector.resolveGap(gaps[0].id, 'benchmark-resolution');
    }
  });

  bench('getCoverageMetrics()', () => {
    detector.getCoverageMetrics('science');
  });

  bench('getPriorityGaps()', () => {
    detector.getPriorityGaps(5);
  });
});

// ============================================================================
// INTEGRATED WORKFLOW BENCHMARKS
// ============================================================================

describe('SCEI: Integrated Workflows', () => {
  bench('full claim lifecycle', () => {
    const calibration = new CalibrationEngine();
    const metabolism = new MetabolismEngine();

    // Record prediction
    calibration.recordPrediction({
      id: 'lifecycle-pred',
      domain: 'test',
      confidence: 0.8,
      timestamp: Date.now(),
    });

    // Birth claim
    const claimId = metabolism.birthClaim({
      domain: 'test',
      subject: 'Lifecycle test',
      confidence: 0.8,
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Local,
        materiality: MaterialityLevel.M1_Temporal,
      },
    });

    // Add evidence
    metabolism.addEvidence(claimId, {
      type: 'supporting',
      strength: 0.7,
      source: 'benchmark',
      timestamp: Date.now(),
    });

    // Resolve prediction
    calibration.recordResolution('lifecycle-pred', true);

    // Decay cycle
    metabolism.runDecayCycle();

    // Generate report
    calibration.generateReport({ domain: 'test' });
  });

  bench('cross-module event flow', async () => {
    resetSCEIEventBus();
    resetSCEIMetrics();

    const eventBus = new SCEIEventBus();
    const metrics = new SCEIMetricsCollector();

    let eventCount = 0;
    eventBus.subscribe(() => {
      eventCount++;
      metrics.incrementCounter('calibration_predictions_recorded', {});
    });

    // Emit 10 events
    for (let i = 0; i < 10; i++) {
      await eventBus.emit(createSCEIEvent('calibration:prediction_recorded', 'calibration', { i }));
    }

    metrics.exportJSON();
  });

  bench('validation + calibration pipeline', () => {
    const engine = new CalibrationEngine();

    // Validate and record 10 predictions
    for (let i = 0; i < 10; i++) {
      const confidence = 0.5 + Math.random() * 0.4;
      const validation = validateConfidence(confidence);
      if (validation.valid) {
        engine.recordPrediction({
          id: `validated-pred-${i}`,
          domain: 'validated',
          confidence,
          timestamp: Date.now(),
        });
      }
    }

    // Resolve and generate report
    for (let i = 0; i < 5; i++) {
      engine.recordResolution(`validated-pred-${i}`, i % 2 === 0);
    }

    engine.generateReport({ domain: 'validated' });
  });
});

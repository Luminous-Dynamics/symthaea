/**
 * @mycelix/sdk Performance Benchmarks
 *
 * Run with: npm run bench
 */

import { bench, describe } from 'vitest';
import {
  // MATL
  createPoGQ,
  compositeScore,
  createReputation,
  recordPositive,
  recordNegative,
  ReputationCache,
  // Epistemic
  claim,
  EmpiricalLevel,
  NormativeLevel,
  EpistemicClaimPool,
  EpistemicBatch,
  // FL
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  trustWeightedAggregation,
  FLCoordinator,
  AggregationMethod,
  // Bridge
  LocalBridge,
  BridgeRouter,
  createReputationQuery,
  // Utilities
  RequestBatcher,
  EventPipeline,
  TimeSeriesAnalytics,
  CircuitBreaker,
  MetricsCollector,
  StructuredLogger,
} from '../src/index.js';

// ============================================================================
// MATL Benchmarks
// ============================================================================

describe('MATL Performance', () => {
  bench('createPoGQ', () => {
    createPoGQ(0.95, 0.88, 0.12);
  });

  bench('compositeScore', () => {
    const pogq = createPoGQ(0.95, 0.88, 0.12);
    compositeScore(pogq, 0.75);
  });

  bench('reputation operations (100 updates)', () => {
    let rep = createReputation();
    for (let i = 0; i < 100; i++) {
      rep = i % 3 === 0 ? recordNegative(rep) : recordPositive(rep);
    }
  });

  bench('ReputationCache lookup', () => {
    const cache = new ReputationCache({ maxSize: 1000 });
    // Pre-populate
    for (let i = 0; i < 100; i++) {
      cache.set(`agent-${i}`, createReputation());
    }
    // Lookup
    cache.get('agent-50');
  });
});

// ============================================================================
// Epistemic Benchmarks
// ============================================================================

describe('Epistemic Performance', () => {
  bench('claim creation', () => {
    claim('Test claim')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .withNormative(NormativeLevel.N1_Local)
      .build();
  });

  bench('EpistemicBatch (10 claims)', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 10; i++) {
      batch.add(
        claim(`Claim ${i}`)
          .withEmpirical(EmpiricalLevel.E1_Testimonial)
          .build()
      );
    }
    batch.summarize();
  });

  bench('EpistemicClaimPool operations', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 50; i++) {
      pool.add(claim(`Pool claim ${i}`).build());
    }
    pool.getActive();
    pool.cleanup();
  });
});

// ============================================================================
// FL Benchmarks
// ============================================================================

describe('FL Aggregation Performance', () => {
  // Create test gradients
  const createGradients = (count: number, size: number) =>
    Array.from({ length: count }, () => ({
      participantId: `p-${Math.random()}`,
      gradients: new Float64Array(size).fill(0).map(() => Math.random()),
      timestamp: Date.now(),
    }));

  const smallGradients = createGradients(10, 100);
  const mediumGradients = createGradients(50, 1000);

  bench('fedAvg (10 participants, 100 dims)', () => {
    fedAvg(smallGradients);
  });

  bench('fedAvg (50 participants, 1000 dims)', () => {
    fedAvg(mediumGradients);
  });

  bench('trimmedMean (50 participants)', () => {
    trimmedMean(mediumGradients, 0.2);
  });

  bench('coordinateMedian (50 participants)', () => {
    coordinateMedian(mediumGradients);
  });

  bench('krum (10 participants)', () => {
    krum(smallGradients);
  });

  bench('trustWeightedAggregation', () => {
    const participants = new Map(
      smallGradients.map((g) => [
        g.participantId,
        { id: g.participantId, reputation: createReputation() },
      ])
    );
    trustWeightedAggregation(smallGradients, participants);
  });
});

describe('FLCoordinator Performance', () => {
  bench('full FL round', () => {
    const coordinator = new FLCoordinator({
      roundTimeoutMs: 30000,
      minParticipants: 3,
      aggregationMethod: AggregationMethod.FedAvg,
    });

    // Register participants
    for (let i = 0; i < 5; i++) {
      coordinator.registerParticipant(`p-${i}`);
    }

    // Start round
    coordinator.startRound();

    // Submit updates
    for (let i = 0; i < 5; i++) {
      coordinator.submitUpdate({
        participantId: `p-${i}`,
        gradients: new Float64Array(100).fill(Math.random()),
        timestamp: Date.now(),
      });
    }

    // Aggregate
    coordinator.aggregateRound();
  });
});

// ============================================================================
// Bridge Benchmarks
// ============================================================================

describe('Bridge Performance', () => {
  bench('LocalBridge message routing', () => {
    const bridge = new LocalBridge();
    bridge.registerHapp('app1');
    bridge.registerHapp('app2');

    for (let i = 0; i < 100; i++) {
      bridge.send('app1', createReputationQuery('agent-1'));
    }
  });

  bench('BridgeRouter with middleware', () => {
    const router = new BridgeRouter({
      handlers: {
        reputation_query: async (msg) => msg,
      },
    });

    router.use(async (msg, next) => {
      await next();
      return msg;
    });

    for (let i = 0; i < 50; i++) {
      router.route(createReputationQuery('agent-1'));
    }
  });
});

// ============================================================================
// Utility Benchmarks
// ============================================================================

describe('Utilities Performance', () => {
  bench('RequestBatcher (100 items)', async () => {
    const batcher = new RequestBatcher<number, number>({
      maxBatchSize: 10,
      maxWaitMs: 5,
      executor: async (items) => items.map((i) => i * 2),
    });

    const promises = Array.from({ length: 100 }, (_, i) => batcher.add(i));
    await Promise.all(promises);
  });

  bench('EventPipeline operations', () => {
    const pipeline = new EventPipeline<number>()
      .filter((n) => n > 0)
      .map((n) => n * 2)
      .tap(() => {});

    for (let i = 0; i < 1000; i++) {
      pipeline.process(i);
    }
  });

  bench('TimeSeriesAnalytics (1000 points)', () => {
    const analytics = new TimeSeriesAnalytics();
    const now = Date.now();

    for (let i = 0; i < 1000; i++) {
      analytics.addPoint(now - i * 1000, Math.random());
    }

    analytics.getMovingAverage(100);
    analytics.getTrend();
    analytics.getStats();
  });

  bench('CircuitBreaker success path', async () => {
    const breaker = new CircuitBreaker({
      failureThreshold: 5,
      resetTimeoutMs: 30000,
    });

    for (let i = 0; i < 100; i++) {
      await breaker.call(async () => i);
    }
  });

  bench('MetricsCollector recording', () => {
    const metrics = new MetricsCollector();
    metrics.registerCounter('requests', 'Total requests');
    metrics.registerHistogram('latency', 'Request latency', [10, 50, 100, 500]);

    for (let i = 0; i < 1000; i++) {
      metrics.incrementCounter('requests');
      metrics.recordHistogram('latency', Math.random() * 500);
    }
  });

  bench('StructuredLogger throughput', () => {
    const logger = new StructuredLogger({
      level: 'warn', // Only log warnings to reduce output
      format: 'json',
      output: () => {}, // No-op output
    });

    for (let i = 0; i < 1000; i++) {
      logger.warn('Benchmark message', { iteration: i });
    }
  });
});

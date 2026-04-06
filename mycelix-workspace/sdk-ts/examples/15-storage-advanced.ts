// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 15: Advanced Storage Patterns
 *
 * Demonstrates advanced UESS features:
 * - Batch operations for high-throughput data
 * - CRDT merge strategies for conflict resolution
 * - Observability (metrics, tracing, health checks)
 * - Cross-hApp storage patterns
 */

import {
  storage,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '@mycelix/sdk';

// =============================================================================
// Part 1: Batch Operations
// =============================================================================

async function batchOperationsExample() {
  console.log('=== Part 1: Batch Operations ===\n');

  const store = storage.createEpistemicStorage({ agentId: 'agent:batch-demo' });

  // Create a batch executor for high-throughput operations
  const executor = storage.createBatchExecutor({
    concurrency: 5,        // Process 5 items at a time
    retries: 3,            // Retry failed items 3 times
    retryDelayMs: 100,     // Wait 100ms between retries
    continueOnError: true, // Don't stop on individual failures
  });

  // Generate test data
  const items: storage.BatchItem<{ value: number; timestamp: number }>[] = [];
  for (let i = 0; i < 100; i++) {
    items.push({
      key: `metric:sensor-${i}`,
      data: { value: Math.random() * 100, timestamp: Date.now() },
      classification: {
        empirical: EmpiricalLevel.E1_Attested,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M1_Temporal,
      },
      options: {
        schema: { id: 'sensor-metric', version: '1.0.0' },
        ttlMs: 3600000, // 1 hour
      },
    });
  }

  console.log('1. Batch storing 100 sensor metrics...');

  const results = await executor.execute(
    items,
    async (item) => {
      return store.store(item.key, item.data, item.classification, item.options);
    },
    (progress) => {
      if (progress.completed % 20 === 0) {
        console.log(`   Progress: ${progress.completed}/${progress.total} (${progress.successCount} success, ${progress.failureCount} failed)`);
      }
    }
  );

  console.log(`\n   Batch complete:`);
  console.log(`   - Total: ${results.stats.total}`);
  console.log(`   - Success: ${results.stats.successCount}`);
  console.log(`   - Failed: ${results.stats.failureCount}`);
  console.log(`   - Duration: ${results.stats.durationMs}ms`);
  console.log(`   - Throughput: ${(results.stats.total / (results.stats.durationMs / 1000)).toFixed(1)} ops/sec\n`);

  // Batch retrieval
  console.log('2. Batch retrieving metrics...');

  const keys = items.slice(0, 20).map(i => i.key);
  const retrieved = await storage.batchRetrieve(
    store.retrieve.bind(store),
    keys,
    { concurrency: 10 }
  );

  console.log(`   Retrieved ${retrieved.filter(r => r.success).length} of ${keys.length} items\n`);

  // Streaming batch for very large datasets
  console.log('3. Streaming batch (memory-efficient)...');

  const streamExecutor = storage.createStreamingBatchExecutor<{ id: number }>({
    concurrency: 3,
    batchSize: 10,
  });

  // Create an async generator for streaming data
  async function* generateData(): AsyncGenerator<storage.BatchItem<{ id: number }>> {
    for (let i = 0; i < 50; i++) {
      yield {
        key: `stream:item-${i}`,
        data: { id: i },
        classification: {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M0_Ephemeral,
        },
      };
    }
  }

  const streamResults = await streamExecutor.execute(
    storage.createArraySource(Array.from({ length: 50 }, (_, i) => ({
      key: `stream:item-${i}`,
      data: { id: i },
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
    }))),
    async (item) => store.store(item.key, item.data, item.classification)
  );

  console.log(`   Streamed ${streamResults.stats.successCount} items\n`);
}

// =============================================================================
// Part 2: CRDT Merge Strategies
// =============================================================================

async function crdtExample() {
  console.log('=== Part 2: CRDT Merge Strategies ===\n');

  // Last-Writer-Wins Register
  console.log('1. Last-Writer-Wins Register...');

  const lww = new storage.LWWRegister<string>('initial value');

  // Simulate concurrent updates from different nodes
  lww.set('update from node A', 1000);
  lww.set('update from node B', 1500); // Later timestamp wins
  lww.set('update from node C', 1200); // Earlier timestamp, ignored

  console.log(`   Current value: "${lww.get()}"`);
  console.log(`   Timestamp: ${lww.timestamp()}\n`);

  // Merge with remote state
  const remoteLWW = new storage.LWWRegister<string>('remote initial');
  remoteLWW.set('remote update', 2000); // Even later

  const merged = lww.merge(remoteLWW);
  console.log(`   After merge: "${merged.get()}" (timestamp: ${merged.timestamp()})\n`);

  // Multi-Value Register (preserves concurrent values)
  console.log('2. Multi-Value Register (preserves conflicts)...');

  const mvr = new storage.MVRegister<string>('node-a');
  mvr.set('value A');

  const mvr2 = new storage.MVRegister<string>('node-b');
  mvr2.set('value B');

  const mergedMV = mvr.merge(mvr2);
  console.log(`   Concurrent values: ${JSON.stringify(mergedMV.values())}`);
  console.log(`   (Application decides how to resolve)\n`);

  // OR-Set (Observed-Remove Set)
  console.log('3. OR-Set (supports add and remove)...');

  const set1 = new storage.ORSet<string>('replica-1');
  set1.add('alice');
  set1.add('bob');
  set1.add('charlie');
  set1.remove('bob');

  const set2 = new storage.ORSet<string>('replica-2');
  set2.add('bob'); // Re-added on different replica
  set2.add('diana');

  const mergedSet = set1.merge(set2);
  console.log(`   Members: ${JSON.stringify([...mergedSet.values()])}`);
  console.log(`   Note: bob is present (add beats concurrent remove)\n`);

  // G-Counter (grow-only counter)
  console.log('4. G-Counter (distributed counting)...');

  const counter1 = new storage.GCounter('node-1');
  counter1.increment(5);

  const counter2 = new storage.GCounter('node-2');
  counter2.increment(3);

  const counter3 = new storage.GCounter('node-3');
  counter3.increment(7);

  let mergedCounter = counter1.merge(counter2);
  mergedCounter = mergedCounter.merge(counter3);

  console.log(`   Total count: ${mergedCounter.value()}`);
  console.log(`   (Sum of all node increments: 5 + 3 + 7 = 15)\n`);

  // PN-Counter (positive-negative counter)
  console.log('5. PN-Counter (supports increment and decrement)...');

  const pnCounter = new storage.PNCounter('replica-1');
  pnCounter.increment(10);
  pnCounter.decrement(3);

  const pnCounter2 = new storage.PNCounter('replica-2');
  pnCounter2.increment(5);
  pnCounter2.decrement(2);

  const mergedPN = pnCounter.merge(pnCounter2);
  console.log(`   Final value: ${mergedPN.value()}`);
  console.log(`   (10 - 3 + 5 - 2 = 10)\n`);

  // Vector Clocks
  console.log('6. Vector Clocks (causality tracking)...');

  let clockA = storage.createVectorClock();
  clockA = storage.incrementClock(clockA, 'node-a');
  clockA = storage.incrementClock(clockA, 'node-a');

  let clockB = storage.createVectorClock();
  clockB = storage.incrementClock(clockB, 'node-b');

  const comparison = storage.compareClocks(clockA, clockB);
  console.log(`   Clock A: ${JSON.stringify(clockA)}`);
  console.log(`   Clock B: ${JSON.stringify(clockB)}`);
  console.log(`   Comparison: ${comparison === 0 ? 'CONCURRENT' : comparison > 0 ? 'A > B' : 'B > A'}`);

  const mergedClock = storage.mergeClocks(clockA, clockB);
  console.log(`   Merged: ${JSON.stringify(mergedClock)}\n`);
}

// =============================================================================
// Part 3: Observability
// =============================================================================

async function observabilityExample() {
  console.log('=== Part 3: Observability ===\n');

  // Create metrics collector
  const metrics = storage.createMetricsCollector();

  // Create tracer
  const tracer = storage.createTracer();

  // Create health checker
  const healthChecker = storage.createHealthChecker();

  // Simulate some operations
  console.log('1. Recording metrics...');

  for (let i = 0; i < 50; i++) {
    const operation = ['store', 'retrieve', 'delete'][Math.floor(Math.random() * 3)] as storage.StorageOperation;
    const backend = ['memory', 'local', 'dht'][Math.floor(Math.random() * 3)] as storage.BackendType;
    const success = Math.random() > 0.1; // 90% success rate
    const latency = Math.random() * 100;

    metrics.recordOperation(operation, backend, success);
    metrics.recordLatency(operation, backend, latency);
  }

  // Export metrics
  const snapshot = metrics.getSnapshot();
  console.log(`   Operations recorded: ${Object.values(snapshot.operations).reduce((a, b) => a + b.success + b.error, 0)}`);
  console.log(`   Error rate: ${(Object.values(snapshot.operations).reduce((a, b) => a + b.error, 0) / 50 * 100).toFixed(1)}%\n`);

  // Prometheus format
  console.log('2. Prometheus metrics export...');
  const prometheus = metrics.toPrometheus();
  console.log(prometheus.split('\n').slice(0, 10).join('\n'));
  console.log('   ...\n');

  // Distributed tracing
  console.log('3. Distributed tracing...');

  const traceContext = tracer.startTrace('storage.batch_operation');

  const span1 = tracer.startSpan('storage.classify', traceContext);
  await new Promise(r => setTimeout(r, 10));
  tracer.endSpan(span1);

  const span2 = tracer.startSpan('storage.route', traceContext);
  await new Promise(r => setTimeout(r, 5));
  tracer.endSpan(span2);

  const span3 = tracer.startSpan('backend.execute', traceContext);
  await new Promise(r => setTimeout(r, 20));
  tracer.endSpan(span3);

  tracer.endTrace(traceContext.traceId);

  const traces = tracer.getTraces();
  console.log(`   Active traces: ${traces.length}`);
  if (traces[0]) {
    console.log(`   Root span: ${traces[0].rootSpan.name}`);
    console.log(`   Total spans: ${1 + (traces[0].childSpans?.length || 0)}\n`);
  }

  // Health checks
  console.log('4. Health checks...');

  // Register health checks
  healthChecker.registerCheck('memory', async () => ({
    name: 'memory',
    status: 'healthy',
    latencyMs: 1,
    lastCheck: Date.now(),
  }));

  healthChecker.registerCheck('local', async () => ({
    name: 'local',
    status: 'healthy',
    latencyMs: 5,
    lastCheck: Date.now(),
  }));

  healthChecker.registerCheck('dht', async () => ({
    name: 'dht',
    status: 'degraded',
    message: '3 of 5 peers connected',
    latencyMs: 45,
    lastCheck: Date.now(),
  }));

  const health = await healthChecker.checkAll();
  console.log(`   Overall status: ${health.status}`);
  for (const check of health.checks) {
    console.log(`   - ${check.name}: ${check.status} (${check.latencyMs}ms)`);
  }
  console.log();

  // Instrumented operations
  console.log('5. Instrumented operation wrapper...');

  const result = await storage.instrumentedOperation(
    'store',
    'memory',
    async () => {
      await new Promise(r => setTimeout(r, 15));
      return { success: true, data: 'test' };
    },
    metrics,
    tracer
  );

  console.log(`   Result: ${JSON.stringify(result)}`);
  console.log(`   Metrics and trace automatically recorded\n`);
}

// =============================================================================
// Part 4: Cross-hApp Storage Patterns
// =============================================================================

async function crossHappExample() {
  console.log('=== Part 4: Cross-hApp Storage Patterns ===\n');

  // Shared storage namespace pattern
  console.log('1. Shared reputation storage across hApps...');

  const store = storage.createEpistemicStorage({ agentId: 'agent:demo' });

  // Each hApp stores reputation in a shared namespace
  const happReputations = [
    { happ: 'marketplace', agent: 'alice', score: 4.8, interactions: 150 },
    { happ: 'praxis', agent: 'alice', score: 4.9, interactions: 45 },
    { happ: 'supplychain', agent: 'alice', score: 4.7, interactions: 80 },
  ];

  for (const rep of happReputations) {
    await store.store(
      `reputation:${rep.happ}:${rep.agent}`,
      rep,
      {
        empirical: EmpiricalLevel.E2_Verified,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      }
    );
    console.log(`   Stored ${rep.happ} reputation for ${rep.agent}: ${rep.score}`);
  }

  // Aggregate reputation query
  console.log('\n2. Aggregating cross-hApp reputation...');

  const reputations = await store.query({
    keyPrefix: 'reputation:',
    normative: { min: NormativeLevel.N2_Network },
  });

  const aliceReps = reputations.items.filter(r =>
    (r.data as { agent: string }).agent === 'alice'
  );

  const weightedAvg = aliceReps.reduce((sum, r) => {
    const rep = r.data as { score: number; interactions: number };
    return sum + (rep.score * rep.interactions);
  }, 0) / aliceReps.reduce((sum, r) => {
    return sum + (r.data as { interactions: number }).interactions;
  }, 0);

  console.log(`   Alice's aggregate reputation: ${weightedAvg.toFixed(2)}`);
  console.log(`   Based on ${aliceReps.length} hApps, ${aliceReps.reduce((s, r) => s + (r.data as { interactions: number }).interactions, 0)} total interactions\n`);

  // Cross-hApp credential sharing
  console.log('3. Cross-hApp credential sharing...');

  const credential = {
    type: 'VerifiedIdentity',
    subject: 'did:mycelix:alice',
    issuer: 'did:mycelix:identity-happ',
    claims: {
      realName: true,
      kycComplete: true,
      humanVerified: true,
    },
    issuedAt: Date.now(),
    expiresAt: Date.now() + 365 * 24 * 3600000,
  };

  await store.store(
    'credential:identity:alice',
    credential,
    {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network, // Other hApps can verify
      materiality: MaterialityLevel.M3_Immutable,
    }
  );

  console.log('   Stored identity credential (accessible by all hApps)');

  // Other hApps can verify the credential
  const retrieved = await store.retrieve<typeof credential>('credential:identity:alice');
  if (retrieved) {
    const isValid = retrieved.data.expiresAt > Date.now();
    console.log(`   Credential valid: ${isValid}`);
    console.log(`   KYC complete: ${retrieved.data.claims.kycComplete}\n`);
  }
}

// =============================================================================
// Main
// =============================================================================

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║          Advanced Storage Patterns Example                    ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  await batchOperationsExample();
  await crdtExample();
  await observabilityExample();
  await crossHappExample();

  console.log('=== All Examples Complete ===');
}

main().catch(console.error);

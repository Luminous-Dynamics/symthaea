// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Advanced Patterns Example
 *
 * This example demonstrates advanced SDK features:
 * - Request batching
 * - Event pipelines
 * - Circuit breakers
 * - Metrics & health checks
 * - Plugin system
 */

import {
  // Utilities
  RequestBatcher,
  EventPipeline,
  CircuitBreaker,
  MetricsCollector,
  HealthChecker,
  StructuredLogger,
  TracingManager,
  PolicyEngine,
  TimeSeriesAnalytics,

  // Plugin system
  PluginManager,
  type Plugin,

  // Events
  sdkEvents,
  MatlEventType,
  type ReputationUpdatedEvent,
} from '@mycelix/sdk';

async function main() {
  console.log('=== @mycelix/sdk Advanced Patterns ===\n');

  // -------------------------------------------------------------------------
  // 1. Structured Logging
  // -------------------------------------------------------------------------
  console.log('--- Structured Logging ---');

  const logger = new StructuredLogger({
    level: 'debug',
    format: 'pretty',
    timestamps: true,
  });

  const moduleLogger = logger.child({ module: 'advanced-example' });
  moduleLogger.info('Starting advanced patterns example');
  moduleLogger.debug('Debug information', { detail: 'value' });

  // -------------------------------------------------------------------------
  // 2. Request Batching
  // -------------------------------------------------------------------------
  console.log('\n--- Request Batching ---');

  const batcher = new RequestBatcher<string, string>({
    maxBatchSize: 5,
    maxWaitMs: 100,
    executor: async (ids) => {
      console.log(`  Batch processing ${ids.length} items:`, ids);
      return ids.map((id) => `result-${id}`);
    },
  });

  // These will be batched together
  const promises = [
    batcher.add('item-1'),
    batcher.add('item-2'),
    batcher.add('item-3'),
  ];

  const results = await Promise.all(promises);
  console.log('Batch results:', results);

  // -------------------------------------------------------------------------
  // 3. Event Pipeline
  // -------------------------------------------------------------------------
  console.log('\n--- Event Pipeline ---');

  type ReputationChange = { agentId: string; delta: number; timestamp: number };

  const reputationPipeline = new EventPipeline<ReputationChange>()
    .filter((event) => Math.abs(event.delta) > 0.1) // Significant changes only
    .map((event) => ({
      ...event,
      severity: event.delta < -0.2 ? 'critical' : 'normal',
    }))
    .tap((event) => {
      console.log(`  Reputation change: ${event.agentId} (${event.delta > 0 ? '+' : ''}${event.delta.toFixed(2)})`);
    });

  // Process some events
  reputationPipeline.process({ agentId: 'alice', delta: 0.15, timestamp: Date.now() });
  reputationPipeline.process({ agentId: 'bob', delta: -0.05, timestamp: Date.now() }); // Filtered out
  reputationPipeline.process({ agentId: 'charlie', delta: -0.25, timestamp: Date.now() });

  // -------------------------------------------------------------------------
  // 4. Circuit Breaker
  // -------------------------------------------------------------------------
  console.log('\n--- Circuit Breaker ---');

  const breaker = new CircuitBreaker({
    failureThreshold: 3,
    resetTimeoutMs: 5000,
    successThreshold: 2,
    callTimeoutMs: 1000,
  });

  async function riskyOperation(shouldFail: boolean): Promise<string> {
    if (shouldFail) throw new Error('Simulated failure');
    return 'success';
  }

  // Successful calls
  for (let i = 0; i < 3; i++) {
    const result = await breaker.call(() => riskyOperation(false));
    console.log(`  Call ${i + 1}: ${result}`);
  }

  console.log('  Circuit state:', breaker.getState());

  // -------------------------------------------------------------------------
  // 5. Metrics Collection
  // -------------------------------------------------------------------------
  console.log('\n--- Metrics Collection ---');

  const metrics = new MetricsCollector();

  // Register metrics
  metrics.registerCounter('requests_total', 'Total requests processed');
  metrics.registerGauge('active_connections', 'Current active connections');
  metrics.registerHistogram('request_duration_ms', 'Request duration', [10, 50, 100, 250, 500]);

  // Record some metrics
  for (let i = 0; i < 100; i++) {
    metrics.incrementCounter('requests_total');
    metrics.recordHistogram('request_duration_ms', Math.random() * 200);
  }
  metrics.setGauge('active_connections', 42);

  // Get metrics output
  const output = metrics.getPrometheusOutput();
  console.log('Sample metrics output (first 500 chars):\n', output.slice(0, 500));

  // -------------------------------------------------------------------------
  // 6. Health Checks
  // -------------------------------------------------------------------------
  console.log('\n--- Health Checks ---');

  const healthChecker = new HealthChecker();

  healthChecker.register('database', async () => {
    // Simulate database check
    return { status: 'healthy', latency: 5 };
  });

  healthChecker.register('cache', async () => {
    // Simulate cache check
    return { status: 'healthy', latency: 1 };
  });

  healthChecker.register('external-api', async () => {
    // Simulate degraded service
    return { status: 'degraded', latency: 500, message: 'High latency' };
  });

  const healthReport = await healthChecker.check();
  console.log('Health status:', healthReport.status);
  console.log('Checks:', Object.keys(healthReport.checks).length);
  for (const [name, check] of Object.entries(healthReport.checks)) {
    console.log(`  ${name}: ${check.status} (${check.latency}ms)`);
  }

  // -------------------------------------------------------------------------
  // 7. Distributed Tracing
  // -------------------------------------------------------------------------
  console.log('\n--- Distributed Tracing ---');

  const tracer = new TracingManager();

  // Start a trace
  const trace = tracer.startTrace('process-request', { userId: 'user-123' });

  // Create child spans
  const dbSpan = tracer.startSpan(trace.traceId, 'database-query');
  await new Promise((r) => setTimeout(r, 10)); // Simulate work
  tracer.endSpan(trace.traceId, dbSpan.spanId, { rows: 42 });

  const cacheSpan = tracer.startSpan(trace.traceId, 'cache-lookup');
  await new Promise((r) => setTimeout(r, 5));
  tracer.endSpan(trace.traceId, cacheSpan.spanId, { hit: true });

  // End trace
  const completedTrace = tracer.endTrace(trace.traceId);
  console.log('Trace ID:', completedTrace?.traceId);
  console.log('Total duration:', completedTrace?.duration, 'ms');
  console.log('Spans:', completedTrace?.spans.length);

  // -------------------------------------------------------------------------
  // 8. Policy Engine
  // -------------------------------------------------------------------------
  console.log('\n--- Policy Engine ---');

  const policyEngine = new PolicyEngine();

  // Define access policies
  policyEngine.addPolicy({
    id: 'admin-full-access',
    resource: 'reputation',
    action: '*',
    effect: 'allow',
    condition: (ctx) => ctx.subject.role === 'admin',
    priority: 100,
  });

  policyEngine.addPolicy({
    id: 'user-read-own',
    resource: 'reputation',
    action: 'read',
    effect: 'allow',
    condition: (ctx) => ctx.subject.id === ctx.resource.ownerId,
    priority: 50,
  });

  policyEngine.addPolicy({
    id: 'default-deny',
    resource: '*',
    action: '*',
    effect: 'deny',
    priority: 0,
  });

  // Check permissions
  const adminContext = {
    subject: { id: 'admin-1', role: 'admin' },
    resource: { type: 'reputation', ownerId: 'user-1' },
    action: 'delete' as const,
  };

  const userContext = {
    subject: { id: 'user-1', role: 'user' },
    resource: { type: 'reputation', ownerId: 'user-1' },
    action: 'read' as const,
  };

  console.log('Admin can delete?', policyEngine.evaluate(adminContext).allowed);
  console.log('User can read own?', policyEngine.evaluate(userContext).allowed);

  // -------------------------------------------------------------------------
  // 9. Time Series Analytics
  // -------------------------------------------------------------------------
  console.log('\n--- Time Series Analytics ---');

  const analytics = new TimeSeriesAnalytics();
  const now = Date.now();

  // Add historical data points
  for (let i = 100; i >= 0; i--) {
    // Simulate trending upward with noise
    const value = 0.5 + (100 - i) * 0.003 + (Math.random() - 0.5) * 0.1;
    analytics.addPoint(now - i * 60000, value);
  }

  console.log('Points:', analytics.getPoints().length);
  console.log('Moving average (10):', analytics.getMovingAverage(10).toFixed(4));
  console.log('Trend:', analytics.getTrend());

  const stats = analytics.getStats();
  console.log('Stats:', {
    min: stats.min.toFixed(4),
    max: stats.max.toFixed(4),
    mean: stats.mean.toFixed(4),
    stdDev: stats.stdDev.toFixed(4),
  });

  // -------------------------------------------------------------------------
  // 10. Plugin System
  // -------------------------------------------------------------------------
  console.log('\n--- Plugin System ---');

  const pluginManager = new PluginManager();

  // Create a logging plugin
  const loggingPlugin: Plugin = {
    metadata: {
      id: 'logging-plugin',
      name: 'Logging Plugin',
      version: '1.0.0',
    },
    async onInit() {
      console.log('  [Plugin] Logging plugin initialized');
    },
    async beforeOperation(operation, params) {
      console.log(`  [Plugin] Before operation: ${operation}`);
      return params;
    },
    async afterOperation(operation, result) {
      console.log(`  [Plugin] After operation: ${operation}`);
      return result;
    },
  };

  // Register and initialize
  await pluginManager.register(loggingPlugin);

  // Execute with hooks
  await pluginManager.executeWithHooks('reputation.update', { agentId: 'test' }, async (params) => {
    console.log('  [Core] Executing operation with params:', params);
    return { success: true };
  });

  console.log('\n=== Advanced Patterns Complete ===');
}

main().catch(console.error);

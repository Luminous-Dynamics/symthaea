/**
 * UESS Observability
 *
 * Metrics, tracing, and health checks for storage operations.
 * @see docs/architecture/uess/UESS-15-OBSERVABILITY.md
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Storage operation types for metrics
 */
export type StorageOperation =
  | 'store'
  | 'retrieve'
  | 'update'
  | 'delete'
  | 'reclassify'
  | 'query'
  | 'verify'
  | 'migrate';

/**
 * Backend types for metrics
 */
export type BackendType = 'memory' | 'local' | 'dht' | 'ipfs' | 'filecoin';

/**
 * Health status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

/**
 * Metric value
 */
export interface MetricValue {
  value: number;
  timestamp: number;
  labels?: Record<string, string>;
}

/**
 * Histogram bucket
 */
export interface HistogramBucket {
  le: number;  // Less than or equal to
  count: number;
}

/**
 * Histogram metric
 */
export interface HistogramMetric {
  buckets: HistogramBucket[];
  sum: number;
  count: number;
}

/**
 * Trace span
 */
export interface TraceSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  operationName: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  status: 'ok' | 'error';
  attributes: Record<string, string | number | boolean>;
  events: Array<{
    name: string;
    timestamp: number;
    attributes?: Record<string, unknown>;
  }>;
}

/**
 * Health check result
 */
export interface HealthCheckResult {
  name: string;
  status: HealthStatus;
  message?: string;
  latencyMs?: number;
  lastCheck: number;
  details?: Record<string, unknown>;
}

/**
 * Storage metrics snapshot
 */
export interface StorageMetricsSnapshot {
  timestamp: number;

  // Counters
  operationsTotal: Record<StorageOperation, number>;
  operationsByBackend: Record<BackendType, Record<StorageOperation, number>>;
  errorsTotal: Record<StorageOperation, number>;

  // Gauges
  itemCount: Record<BackendType, number>;
  totalSizeBytes: Record<BackendType, number>;
  cacheHitRate: number;

  // Histograms
  operationLatencyMs: Record<StorageOperation, HistogramMetric>;

  // Health
  backendHealth: Record<BackendType, HealthStatus>;
  overallHealth: HealthStatus;
}

// =============================================================================
// Metrics Collector
// =============================================================================

/**
 * Storage Metrics Collector
 */
export class StorageMetricsCollector {
  private operationsTotal: Map<string, number> = new Map();
  private operationsByBackend: Map<string, number> = new Map();
  private errorsTotal: Map<string, number> = new Map();
  private itemCounts: Map<BackendType, number> = new Map();
  private sizeCounts: Map<BackendType, number> = new Map();
  private latencyHistograms: Map<StorageOperation, number[]> = new Map();
  private cacheHits = 0;
  private cacheMisses = 0;

  private readonly bucketBoundaries = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

  // ===========================================================================
  // Counter Operations
  // ===========================================================================

  /**
   * Record an operation
   */
  recordOperation(operation: StorageOperation, backend?: BackendType): void {
    const opKey = operation;
    this.operationsTotal.set(opKey, (this.operationsTotal.get(opKey) ?? 0) + 1);

    if (backend) {
      const beKey = `${backend}:${operation}`;
      this.operationsByBackend.set(beKey, (this.operationsByBackend.get(beKey) ?? 0) + 1);
    }
  }

  /**
   * Record an error
   */
  recordError(operation: StorageOperation, _errorType?: string): void {
    this.errorsTotal.set(operation, (this.errorsTotal.get(operation) ?? 0) + 1);
  }

  /**
   * Record cache hit/miss
   */
  recordCacheAccess(hit: boolean): void {
    if (hit) {
      this.cacheHits++;
    } else {
      this.cacheMisses++;
    }
  }

  // ===========================================================================
  // Gauge Operations
  // ===========================================================================

  /**
   * Set item count for a backend
   */
  setItemCount(backend: BackendType, count: number): void {
    this.itemCounts.set(backend, count);
  }

  /**
   * Set total size for a backend
   */
  setTotalSize(backend: BackendType, sizeBytes: number): void {
    this.sizeCounts.set(backend, sizeBytes);
  }

  // ===========================================================================
  // Histogram Operations
  // ===========================================================================

  /**
   * Record operation latency
   */
  recordLatency(operation: StorageOperation, latencyMs: number): void {
    if (!this.latencyHistograms.has(operation)) {
      this.latencyHistograms.set(operation, []);
    }
    this.latencyHistograms.get(operation)!.push(latencyMs);
  }

  /**
   * Get histogram for an operation
   */
  getHistogram(operation: StorageOperation): HistogramMetric {
    const values = this.latencyHistograms.get(operation) ?? [];

    const buckets: HistogramBucket[] = this.bucketBoundaries.map(le => ({
      le,
      count: values.filter(v => v <= le).length,
    }));

    // Add +Inf bucket
    buckets.push({ le: Infinity, count: values.length });

    return {
      buckets,
      sum: values.reduce((a, b) => a + b, 0),
      count: values.length,
    };
  }

  // ===========================================================================
  // Snapshot
  // ===========================================================================

  /**
   * Get current metrics snapshot
   */
  getSnapshot(backendHealthStatus?: Record<BackendType, HealthStatus>): StorageMetricsSnapshot {
    const operations: StorageOperation[] = ['store', 'retrieve', 'update', 'delete', 'reclassify', 'query', 'verify', 'migrate'];
    const backends: BackendType[] = ['memory', 'local', 'dht', 'ipfs', 'filecoin'];

    const operationsTotal: Record<string, number> = {};
    const errorsTotal: Record<string, number> = {};
    const operationLatencyMs: Record<string, HistogramMetric> = {};

    for (const op of operations) {
      operationsTotal[op] = this.operationsTotal.get(op) ?? 0;
      errorsTotal[op] = this.errorsTotal.get(op) ?? 0;
      operationLatencyMs[op] = this.getHistogram(op);
    }

    const operationsByBackend: Record<string, Record<string, number>> = {};
    const itemCount: Record<string, number> = {};
    const totalSizeBytes: Record<string, number> = {};

    for (const backend of backends) {
      operationsByBackend[backend] = {};
      for (const op of operations) {
        operationsByBackend[backend][op] = this.operationsByBackend.get(`${backend}:${op}`) ?? 0;
      }
      itemCount[backend] = this.itemCounts.get(backend) ?? 0;
      totalSizeBytes[backend] = this.sizeCounts.get(backend) ?? 0;
    }

    const totalCacheAccess = this.cacheHits + this.cacheMisses;
    const cacheHitRate = totalCacheAccess > 0 ? this.cacheHits / totalCacheAccess : 0;

    const backendHealth = backendHealthStatus ?? {
      memory: 'unknown' as HealthStatus,
      local: 'unknown' as HealthStatus,
      dht: 'unknown' as HealthStatus,
      ipfs: 'unknown' as HealthStatus,
      filecoin: 'unknown' as HealthStatus,
    };

    const overallHealth = this.calculateOverallHealth(backendHealth);

    return {
      timestamp: Date.now(),
      operationsTotal: operationsTotal as Record<StorageOperation, number>,
      operationsByBackend: operationsByBackend as Record<BackendType, Record<StorageOperation, number>>,
      errorsTotal: errorsTotal as Record<StorageOperation, number>,
      itemCount: itemCount as Record<BackendType, number>,
      totalSizeBytes: totalSizeBytes as Record<BackendType, number>,
      cacheHitRate,
      operationLatencyMs: operationLatencyMs as Record<StorageOperation, HistogramMetric>,
      backendHealth,
      overallHealth,
    };
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    this.operationsTotal.clear();
    this.operationsByBackend.clear();
    this.errorsTotal.clear();
    this.itemCounts.clear();
    this.sizeCounts.clear();
    this.latencyHistograms.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }

  // ===========================================================================
  // Prometheus Export
  // ===========================================================================

  /**
   * Export metrics in Prometheus format
   */
  toPrometheus(): string {
    const lines: string[] = [];
    const snapshot = this.getSnapshot();

    // Operations total
    lines.push('# HELP uess_operations_total Total number of storage operations');
    lines.push('# TYPE uess_operations_total counter');
    for (const [op, count] of Object.entries(snapshot.operationsTotal)) {
      lines.push(`uess_operations_total{operation="${op}"} ${count}`);
    }

    // Errors total
    lines.push('# HELP uess_errors_total Total number of storage errors');
    lines.push('# TYPE uess_errors_total counter');
    for (const [op, count] of Object.entries(snapshot.errorsTotal)) {
      lines.push(`uess_errors_total{operation="${op}"} ${count}`);
    }

    // Item count
    lines.push('# HELP uess_items_count Number of items per backend');
    lines.push('# TYPE uess_items_count gauge');
    for (const [backend, count] of Object.entries(snapshot.itemCount)) {
      lines.push(`uess_items_count{backend="${backend}"} ${count}`);
    }

    // Size bytes
    lines.push('# HELP uess_size_bytes Total size in bytes per backend');
    lines.push('# TYPE uess_size_bytes gauge');
    for (const [backend, size] of Object.entries(snapshot.totalSizeBytes)) {
      lines.push(`uess_size_bytes{backend="${backend}"} ${size}`);
    }

    // Cache hit rate
    lines.push('# HELP uess_cache_hit_rate Cache hit rate');
    lines.push('# TYPE uess_cache_hit_rate gauge');
    lines.push(`uess_cache_hit_rate ${snapshot.cacheHitRate}`);

    // Latency histograms
    lines.push('# HELP uess_operation_latency_ms Operation latency in milliseconds');
    lines.push('# TYPE uess_operation_latency_ms histogram');
    for (const [op, hist] of Object.entries(snapshot.operationLatencyMs)) {
      for (const bucket of hist.buckets) {
        const le = bucket.le === Infinity ? '+Inf' : bucket.le;
        lines.push(`uess_operation_latency_ms_bucket{operation="${op}",le="${le}"} ${bucket.count}`);
      }
      lines.push(`uess_operation_latency_ms_sum{operation="${op}"} ${hist.sum}`);
      lines.push(`uess_operation_latency_ms_count{operation="${op}"} ${hist.count}`);
    }

    return lines.join('\n');
  }

  private calculateOverallHealth(backendHealth: Record<BackendType, HealthStatus>): HealthStatus {
    const statuses = Object.values(backendHealth);
    if (statuses.every(s => s === 'healthy')) return 'healthy';
    if (statuses.some(s => s === 'unhealthy')) return 'unhealthy';
    if (statuses.some(s => s === 'degraded')) return 'degraded';
    return 'unknown';
  }
}

// =============================================================================
// Tracing
// =============================================================================

/**
 * Storage Tracer
 */
export class StorageTracer {
  private spans: Map<string, TraceSpan> = new Map();
  private activeSpans: Map<string, TraceSpan> = new Map();

  /**
   * Start a new trace span
   */
  startSpan(
    operationName: string,
    attributes?: Record<string, string | number | boolean>,
    parentSpanId?: string
  ): TraceSpan {
    const traceId = parentSpanId
      ? this.spans.get(parentSpanId)?.traceId ?? this.generateId()
      : this.generateId();

    const span: TraceSpan = {
      traceId,
      spanId: this.generateId(),
      parentSpanId,
      operationName,
      startTime: Date.now(),
      status: 'ok',
      attributes: attributes ?? {},
      events: [],
    };

    this.activeSpans.set(span.spanId, span);
    return span;
  }

  /**
   * Add an event to a span
   */
  addEvent(
    spanId: string,
    name: string,
    attributes?: Record<string, unknown>
  ): void {
    const span = this.activeSpans.get(spanId);
    if (span) {
      span.events.push({
        name,
        timestamp: Date.now(),
        attributes,
      });
    }
  }

  /**
   * Set span attribute
   */
  setAttribute(spanId: string, key: string, value: string | number | boolean): void {
    const span = this.activeSpans.get(spanId);
    if (span) {
      span.attributes[key] = value;
    }
  }

  /**
   * End a span
   */
  endSpan(spanId: string, status: 'ok' | 'error' = 'ok'): TraceSpan | undefined {
    const span = this.activeSpans.get(spanId);
    if (!span) return undefined;

    span.endTime = Date.now();
    span.duration = span.endTime - span.startTime;
    span.status = status;

    this.activeSpans.delete(spanId);
    this.spans.set(spanId, span);

    return span;
  }

  /**
   * Get completed spans for a trace
   */
  getTrace(traceId: string): TraceSpan[] {
    return Array.from(this.spans.values()).filter(s => s.traceId === traceId);
  }

  /**
   * Get all completed spans
   */
  getAllSpans(): TraceSpan[] {
    return Array.from(this.spans.values());
  }

  /**
   * Clear old spans
   */
  clearOlderThan(maxAgeMs: number): number {
    const cutoff = Date.now() - maxAgeMs;
    let removed = 0;

    for (const [id, span] of this.spans.entries()) {
      if (span.startTime < cutoff) {
        this.spans.delete(id);
        removed++;
      }
    }

    return removed;
  }

  /**
   * Export traces in Jaeger format
   */
  toJaeger(): object {
    const traces: Record<string, object[]> = {};

    for (const span of this.spans.values()) {
      if (!traces[span.traceId]) {
        traces[span.traceId] = [];
      }

      traces[span.traceId].push({
        traceID: span.traceId,
        spanID: span.spanId,
        parentSpanID: span.parentSpanId,
        operationName: span.operationName,
        startTime: span.startTime * 1000, // Jaeger uses microseconds
        duration: (span.duration ?? 0) * 1000,
        tags: Object.entries(span.attributes).map(([key, value]) => ({
          key,
          type: typeof value,
          value,
        })),
        logs: span.events.map(e => ({
          timestamp: e.timestamp * 1000,
          fields: [{ key: 'event', value: e.name }],
        })),
      });
    }

    return { data: Object.values(traces) };
  }

  private generateId(): string {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  }
}

// =============================================================================
// Health Checker
// =============================================================================

/**
 * Health check function type
 */
export type HealthCheckFn = () => Promise<HealthCheckResult>;

/**
 * Storage Health Checker
 */
export class StorageHealthChecker {
  private checks: Map<string, HealthCheckFn> = new Map();
  private lastResults: Map<string, HealthCheckResult> = new Map();
  private checkInterval?: NodeJS.Timeout;

  /**
   * Register a health check
   */
  registerCheck(name: string, checkFn: HealthCheckFn): void {
    this.checks.set(name, checkFn);
  }

  /**
   * Remove a health check
   */
  removeCheck(name: string): void {
    this.checks.delete(name);
    this.lastResults.delete(name);
  }

  /**
   * Run a specific health check
   */
  async runCheck(name: string): Promise<HealthCheckResult> {
    const checkFn = this.checks.get(name);
    if (!checkFn) {
      return {
        name,
        status: 'unknown',
        message: 'Check not found',
        lastCheck: Date.now(),
      };
    }

    const startTime = Date.now();
    try {
      const result = await checkFn();
      result.latencyMs = Date.now() - startTime;
      result.lastCheck = Date.now();
      this.lastResults.set(name, result);
      return result;
    } catch (error) {
      const result: HealthCheckResult = {
        name,
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Check failed',
        latencyMs: Date.now() - startTime,
        lastCheck: Date.now(),
      };
      this.lastResults.set(name, result);
      return result;
    }
  }

  /**
   * Run all health checks
   */
  async runAllChecks(): Promise<HealthCheckResult[]> {
    const results: HealthCheckResult[] = [];

    for (const name of this.checks.keys()) {
      const result = await this.runCheck(name);
      results.push(result);
    }

    return results;
  }

  /**
   * Get last result for a check
   */
  getLastResult(name: string): HealthCheckResult | undefined {
    return this.lastResults.get(name);
  }

  /**
   * Get all last results
   */
  getAllLastResults(): HealthCheckResult[] {
    return Array.from(this.lastResults.values());
  }

  /**
   * Get overall health status
   */
  getOverallStatus(): HealthStatus {
    const results = this.getAllLastResults();
    if (results.length === 0) return 'unknown';
    if (results.some(r => r.status === 'unhealthy')) return 'unhealthy';
    if (results.some(r => r.status === 'degraded')) return 'degraded';
    if (results.every(r => r.status === 'healthy')) return 'healthy';
    return 'unknown';
  }

  /**
   * Start periodic health checks
   */
  startPeriodicChecks(intervalMs: number): void {
    this.stopPeriodicChecks();
    this.checkInterval = setInterval(() => {
      this.runAllChecks().catch(console.error);
    }, intervalMs);
  }

  /**
   * Stop periodic health checks
   */
  stopPeriodicChecks(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = undefined;
    }
  }

  /**
   * Export health status in standard format
   */
  toHealthEndpoint(): {
    status: HealthStatus;
    checks: Record<string, HealthCheckResult>;
    timestamp: number;
  } {
    const checks: Record<string, HealthCheckResult> = {};
    for (const [name, result] of this.lastResults.entries()) {
      checks[name] = result;
    }

    return {
      status: this.getOverallStatus(),
      checks,
      timestamp: Date.now(),
    };
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a metrics collector
 */
export function createMetricsCollector(): StorageMetricsCollector {
  return new StorageMetricsCollector();
}

/**
 * Create a tracer
 */
export function createTracer(): StorageTracer {
  return new StorageTracer();
}

/**
 * Create a health checker
 */
export function createHealthChecker(): StorageHealthChecker {
  return new StorageHealthChecker();
}

/**
 * Create standard backend health checks
 */
export function createBackendHealthChecks(
  healthChecker: StorageHealthChecker,
  backends: {
    memory?: { has: (key: string) => Promise<boolean> };
    local?: { has: (key: string) => Promise<boolean> };
    dht?: { has: (key: string) => Promise<boolean> };
    ipfs?: { has: (key: string) => Promise<boolean> };
  }
): void {
  if (backends.memory) {
    healthChecker.registerCheck('memory', async () => {
      try {
        await backends.memory!.has('__health_check__');
        return { name: 'memory', status: 'healthy', lastCheck: Date.now() };
      } catch (e) {
        return {
          name: 'memory',
          status: 'unhealthy',
          message: e instanceof Error ? e.message : 'Unknown error',
          lastCheck: Date.now(),
        };
      }
    });
  }

  if (backends.local) {
    healthChecker.registerCheck('local', async () => {
      try {
        await backends.local!.has('__health_check__');
        return { name: 'local', status: 'healthy', lastCheck: Date.now() };
      } catch (e) {
        return {
          name: 'local',
          status: 'unhealthy',
          message: e instanceof Error ? e.message : 'Unknown error',
          lastCheck: Date.now(),
        };
      }
    });
  }

  if (backends.dht) {
    healthChecker.registerCheck('dht', async () => {
      try {
        const start = Date.now();
        await backends.dht!.has('__health_check__');
        const latency = Date.now() - start;

        // DHT is degraded if latency > 1s
        const status: HealthStatus = latency > 1000 ? 'degraded' : 'healthy';
        return {
          name: 'dht',
          status,
          latencyMs: latency,
          lastCheck: Date.now(),
        };
      } catch (e) {
        return {
          name: 'dht',
          status: 'unhealthy',
          message: e instanceof Error ? e.message : 'Unknown error',
          lastCheck: Date.now(),
        };
      }
    });
  }

  if (backends.ipfs) {
    healthChecker.registerCheck('ipfs', async () => {
      try {
        const start = Date.now();
        await backends.ipfs!.has('__health_check__');
        const latency = Date.now() - start;

        // IPFS is degraded if latency > 2s
        const status: HealthStatus = latency > 2000 ? 'degraded' : 'healthy';
        return {
          name: 'ipfs',
          status,
          latencyMs: latency,
          lastCheck: Date.now(),
        };
      } catch (e) {
        return {
          name: 'ipfs',
          status: 'unhealthy',
          message: e instanceof Error ? e.message : 'Unknown error',
          lastCheck: Date.now(),
        };
      }
    });
  }
}

/**
 * Create an instrumented operation wrapper
 */
export function instrumentedOperation<T>(
  metrics: StorageMetricsCollector,
  tracer: StorageTracer,
  operation: StorageOperation,
  backend: BackendType,
  fn: () => Promise<T>
): () => Promise<T> {
  return async () => {
    const span = tracer.startSpan(`storage.${operation}`, {
      backend,
      operation,
    });

    const start = Date.now();

    try {
      const result = await fn();

      const latency = Date.now() - start;
      metrics.recordOperation(operation, backend);
      metrics.recordLatency(operation, latency);
      tracer.endSpan(span.spanId, 'ok');

      return result;
    } catch (error) {
      const latency = Date.now() - start;
      metrics.recordOperation(operation, backend);
      metrics.recordLatency(operation, latency);
      metrics.recordError(operation);

      tracer.setAttribute(span.spanId, 'error', true);
      tracer.setAttribute(span.spanId, 'error.message', error instanceof Error ? error.message : 'Unknown');
      tracer.endSpan(span.spanId, 'error');

      throw error;
    }
  };
}

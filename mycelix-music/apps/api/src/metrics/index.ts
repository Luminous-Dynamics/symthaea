// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Prometheus Metrics
 *
 * Comprehensive metrics instrumentation for observability.
 * Includes HTTP request metrics, business metrics, and system health.
 */

import { Request, Response, NextFunction } from 'express';
import { Pool } from 'pg';

/**
 * Histogram bucket boundaries
 */
const HTTP_LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];
const DB_LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1];

/**
 * Metric types
 */
interface Counter {
  name: string;
  help: string;
  labels: string[];
  values: Map<string, number>;
}

interface Gauge {
  name: string;
  help: string;
  labels: string[];
  values: Map<string, number>;
}

interface Histogram {
  name: string;
  help: string;
  labels: string[];
  buckets: number[];
  values: Map<string, { buckets: number[]; sum: number; count: number }>;
}

/**
 * Metrics registry
 */
class MetricsRegistry {
  private counters: Map<string, Counter> = new Map();
  private gauges: Map<string, Gauge> = new Map();
  private histograms: Map<string, Histogram> = new Map();
  private prefix = 'mycelix_';

  /**
   * Create a counter metric
   */
  createCounter(name: string, help: string, labels: string[] = []): void {
    this.counters.set(name, {
      name: this.prefix + name,
      help,
      labels,
      values: new Map(),
    });
  }

  /**
   * Increment a counter
   */
  incCounter(name: string, labels: Record<string, string> = {}, value = 1): void {
    const counter = this.counters.get(name);
    if (!counter) return;

    const key = this.labelsToKey(labels);
    const current = counter.values.get(key) || 0;
    counter.values.set(key, current + value);
  }

  /**
   * Create a gauge metric
   */
  createGauge(name: string, help: string, labels: string[] = []): void {
    this.gauges.set(name, {
      name: this.prefix + name,
      help,
      labels,
      values: new Map(),
    });
  }

  /**
   * Set a gauge value
   */
  setGauge(name: string, value: number, labels: Record<string, string> = {}): void {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    const key = this.labelsToKey(labels);
    gauge.values.set(key, value);
  }

  /**
   * Increment a gauge
   */
  incGauge(name: string, labels: Record<string, string> = {}, value = 1): void {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    const key = this.labelsToKey(labels);
    const current = gauge.values.get(key) || 0;
    gauge.values.set(key, current + value);
  }

  /**
   * Decrement a gauge
   */
  decGauge(name: string, labels: Record<string, string> = {}, value = 1): void {
    this.incGauge(name, labels, -value);
  }

  /**
   * Create a histogram metric
   */
  createHistogram(
    name: string,
    help: string,
    labels: string[] = [],
    buckets: number[] = HTTP_LATENCY_BUCKETS
  ): void {
    this.histograms.set(name, {
      name: this.prefix + name,
      help,
      labels,
      buckets: [...buckets].sort((a, b) => a - b),
      values: new Map(),
    });
  }

  /**
   * Observe a histogram value
   */
  observeHistogram(name: string, value: number, labels: Record<string, string> = {}): void {
    const histogram = this.histograms.get(name);
    if (!histogram) return;

    const key = this.labelsToKey(labels);
    let data = histogram.values.get(key);

    if (!data) {
      data = {
        buckets: new Array(histogram.buckets.length).fill(0),
        sum: 0,
        count: 0,
      };
      histogram.values.set(key, data);
    }

    // Update buckets
    for (let i = 0; i < histogram.buckets.length; i++) {
      if (value <= histogram.buckets[i]) {
        data.buckets[i]++;
      }
    }

    data.sum += value;
    data.count++;
  }

  /**
   * Convert labels to a unique key
   */
  private labelsToKey(labels: Record<string, string>): string {
    if (Object.keys(labels).length === 0) return '';
    return Object.entries(labels)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}="${v}"`)
      .join(',');
  }

  /**
   * Format labels for Prometheus output
   */
  private formatLabels(key: string): string {
    if (!key) return '';
    return `{${key}}`;
  }

  /**
   * Export metrics in Prometheus format
   */
  export(): string {
    const lines: string[] = [];

    // Export counters
    for (const counter of this.counters.values()) {
      lines.push(`# HELP ${counter.name} ${counter.help}`);
      lines.push(`# TYPE ${counter.name} counter`);
      for (const [key, value] of counter.values) {
        lines.push(`${counter.name}${this.formatLabels(key)} ${value}`);
      }
    }

    // Export gauges
    for (const gauge of this.gauges.values()) {
      lines.push(`# HELP ${gauge.name} ${gauge.help}`);
      lines.push(`# TYPE ${gauge.name} gauge`);
      for (const [key, value] of gauge.values) {
        lines.push(`${gauge.name}${this.formatLabels(key)} ${value}`);
      }
    }

    // Export histograms
    for (const histogram of this.histograms.values()) {
      lines.push(`# HELP ${histogram.name} ${histogram.help}`);
      lines.push(`# TYPE ${histogram.name} histogram`);
      for (const [key, data] of histogram.values) {
        const labelPart = key ? `${key},` : '';
        for (let i = 0; i < histogram.buckets.length; i++) {
          const le = histogram.buckets[i];
          const bucketLabel = `${labelPart}le="${le}"`;
          lines.push(`${histogram.name}_bucket{${bucketLabel}} ${data.buckets[i]}`);
        }
        lines.push(`${histogram.name}_bucket{${labelPart}le="+Inf"} ${data.count}`);
        lines.push(`${histogram.name}_sum${this.formatLabels(key)} ${data.sum}`);
        lines.push(`${histogram.name}_count${this.formatLabels(key)} ${data.count}`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    for (const counter of this.counters.values()) {
      counter.values.clear();
    }
    for (const gauge of this.gauges.values()) {
      gauge.values.clear();
    }
    for (const histogram of this.histograms.values()) {
      histogram.values.clear();
    }
  }
}

/**
 * Global metrics registry
 */
let registry: MetricsRegistry | null = null;

export function getMetrics(): MetricsRegistry {
  if (!registry) {
    registry = new MetricsRegistry();
    initializeMetrics(registry);
  }
  return registry;
}

export function resetMetrics(): void {
  registry = null;
}

/**
 * Initialize all application metrics
 */
function initializeMetrics(metrics: MetricsRegistry): void {
  // HTTP request metrics
  metrics.createCounter('http_requests_total', 'Total HTTP requests', [
    'method',
    'path',
    'status',
  ]);
  metrics.createHistogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'path', 'status'],
    HTTP_LATENCY_BUCKETS
  );
  metrics.createGauge('http_requests_in_flight', 'Current in-flight requests', ['method']);

  // Database metrics
  metrics.createCounter('db_queries_total', 'Total database queries', ['operation', 'table']);
  metrics.createHistogram(
    'db_query_duration_seconds',
    'Database query latency',
    ['operation', 'table'],
    DB_LATENCY_BUCKETS
  );
  metrics.createGauge('db_pool_connections', 'Database pool connections', ['state']);

  // Business metrics
  metrics.createCounter('plays_total', 'Total plays recorded', ['source']);
  metrics.createCounter('songs_created_total', 'Total songs created', []);
  metrics.createCounter('earnings_total_wei', 'Total earnings in wei', ['currency']);
  metrics.createGauge('unique_listeners', 'Unique listeners count', ['period']);
  metrics.createGauge('active_songs', 'Active songs count', []);

  // Cache metrics
  metrics.createCounter('cache_hits_total', 'Cache hits', ['cache']);
  metrics.createCounter('cache_misses_total', 'Cache misses', ['cache']);
  metrics.createGauge('cache_size_bytes', 'Cache size in bytes', ['cache']);

  // Job metrics
  metrics.createCounter('job_executions_total', 'Job executions', ['job', 'status']);
  metrics.createHistogram('job_duration_seconds', 'Job execution time', ['job'], [
    0.1, 0.5, 1, 5, 10, 30, 60, 120,
  ]);
  metrics.createGauge('job_last_success_timestamp', 'Last successful job run', ['job']);

  // Rate limit metrics
  metrics.createCounter('rate_limit_hits_total', 'Rate limit hits', ['endpoint', 'tier']);

  // Error metrics
  metrics.createCounter('errors_total', 'Total errors', ['type', 'code']);

  // System metrics
  metrics.createGauge('nodejs_heap_size_bytes', 'Node.js heap size', ['type']);
  metrics.createGauge('nodejs_event_loop_lag_seconds', 'Event loop lag', []);
  metrics.createGauge('process_uptime_seconds', 'Process uptime', []);
}

/**
 * HTTP metrics middleware
 */
export function metricsMiddleware() {
  const metrics = getMetrics();

  return (req: Request, res: Response, next: NextFunction): void => {
    const start = process.hrtime.bigint();
    const method = req.method;

    // Increment in-flight
    metrics.incGauge('http_requests_in_flight', { method });

    // Normalize path for metrics (replace IDs with placeholders)
    const path = normalizePath(req.path);

    // On response finish
    res.on('finish', () => {
      const end = process.hrtime.bigint();
      const durationSeconds = Number(end - start) / 1e9;
      const status = String(res.statusCode);
      const statusClass = `${Math.floor(res.statusCode / 100)}xx`;

      // Record metrics
      metrics.incCounter('http_requests_total', { method, path, status: statusClass });
      metrics.observeHistogram('http_request_duration_seconds', durationSeconds, {
        method,
        path,
        status: statusClass,
      });
      metrics.decGauge('http_requests_in_flight', { method });
    });

    next();
  };
}

/**
 * Normalize path for consistent metrics
 */
function normalizePath(path: string): string {
  return path
    // Replace UUIDs
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, ':id')
    // Replace wallet addresses
    .replace(/0x[a-fA-F0-9]{40}/g, ':address')
    // Replace numeric IDs
    .replace(/\/\d+(?=\/|$)/g, '/:id');
}

/**
 * Database query metrics wrapper
 */
export function createMetricsPool(pool: Pool): Pool {
  const metrics = getMetrics();
  const originalQuery = pool.query.bind(pool);

  // Wrap query method
  (pool as any).query = async (...args: any[]) => {
    const start = process.hrtime.bigint();
    const queryText = typeof args[0] === 'string' ? args[0] : args[0]?.text || '';

    // Extract operation and table
    const operation = extractOperation(queryText);
    const table = extractTable(queryText);

    try {
      const result = await originalQuery(...args);

      const end = process.hrtime.bigint();
      const durationSeconds = Number(end - start) / 1e9;

      metrics.incCounter('db_queries_total', { operation, table });
      metrics.observeHistogram('db_query_duration_seconds', durationSeconds, {
        operation,
        table,
      });

      return result;
    } catch (error) {
      metrics.incCounter('errors_total', { type: 'database', code: 'query_error' });
      throw error;
    }
  };

  // Monitor pool events
  pool.on('connect', () => {
    metrics.incGauge('db_pool_connections', { state: 'active' });
  });

  pool.on('remove', () => {
    metrics.decGauge('db_pool_connections', { state: 'active' });
  });

  return pool;
}

/**
 * Extract SQL operation from query
 */
function extractOperation(query: string): string {
  const normalized = query.trim().toUpperCase();
  if (normalized.startsWith('SELECT')) return 'select';
  if (normalized.startsWith('INSERT')) return 'insert';
  if (normalized.startsWith('UPDATE')) return 'update';
  if (normalized.startsWith('DELETE')) return 'delete';
  return 'other';
}

/**
 * Extract table name from query
 */
function extractTable(query: string): string {
  const normalized = query.trim().toLowerCase();

  // Match FROM clause
  const fromMatch = normalized.match(/from\s+(\w+)/);
  if (fromMatch) return fromMatch[1];

  // Match INTO clause
  const intoMatch = normalized.match(/into\s+(\w+)/);
  if (intoMatch) return intoMatch[1];

  // Match UPDATE clause
  const updateMatch = normalized.match(/update\s+(\w+)/);
  if (updateMatch) return updateMatch[1];

  return 'unknown';
}

/**
 * Record business metrics
 */
export const businessMetrics = {
  recordPlay(source: string = 'api'): void {
    getMetrics().incCounter('plays_total', { source });
  },

  recordSongCreated(): void {
    getMetrics().incCounter('songs_created_total', {});
  },

  recordEarnings(amountWei: bigint, currency: string = 'ETH'): void {
    getMetrics().incCounter('earnings_total_wei', { currency }, Number(amountWei));
  },

  setUniqueListeners(count: number, period: string = '24h'): void {
    getMetrics().setGauge('unique_listeners', count, { period });
  },

  setActiveSongs(count: number): void {
    getMetrics().setGauge('active_songs', count, {});
  },
};

/**
 * Record cache metrics
 */
export const cacheMetrics = {
  hit(cacheName: string): void {
    getMetrics().incCounter('cache_hits_total', { cache: cacheName });
  },

  miss(cacheName: string): void {
    getMetrics().incCounter('cache_misses_total', { cache: cacheName });
  },

  setSize(cacheName: string, sizeBytes: number): void {
    getMetrics().setGauge('cache_size_bytes', sizeBytes, { cache: cacheName });
  },
};

/**
 * Record job metrics
 */
export const jobMetrics = {
  started(jobName: string): () => void {
    const start = process.hrtime.bigint();

    return () => {
      const end = process.hrtime.bigint();
      const durationSeconds = Number(end - start) / 1e9;

      getMetrics().incCounter('job_executions_total', { job: jobName, status: 'success' });
      getMetrics().observeHistogram('job_duration_seconds', durationSeconds, { job: jobName });
      getMetrics().setGauge('job_last_success_timestamp', Date.now() / 1000, { job: jobName });
    };
  },

  failed(jobName: string): void {
    getMetrics().incCounter('job_executions_total', { job: jobName, status: 'failed' });
  },
};

/**
 * Record error metrics
 */
export function recordError(type: string, code: string): void {
  getMetrics().incCounter('errors_total', { type, code });
}

/**
 * Record rate limit hit
 */
export function recordRateLimitHit(endpoint: string, tier: string = 'default'): void {
  getMetrics().incCounter('rate_limit_hits_total', { endpoint, tier });
}

/**
 * Collect system metrics
 */
export function collectSystemMetrics(): void {
  const metrics = getMetrics();
  const memUsage = process.memoryUsage();

  metrics.setGauge('nodejs_heap_size_bytes', memUsage.heapUsed, { type: 'used' });
  metrics.setGauge('nodejs_heap_size_bytes', memUsage.heapTotal, { type: 'total' });
  metrics.setGauge('nodejs_heap_size_bytes', memUsage.external, { type: 'external' });
  metrics.setGauge('nodejs_heap_size_bytes', memUsage.rss, { type: 'rss' });
  metrics.setGauge('process_uptime_seconds', process.uptime(), {});
}

/**
 * Create metrics endpoint handler
 */
export function createMetricsHandler() {
  return (req: Request, res: Response): void => {
    collectSystemMetrics();

    res.set('Content-Type', 'text/plain; charset=utf-8');
    res.send(getMetrics().export());
  };
}

/**
 * Create metrics router
 */
export function createMetricsRouter() {
  const { Router } = require('express');
  const router = Router();

  router.get('/', createMetricsHandler());

  return router;
}

export default {
  getMetrics,
  resetMetrics,
  metricsMiddleware,
  createMetricsPool,
  createMetricsRouter,
  businessMetrics,
  cacheMetrics,
  jobMetrics,
  recordError,
  recordRateLimitHit,
  collectSystemMetrics,
};

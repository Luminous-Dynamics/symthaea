// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Metrics Service
 * Prometheus-compatible metrics collection and export
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';

// Metric types
type MetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

interface MetricLabels {
  [key: string]: string;
}

interface MetricValue {
  value: number;
  timestamp: number;
  labels: MetricLabels;
}

interface HistogramBucket {
  le: number;
  count: number;
}

interface HistogramValue {
  buckets: HistogramBucket[];
  sum: number;
  count: number;
  labels: MetricLabels;
}

// Metric configuration
interface MetricConfig {
  name: string;
  help: string;
  type: MetricType;
  labelNames?: string[];
  buckets?: number[]; // For histograms
}

// Default histogram buckets for latency (in ms)
const DEFAULT_LATENCY_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

// Default histogram buckets for size (in bytes)
const DEFAULT_SIZE_BUCKETS = [100, 1000, 10000, 100000, 1000000, 10000000];

/**
 * Metrics Registry
 */
class MetricsRegistry {
  private metrics: Map<string, Metric> = new Map();
  private prefix: string;

  constructor(prefix = 'mycelix') {
    this.prefix = prefix;
    this.initDefaultMetrics();
  }

  private initDefaultMetrics(): void {
    // HTTP request metrics
    this.createHistogram({
      name: 'http_request_duration_ms',
      help: 'HTTP request duration in milliseconds',
      labelNames: ['method', 'route', 'status_code'],
      buckets: DEFAULT_LATENCY_BUCKETS,
    });

    this.createCounter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'route', 'status_code'],
    });

    this.createHistogram({
      name: 'http_request_size_bytes',
      help: 'HTTP request body size in bytes',
      labelNames: ['method', 'route'],
      buckets: DEFAULT_SIZE_BUCKETS,
    });

    this.createHistogram({
      name: 'http_response_size_bytes',
      help: 'HTTP response body size in bytes',
      labelNames: ['method', 'route'],
      buckets: DEFAULT_SIZE_BUCKETS,
    });

    // Database metrics
    this.createHistogram({
      name: 'db_query_duration_ms',
      help: 'Database query duration in milliseconds',
      labelNames: ['operation', 'table'],
      buckets: DEFAULT_LATENCY_BUCKETS,
    });

    this.createCounter({
      name: 'db_queries_total',
      help: 'Total number of database queries',
      labelNames: ['operation', 'table', 'status'],
    });

    // Cache metrics
    this.createCounter({
      name: 'cache_hits_total',
      help: 'Total number of cache hits',
      labelNames: ['cache'],
    });

    this.createCounter({
      name: 'cache_misses_total',
      help: 'Total number of cache misses',
      labelNames: ['cache'],
    });

    // Business metrics
    this.createCounter({
      name: 'song_plays_total',
      help: 'Total number of song plays',
      labelNames: ['genre', 'subscription_tier'],
    });

    this.createCounter({
      name: 'uploads_total',
      help: 'Total number of uploads',
      labelNames: ['type', 'status'],
    });

    this.createGauge({
      name: 'active_streams',
      help: 'Number of active audio streams',
      labelNames: ['quality'],
    });

    this.createGauge({
      name: 'active_websocket_connections',
      help: 'Number of active WebSocket connections',
      labelNames: ['namespace'],
    });

    // System metrics
    this.createGauge({
      name: 'nodejs_heap_size_bytes',
      help: 'Node.js heap size in bytes',
      labelNames: ['type'],
    });

    this.createGauge({
      name: 'nodejs_active_handles',
      help: 'Number of active handles',
    });

    this.createGauge({
      name: 'nodejs_active_requests',
      help: 'Number of active requests',
    });

    // Event loop metrics
    this.createGauge({
      name: 'nodejs_eventloop_lag_ms',
      help: 'Event loop lag in milliseconds',
    });
  }

  createCounter(config: Omit<MetricConfig, 'type'>): Counter {
    const fullName = `${this.prefix}_${config.name}`;
    const counter = new Counter(fullName, config.help, config.labelNames);
    this.metrics.set(fullName, counter);
    return counter;
  }

  createGauge(config: Omit<MetricConfig, 'type'>): Gauge {
    const fullName = `${this.prefix}_${config.name}`;
    const gauge = new Gauge(fullName, config.help, config.labelNames);
    this.metrics.set(fullName, gauge);
    return gauge;
  }

  createHistogram(config: Omit<MetricConfig, 'type'> & { buckets?: number[] }): Histogram {
    const fullName = `${this.prefix}_${config.name}`;
    const histogram = new Histogram(
      fullName,
      config.help,
      config.labelNames,
      config.buckets || DEFAULT_LATENCY_BUCKETS
    );
    this.metrics.set(fullName, histogram);
    return histogram;
  }

  getMetric(name: string): Metric | undefined {
    return this.metrics.get(`${this.prefix}_${name}`);
  }

  /**
   * Export all metrics in Prometheus format
   */
  export(): string {
    const lines: string[] = [];

    for (const metric of this.metrics.values()) {
      lines.push(...metric.export());
      lines.push('');
    }

    return lines.join('\n');
  }

  /**
   * Export metrics as JSON
   */
  exportJson(): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    for (const [name, metric] of this.metrics) {
      result[name] = metric.exportJson();
    }

    return result;
  }

  /**
   * Collect system metrics
   */
  collectSystemMetrics(): void {
    const memory = process.memoryUsage();

    const heapGauge = this.getMetric('nodejs_heap_size_bytes') as Gauge;
    if (heapGauge) {
      heapGauge.set({ type: 'used' }, memory.heapUsed);
      heapGauge.set({ type: 'total' }, memory.heapTotal);
      heapGauge.set({ type: 'external' }, memory.external);
    }

    // Event loop lag
    const start = Date.now();
    setImmediate(() => {
      const lag = Date.now() - start;
      const lagGauge = this.getMetric('nodejs_eventloop_lag_ms') as Gauge;
      if (lagGauge) {
        lagGauge.set({}, lag);
      }
    });
  }
}

/**
 * Base Metric class
 */
abstract class Metric {
  constructor(
    protected name: string,
    protected help: string,
    protected labelNames: string[] = []
  ) {}

  protected labelsToString(labels: MetricLabels): string {
    const entries = Object.entries(labels);
    if (entries.length === 0) return '';

    const labelStr = entries
      .map(([k, v]) => `${k}="${v}"`)
      .join(',');

    return `{${labelStr}}`;
  }

  abstract export(): string[];
  abstract exportJson(): unknown;
}

/**
 * Counter - monotonically increasing value
 */
class Counter extends Metric {
  private values: Map<string, MetricValue> = new Map();

  inc(labels: MetricLabels = {}, value = 1): void {
    const key = this.labelsToKey(labels);
    const current = this.values.get(key);

    if (current) {
      current.value += value;
      current.timestamp = Date.now();
    } else {
      this.values.set(key, {
        value,
        timestamp: Date.now(),
        labels,
      });
    }
  }

  private labelsToKey(labels: MetricLabels): string {
    return JSON.stringify(labels);
  }

  export(): string[] {
    const lines: string[] = [
      `# HELP ${this.name} ${this.help}`,
      `# TYPE ${this.name} counter`,
    ];

    for (const mv of this.values.values()) {
      lines.push(`${this.name}${this.labelsToString(mv.labels)} ${mv.value}`);
    }

    return lines;
  }

  exportJson(): unknown {
    return {
      type: 'counter',
      help: this.help,
      values: Array.from(this.values.values()),
    };
  }
}

/**
 * Gauge - value that can go up and down
 */
class Gauge extends Metric {
  private values: Map<string, MetricValue> = new Map();

  set(labels: MetricLabels = {}, value: number): void {
    const key = this.labelsToKey(labels);
    this.values.set(key, {
      value,
      timestamp: Date.now(),
      labels,
    });
  }

  inc(labels: MetricLabels = {}, value = 1): void {
    const key = this.labelsToKey(labels);
    const current = this.values.get(key);

    if (current) {
      current.value += value;
      current.timestamp = Date.now();
    } else {
      this.set(labels, value);
    }
  }

  dec(labels: MetricLabels = {}, value = 1): void {
    this.inc(labels, -value);
  }

  private labelsToKey(labels: MetricLabels): string {
    return JSON.stringify(labels);
  }

  export(): string[] {
    const lines: string[] = [
      `# HELP ${this.name} ${this.help}`,
      `# TYPE ${this.name} gauge`,
    ];

    for (const mv of this.values.values()) {
      lines.push(`${this.name}${this.labelsToString(mv.labels)} ${mv.value}`);
    }

    return lines;
  }

  exportJson(): unknown {
    return {
      type: 'gauge',
      help: this.help,
      values: Array.from(this.values.values()),
    };
  }
}

/**
 * Histogram - distribution of values
 */
class Histogram extends Metric {
  private values: Map<string, HistogramValue> = new Map();
  private buckets: number[];

  constructor(name: string, help: string, labelNames: string[] = [], buckets: number[]) {
    super(name, help, labelNames);
    this.buckets = [...buckets].sort((a, b) => a - b);
  }

  observe(labels: MetricLabels = {}, value: number): void {
    const key = this.labelsToKey(labels);
    let histValue = this.values.get(key);

    if (!histValue) {
      histValue = {
        buckets: this.buckets.map((le) => ({ le, count: 0 })),
        sum: 0,
        count: 0,
        labels,
      };
      this.values.set(key, histValue);
    }

    histValue.sum += value;
    histValue.count += 1;

    for (const bucket of histValue.buckets) {
      if (value <= bucket.le) {
        bucket.count += 1;
      }
    }
  }

  /**
   * Create a timer that observes duration
   */
  startTimer(labels: MetricLabels = {}): () => number {
    const start = Date.now();
    return () => {
      const duration = Date.now() - start;
      this.observe(labels, duration);
      return duration;
    };
  }

  private labelsToKey(labels: MetricLabels): string {
    return JSON.stringify(labels);
  }

  export(): string[] {
    const lines: string[] = [
      `# HELP ${this.name} ${this.help}`,
      `# TYPE ${this.name} histogram`,
    ];

    for (const hv of this.values.values()) {
      const labelStr = this.labelsToString(hv.labels);

      for (const bucket of hv.buckets) {
        const bucketLabels = labelStr
          ? labelStr.slice(0, -1) + `,le="${bucket.le}"}`
          : `{le="${bucket.le}"}`;
        lines.push(`${this.name}_bucket${bucketLabels} ${bucket.count}`);
      }

      // +Inf bucket
      const infLabels = labelStr
        ? labelStr.slice(0, -1) + ',le="+Inf"}'
        : '{le="+Inf"}';
      lines.push(`${this.name}_bucket${infLabels} ${hv.count}`);

      lines.push(`${this.name}_sum${labelStr} ${hv.sum}`);
      lines.push(`${this.name}_count${labelStr} ${hv.count}`);
    }

    return lines;
  }

  exportJson(): unknown {
    return {
      type: 'histogram',
      help: this.help,
      buckets: this.buckets,
      values: Array.from(this.values.values()),
    };
  }
}

// Singleton registry
export const metrics = new MetricsRegistry();

/**
 * HTTP metrics middleware
 */
export function httpMetricsMiddleware(): RequestHandler {
  const requestDuration = metrics.getMetric('http_request_duration_ms') as Histogram;
  const requestsTotal = metrics.getMetric('http_requests_total') as Counter;
  const requestSize = metrics.getMetric('http_request_size_bytes') as Histogram;
  const responseSize = metrics.getMetric('http_response_size_bytes') as Histogram;

  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = Date.now();

    // Normalize route path
    const route = req.route?.path || req.path.replace(/\/[a-f0-9-]{36}/gi, '/:id');

    // Track request size
    const reqSize = parseInt(req.headers['content-length'] || '0', 10);
    if (reqSize > 0) {
      requestSize.observe({ method: req.method, route }, reqSize);
    }

    // Intercept response end
    const originalEnd = res.end.bind(res);
    res.end = function (...args: Parameters<typeof res.end>) {
      const duration = Date.now() - startTime;
      const statusCode = res.statusCode.toString();

      requestDuration.observe({ method: req.method, route, status_code: statusCode }, duration);
      requestsTotal.inc({ method: req.method, route, status_code: statusCode });

      // Track response size
      const resSize = parseInt(res.getHeader('content-length')?.toString() || '0', 10);
      if (resSize > 0) {
        responseSize.observe({ method: req.method, route }, resSize);
      }

      return originalEnd(...args);
    };

    next();
  };
}

/**
 * Metrics endpoint handler
 */
export function metricsHandler(): RequestHandler {
  return (req: Request, res: Response): void => {
    // Collect system metrics
    metrics.collectSystemMetrics();

    const format = req.query.format;

    if (format === 'json') {
      res.json(metrics.exportJson());
    } else {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.send(metrics.export());
    }
  };
}

// Export types and utilities
export { Counter, Gauge, Histogram, MetricsRegistry };
export type { MetricLabels, MetricConfig };

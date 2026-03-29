// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Observability Metrics Tests
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  MetricsRegistry,
  Counter,
  Gauge,
  Histogram,
  createMetrics,
  httpRequestDurationMs,
  httpRequestTotal,
  activeConnections,
  dbQueryDurationMs,
} from '../metrics';

describe('MetricsRegistry', () => {
  let registry: MetricsRegistry;

  beforeEach(() => {
    registry = new MetricsRegistry();
  });

  describe('Counter', () => {
    it('creates a counter with name and labels', () => {
      const counter = registry.createCounter('test_counter', 'Test counter', ['label1']);

      expect(counter).toBeInstanceOf(Counter);
      expect(counter.name).toBe('test_counter');
    });

    it('increments counter value', () => {
      const counter = registry.createCounter('requests_total', 'Total requests', ['method']);

      counter.inc({ method: 'GET' });
      counter.inc({ method: 'GET' });
      counter.inc({ method: 'POST' });

      expect(counter.get({ method: 'GET' })).toBe(2);
      expect(counter.get({ method: 'POST' })).toBe(1);
    });

    it('increments by custom value', () => {
      const counter = registry.createCounter('bytes_total', 'Total bytes', ['type']);

      counter.inc({ type: 'upload' }, 1024);
      counter.inc({ type: 'upload' }, 2048);

      expect(counter.get({ type: 'upload' })).toBe(3072);
    });

    it('throws on negative increment', () => {
      const counter = registry.createCounter('test', 'Test', []);

      expect(() => counter.inc({}, -1)).toThrow('Counter can only be incremented');
    });
  });

  describe('Gauge', () => {
    it('sets gauge value', () => {
      const gauge = registry.createGauge('temperature', 'Temperature', ['location']);

      gauge.set({ location: 'server1' }, 72);

      expect(gauge.get({ location: 'server1' })).toBe(72);
    });

    it('increments and decrements gauge', () => {
      const gauge = registry.createGauge('connections', 'Active connections', ['service']);

      gauge.set({ service: 'api' }, 5);
      gauge.inc({ service: 'api' });
      gauge.inc({ service: 'api' }, 3);
      gauge.dec({ service: 'api' });

      expect(gauge.get({ service: 'api' })).toBe(8);
    });

    it('allows negative values', () => {
      const gauge = registry.createGauge('delta', 'Delta value', []);

      gauge.set({}, -10);

      expect(gauge.get({})).toBe(-10);
    });

    it('sets to current time', () => {
      const gauge = registry.createGauge('last_updated', 'Last update timestamp', []);

      const before = Date.now();
      gauge.setToCurrentTime({});
      const after = Date.now();

      const value = gauge.get({});
      expect(value).toBeGreaterThanOrEqual(before);
      expect(value).toBeLessThanOrEqual(after);
    });
  });

  describe('Histogram', () => {
    it('observes values', () => {
      const histogram = registry.createHistogram(
        'request_duration',
        'Request duration',
        ['handler'],
        [10, 50, 100, 500, 1000]
      );

      histogram.observe({ handler: 'api' }, 25);
      histogram.observe({ handler: 'api' }, 75);
      histogram.observe({ handler: 'api' }, 150);

      const metrics = histogram.getMetrics({ handler: 'api' });

      expect(metrics.count).toBe(3);
      expect(metrics.sum).toBe(250);
    });

    it('calculates bucket distribution', () => {
      const histogram = registry.createHistogram(
        'latency',
        'Latency',
        [],
        [10, 50, 100]
      );

      histogram.observe({}, 5);   // <= 10
      histogram.observe({}, 15);  // <= 50
      histogram.observe({}, 45);  // <= 50
      histogram.observe({}, 75);  // <= 100
      histogram.observe({}, 150); // > 100

      const metrics = histogram.getMetrics({});

      expect(metrics.buckets[10]).toBe(1);
      expect(metrics.buckets[50]).toBe(3);  // cumulative
      expect(metrics.buckets[100]).toBe(4); // cumulative
      expect(metrics.buckets['+Inf']).toBe(5);
    });

    it('times async functions', async () => {
      const histogram = registry.createHistogram(
        'async_duration',
        'Async operation duration',
        ['operation'],
        [10, 100, 1000]
      );

      const result = await histogram.time({ operation: 'test' }, async () => {
        await new Promise((r) => setTimeout(r, 50));
        return 'done';
      });

      expect(result).toBe('done');

      const metrics = histogram.getMetrics({ operation: 'test' });
      expect(metrics.count).toBe(1);
      expect(metrics.sum).toBeGreaterThanOrEqual(50);
    });
  });

  describe('Prometheus Export', () => {
    it('exports metrics in Prometheus format', () => {
      const counter = registry.createCounter('http_requests', 'HTTP requests', ['method', 'status']);
      counter.inc({ method: 'GET', status: '200' }, 100);
      counter.inc({ method: 'POST', status: '201' }, 50);

      const output = registry.export();

      expect(output).toContain('# HELP http_requests HTTP requests');
      expect(output).toContain('# TYPE http_requests counter');
      expect(output).toContain('http_requests{method="GET",status="200"} 100');
      expect(output).toContain('http_requests{method="POST",status="201"} 50');
    });

    it('exports histogram with buckets', () => {
      const histogram = registry.createHistogram(
        'request_size',
        'Request size',
        ['endpoint'],
        [100, 1000, 10000]
      );

      histogram.observe({ endpoint: '/api' }, 500);
      histogram.observe({ endpoint: '/api' }, 5000);

      const output = registry.export();

      expect(output).toContain('# TYPE request_size histogram');
      expect(output).toContain('request_size_bucket{endpoint="/api",le="100"}');
      expect(output).toContain('request_size_bucket{endpoint="/api",le="1000"}');
      expect(output).toContain('request_size_count{endpoint="/api"} 2');
      expect(output).toContain('request_size_sum{endpoint="/api"} 5500');
    });

    it('exports gauge values', () => {
      const gauge = registry.createGauge('memory_usage', 'Memory usage bytes', ['type']);
      gauge.set({ type: 'heap' }, 1024000);
      gauge.set({ type: 'rss' }, 2048000);

      const output = registry.export();

      expect(output).toContain('# TYPE memory_usage gauge');
      expect(output).toContain('memory_usage{type="heap"} 1024000');
      expect(output).toContain('memory_usage{type="rss"} 2048000');
    });
  });

  describe('Default Metrics', () => {
    it('collects process metrics', () => {
      registry.collectDefaultMetrics();

      const output = registry.export();

      expect(output).toContain('process_cpu_');
      expect(output).toContain('process_memory_');
      expect(output).toContain('process_uptime_seconds');
    });

    it('collects Node.js specific metrics', () => {
      registry.collectDefaultMetrics();

      const output = registry.export();

      expect(output).toContain('nodejs_heap_size_');
      expect(output).toContain('nodejs_eventloop_lag_');
    });
  });
});

describe('Pre-configured Metrics', () => {
  describe('httpRequestDurationMs', () => {
    it('tracks request duration', () => {
      httpRequestDurationMs.observe(
        { method: 'GET', route: '/api/songs', status: '200' },
        125
      );

      const metrics = httpRequestDurationMs.getMetrics({
        method: 'GET',
        route: '/api/songs',
        status: '200',
      });

      expect(metrics.count).toBe(1);
      expect(metrics.sum).toBe(125);
    });
  });

  describe('httpRequestTotal', () => {
    it('counts total requests', () => {
      httpRequestTotal.inc({ method: 'GET', route: '/api/songs', status: '200' });
      httpRequestTotal.inc({ method: 'GET', route: '/api/songs', status: '200' });
      httpRequestTotal.inc({ method: 'GET', route: '/api/songs', status: '404' });

      expect(
        httpRequestTotal.get({ method: 'GET', route: '/api/songs', status: '200' })
      ).toBe(2);
      expect(
        httpRequestTotal.get({ method: 'GET', route: '/api/songs', status: '404' })
      ).toBe(1);
    });
  });

  describe('activeConnections', () => {
    it('tracks active connections', () => {
      activeConnections.set({ type: 'websocket' }, 0);
      activeConnections.inc({ type: 'websocket' });
      activeConnections.inc({ type: 'websocket' });
      activeConnections.dec({ type: 'websocket' });

      expect(activeConnections.get({ type: 'websocket' })).toBe(1);
    });
  });

  describe('dbQueryDurationMs', () => {
    it('tracks database query duration', async () => {
      const result = await dbQueryDurationMs.time(
        { operation: 'SELECT', table: 'songs' },
        async () => {
          await new Promise((r) => setTimeout(r, 10));
          return [{ id: 1 }];
        }
      );

      expect(result).toEqual([{ id: 1 }]);

      const metrics = dbQueryDurationMs.getMetrics({
        operation: 'SELECT',
        table: 'songs',
      });
      expect(metrics.count).toBe(1);
    });
  });
});

describe('createMetrics factory', () => {
  it('creates metrics with prefix', () => {
    const metrics = createMetrics('mycelix_');

    expect(metrics.httpDuration.name).toBe('mycelix_http_request_duration_ms');
    expect(metrics.httpTotal.name).toBe('mycelix_http_requests_total');
  });

  it('returns singleton registry', () => {
    const metrics1 = createMetrics('app_');
    const metrics2 = createMetrics('app_');

    expect(metrics1.registry).toBe(metrics2.registry);
  });
});

describe('Label Validation', () => {
  let registry: MetricsRegistry;

  beforeEach(() => {
    registry = new MetricsRegistry();
  });

  it('rejects invalid label names', () => {
    expect(() => {
      registry.createCounter('test', 'Test', ['invalid-name']);
    }).toThrow('Invalid label name');
  });

  it('accepts valid label names', () => {
    expect(() => {
      registry.createCounter('test', 'Test', ['valid_name', 'another_valid']);
    }).not.toThrow();
  });

  it('validates label values are strings', () => {
    const counter = registry.createCounter('test', 'Test', ['label']);

    expect(() => {
      counter.inc({ label: 123 as any });
    }).toThrow('Label values must be strings');
  });
});

describe('Metric Aggregation', () => {
  let registry: MetricsRegistry;

  beforeEach(() => {
    registry = new MetricsRegistry();
  });

  it('aggregates counter totals', () => {
    const counter = registry.createCounter('requests', 'Requests', ['method']);

    counter.inc({ method: 'GET' }, 100);
    counter.inc({ method: 'POST' }, 50);
    counter.inc({ method: 'PUT' }, 25);

    const total = counter.total();

    expect(total).toBe(175);
  });

  it('aggregates histogram statistics', () => {
    const histogram = registry.createHistogram(
      'duration',
      'Duration',
      ['handler'],
      [10, 100, 1000]
    );

    histogram.observe({ handler: 'a' }, 50);
    histogram.observe({ handler: 'a' }, 150);
    histogram.observe({ handler: 'b' }, 500);

    const stats = histogram.getAggregatedStats();

    expect(stats.totalCount).toBe(3);
    expect(stats.totalSum).toBe(700);
    expect(stats.mean).toBeCloseTo(233.33, 1);
  });
});

describe('Thread Safety', () => {
  it('handles concurrent increments', async () => {
    const registry = new MetricsRegistry();
    const counter = registry.createCounter('concurrent', 'Concurrent counter', []);

    const increments = 1000;
    const promises = Array.from({ length: increments }, () =>
      Promise.resolve().then(() => counter.inc({}))
    );

    await Promise.all(promises);

    expect(counter.get({})).toBe(increments);
  });

  it('handles concurrent histogram observations', async () => {
    const registry = new MetricsRegistry();
    const histogram = registry.createHistogram(
      'concurrent_histogram',
      'Concurrent histogram',
      [],
      [10, 100, 1000]
    );

    const observations = 1000;
    const promises = Array.from({ length: observations }, (_, i) =>
      Promise.resolve().then(() => histogram.observe({}, i))
    );

    await Promise.all(promises);

    const metrics = histogram.getMetrics({});
    expect(metrics.count).toBe(observations);
  });
});

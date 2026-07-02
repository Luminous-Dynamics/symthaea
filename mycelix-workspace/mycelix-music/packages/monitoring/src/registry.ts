// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Prometheus Registry
 *
 * Centralized metrics registry with default metrics.
 */

import {
  Registry,
  collectDefaultMetrics,
  Counter,
  Histogram,
  Gauge,
} from 'prom-client';

export interface MetricsRegistry {
  registry: Registry;
  getMetrics(): Promise<string>;
  getContentType(): string;
}

export interface RegistryOptions {
  serviceName: string;
  defaultLabels?: Record<string, string>;
  collectDefault?: boolean;
}

export function createMetricsRegistry(options: RegistryOptions): MetricsRegistry {
  const registry = new Registry();

  // Set default labels
  registry.setDefaultLabels({
    service: options.serviceName,
    ...options.defaultLabels,
  });

  // Collect default Node.js metrics (CPU, memory, event loop, etc.)
  if (options.collectDefault !== false) {
    collectDefaultMetrics({
      register: registry,
      prefix: 'mycelix_',
    });
  }

  return {
    registry,
    getMetrics: () => registry.metrics(),
    getContentType: () => registry.contentType,
  };
}

/**
 * Create a counter metric
 */
export function createCounter(
  registry: Registry,
  name: string,
  help: string,
  labelNames?: string[]
): Counter {
  return new Counter({
    name: `mycelix_${name}`,
    help,
    labelNames: labelNames || [],
    registers: [registry],
  });
}

/**
 * Create a histogram metric
 */
export function createHistogram(
  registry: Registry,
  name: string,
  help: string,
  labelNames?: string[],
  buckets?: number[]
): Histogram {
  return new Histogram({
    name: `mycelix_${name}`,
    help,
    labelNames: labelNames || [],
    buckets: buckets || [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    registers: [registry],
  });
}

/**
 * Create a gauge metric
 */
export function createGauge(
  registry: Registry,
  name: string,
  help: string,
  labelNames?: string[]
): Gauge {
  return new Gauge({
    name: `mycelix_${name}`,
    help,
    labelNames: labelNames || [],
    registers: [registry],
  });
}

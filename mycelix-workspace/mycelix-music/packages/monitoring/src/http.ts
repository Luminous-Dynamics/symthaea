// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * HTTP Metrics
 *
 * Express middleware for tracking HTTP request metrics.
 */

import { Registry, Counter, Histogram } from 'prom-client';
import { Request, Response, NextFunction, RequestHandler } from 'express';
import { createCounter, createHistogram } from './registry';

export interface HttpMetrics {
  requestsTotal: Counter;
  requestDuration: Histogram;
  requestSize: Histogram;
  responseSize: Histogram;
}

/**
 * Create HTTP metrics
 */
export function createHttpMetrics(registry: Registry): HttpMetrics {
  return {
    requestsTotal: createCounter(
      registry,
      'http_requests_total',
      'Total number of HTTP requests',
      ['method', 'path', 'status']
    ),
    requestDuration: createHistogram(
      registry,
      'http_request_duration_seconds',
      'HTTP request duration in seconds',
      ['method', 'path', 'status'],
      [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    ),
    requestSize: createHistogram(
      registry,
      'http_request_size_bytes',
      'HTTP request size in bytes',
      ['method', 'path'],
      [100, 1000, 10000, 100000, 1000000]
    ),
    responseSize: createHistogram(
      registry,
      'http_response_size_bytes',
      'HTTP response size in bytes',
      ['method', 'path', 'status'],
      [100, 1000, 10000, 100000, 1000000, 10000000]
    ),
  };
}

/**
 * Normalize path for metrics (remove IDs, etc.)
 */
function normalizePath(path: string): string {
  return path
    // Replace UUIDs
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, ':id')
    // Replace Ethereum addresses
    .replace(/0x[a-fA-F0-9]{40}/g, ':address')
    // Replace numeric IDs
    .replace(/\/\d+/g, '/:id')
    // Remove trailing slashes
    .replace(/\/$/, '') || '/';
}

/**
 * Express middleware for collecting HTTP metrics
 */
export function httpMetricsMiddleware(metrics: HttpMetrics): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = process.hrtime.bigint();
    const path = normalizePath(req.path);
    const method = req.method;

    // Track request size
    const requestSize = parseInt(req.get('content-length') || '0', 10);
    if (requestSize > 0) {
      metrics.requestSize.labels(method, path).observe(requestSize);
    }

    // Hook into response finish
    res.on('finish', () => {
      const duration = Number(process.hrtime.bigint() - startTime) / 1e9;
      const status = res.statusCode.toString();

      // Increment request counter
      metrics.requestsTotal.labels(method, path, status).inc();

      // Record duration
      metrics.requestDuration.labels(method, path, status).observe(duration);

      // Track response size
      const responseSize = parseInt(res.get('content-length') || '0', 10);
      if (responseSize > 0) {
        metrics.responseSize.labels(method, path, status).observe(responseSize);
      }
    });

    next();
  };
}

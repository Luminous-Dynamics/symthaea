// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Observability Module
 * Unified exports for metrics, logging, and tracing
 */

// Metrics
export {
  metrics,
  httpMetricsMiddleware,
  metricsHandler,
  Counter,
  Gauge,
  Histogram,
  MetricsRegistry,
  type MetricLabels,
  type MetricConfig,
} from './metrics';

// Logging
export {
  logger,
  Logger,
  LogLevel,
  requestLoggingMiddleware,
  ConsoleTransport,
  FileTransport,
  HttpTransport,
  type LogEntry,
  type LogTransport,
  type LoggerConfig,
} from './logger';

// Tracing
export {
  tracer,
  Tracer,
  tracingMiddleware,
  SpanStatus,
  SpanKind,
  ConsoleSpanExporter,
  OTLPSpanExporter,
  type Span,
  type SpanAttributes,
  type SpanEvent,
  type SpanLink,
  type TraceContext,
  type SpanExporter,
  type TracerConfig,
} from './tracing';

// Convenience function to apply all observability middleware
import { Application } from 'express';
import { httpMetricsMiddleware, metricsHandler } from './metrics';
import { requestLoggingMiddleware } from './logger';
import { tracingMiddleware } from './tracing';

export interface ObservabilityConfig {
  enableMetrics?: boolean;
  enableLogging?: boolean;
  enableTracing?: boolean;
  metricsPath?: string;
}

export function applyObservability(
  app: Application,
  config: ObservabilityConfig = {}
): void {
  const {
    enableMetrics = true,
    enableLogging = true,
    enableTracing = true,
    metricsPath = '/metrics',
  } = config;

  if (enableTracing) {
    app.use(tracingMiddleware());
  }

  if (enableLogging) {
    app.use(requestLoggingMiddleware());
  }

  if (enableMetrics) {
    app.use(httpMetricsMiddleware());
    app.get(metricsPath, metricsHandler());
  }
}

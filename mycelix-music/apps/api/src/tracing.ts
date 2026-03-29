// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenTelemetry Tracing Configuration
 *
 * This module initializes distributed tracing for the API.
 * Import this BEFORE any other modules in index.ts.
 *
 * Environment variables:
 *   OTEL_ENABLED=true              - Enable tracing (default: false in dev)
 *   OTEL_SERVICE_NAME=mycelix-api  - Service name in traces
 *   OTEL_EXPORTER_OTLP_ENDPOINT    - OTLP endpoint (e.g., http://jaeger:4318)
 *   OTEL_EXPORTER_OTLP_PROTOCOL    - Protocol: grpc or http/protobuf
 *   OTEL_TRACES_SAMPLER            - Sampler: always_on, always_off, traceidratio
 *   OTEL_TRACES_SAMPLER_ARG        - Sampler argument (e.g., 0.1 for 10%)
 */

import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { Resource } from '@opentelemetry/resources';
import {
  SEMRESATTRS_SERVICE_NAME,
  SEMRESATTRS_SERVICE_VERSION,
  SEMRESATTRS_DEPLOYMENT_ENVIRONMENT,
} from '@opentelemetry/semantic-conventions';
import { diag, DiagConsoleLogger, DiagLogLevel } from '@opentelemetry/api';

// Enable debug logging if needed
if (process.env.OTEL_DEBUG === 'true') {
  diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
}

const isEnabled = process.env.OTEL_ENABLED === 'true' || process.env.NODE_ENV === 'production';
const serviceName = process.env.OTEL_SERVICE_NAME || 'mycelix-api';
const serviceVersion = process.env.npm_package_version || '0.1.0';
const environment = process.env.NODE_ENV || 'development';

let sdk: NodeSDK | null = null;

export function initTracing(): void {
  if (!isEnabled) {
    console.log('[tracing] OpenTelemetry disabled (set OTEL_ENABLED=true to enable)');
    return;
  }

  const otlpEndpoint = process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318';

  console.log(`[tracing] Initializing OpenTelemetry for ${serviceName}@${serviceVersion}`);
  console.log(`[tracing] Exporting to: ${otlpEndpoint}`);

  const resource = new Resource({
    [SEMRESATTRS_SERVICE_NAME]: serviceName,
    [SEMRESATTRS_SERVICE_VERSION]: serviceVersion,
    [SEMRESATTRS_DEPLOYMENT_ENVIRONMENT]: environment,
  });

  const traceExporter = new OTLPTraceExporter({
    url: `${otlpEndpoint}/v1/traces`,
  });

  const metricExporter = new OTLPMetricExporter({
    url: `${otlpEndpoint}/v1/metrics`,
  });

  sdk = new NodeSDK({
    resource,
    traceExporter,
    metricReader: new PeriodicExportingMetricReader({
      exporter: metricExporter,
      exportIntervalMillis: 30000,
    }),
    instrumentations: [
      getNodeAutoInstrumentations({
        // Instrument HTTP requests
        '@opentelemetry/instrumentation-http': {
          ignoreIncomingRequestHook: (req) => {
            // Don't trace health checks
            const url = req.url || '';
            return url.startsWith('/health') || url === '/metrics';
          },
        },
        // Instrument Express
        '@opentelemetry/instrumentation-express': {
          enabled: true,
        },
        // Instrument PostgreSQL
        '@opentelemetry/instrumentation-pg': {
          enabled: true,
          enhancedDatabaseReporting: true,
        },
        // Instrument Redis
        '@opentelemetry/instrumentation-redis-4': {
          enabled: true,
        },
        // Instrument fetch/undici for IPFS/Ceramic calls
        '@opentelemetry/instrumentation-undici': {
          enabled: true,
        },
        // Disable noisy instrumentations
        '@opentelemetry/instrumentation-fs': {
          enabled: false,
        },
        '@opentelemetry/instrumentation-dns': {
          enabled: false,
        },
      }),
    ],
  });

  sdk.start();
  console.log('[tracing] OpenTelemetry initialized successfully');

  // Graceful shutdown
  process.on('SIGTERM', () => {
    sdk?.shutdown()
      .then(() => console.log('[tracing] OpenTelemetry shut down'))
      .catch((err) => console.error('[tracing] Error shutting down', err));
  });
}

export function shutdownTracing(): Promise<void> {
  if (sdk) {
    return sdk.shutdown();
  }
  return Promise.resolve();
}

// Custom span helpers for manual instrumentation
import { trace, SpanStatusCode, Span, context } from '@opentelemetry/api';

const tracer = trace.getTracer(serviceName, serviceVersion);

/**
 * Create a span for a database operation
 */
export function withDbSpan<T>(
  operation: string,
  table: string,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  return tracer.startActiveSpan(`db.${operation}`, async (span) => {
    span.setAttribute('db.system', 'postgresql');
    span.setAttribute('db.operation', operation);
    span.setAttribute('db.sql.table', table);
    try {
      const result = await fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error?.message });
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  });
}

/**
 * Create a span for an external service call
 */
export function withExternalSpan<T>(
  service: string,
  operation: string,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  return tracer.startActiveSpan(`${service}.${operation}`, async (span) => {
    span.setAttribute('peer.service', service);
    span.setAttribute('rpc.method', operation);
    try {
      const result = await fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error?.message });
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  });
}

/**
 * Add attributes to the current span
 */
export function addSpanAttributes(attributes: Record<string, string | number | boolean>): void {
  const span = trace.getActiveSpan();
  if (span) {
    for (const [key, value] of Object.entries(attributes)) {
      span.setAttribute(key, value);
    }
  }
}

/**
 * Record an error on the current span
 */
export function recordSpanError(error: Error): void {
  const span = trace.getActiveSpan();
  if (span) {
    span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
    span.recordException(error);
  }
}

export { tracer, trace, context, SpanStatusCode };

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenTelemetry Observability Integration
 *
 * Provides tracing, metrics, and logging for the Mycelix SDK.
 * Enables debugging and monitoring in production deployments.
 */

import {
  trace,
  context,
  SpanStatusCode,
  type Span,
  type Tracer,
  type SpanOptions,
  type Context,
  metrics,
  type Meter,
  type Counter,
  type Histogram,
} from '@opentelemetry/api';

// ============================================================================
// Tracer Configuration
// ============================================================================

const TRACER_NAME = '@mycelix/sdk';
const TRACER_VERSION = '0.6.0';

let globalTracer: Tracer | null = null;
let globalMeter: Meter | null = null;

/**
 * Initialize observability with custom tracer/meter
 */
export function initObservability(options?: { tracer?: Tracer; meter?: Meter }): void {
  globalTracer = options?.tracer ?? trace.getTracer(TRACER_NAME, TRACER_VERSION);
  globalMeter = options?.meter ?? metrics.getMeter(TRACER_NAME, TRACER_VERSION);
  initMetrics();
}

/**
 * Get the SDK tracer
 */
export function getTracer(): Tracer {
  if (!globalTracer) {
    globalTracer = trace.getTracer(TRACER_NAME, TRACER_VERSION);
  }
  return globalTracer;
}

/**
 * Get the SDK meter
 */
export function getMeter(): Meter {
  if (!globalMeter) {
    globalMeter = metrics.getMeter(TRACER_NAME, TRACER_VERSION);
    initMetrics();
  }
  return globalMeter;
}

// ============================================================================
// Metrics
// ============================================================================

interface SDKMetrics {
  operationCounter: Counter;
  operationDuration: Histogram;
  errorCounter: Counter;
  activeOperations: Counter;
  bridgeCallCounter: Counter;
  bridgeCallDuration: Histogram;
  validationErrorCounter: Counter;
  circuitBreakerStateChanges: Counter;
}

let sdkMetrics: SDKMetrics | null = null;

function initMetrics(): void {
  const meter = getMeter();

  sdkMetrics = {
    operationCounter: meter.createCounter('mycelix.operations.total', {
      description: 'Total number of SDK operations',
    }),

    operationDuration: meter.createHistogram('mycelix.operations.duration', {
      description: 'Duration of SDK operations in milliseconds',
      unit: 'ms',
    }),

    errorCounter: meter.createCounter('mycelix.errors.total', {
      description: 'Total number of errors',
    }),

    activeOperations: meter.createCounter('mycelix.operations.active', {
      description: 'Number of currently active operations',
    }),

    bridgeCallCounter: meter.createCounter('mycelix.bridge.calls.total', {
      description: 'Total number of Bridge zome calls',
    }),

    bridgeCallDuration: meter.createHistogram('mycelix.bridge.calls.duration', {
      description: 'Duration of Bridge zome calls in milliseconds',
      unit: 'ms',
    }),

    validationErrorCounter: meter.createCounter('mycelix.validation.errors.total', {
      description: 'Total number of validation errors',
    }),

    circuitBreakerStateChanges: meter.createCounter('mycelix.circuit_breaker.state_changes', {
      description: 'Circuit breaker state changes',
    }),
  };
}

export function getMetrics(): SDKMetrics {
  if (!sdkMetrics) {
    initMetrics();
  }
  return sdkMetrics!;
}

// ============================================================================
// Tracing Decorators and Utilities
// ============================================================================

export interface TraceOptions extends SpanOptions {
  /** Operation name */
  name: string;
  /** Additional attributes */
  attributes?: Record<string, string | number | boolean>;
  /** Record metrics for this operation */
  recordMetrics?: boolean;
}

/**
 * Execute a function with tracing
 */
export async function withTrace<T>(
  options: TraceOptions,
  fn: (span: Span) => Promise<T>
): Promise<T> {
  const tracer = getTracer();
  const metrics = getMetrics();
  const startTime = Date.now();

  return tracer.startActiveSpan(options.name, options, async (span: Span) => {
    if (options.attributes) {
      for (const [key, value] of Object.entries(options.attributes)) {
        span.setAttribute(key, value);
      }
    }

    if (options.recordMetrics !== false) {
      metrics.operationCounter.add(1, { operation: options.name });
      metrics.activeOperations.add(1, { operation: options.name });
    }

    try {
      const result = await fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : 'Unknown error',
      });
      span.recordException(error instanceof Error ? error : new Error(String(error)));

      if (options.recordMetrics !== false) {
        metrics.errorCounter.add(1, {
          operation: options.name,
          error_type: error instanceof Error ? error.name : 'unknown',
        });
      }

      throw error;
    } finally {
      const duration = Date.now() - startTime;

      if (options.recordMetrics !== false) {
        metrics.operationDuration.record(duration, { operation: options.name });
        metrics.activeOperations.add(-1, { operation: options.name });
      }

      span.end();
    }
  });
}

/**
 * Execute a synchronous function with tracing
 */
export function withTraceSync<T>(options: TraceOptions, fn: (span: Span) => T): T {
  const tracer = getTracer();
  const metrics = getMetrics();
  const startTime = Date.now();

  const span = tracer.startSpan(options.name, options);

  if (options.attributes) {
    for (const [key, value] of Object.entries(options.attributes)) {
      span.setAttribute(key, value);
    }
  }

  if (options.recordMetrics !== false) {
    metrics.operationCounter.add(1, { operation: options.name });
  }

  try {
    const result = fn(span);
    span.setStatus({ code: SpanStatusCode.OK });
    return result;
  } catch (error) {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error instanceof Error ? error.message : 'Unknown error',
    });
    span.recordException(error instanceof Error ? error : new Error(String(error)));

    if (options.recordMetrics !== false) {
      metrics.errorCounter.add(1, {
        operation: options.name,
        error_type: error instanceof Error ? error.name : 'unknown',
      });
    }

    throw error;
  } finally {
    const duration = Date.now() - startTime;

    if (options.recordMetrics !== false) {
      metrics.operationDuration.record(duration, { operation: options.name });
    }

    span.end();
  }
}

// ============================================================================
// Bridge-Specific Tracing
// ============================================================================

export interface BridgeTraceOptions {
  happ: string;
  zome: string;
  fn: string;
  payload?: unknown;
}

/**
 * Trace a Bridge zome call
 */
export async function traceBridgeCall<T>(
  options: BridgeTraceOptions,
  fn: () => Promise<T>
): Promise<T> {
  const metrics = getMetrics();
  const spanName = `bridge.${options.happ}.${options.zome}.${options.fn}`;

  return withTrace(
    {
      name: spanName,
      attributes: {
        'bridge.happ': options.happ,
        'bridge.zome': options.zome,
        'bridge.fn': options.fn,
        'bridge.payload_size': options.payload ? JSON.stringify(options.payload).length : 0,
      },
    },
    async (span) => {
      metrics.bridgeCallCounter.add(1, {
        happ: options.happ,
        zome: options.zome,
        fn: options.fn,
      });

      const startTime = Date.now();
      try {
        const result = await fn();
        span.setAttribute('bridge.success', true);
        return result;
      } finally {
        const duration = Date.now() - startTime;
        metrics.bridgeCallDuration.record(duration, {
          happ: options.happ,
          zome: options.zome,
          fn: options.fn,
        });
      }
    }
  );
}

// ============================================================================
// Context Propagation
// ============================================================================

/**
 * Get the current trace context
 */
export function getCurrentContext(): Context {
  return context.active();
}

/**
 * Run a function with a specific context
 */
export function withContext<T>(ctx: Context, fn: () => T): T {
  return context.with(ctx, fn);
}

/**
 * Extract trace context for cross-service propagation
 */
export function extractTraceContext(): Record<string, string> {
  const span = trace.getActiveSpan();
  if (!span) return {};

  const spanContext = span.spanContext();
  return {
    'trace-id': spanContext.traceId,
    'span-id': spanContext.spanId,
    'trace-flags': spanContext.traceFlags.toString(),
  };
}

// ============================================================================
// Logging Integration
// ============================================================================

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
  traceId?: string;
  spanId?: string;
  attributes?: Record<string, unknown>;
}

export type LogHandler = (entry: LogEntry) => void;

let logHandlers: LogHandler[] = [];

/**
 * Add a log handler
 */
export function addLogHandler(handler: LogHandler): void {
  logHandlers.push(handler);
}

/**
 * Remove a log handler
 */
export function removeLogHandler(handler: LogHandler): void {
  logHandlers = logHandlers.filter((h) => h !== handler);
}

/**
 * Log a message with trace context
 */
export function log(level: LogLevel, message: string, attributes?: Record<string, unknown>): void {
  const span = trace.getActiveSpan();
  const spanContext = span?.spanContext();

  const entry: LogEntry = {
    level,
    message,
    timestamp: Date.now(),
    traceId: spanContext?.traceId,
    spanId: spanContext?.spanId,
    attributes,
  };

  // Add log as span event
  if (span) {
    span.addEvent(message, attributes as Record<string, string | number | boolean>);
  }

  // Call handlers
  for (const handler of logHandlers) {
    try {
      handler(entry);
    } catch {
      // Ignore handler errors
    }
  }
}

export const logger = {
  debug: (message: string, attributes?: Record<string, unknown>) =>
    log(LogLevel.DEBUG, message, attributes),
  info: (message: string, attributes?: Record<string, unknown>) =>
    log(LogLevel.INFO, message, attributes),
  warn: (message: string, attributes?: Record<string, unknown>) =>
    log(LogLevel.WARN, message, attributes),
  error: (message: string, attributes?: Record<string, unknown>) =>
    log(LogLevel.ERROR, message, attributes),
};

// ============================================================================
// Console Log Handler (Default)
// ============================================================================

export const consoleLogHandler: LogHandler = (entry) => {
  const prefix = entry.traceId ? `[${entry.traceId.slice(0, 8)}] ` : '';
  const attrs = entry.attributes ? ` ${JSON.stringify(entry.attributes)}` : '';

  switch (entry.level) {
    case LogLevel.DEBUG:
      console.debug(`${prefix}${entry.message}${attrs}`);
      break;
    case LogLevel.INFO:
      console.info(`${prefix}${entry.message}${attrs}`);
      break;
    case LogLevel.WARN:
      console.warn(`${prefix}${entry.message}${attrs}`);
      break;
    case LogLevel.ERROR:
      console.error(`${prefix}${entry.message}${attrs}`);
      break;
  }
};

// Add console handler by default in development
if (typeof process !== 'undefined' && process.env.NODE_ENV !== 'production') {
  addLogHandler(consoleLogHandler);
}

// ============================================================================
// Health Check
// ============================================================================

export interface HealthStatus {
  healthy: boolean;
  timestamp: number;
  components: Record<
    string,
    {
      healthy: boolean;
      message?: string;
      latencyMs?: number;
    }
  >;
}

export type HealthCheck = () => Promise<{ healthy: boolean; message?: string; latencyMs?: number }>;

const healthChecks: Map<string, HealthCheck> = new Map();

/**
 * Register a health check
 */
export function registerHealthCheck(name: string, check: HealthCheck): void {
  healthChecks.set(name, check);
}

/**
 * Run all health checks
 */
export async function checkHealth(): Promise<HealthStatus> {
  const components: HealthStatus['components'] = {};
  let allHealthy = true;

  for (const [name, check] of healthChecks) {
    try {
      const result = await check();
      components[name] = result;
      if (!result.healthy) {
        allHealthy = false;
      }
    } catch (error) {
      components[name] = {
        healthy: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
      allHealthy = false;
    }
  }

  return {
    healthy: allHealthy,
    timestamp: Date.now(),
    components,
  };
}

// ============================================================================
// Exports
// ============================================================================

export { trace, context, SpanStatusCode, type Span, type Tracer, type SpanOptions };

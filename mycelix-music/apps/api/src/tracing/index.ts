// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenTelemetry Distributed Tracing
 *
 * End-to-end request tracing across services.
 * Integrates with Jaeger, Zipkin, or any OTLP-compatible backend.
 */

import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

/**
 * Span status
 */
export enum SpanStatus {
  UNSET = 0,
  OK = 1,
  ERROR = 2,
}

/**
 * Span kind
 */
export enum SpanKind {
  INTERNAL = 0,
  SERVER = 1,
  CLIENT = 2,
  PRODUCER = 3,
  CONSUMER = 4,
}

/**
 * Span attributes
 */
export interface SpanAttributes {
  [key: string]: string | number | boolean | undefined;
}

/**
 * Span event
 */
export interface SpanEvent {
  name: string;
  timestamp: number;
  attributes?: SpanAttributes;
}

/**
 * Span context for propagation
 */
export interface SpanContext {
  traceId: string;
  spanId: string;
  traceFlags: number;
  traceState?: string;
}

/**
 * Span interface
 */
export interface Span {
  readonly name: string;
  readonly traceId: string;
  readonly spanId: string;
  readonly parentSpanId?: string;
  readonly startTime: number;
  endTime?: number;
  status: SpanStatus;
  kind: SpanKind;
  attributes: SpanAttributes;
  events: SpanEvent[];

  setAttribute(key: string, value: string | number | boolean): Span;
  setAttributes(attributes: SpanAttributes): Span;
  addEvent(name: string, attributes?: SpanAttributes): Span;
  setStatus(status: SpanStatus, message?: string): Span;
  recordException(error: Error): Span;
  end(): void;
  getContext(): SpanContext;
}

/**
 * Span implementation
 */
class SpanImpl implements Span {
  readonly startTime: number;
  endTime?: number;
  status: SpanStatus = SpanStatus.UNSET;
  attributes: SpanAttributes = {};
  events: SpanEvent[] = [];

  constructor(
    readonly name: string,
    readonly traceId: string,
    readonly spanId: string,
    readonly parentSpanId: string | undefined,
    readonly kind: SpanKind
  ) {
    this.startTime = Date.now();
  }

  setAttribute(key: string, value: string | number | boolean): Span {
    this.attributes[key] = value;
    return this;
  }

  setAttributes(attributes: SpanAttributes): Span {
    Object.assign(this.attributes, attributes);
    return this;
  }

  addEvent(name: string, attributes?: SpanAttributes): Span {
    this.events.push({
      name,
      timestamp: Date.now(),
      attributes,
    });
    return this;
  }

  setStatus(status: SpanStatus, message?: string): Span {
    this.status = status;
    if (message) {
      this.attributes['status.message'] = message;
    }
    return this;
  }

  recordException(error: Error): Span {
    this.addEvent('exception', {
      'exception.type': error.name,
      'exception.message': error.message,
      'exception.stacktrace': error.stack,
    });
    this.setStatus(SpanStatus.ERROR, error.message);
    return this;
  }

  end(): void {
    this.endTime = Date.now();
    getTracer().exportSpan(this);
  }

  getContext(): SpanContext {
    return {
      traceId: this.traceId,
      spanId: this.spanId,
      traceFlags: 1,
    };
  }
}

/**
 * Tracer configuration
 */
export interface TracerConfig {
  serviceName: string;
  serviceVersion: string;
  environment: string;
  enabled: boolean;
  samplingRate: number;
  exporterEndpoint?: string;
  exporterType: 'console' | 'otlp' | 'jaeger' | 'zipkin' | 'none';
}

/**
 * Default configuration
 */
const defaultConfig: TracerConfig = {
  serviceName: 'mycelix-api',
  serviceVersion: process.env.npm_package_version || '1.0.0',
  environment: process.env.NODE_ENV || 'development',
  enabled: true,
  samplingRate: 1.0,
  exporterType: process.env.NODE_ENV === 'production' ? 'otlp' : 'console',
};

/**
 * Tracer implementation
 */
class Tracer {
  private config: TracerConfig;
  private activeSpans: Map<string, Span> = new Map();
  private exportBuffer: Span[] = [];
  private exportInterval?: NodeJS.Timeout;

  constructor(config: Partial<TracerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };

    if (this.config.enabled && this.config.exporterType !== 'none') {
      this.startExportInterval();
    }
  }

  /**
   * Create a new span
   */
  startSpan(name: string, options: {
    kind?: SpanKind;
    parent?: SpanContext;
    attributes?: SpanAttributes;
  } = {}): Span {
    // Sampling decision
    if (!this.shouldSample()) {
      return new NoOpSpan(name);
    }

    const traceId = options.parent?.traceId || this.generateTraceId();
    const spanId = this.generateSpanId();
    const parentSpanId = options.parent?.spanId;

    const span = new SpanImpl(
      name,
      traceId,
      spanId,
      parentSpanId,
      options.kind || SpanKind.INTERNAL
    );

    // Add default attributes
    span.setAttributes({
      'service.name': this.config.serviceName,
      'service.version': this.config.serviceVersion,
      'deployment.environment': this.config.environment,
      ...options.attributes,
    });

    this.activeSpans.set(spanId, span);

    return span;
  }

  /**
   * Get the current active span
   */
  getActiveSpan(): Span | undefined {
    // In a real implementation, this would use async context
    return undefined;
  }

  /**
   * Export a completed span
   */
  exportSpan(span: Span): void {
    this.activeSpans.delete(span.spanId);

    if (this.config.exporterType === 'none') return;

    this.exportBuffer.push(span);

    // Flush if buffer is large
    if (this.exportBuffer.length >= 100) {
      this.flush();
    }
  }

  /**
   * Flush export buffer
   */
  async flush(): Promise<void> {
    if (this.exportBuffer.length === 0) return;

    const spans = [...this.exportBuffer];
    this.exportBuffer = [];

    switch (this.config.exporterType) {
      case 'console':
        this.exportToConsole(spans);
        break;
      case 'otlp':
        await this.exportToOTLP(spans);
        break;
      case 'jaeger':
        await this.exportToJaeger(spans);
        break;
      case 'zipkin':
        await this.exportToZipkin(spans);
        break;
    }
  }

  /**
   * Export spans to console (development)
   */
  private exportToConsole(spans: Span[]): void {
    for (const span of spans) {
      const duration = (span.endTime || Date.now()) - span.startTime;
      const statusIcon = span.status === SpanStatus.ERROR ? '❌' :
                         span.status === SpanStatus.OK ? '✅' : '⚪';

      console.log(
        `${statusIcon} [${span.traceId.slice(0, 8)}] ${span.name} ` +
        `(${duration}ms) ${JSON.stringify(span.attributes)}`
      );

      for (const event of span.events) {
        console.log(`   📍 ${event.name}`, event.attributes || '');
      }
    }
  }

  /**
   * Export spans to OTLP endpoint
   */
  private async exportToOTLP(spans: Span[]): Promise<void> {
    if (!this.config.exporterEndpoint) return;

    const payload = {
      resourceSpans: [{
        resource: {
          attributes: [
            { key: 'service.name', value: { stringValue: this.config.serviceName } },
            { key: 'service.version', value: { stringValue: this.config.serviceVersion } },
          ],
        },
        scopeSpans: [{
          scope: { name: 'mycelix-tracer', version: '1.0.0' },
          spans: spans.map(span => this.spanToOTLP(span)),
        }],
      }],
    };

    try {
      await fetch(this.config.exporterEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      console.error('Failed to export traces to OTLP:', error);
    }
  }

  /**
   * Convert span to OTLP format
   */
  private spanToOTLP(span: Span): object {
    return {
      traceId: span.traceId,
      spanId: span.spanId,
      parentSpanId: span.parentSpanId,
      name: span.name,
      kind: span.kind,
      startTimeUnixNano: span.startTime * 1e6,
      endTimeUnixNano: (span.endTime || Date.now()) * 1e6,
      attributes: Object.entries(span.attributes).map(([key, value]) => ({
        key,
        value: { stringValue: String(value) },
      })),
      events: span.events.map(event => ({
        name: event.name,
        timeUnixNano: event.timestamp * 1e6,
        attributes: Object.entries(event.attributes || {}).map(([key, value]) => ({
          key,
          value: { stringValue: String(value) },
        })),
      })),
      status: { code: span.status },
    };
  }

  /**
   * Export to Jaeger
   */
  private async exportToJaeger(spans: Span[]): Promise<void> {
    // Jaeger accepts Thrift or OTLP - using OTLP
    await this.exportToOTLP(spans);
  }

  /**
   * Export to Zipkin
   */
  private async exportToZipkin(spans: Span[]): Promise<void> {
    if (!this.config.exporterEndpoint) return;

    const zipkinSpans = spans.map(span => ({
      traceId: span.traceId,
      id: span.spanId,
      parentId: span.parentSpanId,
      name: span.name,
      timestamp: span.startTime * 1000,
      duration: ((span.endTime || Date.now()) - span.startTime) * 1000,
      localEndpoint: { serviceName: this.config.serviceName },
      tags: span.attributes as Record<string, string>,
      annotations: span.events.map(e => ({
        timestamp: e.timestamp * 1000,
        value: e.name,
      })),
    }));

    try {
      await fetch(this.config.exporterEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zipkinSpans),
      });
    } catch (error) {
      console.error('Failed to export traces to Zipkin:', error);
    }
  }

  /**
   * Check if we should sample this trace
   */
  private shouldSample(): boolean {
    return Math.random() < this.config.samplingRate;
  }

  /**
   * Generate trace ID (32 hex characters)
   */
  private generateTraceId(): string {
    return randomUUID().replace(/-/g, '');
  }

  /**
   * Generate span ID (16 hex characters)
   */
  private generateSpanId(): string {
    return randomUUID().replace(/-/g, '').slice(0, 16);
  }

  /**
   * Start periodic export
   */
  private startExportInterval(): void {
    this.exportInterval = setInterval(() => {
      this.flush().catch(console.error);
    }, 5000);
  }

  /**
   * Shutdown tracer
   */
  async shutdown(): Promise<void> {
    if (this.exportInterval) {
      clearInterval(this.exportInterval);
    }
    await this.flush();
  }
}

/**
 * No-op span for when sampling rejects
 */
class NoOpSpan implements Span {
  readonly traceId = '0'.repeat(32);
  readonly spanId = '0'.repeat(16);
  readonly parentSpanId = undefined;
  readonly startTime = 0;
  endTime = 0;
  status = SpanStatus.UNSET;
  kind = SpanKind.INTERNAL;
  attributes = {};
  events = [];

  constructor(readonly name: string) {}

  setAttribute(): Span { return this; }
  setAttributes(): Span { return this; }
  addEvent(): Span { return this; }
  setStatus(): Span { return this; }
  recordException(): Span { return this; }
  end(): void {}
  getContext(): SpanContext {
    return { traceId: this.traceId, spanId: this.spanId, traceFlags: 0 };
  }
}

/**
 * Global tracer instance
 */
let tracer: Tracer | null = null;

export function initTracer(config?: Partial<TracerConfig>): Tracer {
  tracer = new Tracer(config);
  return tracer;
}

export function getTracer(): Tracer {
  if (!tracer) {
    tracer = new Tracer();
  }
  return tracer;
}

/**
 * Express middleware for HTTP tracing
 */
export function tracingMiddleware() {
  return (req: Request, res: Response, next: NextFunction): void => {
    const tracer = getTracer();

    // Extract parent context from headers
    const parentContext = extractContext(req);

    // Start span for this request
    const span = tracer.startSpan(`${req.method} ${req.path}`, {
      kind: SpanKind.SERVER,
      parent: parentContext,
      attributes: {
        'http.method': req.method,
        'http.url': req.url,
        'http.target': req.path,
        'http.host': req.hostname,
        'http.scheme': req.protocol,
        'http.user_agent': req.headers['user-agent'],
        'net.peer.ip': req.ip,
      },
    });

    // Inject context into response headers
    const context = span.getContext();
    res.setHeader('X-Trace-ID', context.traceId);
    res.setHeader('X-Span-ID', context.spanId);

    // Attach to request for child spans
    (req as any).span = span;
    (req as any).traceId = context.traceId;

    // Capture response
    res.on('finish', () => {
      span.setAttribute('http.status_code', res.statusCode);
      span.setAttribute('http.response_content_length', res.get('Content-Length') || 0);

      if (res.statusCode >= 400) {
        span.setStatus(SpanStatus.ERROR, `HTTP ${res.statusCode}`);
      } else {
        span.setStatus(SpanStatus.OK);
      }

      span.end();
    });

    next();
  };
}

/**
 * Extract trace context from request headers
 */
function extractContext(req: Request): SpanContext | undefined {
  // W3C Trace Context format
  const traceparent = req.headers['traceparent'] as string;
  if (traceparent) {
    const parts = traceparent.split('-');
    if (parts.length === 4) {
      return {
        traceId: parts[1],
        spanId: parts[2],
        traceFlags: parseInt(parts[3], 16),
      };
    }
  }

  // Also check for Jaeger/Zipkin headers
  const uberTraceId = req.headers['uber-trace-id'] as string;
  if (uberTraceId) {
    const parts = uberTraceId.split(':');
    if (parts.length >= 2) {
      return {
        traceId: parts[0],
        spanId: parts[1],
        traceFlags: 1,
      };
    }
  }

  return undefined;
}

/**
 * Create a child span
 */
export function startSpan(name: string, parent?: Span): Span {
  return getTracer().startSpan(name, {
    parent: parent?.getContext(),
  });
}

/**
 * Trace a function execution
 */
export async function trace<T>(
  name: string,
  fn: (span: Span) => Promise<T>,
  parent?: Span
): Promise<T> {
  const span = startSpan(name, parent);
  try {
    const result = await fn(span);
    span.setStatus(SpanStatus.OK);
    return result;
  } catch (error) {
    span.recordException(error as Error);
    throw error;
  } finally {
    span.end();
  }
}

export default {
  initTracer,
  getTracer,
  tracingMiddleware,
  startSpan,
  trace,
  SpanKind,
  SpanStatus,
};

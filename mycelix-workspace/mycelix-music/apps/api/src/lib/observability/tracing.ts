// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Distributed Tracing Service
 * OpenTelemetry-compatible tracing for request flows
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import { randomBytes } from 'crypto';

// Trace context propagation headers (W3C Trace Context)
const TRACEPARENT_HEADER = 'traceparent';
const TRACESTATE_HEADER = 'tracestate';
const BAGGAGE_HEADER = 'baggage';

// Span status
export enum SpanStatus {
  UNSET = 0,
  OK = 1,
  ERROR = 2,
}

// Span kind
export enum SpanKind {
  INTERNAL = 0,
  SERVER = 1,
  CLIENT = 2,
  PRODUCER = 3,
  CONSUMER = 4,
}

// Span attributes
export interface SpanAttributes {
  [key: string]: string | number | boolean | string[] | number[] | boolean[];
}

// Span event
export interface SpanEvent {
  name: string;
  timestamp: number;
  attributes?: SpanAttributes;
}

// Span link
export interface SpanLink {
  traceId: string;
  spanId: string;
  attributes?: SpanAttributes;
}

// Span interface
export interface Span {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: SpanKind;
  startTime: number;
  endTime?: number;
  status: SpanStatus;
  statusMessage?: string;
  attributes: SpanAttributes;
  events: SpanEvent[];
  links: SpanLink[];
}

// Trace context
export interface TraceContext {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  traceFlags: number;
  traceState?: string;
  baggage?: Record<string, string>;
}

// Exporter interface
export interface SpanExporter {
  export(spans: Span[]): Promise<void>;
}

// Console exporter (for development)
class ConsoleSpanExporter implements SpanExporter {
  async export(spans: Span[]): Promise<void> {
    for (const span of spans) {
      const duration = span.endTime ? span.endTime - span.startTime : 0;
      console.log(
        `[Trace] ${span.name} traceId=${span.traceId.slice(0, 8)}... ` +
          `spanId=${span.spanId.slice(0, 8)}... duration=${duration}ms`
      );
    }
  }
}

// OTLP exporter (OpenTelemetry Protocol)
class OTLPSpanExporter implements SpanExporter {
  private buffer: Span[] = [];
  private flushInterval: NodeJS.Timeout | null = null;

  constructor(
    private endpoint: string,
    private headers: Record<string, string> = {},
    private maxBufferSize = 100,
    private flushIntervalMs = 5000
  ) {
    this.startFlushInterval();
  }

  private startFlushInterval(): void {
    this.flushInterval = setInterval(() => {
      this.flush();
    }, this.flushIntervalMs);
  }

  async export(spans: Span[]): Promise<void> {
    this.buffer.push(...spans);
    if (this.buffer.length >= this.maxBufferSize) {
      await this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const spans = this.buffer;
    this.buffer = [];

    try {
      await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this.headers,
        },
        body: JSON.stringify({
          resourceSpans: [
            {
              resource: {
                attributes: [
                  { key: 'service.name', value: { stringValue: 'mycelix-api' } },
                ],
              },
              scopeSpans: [
                {
                  spans: spans.map(this.convertToOTLP),
                },
              ],
            },
          ],
        }),
      });
    } catch (error) {
      console.error('[OTLP] Failed to export spans:', error);
      // Re-add to buffer
      this.buffer.unshift(...spans);
    }
  }

  private convertToOTLP(span: Span): object {
    return {
      traceId: span.traceId,
      spanId: span.spanId,
      parentSpanId: span.parentSpanId,
      name: span.name,
      kind: span.kind,
      startTimeUnixNano: span.startTime * 1_000_000,
      endTimeUnixNano: (span.endTime || Date.now()) * 1_000_000,
      status: {
        code: span.status,
        message: span.statusMessage,
      },
      attributes: Object.entries(span.attributes).map(([key, value]) => ({
        key,
        value: { stringValue: String(value) },
      })),
      events: span.events.map((e) => ({
        timeUnixNano: e.timestamp * 1_000_000,
        name: e.name,
        attributes: e.attributes
          ? Object.entries(e.attributes).map(([key, value]) => ({
              key,
              value: { stringValue: String(value) },
            }))
          : [],
      })),
    };
  }

  destroy(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flush();
  }
}

// Tracer configuration
export interface TracerConfig {
  serviceName?: string;
  serviceVersion?: string;
  exporter?: SpanExporter;
  samplingRate?: number;
}

/**
 * Tracer class
 */
export class Tracer {
  private serviceName: string;
  private serviceVersion: string;
  private exporter: SpanExporter;
  private samplingRate: number;
  private activeSpans: Map<string, Span> = new Map();
  private completedSpans: Span[] = [];

  constructor(config: TracerConfig = {}) {
    this.serviceName = config.serviceName || 'mycelix-api';
    this.serviceVersion = config.serviceVersion || '1.0.0';
    this.samplingRate = config.samplingRate ?? 1.0;
    this.exporter = config.exporter || new ConsoleSpanExporter();

    // Start flush interval
    setInterval(() => {
      this.flushCompletedSpans();
    }, 5000);
  }

  /**
   * Generate trace ID (32 hex chars = 16 bytes)
   */
  private generateTraceId(): string {
    return randomBytes(16).toString('hex');
  }

  /**
   * Generate span ID (16 hex chars = 8 bytes)
   */
  private generateSpanId(): string {
    return randomBytes(8).toString('hex');
  }

  /**
   * Parse W3C traceparent header
   */
  parseTraceparent(header: string | undefined): TraceContext | null {
    if (!header) return null;

    const parts = header.split('-');
    if (parts.length !== 4) return null;

    const [version, traceId, spanId, flags] = parts;

    if (version !== '00') return null;
    if (traceId.length !== 32) return null;
    if (spanId.length !== 16) return null;

    return {
      traceId,
      spanId,
      traceFlags: parseInt(flags, 16),
    };
  }

  /**
   * Create traceparent header
   */
  createTraceparent(context: TraceContext): string {
    const flags = context.traceFlags.toString(16).padStart(2, '0');
    return `00-${context.traceId}-${context.spanId}-${flags}`;
  }

  /**
   * Start a new span
   */
  startSpan(
    name: string,
    options: {
      kind?: SpanKind;
      parentContext?: TraceContext;
      attributes?: SpanAttributes;
      links?: SpanLink[];
    } = {}
  ): Span {
    // Check sampling
    if (Math.random() > this.samplingRate) {
      // Return a no-op span
      return this.createNoopSpan(name);
    }

    const traceId = options.parentContext?.traceId || this.generateTraceId();
    const spanId = this.generateSpanId();
    const parentSpanId = options.parentContext?.spanId;

    const span: Span = {
      traceId,
      spanId,
      parentSpanId,
      name,
      kind: options.kind ?? SpanKind.INTERNAL,
      startTime: Date.now(),
      status: SpanStatus.UNSET,
      attributes: {
        'service.name': this.serviceName,
        'service.version': this.serviceVersion,
        ...options.attributes,
      },
      events: [],
      links: options.links || [],
    };

    this.activeSpans.set(spanId, span);
    return span;
  }

  private createNoopSpan(name: string): Span {
    return {
      traceId: '0'.repeat(32),
      spanId: '0'.repeat(16),
      name,
      kind: SpanKind.INTERNAL,
      startTime: Date.now(),
      status: SpanStatus.OK,
      attributes: {},
      events: [],
      links: [],
    };
  }

  /**
   * End a span
   */
  endSpan(span: Span, status?: SpanStatus, statusMessage?: string): void {
    span.endTime = Date.now();
    span.status = status ?? SpanStatus.OK;
    if (statusMessage) span.statusMessage = statusMessage;

    this.activeSpans.delete(span.spanId);
    this.completedSpans.push(span);
  }

  /**
   * Add an event to a span
   */
  addEvent(span: Span, name: string, attributes?: SpanAttributes): void {
    span.events.push({
      name,
      timestamp: Date.now(),
      attributes,
    });
  }

  /**
   * Set span attributes
   */
  setAttributes(span: Span, attributes: SpanAttributes): void {
    Object.assign(span.attributes, attributes);
  }

  /**
   * Record an exception on a span
   */
  recordException(span: Span, error: Error): void {
    this.addEvent(span, 'exception', {
      'exception.type': error.name,
      'exception.message': error.message,
      'exception.stacktrace': error.stack || '',
    });
    span.status = SpanStatus.ERROR;
    span.statusMessage = error.message;
  }

  /**
   * Flush completed spans to exporter
   */
  private async flushCompletedSpans(): Promise<void> {
    if (this.completedSpans.length === 0) return;

    const spans = this.completedSpans;
    this.completedSpans = [];

    try {
      await this.exporter.export(spans);
    } catch (error) {
      console.error('[Tracer] Failed to export spans:', error);
      // Re-add to completed spans
      this.completedSpans.push(...spans);
    }
  }

  /**
   * Create a child span
   */
  childSpan(parent: Span, name: string, options: { kind?: SpanKind; attributes?: SpanAttributes } = {}): Span {
    return this.startSpan(name, {
      parentContext: {
        traceId: parent.traceId,
        spanId: parent.spanId,
        traceFlags: 1,
      },
      ...options,
    });
  }

  /**
   * Wrap a function with tracing
   */
  async trace<T>(
    name: string,
    fn: (span: Span) => Promise<T>,
    options: { kind?: SpanKind; attributes?: SpanAttributes } = {}
  ): Promise<T> {
    const span = this.startSpan(name, options);

    try {
      const result = await fn(span);
      this.endSpan(span, SpanStatus.OK);
      return result;
    } catch (error) {
      this.recordException(span, error as Error);
      this.endSpan(span, SpanStatus.ERROR);
      throw error;
    }
  }
}

// Singleton tracer
export const tracer = new Tracer();

/**
 * Tracing middleware
 */
export function tracingMiddleware(): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    // Parse incoming trace context
    const parentContext = tracer.parseTraceparent(req.headers[TRACEPARENT_HEADER] as string);

    // Start server span
    const span = tracer.startSpan(`${req.method} ${req.path}`, {
      kind: SpanKind.SERVER,
      parentContext: parentContext || undefined,
      attributes: {
        'http.method': req.method,
        'http.url': req.url,
        'http.target': req.path,
        'http.host': req.hostname,
        'http.user_agent': req.headers['user-agent'] || '',
        'http.request_content_length': req.headers['content-length'] || '0',
        'net.peer.ip': req.ip || req.socket.remoteAddress || '',
      },
    });

    // Attach span to request
    (req as any).span = span;

    // Set response headers for trace propagation
    res.setHeader(
      TRACEPARENT_HEADER,
      tracer.createTraceparent({
        traceId: span.traceId,
        spanId: span.spanId,
        traceFlags: 1,
      })
    );

    // End span on response finish
    res.on('finish', () => {
      tracer.setAttributes(span, {
        'http.status_code': res.statusCode,
        'http.response_content_length': res.getHeader('content-length')?.toString() || '0',
      });

      const status = res.statusCode >= 400 ? SpanStatus.ERROR : SpanStatus.OK;
      tracer.endSpan(span, status);
    });

    next();
  };
}

// Export types and utilities
export { ConsoleSpanExporter, OTLPSpanExporter };

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Distributed Tracing
 *
 * OpenTelemetry-compatible distributed tracing for cross-hApp operations.
 * Enables end-to-end visibility across the Mycelix ecosystem.
 *
 * @packageDocumentation
 * @module utils/tracing
 */

/** Branded timestamp type */
type BrandedTimestampMs = number & { readonly __brand: 'TimestampMs' };

/** Create branded timestamp */
function brandTimestamp(ms: number): BrandedTimestampMs {
  return ms as BrandedTimestampMs;
}

// ============================================================================
// Types
// ============================================================================

/**
 * Trace context for propagating across hApp boundaries
 */
export interface TraceContext {
  /** Unique trace ID (128-bit, hex encoded) */
  traceId: string;
  /** Current span ID (64-bit, hex encoded) */
  spanId: string;
  /** Parent span ID */
  parentSpanId?: string;
  /** Trace flags (sampling, etc) */
  flags: number;
  /** Baggage items (key-value pairs) */
  baggage?: Record<string, string>;
}

/**
 * Span status
 */
export enum SpanStatus {
  UNSET = 'unset',
  OK = 'ok',
  ERROR = 'error',
}

/**
 * Span kind
 */
export enum SpanKind {
  INTERNAL = 'internal',
  SERVER = 'server',
  CLIENT = 'client',
  PRODUCER = 'producer',
  CONSUMER = 'consumer',
}

/**
 * Span attributes
 */
export type SpanAttributes = Record<string, string | number | boolean | string[] | number[]>;

/**
 * Span event
 */
export interface SpanEvent {
  name: string;
  timestamp: BrandedTimestampMs | number;
  attributes?: SpanAttributes;
}

/**
 * Span link to related traces
 */
export interface SpanLink {
  traceId: string;
  spanId: string;
  attributes?: SpanAttributes;
}

/**
 * Completed span data
 */
export interface SpanData {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: SpanKind;
  startTime: number;
  endTime: number;
  durationMs: number;
  status: SpanStatus;
  statusMessage?: string;
  attributes: SpanAttributes;
  events: SpanEvent[];
  links: SpanLink[];
}

/**
 * Span exporter interface
 */
export interface SpanExporter {
  export(spans: SpanData[]): Promise<void>;
  shutdown(): Promise<void>;
}

// ============================================================================
// Implementation
// ============================================================================

/**
 * Active span for recording operations
 */
export class Span {
  private _traceId: string;
  private _spanId: string;
  private _parentSpanId?: string;
  private _name: string;
  private _kind: SpanKind;
  private _startTime: number;
  private _endTime?: number;
  private _status: SpanStatus = SpanStatus.UNSET;
  private _statusMessage?: string;
  private _attributes: SpanAttributes = {};
  private _events: SpanEvent[] = [];
  private _links: SpanLink[] = [];
  private _ended = false;
  private _tracer: Tracer;

  constructor(
    tracer: Tracer,
    name: string,
    kind: SpanKind,
    traceId: string,
    spanId: string,
    parentSpanId?: string
  ) {
    this._tracer = tracer;
    this._name = name;
    this._kind = kind;
    this._traceId = traceId;
    this._spanId = spanId;
    this._parentSpanId = parentSpanId;
    this._startTime = Date.now();
  }

  /**
   * Set a single attribute
   */
  setAttribute(key: string, value: string | number | boolean): this {
    this._attributes[key] = value;
    return this;
  }

  /**
   * Set multiple attributes
   */
  setAttributes(attributes: SpanAttributes): this {
    Object.assign(this._attributes, attributes);
    return this;
  }

  /**
   * Add an event to the span
   */
  addEvent(name: string, attributes?: SpanAttributes): this {
    this._events.push({
      name,
      timestamp: brandTimestamp(Date.now()),
      attributes,
    });
    return this;
  }

  /**
   * Add a link to a related span
   */
  addLink(traceId: string, spanId: string, attributes?: SpanAttributes): this {
    this._links.push({ traceId, spanId, attributes });
    return this;
  }

  /**
   * Set the span status
   */
  setStatus(status: SpanStatus, message?: string): this {
    this._status = status;
    this._statusMessage = message;
    return this;
  }

  /**
   * Record an exception
   */
  recordException(error: Error): this {
    this.addEvent('exception', {
      'exception.type': error.name,
      'exception.message': error.message,
      'exception.stacktrace': error.stack ?? '',
    });
    this.setStatus(SpanStatus.ERROR, error.message);
    return this;
  }

  /**
   * Get the trace context for propagation
   */
  getContext(): TraceContext {
    return {
      traceId: this._traceId,
      spanId: this._spanId,
      parentSpanId: this._parentSpanId,
      flags: 1, // Sampled
    };
  }

  /**
   * End the span
   */
  end(): void {
    if (this._ended) return;
    this._ended = true;
    this._endTime = Date.now();
    this._tracer._recordSpan(this._toSpanData());
  }

  /** @internal */
  _toSpanData(): SpanData {
    return {
      traceId: this._traceId,
      spanId: this._spanId,
      parentSpanId: this._parentSpanId,
      name: this._name,
      kind: this._kind,
      startTime: this._startTime,
      endTime: this._endTime ?? Date.now(),
      durationMs: (this._endTime ?? Date.now()) - this._startTime,
      status: this._status,
      statusMessage: this._statusMessage,
      attributes: this._attributes,
      events: this._events,
      links: this._links,
    };
  }
}

/**
 * Tracer for creating and managing spans
 */
export class Tracer {
  private _name: string;
  private _version: string;
  private _spans: SpanData[] = [];
  private _exporters: SpanExporter[] = [];
  private _batchSize = 100;
  private _flushIntervalMs = 5000;
  private _flushTimer?: NodeJS.Timeout;
  private _currentSpan?: Span;

  constructor(name: string, version: string = '1.0.0') {
    this._name = name;
    this._version = version;
  }

  /**
   * Start a new span
   */
  startSpan(name: string, options: {
    kind?: SpanKind;
    parent?: TraceContext;
    attributes?: SpanAttributes;
    links?: SpanLink[];
  } = {}): Span {
    const traceId = options.parent?.traceId ?? this._generateTraceId();
    const spanId = this._generateSpanId();
    const parentSpanId = options.parent?.spanId;

    const span = new Span(
      this,
      name,
      options.kind ?? SpanKind.INTERNAL,
      traceId,
      spanId,
      parentSpanId
    );

    if (options.attributes) {
      span.setAttributes(options.attributes);
    }

    if (options.links) {
      for (const link of options.links) {
        span.addLink(link.traceId, link.spanId, link.attributes);
      }
    }

    // Set standard attributes
    span.setAttributes({
      'service.name': this._name,
      'service.version': this._version,
    });

    this._currentSpan = span;
    return span;
  }

  /**
   * Get the current active span
   */
  getCurrentSpan(): Span | undefined {
    return this._currentSpan;
  }

  /**
   * Add an exporter
   */
  addExporter(exporter: SpanExporter): void {
    this._exporters.push(exporter);

    // Start flush timer if first exporter
    if (this._exporters.length === 1 && !this._flushTimer) {
      this._flushTimer = setInterval(() => {
        void this.flush();
      }, this._flushIntervalMs);
    }
  }

  /**
   * Flush buffered spans to exporters
   */
  async flush(): Promise<void> {
    if (this._spans.length === 0) return;

    const spansToExport = this._spans.splice(0, this._batchSize);

    await Promise.all(
      this._exporters.map((exporter) =>
        exporter.export(spansToExport).catch((err) =>
          console.error('Span export failed:', err)
        )
      )
    );
  }

  /**
   * Shutdown the tracer
   */
  async shutdown(): Promise<void> {
    if (this._flushTimer) {
      clearInterval(this._flushTimer);
      this._flushTimer = undefined;
    }

    await this.flush();

    await Promise.all(
      this._exporters.map((exporter) => exporter.shutdown())
    );
  }

  /** @internal */
  _recordSpan(span: SpanData): void {
    this._spans.push(span);

    // Auto-flush if batch size reached
    if (this._spans.length >= this._batchSize) {
      this.flush().catch((err) => console.error('Auto-flush failed:', err));
    }
  }

  private _generateTraceId(): string {
    return Array.from({ length: 32 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }

  private _generateSpanId(): string {
    return Array.from({ length: 16 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }
}

// ============================================================================
// Exporters
// ============================================================================

/**
 * Console exporter for development
 */
export class ConsoleSpanExporter implements SpanExporter {
  async export(spans: SpanData[]): Promise<void> {
    for (const span of spans) {
      const status = span.status === SpanStatus.ERROR ? '❌' : '✓';
      console.log(
        `[TRACE] ${status} ${span.name} (${span.durationMs}ms)`,
        `trace=${span.traceId.substring(0, 8)}`,
        `span=${span.spanId.substring(0, 8)}`
      );

      if (span.events.length > 0) {
        for (const event of span.events) {
          console.log(`  └─ ${event.name}`, event.attributes);
        }
      }
    }
  }

  async shutdown(): Promise<void> {
    // No cleanup needed
  }
}

/**
 * In-memory exporter for testing
 */
export class InMemorySpanExporter implements SpanExporter {
  private _spans: SpanData[] = [];

  async export(spans: SpanData[]): Promise<void> {
    this._spans.push(...spans);
  }

  async shutdown(): Promise<void> {
    this._spans = [];
  }

  getSpans(): SpanData[] {
    return [...this._spans];
  }

  clear(): void {
    this._spans = [];
  }
}

/**
 * HTTP exporter for OTLP-compatible collectors
 */
export class OTLPHttpExporter implements SpanExporter {
  private _endpoint: string;
  private _headers: Record<string, string>;

  constructor(endpoint: string, headers: Record<string, string> = {}) {
    this._endpoint = endpoint;
    this._headers = headers;
  }

  async export(spans: SpanData[]): Promise<void> {
    const payload = {
      resourceSpans: [{
        resource: {
          attributes: [],
        },
        scopeSpans: [{
          scope: {},
          spans: spans.map(this._convertSpan),
        }],
      }],
    };

    try {
      const response = await fetch(this._endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this._headers,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        console.error('OTLP export failed:', response.statusText);
      }
    } catch (error) {
      console.error('OTLP export error:', error);
    }
  }

  async shutdown(): Promise<void> {
    // No cleanup needed
  }

  private _convertSpan(span: SpanData): Record<string, unknown> {
    return {
      traceId: span.traceId,
      spanId: span.spanId,
      parentSpanId: span.parentSpanId,
      name: span.name,
      kind: span.kind,
      startTimeUnixNano: span.startTime * 1000000,
      endTimeUnixNano: span.endTime * 1000000,
      status: { code: span.status === SpanStatus.OK ? 1 : span.status === SpanStatus.ERROR ? 2 : 0 },
      attributes: Object.entries(span.attributes).map(([key, value]) => ({
        key,
        value: { stringValue: String(value) },
      })),
      events: span.events.map((e) => ({
        name: e.name,
        timeUnixNano: (e.timestamp as number) * 1000000,
      })),
    };
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/** Global tracer instance */
let globalTracer: Tracer | null = null;

/**
 * Get or create the global tracer
 */
export function getTracer(name: string = 'mycelix', version: string = '1.0.0'): Tracer {
  if (!globalTracer) {
    globalTracer = new Tracer(name, version);
  }
  return globalTracer;
}

/**
 * Create a new isolated tracer (for testing)
 */
export function createTracer(name: string, version: string = '1.0.0'): Tracer {
  return new Tracer(name, version);
}

/**
 * Trace a function execution
 */
export async function trace<T>(
  name: string,
  fn: (span: Span) => Promise<T>,
  options?: {
    tracer?: Tracer;
    parent?: TraceContext;
    attributes?: SpanAttributes;
  }
): Promise<T> {
  const tracer = options?.tracer ?? getTracer();
  const span = tracer.startSpan(name, {
    parent: options?.parent,
    attributes: options?.attributes,
  });

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

/**
 * Trace a synchronous function
 */
export function traceSync<T>(
  name: string,
  fn: (span: Span) => T,
  options?: {
    tracer?: Tracer;
    parent?: TraceContext;
    attributes?: SpanAttributes;
  }
): T {
  const tracer = options?.tracer ?? getTracer();
  const span = tracer.startSpan(name, {
    parent: options?.parent,
    attributes: options?.attributes,
  });

  try {
    const result = fn(span);
    span.setStatus(SpanStatus.OK);
    return result;
  } catch (error) {
    span.recordException(error as Error);
    throw error;
  } finally {
    span.end();
  }
}

// ============================================================================
// Cross-hApp Trace Propagation
// ============================================================================

/**
 * Serialize trace context for cross-hApp propagation
 */
export function serializeTraceContext(context: TraceContext): string {
  const base = `00-${context.traceId}-${context.spanId}-${context.flags.toString(16).padStart(2, '0')}`;

  if (context.baggage && Object.keys(context.baggage).length > 0) {
    const baggageStr = Object.entries(context.baggage)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
      .join(',');
    return `${base}|${baggageStr}`;
  }

  return base;
}

/**
 * Deserialize trace context from string
 */
export function deserializeTraceContext(header: string): TraceContext | null {
  const [traceparent, baggageStr] = header.split('|');
  const parts = traceparent.split('-');

  if (parts.length !== 4 || parts[0] !== '00') {
    return null;
  }

  const context: TraceContext = {
    traceId: parts[1],
    spanId: parts[2],
    flags: parseInt(parts[3], 16),
  };

  if (baggageStr) {
    context.baggage = {};
    for (const pair of baggageStr.split(',')) {
      const [key, value] = pair.split('=');
      if (key && value) {
        context.baggage[decodeURIComponent(key)] = decodeURIComponent(value);
      }
    }
  }

  return context;
}

/**
 * Standard attribute keys for Mycelix
 */
export const TraceAttributes = {
  // hApp identification
  HAPP_ID: 'mycelix.happ.id',
  HAPP_NAME: 'mycelix.happ.name',
  ZOME_NAME: 'mycelix.zome.name',
  FUNCTION_NAME: 'mycelix.function.name',

  // Agent identification
  AGENT_ID: 'mycelix.agent.id',
  AGENT_ROLE: 'mycelix.agent.role',

  // Operation details
  OPERATION_TYPE: 'mycelix.operation.type',
  OPERATION_SUCCESS: 'mycelix.operation.success',

  // MATL
  TRUST_SCORE: 'mycelix.matl.trust_score',
  REPUTATION_SCORE: 'mycelix.matl.reputation_score',
  BYZANTINE_DETECTED: 'mycelix.matl.byzantine_detected',

  // Epistemic
  CLAIM_ID: 'mycelix.epistemic.claim_id',
  EMPIRICAL_LEVEL: 'mycelix.epistemic.empirical',
  NORMATIVE_LEVEL: 'mycelix.epistemic.normative',
  MATERIALITY_LEVEL: 'mycelix.epistemic.materiality',

  // FL
  FL_ROUND_ID: 'mycelix.fl.round_id',
  FL_PARTICIPANT_COUNT: 'mycelix.fl.participant_count',
  FL_AGGREGATION_METHOD: 'mycelix.fl.aggregation_method',

  // Bridge
  BRIDGE_SOURCE_HAPP: 'mycelix.bridge.source_happ',
  BRIDGE_TARGET_HAPP: 'mycelix.bridge.target_happ',
  BRIDGE_MESSAGE_TYPE: 'mycelix.bridge.message_type',
} as const;

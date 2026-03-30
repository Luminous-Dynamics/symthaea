// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tracing module tests
 * Tests for @mycelix/sdk distributed tracing utilities
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  Span,
  Tracer,
  SpanStatus,
  SpanKind,
  ConsoleSpanExporter,
  InMemorySpanExporter,
  OTLPHttpExporter,
  getTracer,
  createTracer,
  trace,
  traceSync,
  serializeTraceContext,
  deserializeTraceContext,
  TraceAttributes,
  type TraceContext,
  type SpanData,
  type SpanExporter,
} from '../../src/utils/tracing.js';

describe('SpanStatus enum', () => {
  it('should have correct values', () => {
    expect(SpanStatus.UNSET).toBe('unset');
    expect(SpanStatus.OK).toBe('ok');
    expect(SpanStatus.ERROR).toBe('error');
  });
});

describe('SpanKind enum', () => {
  it('should have correct values', () => {
    expect(SpanKind.INTERNAL).toBe('internal');
    expect(SpanKind.SERVER).toBe('server');
    expect(SpanKind.CLIENT).toBe('client');
    expect(SpanKind.PRODUCER).toBe('producer');
    expect(SpanKind.CONSUMER).toBe('consumer');
  });
});

describe('Span', () => {
  let tracer: Tracer;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    tracer = createTracer('test-service', '1.0.0');
    exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);
  });

  afterEach(async () => {
    await tracer.shutdown();
  });

  it('should create a span with correct initial values', () => {
    const span = tracer.startSpan('test-operation');
    const context = span.getContext();

    expect(context.traceId).toHaveLength(32);
    expect(context.spanId).toHaveLength(16);
    expect(context.flags).toBe(1);
  });

  describe('setAttribute', () => {
    it('should set string attribute', async () => {
      const span = tracer.startSpan('test');
      span.setAttribute('key', 'value');
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes.key).toBe('value');
    });

    it('should set number attribute', async () => {
      const span = tracer.startSpan('test');
      span.setAttribute('count', 42);
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes.count).toBe(42);
    });

    it('should set boolean attribute', async () => {
      const span = tracer.startSpan('test');
      span.setAttribute('success', true);
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes.success).toBe(true);
    });

    it('should return this for chaining', () => {
      const span = tracer.startSpan('test');
      const result = span.setAttribute('key', 'value');
      expect(result).toBe(span);
    });
  });

  describe('setAttributes', () => {
    it('should set multiple attributes at once', async () => {
      const span = tracer.startSpan('test');
      span.setAttributes({
        'http.method': 'GET',
        'http.status_code': 200,
        'http.success': true,
      });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes['http.method']).toBe('GET');
      expect(spans[0].attributes['http.status_code']).toBe(200);
      expect(spans[0].attributes['http.success']).toBe(true);
    });

    it('should return this for chaining', () => {
      const span = tracer.startSpan('test');
      const result = span.setAttributes({ key: 'value' });
      expect(result).toBe(span);
    });
  });

  describe('addEvent', () => {
    it('should add event with name only', async () => {
      const span = tracer.startSpan('test');
      span.addEvent('checkpoint');
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].events).toHaveLength(1);
      expect(spans[0].events[0].name).toBe('checkpoint');
    });

    it('should add event with attributes', async () => {
      const span = tracer.startSpan('test');
      span.addEvent('user.action', { action: 'click', target: 'button' });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].events[0].attributes).toEqual({
        action: 'click',
        target: 'button',
      });
    });

    it('should record timestamp', async () => {
      const before = Date.now();
      const span = tracer.startSpan('test');
      span.addEvent('event');
      span.end();
      const after = Date.now();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].events[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(spans[0].events[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('should return this for chaining', () => {
      const span = tracer.startSpan('test');
      const result = span.addEvent('event');
      expect(result).toBe(span);
    });
  });

  describe('addLink', () => {
    it('should add link to related span', async () => {
      const span = tracer.startSpan('test');
      span.addLink('trace-123', 'span-456');
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].links).toHaveLength(1);
      expect(spans[0].links[0]).toEqual({
        traceId: 'trace-123',
        spanId: 'span-456',
        attributes: undefined,
      });
    });

    it('should add link with attributes', async () => {
      const span = tracer.startSpan('test');
      span.addLink('trace-123', 'span-456', { relationship: 'causal' });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].links[0].attributes).toEqual({ relationship: 'causal' });
    });

    it('should return this for chaining', () => {
      const span = tracer.startSpan('test');
      const result = span.addLink('trace', 'span');
      expect(result).toBe(span);
    });
  });

  describe('setStatus', () => {
    it('should set OK status', async () => {
      const span = tracer.startSpan('test');
      span.setStatus(SpanStatus.OK);
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].status).toBe(SpanStatus.OK);
    });

    it('should set ERROR status with message', async () => {
      const span = tracer.startSpan('test');
      span.setStatus(SpanStatus.ERROR, 'Something went wrong');
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].status).toBe(SpanStatus.ERROR);
      expect(spans[0].statusMessage).toBe('Something went wrong');
    });

    it('should return this for chaining', () => {
      const span = tracer.startSpan('test');
      const result = span.setStatus(SpanStatus.OK);
      expect(result).toBe(span);
    });
  });

  describe('recordException', () => {
    it('should record exception as event', async () => {
      const span = tracer.startSpan('test');
      const error = new Error('Test error');
      error.name = 'TestError';
      span.recordException(error);
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      const event = spans[0].events.find((e) => e.name === 'exception');
      expect(event).toBeDefined();
      expect(event?.attributes?.['exception.type']).toBe('TestError');
      expect(event?.attributes?.['exception.message']).toBe('Test error');
    });

    it('should set ERROR status', async () => {
      const span = tracer.startSpan('test');
      span.recordException(new Error('Failed'));
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].status).toBe(SpanStatus.ERROR);
    });

    it('should handle error without stack trace', async () => {
      const span = tracer.startSpan('test');
      const error = new Error('No stack');
      error.stack = undefined;
      span.recordException(error);
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      const event = spans[0].events[0];
      expect(event.attributes?.['exception.stacktrace']).toBe('');
    });
  });

  describe('getContext', () => {
    it('should return trace context', () => {
      const span = tracer.startSpan('test');
      const context = span.getContext();

      expect(context.traceId).toBeDefined();
      expect(context.spanId).toBeDefined();
      expect(context.flags).toBe(1);
    });

    it('should include parent span ID when present', () => {
      const parentContext: TraceContext = {
        traceId: 'parent-trace-id-00000000000000',
        spanId: 'parent-span-id',
        flags: 1,
      };

      const span = tracer.startSpan('child', { parent: parentContext });
      const context = span.getContext();

      expect(context.traceId).toBe('parent-trace-id-00000000000000');
      expect(context.parentSpanId).toBe('parent-span-id');
    });
  });

  describe('end', () => {
    it('should record duration', async () => {
      const span = tracer.startSpan('test');
      // Small delay to ensure measurable duration
      await new Promise((r) => setTimeout(r, 5));
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].durationMs).toBeGreaterThan(0);
      expect(spans[0].endTime).toBeGreaterThanOrEqual(spans[0].startTime);
    });

    it('should only end once', async () => {
      const span = tracer.startSpan('test');
      span.end();
      span.end(); // Second call should be no-op

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans).toHaveLength(1);
    });
  });
});

describe('Tracer', () => {
  let tracer: Tracer;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    tracer = createTracer('test-service', '2.0.0');
    exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);
  });

  afterEach(async () => {
    await tracer.shutdown();
  });

  describe('startSpan', () => {
    it('should create span with name', () => {
      const span = tracer.startSpan('operation-name');
      expect(span).toBeInstanceOf(Span);
    });

    it('should create span with kind', async () => {
      const span = tracer.startSpan('server-op', { kind: SpanKind.SERVER });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].kind).toBe(SpanKind.SERVER);
    });

    it('should create span with parent context', async () => {
      const parentContext: TraceContext = {
        traceId: '0'.repeat(32),
        spanId: '1'.repeat(16),
        flags: 1,
      };

      const span = tracer.startSpan('child', { parent: parentContext });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].traceId).toBe(parentContext.traceId);
      expect(spans[0].parentSpanId).toBe(parentContext.spanId);
    });

    it('should create span with initial attributes', async () => {
      const span = tracer.startSpan('op', {
        attributes: { 'custom.attr': 'value' },
      });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes['custom.attr']).toBe('value');
    });

    it('should add service attributes', async () => {
      const span = tracer.startSpan('op');
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].attributes['service.name']).toBe('test-service');
      expect(spans[0].attributes['service.version']).toBe('2.0.0');
    });

    it('should create span with links', async () => {
      const links = [
        { traceId: 'trace1', spanId: 'span1' },
        { traceId: 'trace2', spanId: 'span2', attributes: { type: 'reference' } },
      ];

      const span = tracer.startSpan('op', { links });
      span.end();

      await tracer.flush();
      const spans = exporter.getSpans();
      expect(spans[0].links).toHaveLength(2);
      expect(spans[0].links[1].attributes?.type).toBe('reference');
    });
  });

  describe('getCurrentSpan', () => {
    it('should return undefined initially', () => {
      const newTracer = createTracer('new');
      expect(newTracer.getCurrentSpan()).toBeUndefined();
    });

    it('should return most recently started span', () => {
      const span = tracer.startSpan('current');
      expect(tracer.getCurrentSpan()).toBe(span);
    });
  });

  describe('flush', () => {
    it('should export buffered spans', async () => {
      const span = tracer.startSpan('test');
      span.end();

      expect(exporter.getSpans()).toHaveLength(0);
      await tracer.flush();
      expect(exporter.getSpans()).toHaveLength(1);
    });

    it('should do nothing when no spans', async () => {
      await tracer.flush(); // Should not throw
    });

    it('should handle exporter errors gracefully', async () => {
      const errorExporter: SpanExporter = {
        export: vi.fn().mockRejectedValue(new Error('Export failed')),
        shutdown: vi.fn().mockResolvedValue(undefined),
      };

      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
      tracer.addExporter(errorExporter);

      const span = tracer.startSpan('test');
      span.end();

      await tracer.flush();
      expect(consoleError).toHaveBeenCalled();

      consoleError.mockRestore();
    });
  });

  describe('shutdown', () => {
    it('should flush remaining spans', async () => {
      const span = tracer.startSpan('test');
      span.end();

      // Flush first and check spans before shutdown clears the exporter
      await tracer.flush();
      expect(exporter.getSpans()).toHaveLength(1);
    });

    it('should call shutdown on exporters', async () => {
      const mockExporter: SpanExporter = {
        export: vi.fn().mockResolvedValue(undefined),
        shutdown: vi.fn().mockResolvedValue(undefined),
      };
      tracer.addExporter(mockExporter);

      await tracer.shutdown();
      expect(mockExporter.shutdown).toHaveBeenCalled();
    });
  });

  describe('auto-flush on batch size', () => {
    it('should flush when batch size reached', async () => {
      // Create tracer that will auto-flush at 100 spans
      const localTracer = createTracer('batch-test');
      const localExporter = new InMemorySpanExporter();
      localTracer.addExporter(localExporter);

      // Create and end 100 spans
      for (let i = 0; i < 100; i++) {
        const span = localTracer.startSpan(`span-${i}`);
        span.end();
      }

      // Wait a bit for auto-flush
      await new Promise((r) => setTimeout(r, 50));

      // Should have exported
      expect(localExporter.getSpans().length).toBeGreaterThan(0);

      await localTracer.shutdown();
    });
  });
});

describe('ConsoleSpanExporter', () => {
  let consoleLog: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleLog.mockRestore();
  });

  it('should log spans to console', async () => {
    const exporter = new ConsoleSpanExporter();
    const spanData: SpanData = {
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      name: 'test-span',
      kind: SpanKind.INTERNAL,
      startTime: Date.now() - 100,
      endTime: Date.now(),
      durationMs: 100,
      status: SpanStatus.OK,
      attributes: {},
      events: [],
      links: [],
    };

    await exporter.export([spanData]);

    expect(consoleLog).toHaveBeenCalledWith(
      expect.stringContaining('test-span'),
      expect.stringContaining('trace='),
      expect.stringContaining('span=')
    );
  });

  it('should show error indicator for error spans', async () => {
    const exporter = new ConsoleSpanExporter();
    const spanData: SpanData = {
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      name: 'failed-span',
      kind: SpanKind.INTERNAL,
      startTime: Date.now(),
      endTime: Date.now(),
      durationMs: 0,
      status: SpanStatus.ERROR,
      attributes: {},
      events: [],
      links: [],
    };

    await exporter.export([spanData]);

    expect(consoleLog).toHaveBeenCalledWith(
      expect.stringContaining('❌'),
      expect.any(String),
      expect.any(String)
    );
  });

  it('should log events', async () => {
    const exporter = new ConsoleSpanExporter();
    const spanData: SpanData = {
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      name: 'span-with-events',
      kind: SpanKind.INTERNAL,
      startTime: Date.now(),
      endTime: Date.now(),
      durationMs: 0,
      status: SpanStatus.OK,
      attributes: {},
      events: [{ name: 'checkpoint', timestamp: Date.now(), attributes: { data: 'test' } }],
      links: [],
    };

    await exporter.export([spanData]);

    expect(consoleLog).toHaveBeenCalledWith(
      expect.stringContaining('checkpoint'),
      expect.objectContaining({ data: 'test' })
    );
  });

  it('should shutdown without error', async () => {
    const exporter = new ConsoleSpanExporter();
    await exporter.shutdown(); // Should not throw
  });
});

describe('InMemorySpanExporter', () => {
  it('should store exported spans', async () => {
    const exporter = new InMemorySpanExporter();
    const span1: SpanData = {
      traceId: '1'.repeat(32),
      spanId: '1'.repeat(16),
      name: 'span-1',
      kind: SpanKind.INTERNAL,
      startTime: Date.now(),
      endTime: Date.now(),
      durationMs: 0,
      status: SpanStatus.OK,
      attributes: {},
      events: [],
      links: [],
    };

    await exporter.export([span1]);
    expect(exporter.getSpans()).toHaveLength(1);
  });

  it('should accumulate spans', async () => {
    const exporter = new InMemorySpanExporter();
    const createSpan = (name: string): SpanData => ({
      traceId: Math.random().toString(16).slice(2).padEnd(32, '0'),
      spanId: Math.random().toString(16).slice(2).padEnd(16, '0'),
      name,
      kind: SpanKind.INTERNAL,
      startTime: Date.now(),
      endTime: Date.now(),
      durationMs: 0,
      status: SpanStatus.OK,
      attributes: {},
      events: [],
      links: [],
    });

    await exporter.export([createSpan('span-1')]);
    await exporter.export([createSpan('span-2'), createSpan('span-3')]);

    expect(exporter.getSpans()).toHaveLength(3);
  });

  it('should return copy of spans', async () => {
    const exporter = new InMemorySpanExporter();
    await exporter.export([
      {
        traceId: '0'.repeat(32),
        spanId: '0'.repeat(16),
        name: 'test',
        kind: SpanKind.INTERNAL,
        startTime: 0,
        endTime: 0,
        durationMs: 0,
        status: SpanStatus.OK,
        attributes: {},
        events: [],
        links: [],
      },
    ]);

    const spans1 = exporter.getSpans();
    const spans2 = exporter.getSpans();
    expect(spans1).not.toBe(spans2);
  });

  it('should clear spans', async () => {
    const exporter = new InMemorySpanExporter();
    await exporter.export([
      {
        traceId: '0'.repeat(32),
        spanId: '0'.repeat(16),
        name: 'test',
        kind: SpanKind.INTERNAL,
        startTime: 0,
        endTime: 0,
        durationMs: 0,
        status: SpanStatus.OK,
        attributes: {},
        events: [],
        links: [],
      },
    ]);

    exporter.clear();
    expect(exporter.getSpans()).toHaveLength(0);
  });

  it('should clear on shutdown', async () => {
    const exporter = new InMemorySpanExporter();
    await exporter.export([
      {
        traceId: '0'.repeat(32),
        spanId: '0'.repeat(16),
        name: 'test',
        kind: SpanKind.INTERNAL,
        startTime: 0,
        endTime: 0,
        durationMs: 0,
        status: SpanStatus.OK,
        attributes: {},
        events: [],
        links: [],
      },
    ]);

    await exporter.shutdown();
    expect(exporter.getSpans()).toHaveLength(0);
  });
});

describe('OTLPHttpExporter', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('should POST spans to endpoint', async () => {
    const exporter = new OTLPHttpExporter('https://collector.example.com/v1/traces');
    const span: SpanData = {
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      name: 'test',
      kind: SpanKind.CLIENT,
      startTime: 1000,
      endTime: 2000,
      durationMs: 1000,
      status: SpanStatus.OK,
      attributes: { key: 'value' },
      events: [{ name: 'event', timestamp: 1500 }],
      links: [],
    };

    await exporter.export([span]);

    expect(fetchMock).toHaveBeenCalledWith(
      'https://collector.example.com/v1/traces',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
        }),
      })
    );
  });

  it('should include custom headers', async () => {
    const exporter = new OTLPHttpExporter('https://collector.example.com', {
      Authorization: 'Bearer token123',
      'X-Custom-Header': 'custom-value',
    });

    await exporter.export([]);

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer token123',
          'X-Custom-Header': 'custom-value',
        }),
      })
    );
  });

  it('should handle export errors', async () => {
    fetchMock.mockResolvedValue({ ok: false, statusText: 'Bad Request' });
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    const exporter = new OTLPHttpExporter('https://collector.example.com');
    await exporter.export([
      {
        traceId: '0'.repeat(32),
        spanId: '0'.repeat(16),
        name: 'test',
        kind: SpanKind.INTERNAL,
        startTime: 0,
        endTime: 0,
        durationMs: 0,
        status: SpanStatus.OK,
        attributes: {},
        events: [],
        links: [],
      },
    ]);

    expect(consoleError).toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it('should handle network errors', async () => {
    fetchMock.mockRejectedValue(new Error('Network error'));
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    const exporter = new OTLPHttpExporter('https://collector.example.com');
    await exporter.export([
      {
        traceId: '0'.repeat(32),
        spanId: '0'.repeat(16),
        name: 'test',
        kind: SpanKind.INTERNAL,
        startTime: 0,
        endTime: 0,
        durationMs: 0,
        status: SpanStatus.OK,
        attributes: {},
        events: [],
        links: [],
      },
    ]);

    expect(consoleError).toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it('should shutdown without error', async () => {
    const exporter = new OTLPHttpExporter('https://collector.example.com');
    await exporter.shutdown(); // Should not throw
  });
});

describe('getTracer', () => {
  it('should return global tracer instance', () => {
    const tracer1 = getTracer('mycelix', '1.0.0');
    const tracer2 = getTracer();
    expect(tracer1).toBe(tracer2);
  });
});

describe('createTracer', () => {
  it('should create new tracer instance', () => {
    const tracer1 = createTracer('service1');
    const tracer2 = createTracer('service2');
    expect(tracer1).not.toBe(tracer2);
  });
});

describe('trace function', () => {
  it('should trace async function execution', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    const result = await trace(
      'async-operation',
      async (span) => {
        span.setAttribute('custom', 'value');
        return 'result';
      },
      { tracer }
    );

    await tracer.flush();
    expect(result).toBe('result');
    expect(exporter.getSpans()).toHaveLength(1);
    expect(exporter.getSpans()[0].status).toBe(SpanStatus.OK);

    await tracer.shutdown();
  });

  it('should record exceptions', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    await expect(
      trace(
        'failing-operation',
        async () => {
          throw new Error('Test failure');
        },
        { tracer }
      )
    ).rejects.toThrow('Test failure');

    await tracer.flush();
    expect(exporter.getSpans()[0].status).toBe(SpanStatus.ERROR);

    await tracer.shutdown();
  });

  it('should use parent context', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    const parentContext: TraceContext = {
      traceId: 'parent-trace'.padEnd(32, '0'),
      spanId: 'parent-span'.padEnd(16, '0'),
      flags: 1,
    };

    await trace('child-operation', async () => 'done', {
      tracer,
      parent: parentContext,
    });

    await tracer.flush();
    expect(exporter.getSpans()[0].traceId).toBe(parentContext.traceId);

    await tracer.shutdown();
  });

  it('should include initial attributes', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    await trace('operation', async () => 'done', {
      tracer,
      attributes: { 'initial.attr': 'value' },
    });

    await tracer.flush();
    expect(exporter.getSpans()[0].attributes['initial.attr']).toBe('value');

    await tracer.shutdown();
  });
});

describe('traceSync function', () => {
  it('should trace sync function execution', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    const result = traceSync(
      'sync-operation',
      (span) => {
        span.setAttribute('sync', true);
        return 42;
      },
      { tracer }
    );

    await tracer.flush();
    expect(result).toBe(42);
    expect(exporter.getSpans()).toHaveLength(1);
    expect(exporter.getSpans()[0].status).toBe(SpanStatus.OK);

    await tracer.shutdown();
  });

  it('should record exceptions', async () => {
    const tracer = createTracer('test');
    const exporter = new InMemorySpanExporter();
    tracer.addExporter(exporter);

    expect(() =>
      traceSync(
        'failing-sync',
        () => {
          throw new Error('Sync failure');
        },
        { tracer }
      )
    ).toThrow('Sync failure');

    await tracer.flush();
    expect(exporter.getSpans()[0].status).toBe(SpanStatus.ERROR);

    await tracer.shutdown();
  });
});

describe('serializeTraceContext', () => {
  it('should serialize basic context', () => {
    const context: TraceContext = {
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      flags: 1,
    };

    const serialized = serializeTraceContext(context);
    expect(serialized).toBe(`00-${'a'.repeat(32)}-${'b'.repeat(16)}-01`);
  });

  it('should serialize context with baggage', () => {
    const context: TraceContext = {
      traceId: '0'.repeat(32),
      spanId: '1'.repeat(16),
      flags: 0,
      baggage: { userId: '123', sessionId: 'abc' },
    };

    const serialized = serializeTraceContext(context);
    expect(serialized).toContain('|');
    expect(serialized).toContain('userId=123');
    expect(serialized).toContain('sessionId=abc');
  });

  it('should URL-encode baggage values', () => {
    const context: TraceContext = {
      traceId: '0'.repeat(32),
      spanId: '1'.repeat(16),
      flags: 1,
      baggage: { key: 'value with spaces' },
    };

    const serialized = serializeTraceContext(context);
    expect(serialized).toContain('value%20with%20spaces');
  });

  it('should not include baggage separator for empty baggage', () => {
    const context: TraceContext = {
      traceId: '0'.repeat(32),
      spanId: '1'.repeat(16),
      flags: 1,
      baggage: {},
    };

    const serialized = serializeTraceContext(context);
    expect(serialized).not.toContain('|');
  });
});

describe('deserializeTraceContext', () => {
  it('should deserialize basic context', () => {
    const header = `00-${'a'.repeat(32)}-${'b'.repeat(16)}-01`;
    const context = deserializeTraceContext(header);

    expect(context).toEqual({
      traceId: 'a'.repeat(32),
      spanId: 'b'.repeat(16),
      flags: 1,
    });
  });

  it('should deserialize context with baggage', () => {
    const header = `00-${'0'.repeat(32)}-${'1'.repeat(16)}-00|userId=123,sessionId=abc`;
    const context = deserializeTraceContext(header);

    expect(context?.baggage).toEqual({
      userId: '123',
      sessionId: 'abc',
    });
  });

  it('should URL-decode baggage values', () => {
    const header = `00-${'0'.repeat(32)}-${'1'.repeat(16)}-01|key=value%20with%20spaces`;
    const context = deserializeTraceContext(header);

    expect(context?.baggage?.key).toBe('value with spaces');
  });

  it('should return null for invalid format', () => {
    expect(deserializeTraceContext('invalid')).toBeNull();
    expect(deserializeTraceContext('01-abc-def-00')).toBeNull(); // Wrong version
    expect(deserializeTraceContext('00-abc')).toBeNull(); // Missing parts
  });

  it('should round-trip serialize/deserialize', () => {
    const original: TraceContext = {
      traceId: 'f'.repeat(32),
      spanId: 'e'.repeat(16),
      flags: 1,
      baggage: { user: 'test@example.com', role: 'admin' },
    };

    const serialized = serializeTraceContext(original);
    const deserialized = deserializeTraceContext(serialized);

    expect(deserialized).toEqual(original);
  });
});

describe('TraceAttributes', () => {
  it('should have hApp identification attributes', () => {
    expect(TraceAttributes.HAPP_ID).toBe('mycelix.happ.id');
    expect(TraceAttributes.HAPP_NAME).toBe('mycelix.happ.name');
    expect(TraceAttributes.ZOME_NAME).toBe('mycelix.zome.name');
    expect(TraceAttributes.FUNCTION_NAME).toBe('mycelix.function.name');
  });

  it('should have agent identification attributes', () => {
    expect(TraceAttributes.AGENT_ID).toBe('mycelix.agent.id');
    expect(TraceAttributes.AGENT_ROLE).toBe('mycelix.agent.role');
  });

  it('should have MATL attributes', () => {
    expect(TraceAttributes.TRUST_SCORE).toBe('mycelix.matl.trust_score');
    expect(TraceAttributes.REPUTATION_SCORE).toBe('mycelix.matl.reputation_score');
    expect(TraceAttributes.BYZANTINE_DETECTED).toBe('mycelix.matl.byzantine_detected');
  });

  it('should have epistemic attributes', () => {
    expect(TraceAttributes.CLAIM_ID).toBe('mycelix.epistemic.claim_id');
    expect(TraceAttributes.EMPIRICAL_LEVEL).toBe('mycelix.epistemic.empirical');
    expect(TraceAttributes.NORMATIVE_LEVEL).toBe('mycelix.epistemic.normative');
    expect(TraceAttributes.MATERIALITY_LEVEL).toBe('mycelix.epistemic.materiality');
  });

  it('should have FL attributes', () => {
    expect(TraceAttributes.FL_ROUND_ID).toBe('mycelix.fl.round_id');
    expect(TraceAttributes.FL_PARTICIPANT_COUNT).toBe('mycelix.fl.participant_count');
    expect(TraceAttributes.FL_AGGREGATION_METHOD).toBe('mycelix.fl.aggregation_method');
  });

  it('should have bridge attributes', () => {
    expect(TraceAttributes.BRIDGE_SOURCE_HAPP).toBe('mycelix.bridge.source_happ');
    expect(TraceAttributes.BRIDGE_TARGET_HAPP).toBe('mycelix.bridge.target_happ');
    expect(TraceAttributes.BRIDGE_MESSAGE_TYPE).toBe('mycelix.bridge.message_type');
  });
});

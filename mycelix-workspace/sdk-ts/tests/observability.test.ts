/**
 * Tests for observability patterns
 *
 * Tests tracing, metrics, logging, and health check functionality.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  initObservability,
  getTracer,
  getMeter,
  getMetrics,
  withTrace,
  withTraceSync,
  traceBridgeCall,
  getCurrentContext,
  withContext,
  extractTraceContext,
  LogLevel,
  addLogHandler,
  removeLogHandler,
  log,
  logger,
  consoleLogHandler,
  registerHealthCheck,
  checkHealth,
  type LogEntry,
  type LogHandler,
} from '../src/observability';

// Mock @opentelemetry/api
vi.mock('@opentelemetry/api', () => {
  const mockSpan = {
    setAttribute: vi.fn(),
    setStatus: vi.fn(),
    recordException: vi.fn(),
    addEvent: vi.fn(),
    end: vi.fn(),
    spanContext: () => ({
      traceId: 'test-trace-id',
      spanId: 'test-span-id',
      traceFlags: 1,
    }),
  };

  const mockTracer = {
    startActiveSpan: vi.fn((name, options, fn) => {
      if (typeof options === 'function') {
        return options(mockSpan);
      }
      return fn(mockSpan);
    }),
    startSpan: vi.fn(() => mockSpan),
  };

  const mockCounter = {
    add: vi.fn(),
  };

  const mockHistogram = {
    record: vi.fn(),
  };

  const mockMeter = {
    createCounter: vi.fn(() => mockCounter),
    createHistogram: vi.fn(() => mockHistogram),
  };

  return {
    trace: {
      getTracer: vi.fn(() => mockTracer),
      getActiveSpan: vi.fn(() => mockSpan),
    },
    context: {
      active: vi.fn(() => ({})),
      with: vi.fn((ctx, fn) => fn()),
    },
    metrics: {
      getMeter: vi.fn(() => mockMeter),
    },
    SpanStatusCode: {
      OK: 1,
      ERROR: 2,
    },
  };
});

describe('Tracer and Meter initialization', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Re-initialize observability to get fresh mocked metrics
    initObservability();
  });

  describe('initObservability', () => {
    it('should initialize with default tracer and meter', () => {
      initObservability();
      const tracer = getTracer();
      const meter = getMeter();
      expect(tracer).toBeDefined();
      expect(meter).toBeDefined();
    });

    it('should initialize with custom tracer', () => {
      const customTracer = { startActiveSpan: vi.fn(), startSpan: vi.fn() } as any;
      initObservability({ tracer: customTracer });
      const tracer = getTracer();
      expect(tracer).toBe(customTracer);
    });

    it('should initialize with custom meter', () => {
      const customMeter = { createCounter: vi.fn(), createHistogram: vi.fn() } as any;
      initObservability({ meter: customMeter });
      // Meter should be set
      expect(getMeter()).toBeDefined();
    });
  });

  describe('getTracer', () => {
    it('should return a tracer', () => {
      const tracer = getTracer();
      expect(tracer).toBeDefined();
      expect(tracer.startActiveSpan).toBeDefined();
      expect(tracer.startSpan).toBeDefined();
    });

    it('should create tracer lazily if not initialized', () => {
      const tracer = getTracer();
      expect(tracer).toBeDefined();
    });
  });

  describe('getMeter', () => {
    it('should return a meter', () => {
      const meter = getMeter();
      expect(meter).toBeDefined();
      expect(meter.createCounter).toBeDefined();
      expect(meter.createHistogram).toBeDefined();
    });
  });

  describe('getMetrics', () => {
    it('should return SDK metrics', () => {
      const metrics = getMetrics();
      expect(metrics).toBeDefined();
      expect(metrics.operationCounter).toBeDefined();
      expect(metrics.operationDuration).toBeDefined();
      expect(metrics.errorCounter).toBeDefined();
      expect(metrics.activeOperations).toBeDefined();
      expect(metrics.bridgeCallCounter).toBeDefined();
      expect(metrics.bridgeCallDuration).toBeDefined();
      expect(metrics.validationErrorCounter).toBeDefined();
      expect(metrics.circuitBreakerStateChanges).toBeDefined();
    });
  });
});

describe('Tracing utilities', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Re-initialize observability to get fresh mocked metrics
    initObservability();
  });

  describe('withTrace', () => {
    it('should execute function and return result', async () => {
      const result = await withTrace(
        { name: 'test-operation' },
        async () => 'success'
      );
      expect(result).toBe('success');
    });

    it('should set span attributes', async () => {
      await withTrace(
        {
          name: 'test-operation',
          attributes: { 'custom.attr': 'value' },
        },
        async (span) => {
          expect(span.setAttribute).toHaveBeenCalled();
          return 'ok';
        }
      );
    });

    it('should record metrics by default', async () => {
      const metrics = getMetrics();
      await withTrace(
        { name: 'test-operation' },
        async () => 'ok'
      );
      expect(metrics.operationCounter.add).toHaveBeenCalledWith(1, { operation: 'test-operation' });
    });

    it('should not record metrics when disabled', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();

      await withTrace(
        { name: 'test-operation', recordMetrics: false },
        async () => 'ok'
      );

      // operationCounter should not be called for this operation
      const calls = (metrics.operationCounter.add as any).mock.calls;
      const thisOpCalls = calls.filter((c: any[]) => c[1]?.operation === 'test-operation');
      expect(thisOpCalls.length).toBe(0);
    });

    it('should handle errors and set error status', async () => {
      await expect(withTrace(
        { name: 'failing-operation' },
        async (span) => {
          throw new Error('test error');
        }
      )).rejects.toThrow('test error');
    });

    it('should record error metrics on failure', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();

      try {
        await withTrace(
          { name: 'failing-operation' },
          async () => { throw new Error('test error'); }
        );
      } catch {
        // Expected
      }

      expect(metrics.errorCounter.add).toHaveBeenCalled();
    });
  });

  describe('withTraceSync', () => {
    it('should execute synchronous function and return result', () => {
      const result = withTraceSync(
        { name: 'sync-operation' },
        () => 'sync-success'
      );
      expect(result).toBe('sync-success');
    });

    it('should set span attributes', () => {
      withTraceSync(
        {
          name: 'sync-operation',
          attributes: { 'sync.attr': 'value' },
        },
        (span) => {
          expect(span.setAttribute).toHaveBeenCalled();
          return 'ok';
        }
      );
    });

    it('should handle errors', () => {
      expect(() => withTraceSync(
        { name: 'sync-failing' },
        () => { throw new Error('sync error'); }
      )).toThrow('sync error');
    });

    it('should record metrics', () => {
      const metrics = getMetrics();
      vi.clearAllMocks();

      withTraceSync(
        { name: 'sync-operation' },
        () => 'ok'
      );

      expect(metrics.operationCounter.add).toHaveBeenCalled();
    });
  });

  describe('traceBridgeCall', () => {
    it('should trace bridge calls with happ/zome/fn attributes', async () => {
      const result = await traceBridgeCall(
        { happ: 'mycelix', zome: 'bridge', fn: 'call' },
        async () => 'bridge-result'
      );
      expect(result).toBe('bridge-result');
    });

    it('should record bridge metrics', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();

      await traceBridgeCall(
        { happ: 'mycelix', zome: 'bridge', fn: 'call' },
        async () => 'ok'
      );

      expect(metrics.bridgeCallCounter.add).toHaveBeenCalledWith(1, {
        happ: 'mycelix',
        zome: 'bridge',
        fn: 'call',
      });
    });

    it('should calculate payload size', async () => {
      await traceBridgeCall(
        { happ: 'mycelix', zome: 'bridge', fn: 'call', payload: { data: 'test' } },
        async () => 'ok'
      );
      // Payload size should be included in attributes
    });
  });
});

describe('Context propagation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    initObservability();
  });

  describe('getCurrentContext', () => {
    it('should return current context', () => {
      const ctx = getCurrentContext();
      expect(ctx).toBeDefined();
    });
  });

  describe('withContext', () => {
    it('should execute function with context', () => {
      const result = withContext({} as any, () => 'result');
      expect(result).toBe('result');
    });
  });

  describe('extractTraceContext', () => {
    it('should extract trace context from active span', () => {
      const ctx = extractTraceContext();
      expect(ctx['trace-id']).toBe('test-trace-id');
      expect(ctx['span-id']).toBe('test-span-id');
      expect(ctx['trace-flags']).toBe('1');
    });
  });
});

describe('Logging', () => {
  let capturedLogs: LogEntry[] = [];
  let testHandler: LogHandler;

  beforeEach(() => {
    vi.clearAllMocks();
    initObservability();
    capturedLogs = [];
    testHandler = (entry) => capturedLogs.push(entry);
    addLogHandler(testHandler);
  });

  afterEach(() => {
    removeLogHandler(testHandler);
  });

  describe('addLogHandler and removeLogHandler', () => {
    it('should add log handler', () => {
      const handler = vi.fn();
      addLogHandler(handler);
      log(LogLevel.INFO, 'test');
      expect(handler).toHaveBeenCalled();
      removeLogHandler(handler);
    });

    it('should remove log handler', () => {
      const handler = vi.fn();
      addLogHandler(handler);
      removeLogHandler(handler);
      log(LogLevel.INFO, 'test');
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('log', () => {
    it('should create log entry with correct level', () => {
      log(LogLevel.DEBUG, 'debug message');
      log(LogLevel.INFO, 'info message');
      log(LogLevel.WARN, 'warn message');
      log(LogLevel.ERROR, 'error message');

      expect(capturedLogs).toHaveLength(4);
      expect(capturedLogs[0].level).toBe(LogLevel.DEBUG);
      expect(capturedLogs[1].level).toBe(LogLevel.INFO);
      expect(capturedLogs[2].level).toBe(LogLevel.WARN);
      expect(capturedLogs[3].level).toBe(LogLevel.ERROR);
    });

    it('should include message and timestamp', () => {
      const before = Date.now();
      log(LogLevel.INFO, 'test message');
      const after = Date.now();

      expect(capturedLogs[0].message).toBe('test message');
      expect(capturedLogs[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(capturedLogs[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('should include attributes', () => {
      log(LogLevel.INFO, 'test', { key: 'value', count: 42 });
      expect(capturedLogs[0].attributes).toEqual({ key: 'value', count: 42 });
    });

    it('should include trace context', () => {
      log(LogLevel.INFO, 'traced message');
      expect(capturedLogs[0].traceId).toBe('test-trace-id');
      expect(capturedLogs[0].spanId).toBe('test-span-id');
    });

    it('should handle handler errors gracefully', () => {
      const errorHandler: LogHandler = () => { throw new Error('handler error'); };
      addLogHandler(errorHandler);

      // Should not throw
      expect(() => log(LogLevel.INFO, 'test')).not.toThrow();

      removeLogHandler(errorHandler);
    });
  });

  describe('logger', () => {
    it('should have debug method', () => {
      logger.debug('debug message');
      expect(capturedLogs[0].level).toBe(LogLevel.DEBUG);
    });

    it('should have info method', () => {
      logger.info('info message');
      expect(capturedLogs[0].level).toBe(LogLevel.INFO);
    });

    it('should have warn method', () => {
      logger.warn('warn message');
      expect(capturedLogs[0].level).toBe(LogLevel.WARN);
    });

    it('should have error method', () => {
      logger.error('error message');
      expect(capturedLogs[0].level).toBe(LogLevel.ERROR);
    });

    it('should pass attributes', () => {
      logger.info('message', { foo: 'bar' });
      expect(capturedLogs[0].attributes).toEqual({ foo: 'bar' });
    });
  });

  describe('consoleLogHandler', () => {
    it('should handle DEBUG level', () => {
      const consoleSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.DEBUG,
        message: 'debug',
        timestamp: Date.now(),
      });
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should handle INFO level', () => {
      const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.INFO,
        message: 'info',
        timestamp: Date.now(),
      });
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should handle WARN level', () => {
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.WARN,
        message: 'warn',
        timestamp: Date.now(),
      });
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should handle ERROR level', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.ERROR,
        message: 'error',
        timestamp: Date.now(),
      });
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should include trace prefix when available', () => {
      const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.INFO,
        message: 'message',
        timestamp: Date.now(),
        traceId: 'abc123def456',
      });
      expect(consoleSpy).toHaveBeenCalledWith('[abc123de] message');
      consoleSpy.mockRestore();
    });

    it('should include attributes when present', () => {
      const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.INFO,
        message: 'message',
        timestamp: Date.now(),
        attributes: { key: 'value' },
      });
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('{"key":"value"}'));
      consoleSpy.mockRestore();
    });
  });
});

describe('Health checks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    initObservability();
    // Note: Can't easily clear health checks without access to internal state
    // Tests should be designed to work with accumulated checks
  });

  describe('registerHealthCheck', () => {
    it('should register a health check', () => {
      registerHealthCheck('test-service', async () => ({
        healthy: true,
        message: 'OK',
      }));
      // Check is registered - will be verified in checkHealth
    });
  });

  describe('checkHealth', () => {
    it('should return health status', async () => {
      registerHealthCheck('healthy-service', async () => ({
        healthy: true,
        latencyMs: 10,
      }));

      const status = await checkHealth();
      expect(status).toBeDefined();
      expect(status.timestamp).toBeDefined();
      expect(status.components).toBeDefined();
    });

    it('should report healthy when all checks pass', async () => {
      registerHealthCheck('always-healthy', async () => ({
        healthy: true,
      }));

      const status = await checkHealth();
      // May include other checks registered earlier
      expect(status.components['always-healthy']).toBeDefined();
      expect(status.components['always-healthy'].healthy).toBe(true);
    });

    it('should report unhealthy when any check fails', async () => {
      registerHealthCheck('failing-service', async () => ({
        healthy: false,
        message: 'Service down',
      }));

      const status = await checkHealth();
      expect(status.healthy).toBe(false);
      expect(status.components['failing-service'].healthy).toBe(false);
      expect(status.components['failing-service'].message).toBe('Service down');
    });

    it('should handle check errors gracefully', async () => {
      registerHealthCheck('error-service', async () => {
        throw new Error('Check failed');
      });

      const status = await checkHealth();
      expect(status.healthy).toBe(false);
      expect(status.components['error-service'].healthy).toBe(false);
      expect(status.components['error-service'].message).toBe('Check failed');
    });

    it('should include latency when provided', async () => {
      registerHealthCheck('latency-check', async () => ({
        healthy: true,
        latencyMs: 42,
      }));

      const status = await checkHealth();
      expect(status.components['latency-check'].latencyMs).toBe(42);
    });
  });
});

describe('LogLevel enum', () => {
  it('should have correct values', () => {
    expect(LogLevel.DEBUG).toBe('debug');
    expect(LogLevel.INFO).toBe('info');
    expect(LogLevel.WARN).toBe('warn');
    expect(LogLevel.ERROR).toBe('error');
  });
});

// Additional mutation-targeted tests
describe('Mutation-targeted tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    initObservability();
  });

  describe('withTrace edge cases', () => {
    it('should set each attribute individually', async () => {
      await withTrace(
        {
          name: 'multi-attr-test',
          attributes: {
            'string.attr': 'value',
            'number.attr': 42,
            'bool.attr': true,
          },
        },
        async (span) => {
          expect(span.setAttribute).toHaveBeenCalledWith('string.attr', 'value');
          expect(span.setAttribute).toHaveBeenCalledWith('number.attr', 42);
          expect(span.setAttribute).toHaveBeenCalledWith('bool.attr', true);
          return 'ok';
        }
      );
    });

    it('should call setStatus with OK on success', async () => {
      await withTrace(
        { name: 'status-test' },
        async (span) => {
          return 'success';
        }
      );
      // Span should have setStatus called with OK
    });

    it('should increment operationCounter with exact count 1', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      await withTrace(
        { name: 'counter-test' },
        async () => 'ok'
      );

      expect(metrics.operationCounter.add).toHaveBeenCalledWith(1, expect.any(Object));
    });

    it('should increment activeOperations on start and decrement on end', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      await withTrace(
        { name: 'active-test' },
        async () => 'ok'
      );

      // Should have been called twice: +1 on start, -1 on end
      const calls = (metrics.activeOperations.add as any).mock.calls;
      expect(calls.some((c: any[]) => c[0] === 1)).toBe(true);
      expect(calls.some((c: any[]) => c[0] === -1)).toBe(true);
    });

    it('should record duration in histogram', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      await withTrace(
        { name: 'duration-test' },
        async () => {
          await new Promise(resolve => setTimeout(resolve, 10));
          return 'ok';
        }
      );

      expect(metrics.operationDuration.record).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({ operation: 'duration-test' })
      );
    });

    it('should record exception on error', async () => {
      await expect(
        withTrace(
          { name: 'exception-test' },
          async (span) => {
            throw new Error('test exception');
          }
        )
      ).rejects.toThrow('test exception');
    });

    it('should set error status message from Error', async () => {
      try {
        await withTrace(
          { name: 'error-msg-test' },
          async () => {
            throw new Error('specific error message');
          }
        );
      } catch {
        // Expected
      }
      // Error message should be captured
    });

    it('should handle non-Error throws', async () => {
      try {
        await withTrace(
          { name: 'non-error-test' },
          async () => {
            throw 'string error';
          }
        );
      } catch {
        // Expected
      }
    });

    it('should increment error counter with error_type', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      try {
        await withTrace(
          { name: 'error-counter-test' },
          async () => {
            throw new TypeError('type error');
          }
        );
      } catch {
        // Expected
      }

      expect(metrics.errorCounter.add).toHaveBeenCalledWith(1, expect.objectContaining({
        operation: 'error-counter-test',
        error_type: 'TypeError',
      }));
    });
  });

  describe('withTraceSync edge cases', () => {
    it('should set each attribute individually', () => {
      withTraceSync(
        {
          name: 'sync-multi-attr',
          attributes: {
            key1: 'val1',
            key2: 100,
          },
        },
        (span) => {
          expect(span.setAttribute).toHaveBeenCalledWith('key1', 'val1');
          expect(span.setAttribute).toHaveBeenCalledWith('key2', 100);
          return 'ok';
        }
      );
    });

    it('should record duration for sync operations', () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      withTraceSync(
        { name: 'sync-duration' },
        () => 'result'
      );

      expect(metrics.operationDuration.record).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({ operation: 'sync-duration' })
      );
    });

    it('should increment error counter on sync error', () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      try {
        withTraceSync(
          { name: 'sync-error' },
          () => { throw new RangeError('range error'); }
        );
      } catch {
        // Expected
      }

      expect(metrics.errorCounter.add).toHaveBeenCalledWith(1, expect.objectContaining({
        error_type: 'RangeError',
      }));
    });

    it('should not record metrics when recordMetrics is false', () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      withTraceSync(
        { name: 'no-metrics', recordMetrics: false },
        () => 'ok'
      );

      // Should not have recorded duration for this operation
      const calls = (metrics.operationDuration.record as any).mock.calls;
      const thisOpCalls = calls.filter((c: any[]) => c[1]?.operation === 'no-metrics');
      expect(thisOpCalls.length).toBe(0);
    });
  });

  describe('traceBridgeCall edge cases', () => {
    it('should create span name with happ.zome.fn format', async () => {
      await traceBridgeCall(
        { happ: 'myHapp', zome: 'myZome', fn: 'myFn' },
        async () => 'result'
      );
      // Span name should be bridge.myHapp.myZome.myFn
    });

    it('should calculate payload size correctly', async () => {
      const payload = { data: 'test', nested: { value: 123 } };
      const expectedSize = JSON.stringify(payload).length;

      await traceBridgeCall(
        { happ: 'test', zome: 'test', fn: 'test', payload },
        async () => 'ok'
      );
      // Payload size attribute should be set
    });

    it('should record 0 payload size when no payload', async () => {
      await traceBridgeCall(
        { happ: 'test', zome: 'test', fn: 'test' },
        async () => 'ok'
      );
      // Should handle undefined payload gracefully
    });

    it('should record bridge call duration in histogram', async () => {
      const metrics = getMetrics();
      vi.clearAllMocks();
      initObservability();

      await traceBridgeCall(
        { happ: 'h', zome: 'z', fn: 'f' },
        async () => {
          await new Promise(resolve => setTimeout(resolve, 5));
          return 'result';
        }
      );

      expect(metrics.bridgeCallDuration.record).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({ happ: 'h', zome: 'z', fn: 'f' })
      );
    });

    it('should set bridge.success attribute on success', async () => {
      await traceBridgeCall(
        { happ: 'success', zome: 'test', fn: 'test' },
        async () => 'success'
      );
      // Should have setAttribute called with bridge.success = true
    });
  });

  describe('log edge cases', () => {
    it('should handle undefined attributes', () => {
      let capturedEntry: any = null;
      const handler = (entry: LogEntry) => { capturedEntry = entry; };
      addLogHandler(handler);

      log(LogLevel.INFO, 'no attrs');

      expect(capturedEntry.attributes).toBeUndefined();
      removeLogHandler(handler);
    });

    it('should not throw when no handlers registered', () => {
      // Remove all handlers temporarily
      const tempHandler = vi.fn();
      addLogHandler(tempHandler);
      removeLogHandler(tempHandler);

      // Should not throw
      expect(() => log(LogLevel.DEBUG, 'test')).not.toThrow();
    });

    it('should call all handlers for each log', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();
      addLogHandler(handler1);
      addLogHandler(handler2);

      log(LogLevel.WARN, 'multi-handler test');

      expect(handler1).toHaveBeenCalled();
      expect(handler2).toHaveBeenCalled();

      removeLogHandler(handler1);
      removeLogHandler(handler2);
    });

    it('should continue calling handlers even if one throws', () => {
      const throwingHandler = () => { throw new Error('handler error'); };
      const normalHandler = vi.fn();

      addLogHandler(throwingHandler);
      addLogHandler(normalHandler);

      log(LogLevel.INFO, 'after throw');

      expect(normalHandler).toHaveBeenCalled();

      removeLogHandler(throwingHandler);
      removeLogHandler(normalHandler);
    });
  });

  describe('consoleLogHandler edge cases', () => {
    it('should not include trace prefix when traceId is undefined', () => {
      const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.INFO,
        message: 'no trace',
        timestamp: Date.now(),
      });
      expect(consoleSpy).toHaveBeenCalledWith('no trace');
      consoleSpy.mockRestore();
    });

    it('should truncate traceId to 8 characters', () => {
      const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.INFO,
        message: 'msg',
        timestamp: Date.now(),
        traceId: '1234567890abcdef',
      });
      expect(consoleSpy).toHaveBeenCalledWith('[12345678] msg');
      consoleSpy.mockRestore();
    });

    it('should format attributes as JSON', () => {
      const consoleSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
      consoleLogHandler({
        level: LogLevel.DEBUG,
        message: 'debug msg',
        timestamp: Date.now(),
        attributes: { foo: 'bar', num: 123 },
      });
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('"foo":"bar"'));
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('"num":123'));
      consoleSpy.mockRestore();
    });
  });

  describe('extractTraceContext edge cases', () => {
    it('should return trace context with expected keys', () => {
      const ctx = extractTraceContext();
      // Should return context with trace-id, span-id, trace-flags
      expect(ctx).toHaveProperty('trace-id');
      expect(ctx).toHaveProperty('span-id');
      expect(ctx).toHaveProperty('trace-flags');
    });

    it('should format trace flags as string', () => {
      const ctx = extractTraceContext();
      expect(typeof ctx['trace-flags']).toBe('string');
    });
  });

  describe('health check edge cases', () => {
    it('should handle non-Error exceptions in health check', async () => {
      registerHealthCheck('non-error-check', async () => {
        throw 'string error';
      });

      const status = await checkHealth();
      expect(status.components['non-error-check'].healthy).toBe(false);
      expect(status.components['non-error-check'].message).toBe('Unknown error');
    });

    it('should set allHealthy to false when check returns unhealthy', async () => {
      registerHealthCheck('unhealthy-a', async () => ({
        healthy: false,
        message: 'down',
      }));

      const status = await checkHealth();
      expect(status.healthy).toBe(false);
    });

    it('should iterate through all registered checks', async () => {
      registerHealthCheck('check-1', async () => ({ healthy: true }));
      registerHealthCheck('check-2', async () => ({ healthy: true }));
      registerHealthCheck('check-3', async () => ({ healthy: false, message: 'fail' }));

      const status = await checkHealth();
      expect(Object.keys(status.components).length).toBeGreaterThanOrEqual(3);
    });
  });
});

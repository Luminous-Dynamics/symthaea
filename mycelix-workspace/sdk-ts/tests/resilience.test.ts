/**
 * Tests for resilience patterns
 *
 * Tests circuit breaker, retry logic, timeout, rate limiter, and bulkhead patterns.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  CircuitBreaker,
  CircuitState,
  CircuitOpenError,
  withRetry,
  withTimeout,
  TimeoutError,
  RateLimiter,
  Bulkhead,
  BulkheadFullError,
  ResiliencePolicy,
  holochainPolicy,
  criticalPolicy,
  fastFailPolicy,
} from '../src/resilience';

describe('CircuitBreaker', () => {
  let cb: CircuitBreaker;

  beforeEach(() => {
    cb = new CircuitBreaker({
      failureThreshold: 3,
      recoveryTimeout: 100, // Short timeout for tests
      successThreshold: 2,
      name: 'test',
    });
  });

  describe('initial state', () => {
    it('should start in CLOSED state', () => {
      expect(cb.currentState).toBe(CircuitState.CLOSED);
    });

    it('should have zero failures and successes initially', () => {
      const stats = cb.stats;
      expect(stats.failures).toBe(0);
      expect(stats.successes).toBe(0);
      expect(stats.lastFailureTime).toBe(0);
    });
  });

  describe('execute with success', () => {
    it('should execute function and return result', async () => {
      const result = await cb.execute(async () => 'success');
      expect(result).toBe('success');
    });

    it('should stay CLOSED after successful executions', async () => {
      await cb.execute(async () => 'ok');
      await cb.execute(async () => 'ok');
      await cb.execute(async () => 'ok');
      expect(cb.currentState).toBe(CircuitState.CLOSED);
    });

    it('should reset failure count on success', async () => {
      // Cause 2 failures (below threshold)
      await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      expect(cb.stats.failures).toBe(2);

      // Success should reset failures
      await cb.execute(async () => 'ok');
      expect(cb.stats.failures).toBe(0);
    });
  });

  describe('execute with failures', () => {
    it('should open circuit after threshold failures', async () => {
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }
      expect(cb.currentState).toBe(CircuitState.OPEN);
    });

    it('should reject calls when OPEN', async () => {
      // Open the circuit
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }

      // Should reject with CircuitOpenError
      await expect(cb.execute(async () => 'ok')).rejects.toThrow(CircuitOpenError);
    });

    it('should include retry time in CircuitOpenError', async () => {
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }

      try {
        await cb.execute(async () => 'ok');
      } catch (error) {
        expect(error).toBeInstanceOf(CircuitOpenError);
        expect((error as CircuitOpenError).retryAfterMs).toBeGreaterThan(0);
        expect((error as CircuitOpenError).retryAfterMs).toBeLessThanOrEqual(100);
      }
    });
  });

  describe('recovery', () => {
    it('should transition to HALF_OPEN after recovery timeout', async () => {
      // Open the circuit
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }
      expect(cb.currentState).toBe(CircuitState.OPEN);

      // Wait for recovery timeout
      await new Promise(resolve => setTimeout(resolve, 110));

      // Next call should transition to HALF_OPEN
      await cb.execute(async () => 'ok');
      expect(cb.currentState).toBe(CircuitState.HALF_OPEN);
    });

    it('should close circuit after success threshold in HALF_OPEN', async () => {
      // Open the circuit
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }

      // Wait for recovery timeout
      await new Promise(resolve => setTimeout(resolve, 110));

      // Success calls to close
      await cb.execute(async () => 'ok');
      expect(cb.currentState).toBe(CircuitState.HALF_OPEN);
      await cb.execute(async () => 'ok');
      expect(cb.currentState).toBe(CircuitState.CLOSED);
    });

    it('should reopen on failure during HALF_OPEN', async () => {
      // Create a circuit breaker with failureThreshold=1 for easier testing
      const cbTest = new CircuitBreaker({
        failureThreshold: 1,
        recoveryTimeout: 100,
        successThreshold: 2,
        name: 'test-reopen',
      });

      // Open the circuit with 1 failure
      await expect(cbTest.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      expect(cbTest.currentState).toBe(CircuitState.OPEN);

      // Wait for recovery timeout
      await new Promise(resolve => setTimeout(resolve, 110));

      // First execution transitions to HALF_OPEN and succeeds
      await cbTest.execute(async () => 'ok');
      expect(cbTest.currentState).toBe(CircuitState.HALF_OPEN);

      // Failure reopens circuit (because we're in HALF_OPEN and failure resets failures to threshold)
      await expect(cbTest.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      // After failure in HALF_OPEN, failures >= threshold (1), so should be OPEN
      expect(cbTest.currentState).toBe(CircuitState.OPEN);
    });
  });

  describe('reset', () => {
    it('should reset all state', async () => {
      // Open the circuit
      for (let i = 0; i < 3; i++) {
        await expect(cb.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      }
      expect(cb.currentState).toBe(CircuitState.OPEN);

      cb.reset();

      expect(cb.currentState).toBe(CircuitState.CLOSED);
      expect(cb.stats.failures).toBe(0);
      expect(cb.stats.successes).toBe(0);
      expect(cb.stats.lastFailureTime).toBe(0);
    });
  });

  describe('default options', () => {
    it('should use defaults when no options provided', () => {
      const defaultCb = new CircuitBreaker();
      expect(defaultCb.currentState).toBe(CircuitState.CLOSED);
    });
  });
});

describe('CircuitOpenError', () => {
  it('should have correct name and properties', () => {
    const error = new CircuitOpenError('test message', 5000);
    expect(error.name).toBe('CircuitOpenError');
    expect(error.message).toBe('test message');
    expect(error.retryAfterMs).toBe(5000);
  });
});

describe('withRetry', () => {
  it('should return result on first success', async () => {
    const fn = vi.fn().mockResolvedValue('success');
    const result = await withRetry(fn);
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on failure', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValue('success');

    const result = await withRetry(fn, { initialDelay: 10, jitter: false });
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should throw after max retries', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('always fails'));

    await expect(withRetry(fn, { maxRetries: 2, initialDelay: 10, jitter: false }))
      .rejects.toThrow('always fails');
    expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
  });

  it('should not retry if isRetryable returns false', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('non-retryable'));

    await expect(withRetry(fn, {
      maxRetries: 3,
      initialDelay: 10,
      isRetryable: () => false,
    })).rejects.toThrow('non-retryable');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should call onRetry callback', async () => {
    const onRetry = vi.fn();
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    await withRetry(fn, { initialDelay: 10, jitter: false, onRetry });
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry).toHaveBeenCalledWith(expect.any(Error), 1, 10);
  });

  it('should apply exponential backoff', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValue('success');

    const delays: number[] = [];
    const onRetry = (_: unknown, __: number, delay: number) => delays.push(delay);

    await withRetry(fn, {
      initialDelay: 100,
      backoffFactor: 2,
      jitter: false,
      onRetry,
    });

    expect(delays[0]).toBe(100);
    expect(delays[1]).toBe(200);
  });

  it('should respect maxDelay', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValue('success');

    const delays: number[] = [];
    const onRetry = (_: unknown, __: number, delay: number) => delays.push(delay);

    await withRetry(fn, {
      initialDelay: 100,
      maxDelay: 150,
      backoffFactor: 2,
      jitter: false,
      onRetry,
    });

    expect(delays[0]).toBe(100);
    expect(delays[1]).toBe(150); // Capped at maxDelay
  });
});

describe('withTimeout', () => {
  it('should return result if function completes in time', async () => {
    const result = await withTimeout(
      async () => 'success',
      1000
    );
    expect(result).toBe('success');
  });

  it('should throw TimeoutError if function takes too long', async () => {
    await expect(withTimeout(
      async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        return 'too slow';
      },
      10
    )).rejects.toThrow(TimeoutError);
  });

  it('should include timeout duration in TimeoutError', async () => {
    try {
      await withTimeout(
        async () => {
          await new Promise(resolve => setTimeout(resolve, 100));
        },
        10,
        'Custom timeout message'
      );
    } catch (error) {
      expect(error).toBeInstanceOf(TimeoutError);
      expect((error as TimeoutError).timeoutMs).toBe(10);
      expect((error as TimeoutError).message).toBe('Custom timeout message');
    }
  });

  it('should use default message if not provided', async () => {
    try {
      await withTimeout(
        async () => {
          await new Promise(resolve => setTimeout(resolve, 100));
        },
        10
      );
    } catch (error) {
      expect((error as TimeoutError).message).toContain('10ms');
    }
  });

  it('should propagate errors from the function', async () => {
    await expect(withTimeout(
      async () => { throw new Error('inner error'); },
      1000
    )).rejects.toThrow('inner error');
  });
});

describe('TimeoutError', () => {
  it('should have correct name and properties', () => {
    const error = new TimeoutError('test timeout', 5000);
    expect(error.name).toBe('TimeoutError');
    expect(error.message).toBe('test timeout');
    expect(error.timeoutMs).toBe(5000);
  });
});

describe('RateLimiter', () => {
  let rl: RateLimiter;

  beforeEach(() => {
    rl = new RateLimiter({
      maxTokens: 5,
      refillRate: 2,
      refillInterval: 100,
    });
  });

  describe('tryAcquire', () => {
    it('should acquire tokens when available', () => {
      expect(rl.tryAcquire(1)).toBe(true);
      expect(rl.tryAcquire(1)).toBe(true);
      expect(rl.availableTokens).toBe(3);
    });

    it('should fail when not enough tokens', () => {
      expect(rl.tryAcquire(6)).toBe(false);
      expect(rl.availableTokens).toBe(5); // No tokens consumed
    });

    it('should acquire multiple tokens', () => {
      expect(rl.tryAcquire(3)).toBe(true);
      expect(rl.availableTokens).toBe(2);
    });

    it('should refill tokens over time', async () => {
      // Use all tokens
      expect(rl.tryAcquire(5)).toBe(true);
      expect(rl.availableTokens).toBe(0);

      // Wait for refill
      await new Promise(resolve => setTimeout(resolve, 110));

      // Should have 2 tokens now (refillRate = 2)
      expect(rl.availableTokens).toBeGreaterThanOrEqual(2);
    });

    it('should not exceed maxTokens on refill', async () => {
      // Start with 5 tokens
      expect(rl.availableTokens).toBe(5);

      // Wait for refill (should not exceed max)
      await new Promise(resolve => setTimeout(resolve, 300));

      expect(rl.availableTokens).toBe(5); // Still at max
    });
  });

  describe('acquire', () => {
    it('should acquire immediately when tokens available', async () => {
      await rl.acquire(1);
      expect(rl.availableTokens).toBe(4);
    });

    it('should wait for refill when no tokens', async () => {
      // Use all tokens
      rl.tryAcquire(5);
      expect(rl.availableTokens).toBe(0);

      // acquire should wait for refill
      const startTime = Date.now();
      await rl.acquire(1);
      const elapsed = Date.now() - startTime;

      expect(elapsed).toBeGreaterThanOrEqual(90); // At least one refill interval
    });
  });

  describe('execute', () => {
    it('should execute function after acquiring tokens', async () => {
      const result = await rl.execute(async () => 'done');
      expect(result).toBe('done');
    });

    it('should respect token cost', async () => {
      const result = await rl.execute(async () => 'done', 3);
      expect(result).toBe('done');
      expect(rl.availableTokens).toBe(2);
    });
  });

  describe('default options', () => {
    it('should use defaults when no options provided', () => {
      const defaultRl = new RateLimiter();
      expect(defaultRl.availableTokens).toBe(100); // Default maxTokens
    });
  });
});

describe('Bulkhead', () => {
  let bh: Bulkhead;

  beforeEach(() => {
    bh = new Bulkhead({
      maxConcurrent: 2,
      maxQueue: 2,
    });
  });

  describe('execute', () => {
    it('should execute function and return result', async () => {
      const result = await bh.execute(async () => 'success');
      expect(result).toBe('success');
    });

    it('should allow concurrent executions up to limit', async () => {
      const results = await Promise.all([
        bh.execute(async () => { await delay(10); return 1; }),
        bh.execute(async () => { await delay(10); return 2; }),
      ]);
      expect(results).toEqual([1, 2]);
    });

    it('should queue executions beyond concurrent limit', async () => {
      const order: number[] = [];

      const promises = [
        bh.execute(async () => { await delay(50); order.push(1); return 1; }),
        bh.execute(async () => { await delay(50); order.push(2); return 2; }),
        bh.execute(async () => { order.push(3); return 3; }), // Queued
      ];

      await Promise.all(promises);

      // Third execution should complete after first two
      expect(order.length).toBe(3);
    });

    it('should reject when queue is full', async () => {
      const bh2 = new Bulkhead({ maxConcurrent: 1, maxQueue: 1 });

      // Start one running
      const p1 = bh2.execute(async () => { await delay(100); return 1; });
      // Queue one
      const p2 = bh2.execute(async () => 2);
      // This should be rejected
      const p3 = bh2.execute(async () => 3);

      await expect(p3).rejects.toThrow(BulkheadFullError);

      // Clean up
      await p1;
      await p2;
    });
  });

  describe('stats', () => {
    it('should report current stats', () => {
      const stats = bh.stats;
      expect(stats.running).toBe(0);
      expect(stats.queued).toBe(0);
      expect(stats.available).toBe(2);
    });
  });

  describe('default options', () => {
    it('should use defaults when no options provided', () => {
      const defaultBh = new Bulkhead();
      expect(defaultBh.stats.available).toBe(10); // Default maxConcurrent
    });
  });
});

describe('BulkheadFullError', () => {
  it('should have correct name and properties', () => {
    const error = new BulkheadFullError('queue full', 5, 10);
    expect(error.name).toBe('BulkheadFullError');
    expect(error.message).toBe('queue full');
    expect(error.running).toBe(5);
    expect(error.queued).toBe(10);
  });
});

describe('ResiliencePolicy', () => {
  describe('with no options', () => {
    it('should execute function directly', async () => {
      const policy = new ResiliencePolicy();
      const result = await policy.execute(async () => 'success');
      expect(result).toBe('success');
    });
  });

  describe('with timeout', () => {
    it('should apply timeout', async () => {
      const policy = new ResiliencePolicy({ timeout: 10 });
      await expect(policy.execute(async () => {
        await delay(100);
        return 'too slow';
      })).rejects.toThrow(TimeoutError);
    });
  });

  describe('with retry', () => {
    it('should apply retry logic', async () => {
      const policy = new ResiliencePolicy({
        retry: { maxRetries: 2, initialDelay: 10, jitter: false },
      });

      let attempts = 0;
      const result = await policy.execute(async () => {
        attempts++;
        if (attempts < 3) throw new Error('fail');
        return 'success';
      });

      expect(result).toBe('success');
      expect(attempts).toBe(3);
    });
  });

  describe('with circuit breaker', () => {
    it('should apply circuit breaker', async () => {
      const policy = new ResiliencePolicy({
        circuitBreaker: { failureThreshold: 2, recoveryTimeout: 100 },
      });

      // Cause failures to open circuit
      await expect(policy.execute(async () => { throw new Error('fail'); })).rejects.toThrow();
      await expect(policy.execute(async () => { throw new Error('fail'); })).rejects.toThrow();

      // Should be open
      await expect(policy.execute(async () => 'ok')).rejects.toThrow(CircuitOpenError);
    });
  });

  describe('with rate limiter', () => {
    it('should apply rate limiting', async () => {
      const policy = new ResiliencePolicy({
        rateLimiter: { maxTokens: 2, refillRate: 1, refillInterval: 100 },
      });

      // Use all tokens
      await policy.execute(async () => 'ok');
      await policy.execute(async () => 'ok');

      // Next execution should wait for refill
      const startTime = Date.now();
      await policy.execute(async () => 'ok');
      const elapsed = Date.now() - startTime;

      expect(elapsed).toBeGreaterThanOrEqual(90);
    });
  });

  describe('with bulkhead', () => {
    it('should apply concurrency limiting', async () => {
      const policy = new ResiliencePolicy({
        bulkhead: { maxConcurrent: 2, maxQueue: 0 },
      });

      // Start two concurrent
      const p1 = policy.execute(async () => { await delay(100); return 1; });
      const p2 = policy.execute(async () => { await delay(100); return 2; });

      // Third should fail immediately (no queue)
      await expect(policy.execute(async () => 3)).rejects.toThrow(BulkheadFullError);

      await Promise.all([p1, p2]);
    });
  });

  describe('stats', () => {
    it('should return stats for all components', () => {
      const policy = new ResiliencePolicy({
        circuitBreaker: { failureThreshold: 5 },
        rateLimiter: { maxTokens: 10 },
        bulkhead: { maxConcurrent: 5 },
      });

      const stats = policy.stats;
      expect(stats.circuitBreaker).toBeDefined();
      expect(stats.rateLimiter).toBeDefined();
      expect(stats.bulkhead).toBeDefined();
    });

    it('should return undefined for unconfigured components', () => {
      const policy = new ResiliencePolicy();
      const stats = policy.stats;
      expect(stats.circuitBreaker).toBeUndefined();
      expect(stats.rateLimiter).toBeUndefined();
      expect(stats.bulkhead).toBeUndefined();
    });
  });

  describe('combined policies', () => {
    it('should apply all policies in correct order', async () => {
      const policy = new ResiliencePolicy({
        timeout: 1000,
        retry: { maxRetries: 1, initialDelay: 10, jitter: false },
        circuitBreaker: { failureThreshold: 5, recoveryTimeout: 100 },
        rateLimiter: { maxTokens: 10, refillRate: 1, refillInterval: 100 },
        bulkhead: { maxConcurrent: 5, maxQueue: 10 },
      });

      const result = await policy.execute(async () => 'success');
      expect(result).toBe('success');
    });
  });
});

describe('Pre-configured policies', () => {
  it('should export holochainPolicy', () => {
    expect(holochainPolicy).toBeInstanceOf(ResiliencePolicy);
  });

  it('should export criticalPolicy', () => {
    expect(criticalPolicy).toBeInstanceOf(ResiliencePolicy);
  });

  it('should export fastFailPolicy', () => {
    expect(fastFailPolicy).toBeInstanceOf(ResiliencePolicy);
  });

  it('holochainPolicy should execute successfully', async () => {
    const result = await holochainPolicy.execute(async () => 'ok');
    expect(result).toBe('ok');
  });

  it('criticalPolicy should execute successfully', async () => {
    const result = await criticalPolicy.execute(async () => 'ok');
    expect(result).toBe('ok');
  });

  it('fastFailPolicy should execute successfully', async () => {
    const result = await fastFailPolicy.execute(async () => 'ok');
    expect(result).toBe('ok');
  });
});

// Helper function
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

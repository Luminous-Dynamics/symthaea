/**
 * Resilience Utilities Tests
 *
 * Tests for error handling, retry logic, circuit breaker,
 * and fallback mechanisms.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  withRetry,
  withFallback,
  withTimeout,
  withResilience,
  createCircuitBreaker,
  CircuitOpenError,
  TimeoutError,
  classifyError,
  isRetryableError,
  processBatched,
  debounce,
  throttle,
} from '../utils/resilience';

describe('withRetry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should succeed on first attempt', async () => {
    const fn = vi.fn().mockResolvedValue('success');
    const result = await withRetry(fn);
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on failure and succeed', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail 1'))
      .mockRejectedValueOnce(new Error('fail 2'))
      .mockResolvedValue('success');

    const promise = withRetry(fn, { maxAttempts: 3, initialDelay: 100 });

    // Fast-forward through delays - need to advance enough for all retries
    await vi.advanceTimersByTimeAsync(300);

    const result = await promise;
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should throw after max attempts', async () => {
    // Use mockImplementation to create fresh rejections that get properly handled
    const fn = vi.fn().mockImplementation(() => Promise.reject(new Error('always fails')));

    let caughtError: Error | null = null;
    const runTest = async () => {
      try {
        await withRetry(fn, { maxAttempts: 3, initialDelay: 100 });
      } catch (error) {
        caughtError = error as Error;
      }
    };

    const testPromise = runTest();

    // Fast-forward through all delays
    await vi.advanceTimersByTimeAsync(500);
    await testPromise;

    expect(caughtError).not.toBeNull();
    expect((caughtError as unknown as Error).message).toBe('always fails');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('should respect shouldRetry predicate', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('non-retryable'));

    await expect(
      withRetry(fn, {
        maxAttempts: 3,
        shouldRetry: () => false,
      })
    ).rejects.toThrow('non-retryable');

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should call onRetry callback', async () => {
    const onRetry = vi.fn();
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    const promise = withRetry(fn, {
      maxAttempts: 2,
      initialDelay: 100,
      onRetry,
    });

    await vi.advanceTimersByTimeAsync(200);
    await promise;

    expect(onRetry).toHaveBeenCalledWith(1, expect.any(Error), 100);
  });

  it('should apply exponential backoff', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('fail 1'))
      .mockRejectedValueOnce(new Error('fail 2'))
      .mockResolvedValue('success');

    const onRetry = vi.fn();
    const promise = withRetry(fn, {
      maxAttempts: 3,
      initialDelay: 100,
      backoffMultiplier: 2,
      onRetry,
    });

    // Advance enough time to cover all retries (100 + 200 = 300ms)
    await vi.advanceTimersByTimeAsync(400);
    await promise;

    expect(onRetry).toHaveBeenNthCalledWith(1, 1, expect.any(Error), 100);
    expect(onRetry).toHaveBeenNthCalledWith(2, 2, expect.any(Error), 200);
  });
});

describe('withFallback', () => {
  it('should return result on success', async () => {
    const fn = vi.fn().mockResolvedValue('primary');
    const result = await withFallback(fn, { fallback: 'backup' });
    expect(result).toBe('primary');
  });

  it('should return fallback value on failure', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('fail'));
    const result = await withFallback(fn, { fallback: 'backup' });
    expect(result).toBe('backup');
  });

  it('should call fallback function on failure', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('fail'));
    const fallbackFn = vi.fn().mockReturnValue('dynamic backup');

    const result = await withFallback(fn, { fallback: fallbackFn });
    expect(result).toBe('dynamic backup');
    expect(fallbackFn).toHaveBeenCalled();
  });

  it('should support async fallback function', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('fail'));
    const fallbackFn = vi.fn().mockResolvedValue('async backup');

    const result = await withFallback(fn, { fallback: fallbackFn });
    expect(result).toBe('async backup');
  });
});

describe('withTimeout', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should resolve before timeout', async () => {
    const fn = vi.fn().mockResolvedValue('fast');
    const promise = withTimeout(fn, { timeout: 1000 });

    const result = await promise;
    expect(result).toBe('fast');
  });

  it('should throw TimeoutError on timeout', async () => {
    const fn = vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve('slow'), 2000))
    );

    const promise = withTimeout(fn, { timeout: 1000 });
    vi.advanceTimersByTime(1001);

    await expect(promise).rejects.toThrow(TimeoutError);
  });

  it('should use custom timeout message', async () => {
    const fn = vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve('slow'), 2000))
    );

    const promise = withTimeout(fn, {
      timeout: 1000,
      message: 'Custom timeout message',
    });
    vi.advanceTimersByTime(1001);

    await expect(promise).rejects.toThrow('Custom timeout message');
  });
});

describe('CircuitBreaker', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should allow requests when closed', async () => {
    const breaker = createCircuitBreaker({ failureThreshold: 3 });
    const fn = vi.fn().mockResolvedValue('success');

    const result = await breaker.execute(fn);
    expect(result).toBe('success');
    expect(breaker.getState()).toBe('closed');
  });

  it('should open after failure threshold', async () => {
    const onOpen = vi.fn();
    const breaker = createCircuitBreaker({ failureThreshold: 3, onOpen });
    const fn = vi.fn().mockRejectedValue(new Error('fail'));

    // Fail 3 times to trigger open
    for (let i = 0; i < 3; i++) {
      try {
        await breaker.execute(fn);
      } catch {
        // Expected
      }
    }

    expect(breaker.getState()).toBe('open');
    expect(onOpen).toHaveBeenCalled();
  });

  it('should reject immediately when open', async () => {
    const breaker = createCircuitBreaker({ failureThreshold: 1 });
    const fn = vi.fn().mockRejectedValue(new Error('fail'));

    try {
      await breaker.execute(fn);
    } catch {
      // Open the circuit
    }

    await expect(breaker.execute(fn)).rejects.toThrow(CircuitOpenError);
    expect(fn).toHaveBeenCalledTimes(1); // Not called again
  });

  it('should transition to half-open after recovery timeout', async () => {
    const breaker = createCircuitBreaker({
      failureThreshold: 1,
      recoveryTimeout: 5000,
    });

    const failingFn = vi.fn().mockRejectedValue(new Error('fail'));
    try {
      await breaker.execute(failingFn);
    } catch {
      // Open circuit
    }

    expect(breaker.getState()).toBe('open');

    // Advance time past recovery timeout
    vi.advanceTimersByTime(5001);

    // Try again - should be half-open now
    const successFn = vi.fn().mockResolvedValue('recovered');
    const result = await breaker.execute(successFn);

    expect(result).toBe('recovered');
    expect(breaker.getState()).toBe('closed');
  });

  it('should reset properly', () => {
    const breaker = createCircuitBreaker({ failureThreshold: 1 });
    const fn = vi.fn().mockRejectedValue(new Error('fail'));

    // Open circuit
    breaker.execute(fn).catch(() => {});

    breaker.reset();
    expect(breaker.getState()).toBe('closed');
  });
});

describe('classifyError', () => {
  it('should classify TimeoutError', () => {
    const error = new TimeoutError('timed out');
    expect(classifyError(error)).toBe('timeout');
  });

  it('should classify network errors', () => {
    const error = new TypeError('Failed to fetch');
    expect(classifyError(error)).toBe('network');
  });

  it('should classify validation errors', () => {
    const error = new Error('Validation failed: invalid input');
    expect(classifyError(error)).toBe('validation');
  });

  it('should classify auth errors', () => {
    const error = new Error('Unauthorized access');
    expect(classifyError(error)).toBe('authentication');
  });

  it('should return unknown for unrecognized errors', () => {
    const error = new Error('Something weird happened');
    expect(classifyError(error)).toBe('unknown');
  });
});

describe('isRetryableError', () => {
  it('should return true for network errors', () => {
    const error = new TypeError('Failed to fetch');
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return true for timeout errors', () => {
    const error = new TimeoutError('timed out');
    expect(isRetryableError(error)).toBe(true);
  });

  it('should return false for validation errors', () => {
    const error = new Error('Validation failed');
    expect(isRetryableError(error)).toBe(false);
  });
});

describe('processBatched', () => {
  it('should process all items', async () => {
    const items = [1, 2, 3, 4, 5];
    const processor = vi.fn().mockImplementation((n) => Promise.resolve(n * 2));

    const { results, errors } = await processBatched(items, processor, {
      batchSize: 2,
      batchDelay: 0,
    });

    expect(results).toEqual([2, 4, 6, 8, 10]);
    expect(errors).toHaveLength(0);
    expect(processor).toHaveBeenCalledTimes(5);
  });

  it('should continue on error when configured', async () => {
    const items = [1, 2, 3];
    const processor = vi.fn()
      .mockResolvedValueOnce(1)
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValueOnce(3);

    const { results, errors } = await processBatched(items, processor, {
      batchSize: 3,
      continueOnError: true,
    });

    expect(results).toEqual([1, 3]);
    expect(errors).toHaveLength(1);
    expect(errors[0].item).toBe(2);
  });

  it('should stop on error when configured', async () => {
    const items = [1, 2, 3];
    const processor = vi.fn()
      .mockResolvedValueOnce(1)
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValueOnce(3);

    await expect(
      processBatched(items, processor, { batchSize: 3, continueOnError: false })
    ).rejects.toThrow('fail');
  });
});

describe('debounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should debounce calls', () => {
    const fn = vi.fn();
    const debounced = debounce(fn, 100);

    debounced('a');
    debounced('b');
    debounced('c');

    expect(fn).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);

    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('c');
  });
});

describe('throttle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should throttle calls', () => {
    const fn = vi.fn();
    const throttled = throttle(fn, 100);

    throttled('a');
    throttled('b');
    throttled('c');

    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('a');

    vi.advanceTimersByTime(100);
    throttled('d');

    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenLastCalledWith('d');
  });
});

describe('withResilience', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should combine retry and fallback', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('fail'));

    // Start the resilience operation
    const promise = withResilience(fn, {
      maxAttempts: 2,
      initialDelay: 100,
      fallback: 'backup',
      timeout: 5000,
    });

    // Advance time to complete all retries
    await vi.advanceTimersByTimeAsync(500);

    const result = await promise;
    expect(result).toBe('backup');
  });
});

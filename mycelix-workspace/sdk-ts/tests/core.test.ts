/**
 * Core Module Tests
 *
 * Tests for retry logic, error handling, and error classification.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  SdkError,
  SdkErrorCode,
  NetworkError,
  ValidationError,
  NotFoundError,
  TimeoutError,
  UnauthorizedError,
  ZomeCallError,
  isRetryableCode,
  wrapError,
  withRetry,
  tryWithRetry,
  withTimeout,
  withRetryAndTimeout,
  createRetryWrapper,
  RetryPolicies,
  DEFAULT_RETRY_CONFIG,
} from '../src/core/index.js';

// =============================================================================
// SdkError Tests
// =============================================================================

describe('SdkError', () => {
  it('should create error with code and message', () => {
    const error = new SdkError(SdkErrorCode.VALIDATION_ERROR, 'Bad input');

    expect(error.code).toBe(SdkErrorCode.VALIDATION_ERROR);
    expect(error.message).toBe('Bad input');
    expect(error.name).toBe('SdkError');
    expect(error.timestamp).toBeLessThanOrEqual(Date.now());
    expect(error instanceof Error).toBe(true);
  });

  it('should store details', () => {
    const error = new SdkError(SdkErrorCode.ZOME_CALL_FAILED, 'Call failed', {
      zomeName: 'identity',
      fnName: 'get_did',
    });

    expect(error.details.zomeName).toBe('identity');
    expect(error.details.fnName).toBe('get_did');
  });

  it('should preserve cause stack trace', () => {
    const cause = new Error('Original error');
    const error = new SdkError(SdkErrorCode.INTERNAL_ERROR, 'Wrapped', { cause });

    expect(error.causeStack).toBeDefined();
    expect(error.causeStack).toContain('Original error');
  });

  it('should check isRetryable', () => {
    const networkErr = new SdkError(SdkErrorCode.NETWORK_ERROR, 'Network failed');
    const validationErr = new SdkError(SdkErrorCode.VALIDATION_ERROR, 'Bad input');

    expect(networkErr.isRetryable()).toBe(true);
    expect(validationErr.isRetryable()).toBe(false);
  });

  it('should check isCode', () => {
    const error = new SdkError(SdkErrorCode.NOT_FOUND, 'Missing');

    expect(error.isCode(SdkErrorCode.NOT_FOUND)).toBe(true);
    expect(error.isCode(SdkErrorCode.TIMEOUT)).toBe(false);
  });

  it('should check isConnectionError', () => {
    const netErr = new SdkError(SdkErrorCode.NETWORK_ERROR, 'Net');
    const timeoutErr = new SdkError(SdkErrorCode.TIMEOUT, 'Timeout');
    const validErr = new SdkError(SdkErrorCode.VALIDATION_ERROR, 'Valid');

    expect(netErr.isConnectionError()).toBe(true);
    expect(timeoutErr.isConnectionError()).toBe(true);
    expect(validErr.isConnectionError()).toBe(false);
  });

  it('should check isAuthError', () => {
    const authErr = new SdkError(SdkErrorCode.UNAUTHORIZED, 'Auth');
    const forbidErr = new SdkError(SdkErrorCode.FORBIDDEN, 'Forbidden');
    const netErr = new SdkError(SdkErrorCode.NETWORK_ERROR, 'Net');

    expect(authErr.isAuthError()).toBe(true);
    expect(forbidErr.isAuthError()).toBe(true);
    expect(netErr.isAuthError()).toBe(false);
  });

  it('should convert to JSON', () => {
    const error = new SdkError(SdkErrorCode.TIMEOUT, 'Timed out', { operation: 'fetch' });
    const json = error.toJSON();

    expect(json.code).toBe('TIMEOUT');
    expect(json.message).toBe('Timed out');
    expect(json.name).toBe('SdkError');
    expect(json.timestamp).toBeDefined();
  });

  it('should format toString', () => {
    const error = new SdkError(SdkErrorCode.NOT_FOUND, 'Resource missing');

    expect(error.toString()).toBe('[NOT_FOUND] Resource missing');
  });

  it('should truncate large payloads in JSON', () => {
    const largePayload = 'x'.repeat(2000);
    const error = new SdkError(SdkErrorCode.INTERNAL_ERROR, 'Big', { payload: largePayload });
    const json = error.toJSON();

    expect((json.details as any).payload).toBe('[truncated]');
  });
});

// =============================================================================
// Specialized Error Types
// =============================================================================

describe('Specialized Error Types', () => {
  it('should create NetworkError', () => {
    const error = new NetworkError('Connection refused');
    expect(error.code).toBe(SdkErrorCode.NETWORK_ERROR);
    expect(error.name).toBe('NetworkError');
    expect(error.isRetryable()).toBe(true);
  });

  it('should create ValidationError', () => {
    const error = new ValidationError('Invalid DID format');
    expect(error.code).toBe(SdkErrorCode.VALIDATION_ERROR);
    expect(error.name).toBe('ValidationError');
    expect(error.isRetryable()).toBe(false);
  });

  it('should create NotFoundError with resource info', () => {
    const error = new NotFoundError('DID', 'did:mycelix:abc123');
    expect(error.code).toBe(SdkErrorCode.NOT_FOUND);
    expect(error.message).toContain('DID');
    expect(error.message).toContain('did:mycelix:abc123');
    expect(error.details.resourceType).toBe('DID');
    expect(error.details.resourceId).toBe('did:mycelix:abc123');
  });

  it('should create TimeoutError', () => {
    const error = new TimeoutError('fetchData', 5000);
    expect(error.code).toBe(SdkErrorCode.TIMEOUT);
    expect(error.message).toContain('fetchData');
    expect(error.message).toContain('5000');
    expect(error.isRetryable()).toBe(true);
  });

  it('should create UnauthorizedError', () => {
    const error = new UnauthorizedError('Token expired');
    expect(error.code).toBe(SdkErrorCode.UNAUTHORIZED);
    expect(error.isAuthError()).toBe(true);
  });

  it('should create ZomeCallError', () => {
    const error = new ZomeCallError('identity', 'get_did', 'Agent not found');
    expect(error.code).toBe(SdkErrorCode.ZOME_CALL_FAILED);
    expect(error.message).toContain('identity.get_did');
    expect(error.details.zomeName).toBe('identity');
    expect(error.details.fnName).toBe('get_did');
  });
});

// =============================================================================
// isRetryableCode
// =============================================================================

describe('isRetryableCode', () => {
  it('should return true for retryable codes', () => {
    expect(isRetryableCode(SdkErrorCode.NETWORK_ERROR)).toBe(true);
    expect(isRetryableCode(SdkErrorCode.CONNECTION_FAILED)).toBe(true);
    expect(isRetryableCode(SdkErrorCode.CONNECTION_CLOSED)).toBe(true);
    expect(isRetryableCode(SdkErrorCode.TIMEOUT)).toBe(true);
    expect(isRetryableCode(SdkErrorCode.ZOME_CALL_FAILED)).toBe(true);
  });

  it('should return false for non-retryable codes', () => {
    expect(isRetryableCode(SdkErrorCode.VALIDATION_ERROR)).toBe(false);
    expect(isRetryableCode(SdkErrorCode.NOT_FOUND)).toBe(false);
    expect(isRetryableCode(SdkErrorCode.UNAUTHORIZED)).toBe(false);
    expect(isRetryableCode(SdkErrorCode.FORBIDDEN)).toBe(false);
  });
});

// =============================================================================
// wrapError
// =============================================================================

describe('wrapError', () => {
  it('should pass through existing SdkError', () => {
    const original = new SdkError(SdkErrorCode.TIMEOUT, 'Timeout');
    const wrapped = wrapError(original);
    expect(wrapped).toBe(original);
  });

  it('should wrap regular Error', () => {
    const error = new Error('Something went wrong');
    const wrapped = wrapError(error);

    expect(wrapped).toBeInstanceOf(SdkError);
    expect(wrapped.message).toBe('Something went wrong');
    expect(wrapped.details.cause).toBe(error);
  });

  it('should detect timeout from message', () => {
    const error = new Error('Request timed out');
    const wrapped = wrapError(error);
    expect(wrapped.code).toBe(SdkErrorCode.TIMEOUT);
  });

  it('should detect network error from message', () => {
    const error = new Error('ECONNREFUSED');
    const wrapped = wrapError(error);
    expect(wrapped.code).toBe(SdkErrorCode.NETWORK_ERROR);
  });

  it('should detect not found from message', () => {
    const error = new Error('Resource not found');
    const wrapped = wrapError(error);
    expect(wrapped.code).toBe(SdkErrorCode.NOT_FOUND);
  });

  it('should wrap non-Error values', () => {
    const wrapped = wrapError('string error');
    expect(wrapped).toBeInstanceOf(SdkError);
    expect(wrapped.message).toBe('string error');
  });

  it('should use provided code override', () => {
    const error = new Error('Something');
    const wrapped = wrapError(error, { code: SdkErrorCode.CONFLICT });
    expect(wrapped.code).toBe(SdkErrorCode.CONFLICT);
  });

  it('should use provided message override', () => {
    const error = new Error('Original message');
    const wrapped = wrapError(error, { message: 'Custom message' });
    expect(wrapped.message).toBe('Custom message');
  });
});

// =============================================================================
// withRetry
// =============================================================================

describe('withRetry', () => {
  it('should return result on first success', async () => {
    const fn = vi.fn().mockResolvedValue('success');
    const result = await withRetry(fn, { maxRetries: 3, baseDelay: 10 });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should retry on retryable error', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('network error'))
      .mockResolvedValue('success');

    const result = await withRetry(fn, { maxRetries: 3, baseDelay: 10, jitter: false });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('should throw after max retries', async () => {
    const fn = vi.fn().mockRejectedValue(new Error('network error'));

    await expect(
      withRetry(fn, { maxRetries: 2, baseDelay: 10, jitter: false })
    ).rejects.toThrow();

    expect(fn).toHaveBeenCalledTimes(3); // 1 initial + 2 retries
  });

  it('should not retry non-retryable errors', async () => {
    const fn = vi.fn().mockRejectedValue(new ValidationError('Bad input'));

    await expect(
      withRetry(fn, { maxRetries: 3, baseDelay: 10 })
    ).rejects.toThrow();

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should call onRetry callback', async () => {
    const onRetry = vi.fn();
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('connection failed'))
      .mockResolvedValue('ok');

    await withRetry(fn, { maxRetries: 3, baseDelay: 10, jitter: false, onRetry });

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry).toHaveBeenCalledWith(1, expect.any(Error), expect.any(Number));
  });

  it('should support custom isRetryable', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('custom retryable'))
      .mockResolvedValue('ok');

    const result = await withRetry(fn, {
      maxRetries: 3,
      baseDelay: 10,
      jitter: false,
      isRetryable: (err) => err instanceof Error && err.message.includes('custom'),
    });

    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(2);
  });
});

// =============================================================================
// tryWithRetry
// =============================================================================

describe('tryWithRetry', () => {
  it('should return success result', async () => {
    const fn = vi.fn().mockResolvedValue(42);
    const result = await tryWithRetry(fn, { maxRetries: 1, baseDelay: 10 });

    expect(result.success).toBe(true);
    expect(result.value).toBe(42);
    expect(result.attempts).toBe(1);
    expect(result.totalTimeMs).toBeGreaterThanOrEqual(0);
  });

  it('should return failure result without throwing', async () => {
    const fn = vi.fn().mockRejectedValue(new ValidationError('Bad'));
    const result = await tryWithRetry(fn, { maxRetries: 1, baseDelay: 10 });

    expect(result.success).toBe(false);
    expect(result.error).toBeInstanceOf(SdkError);
    expect(result.value).toBeUndefined();
  });

  it('should track attempts count', async () => {
    const fn = vi.fn()
      .mockRejectedValueOnce(new Error('network fail'))
      .mockRejectedValueOnce(new Error('network fail'))
      .mockResolvedValue('ok');

    const result = await tryWithRetry(fn, { maxRetries: 3, baseDelay: 10, jitter: false });

    expect(result.success).toBe(true);
    expect(result.attempts).toBe(3);
  });
});

// =============================================================================
// withTimeout
// =============================================================================

describe('withTimeout', () => {
  it('should return result before timeout', async () => {
    const fn = vi.fn().mockResolvedValue('fast');
    const result = await withTimeout(fn, 1000, 'fastOp');

    expect(result).toBe('fast');
  });

  it('should throw TimeoutError when exceeded', async () => {
    const fn = () => new Promise((resolve) => setTimeout(resolve, 500));

    await expect(
      withTimeout(fn, 50, 'slowOp')
    ).rejects.toThrow(TimeoutError);
  });
});

// =============================================================================
// withRetryAndTimeout
// =============================================================================

describe('withRetryAndTimeout', () => {
  it('should succeed within retry+timeout', async () => {
    const fn = vi.fn().mockResolvedValue('ok');
    const result = await withRetryAndTimeout(fn, { maxRetries: 2, baseDelay: 10 }, 1000);

    expect(result).toBe('ok');
  });

  it('should retry on timeout', async () => {
    let callCount = 0;
    const fn = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return new Promise((resolve) => setTimeout(resolve, 500));
      }
      return Promise.resolve('ok');
    });

    const result = await withRetryAndTimeout(
      fn,
      { maxRetries: 2, baseDelay: 10, jitter: false },
      50
    );

    expect(result).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(2);
  });
});

// =============================================================================
// createRetryWrapper
// =============================================================================

describe('createRetryWrapper', () => {
  it('should create wrapped function', async () => {
    const original = vi.fn().mockResolvedValue('result');
    const wrapped = createRetryWrapper(original, { maxRetries: 2, baseDelay: 10 });

    const result = await wrapped('arg1', 'arg2');

    expect(result).toBe('result');
    expect(original).toHaveBeenCalledWith('arg1', 'arg2');
  });

  it('should retry wrapped function', async () => {
    const original = vi.fn()
      .mockRejectedValueOnce(new Error('network error'))
      .mockResolvedValue('ok');

    const wrapped = createRetryWrapper(original, { maxRetries: 2, baseDelay: 10, jitter: false });
    const result = await wrapped();

    expect(result).toBe('ok');
    expect(original).toHaveBeenCalledTimes(2);
  });
});

// =============================================================================
// RetryPolicies
// =============================================================================

describe('RetryPolicies', () => {
  it('should have none policy with 0 retries', () => {
    const policy = RetryPolicies.none();
    expect(policy.maxRetries).toBe(0);
  });

  it('should have light policy', () => {
    const policy = RetryPolicies.light();
    expect(policy.maxRetries).toBe(2);
    expect(policy.baseDelay).toBe(500);
  });

  it('should have standard policy', () => {
    const policy = RetryPolicies.standard();
    expect(policy.maxRetries).toBe(3);
  });

  it('should have aggressive policy', () => {
    const policy = RetryPolicies.aggressive();
    expect(policy.maxRetries).toBe(5);
  });

  it('should have network policy with jitter', () => {
    const policy = RetryPolicies.network();
    expect(policy.maxRetries).toBe(4);
    expect(policy.jitter).toBe(true);
  });
});

// =============================================================================
// DEFAULT_RETRY_CONFIG
// =============================================================================

describe('DEFAULT_RETRY_CONFIG', () => {
  it('should have sensible defaults', () => {
    expect(DEFAULT_RETRY_CONFIG.maxRetries).toBe(3);
    expect(DEFAULT_RETRY_CONFIG.baseDelay).toBe(1000);
    expect(DEFAULT_RETRY_CONFIG.maxDelay).toBe(30000);
    expect(DEFAULT_RETRY_CONFIG.backoffMultiplier).toBe(2);
    expect(DEFAULT_RETRY_CONFIG.jitter).toBe(true);
  });
});

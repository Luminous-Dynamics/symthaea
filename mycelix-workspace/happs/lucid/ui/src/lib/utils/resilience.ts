// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Resilience Utilities
 *
 * Provides error handling, retry logic, and graceful degradation patterns
 * for robust service operations.
 */

// ============================================================================
// TYPES
// ============================================================================

export interface RetryOptions {
  /** Maximum number of retry attempts */
  maxAttempts?: number;
  /** Initial delay in milliseconds before first retry */
  initialDelay?: number;
  /** Maximum delay in milliseconds */
  maxDelay?: number;
  /** Backoff multiplier for exponential backoff */
  backoffMultiplier?: number;
  /** Optional predicate to determine if error is retryable */
  shouldRetry?: (error: unknown) => boolean;
  /** Callback for retry attempts */
  onRetry?: (attempt: number, error: unknown, delay: number) => void;
}

export interface CircuitBreakerOptions {
  /** Number of failures before opening circuit */
  failureThreshold?: number;
  /** Time in ms before attempting to close circuit */
  recoveryTimeout?: number;
  /** Callback when circuit opens */
  onOpen?: () => void;
  /** Callback when circuit closes */
  onClose?: () => void;
}

export type CircuitState = 'closed' | 'open' | 'half-open';

export interface FallbackOptions<T> {
  /** Fallback value or function */
  fallback: T | (() => T) | (() => Promise<T>);
  /** Log errors when using fallback */
  logErrors?: boolean;
}

export interface TimeoutOptions {
  /** Timeout in milliseconds */
  timeout: number;
  /** Error message for timeout */
  message?: string;
}

// ============================================================================
// RETRY WITH EXPONENTIAL BACKOFF
// ============================================================================

const DEFAULT_RETRY_OPTIONS: Required<RetryOptions> = {
  maxAttempts: 3,
  initialDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  shouldRetry: () => true,
  onRetry: () => {},
};

/**
 * Execute a function with automatic retry on failure
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: unknown;
  let delay = opts.initialDelay;

  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === opts.maxAttempts || !opts.shouldRetry(error)) {
        throw error;
      }

      opts.onRetry(attempt, error, delay);
      await sleep(delay);
      delay = Math.min(delay * opts.backoffMultiplier, opts.maxDelay);
    }
  }

  throw lastError;
}

/**
 * Create a retry wrapper for a function
 */
export function createRetryable<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  options: RetryOptions = {}
): T {
  return ((...args: unknown[]) => withRetry(() => fn(...args), options)) as T;
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

const DEFAULT_CIRCUIT_OPTIONS: Required<CircuitBreakerOptions> = {
  failureThreshold: 5,
  recoveryTimeout: 60000,
  onOpen: () => {},
  onClose: () => {},
};

class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures = 0;
  private lastFailure = 0;
  private options: Required<CircuitBreakerOptions>;

  constructor(options: CircuitBreakerOptions = {}) {
    this.options = { ...DEFAULT_CIRCUIT_OPTIONS, ...options };
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure >= this.options.recoveryTimeout) {
        this.state = 'half-open';
      } else {
        throw new CircuitOpenError('Circuit breaker is open');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    if (this.state === 'half-open') {
      this.state = 'closed';
      this.failures = 0;
      this.options.onClose();
    }
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();

    if (this.failures >= this.options.failureThreshold) {
      this.state = 'open';
      this.options.onOpen();
    }
  }

  getState(): CircuitState {
    return this.state;
  }

  reset(): void {
    this.state = 'closed';
    this.failures = 0;
  }
}

export class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

/**
 * Create a circuit breaker for a set of operations
 */
export function createCircuitBreaker(options?: CircuitBreakerOptions): CircuitBreaker {
  return new CircuitBreaker(options);
}

// ============================================================================
// FALLBACK
// ============================================================================

/**
 * Execute with a fallback value on failure
 */
export async function withFallback<T>(
  fn: () => Promise<T>,
  options: FallbackOptions<T>
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (options.logErrors) {
      console.warn('Operation failed, using fallback:', error);
    }

    if (typeof options.fallback === 'function') {
      return (options.fallback as () => T | Promise<T>)();
    }
    return options.fallback;
  }
}

/**
 * Create a fallback wrapper for a function
 */
export function createWithFallback<T>(
  fn: () => Promise<T>,
  fallback: T | (() => T) | (() => Promise<T>),
  logErrors = true
): () => Promise<T> {
  return () => withFallback(fn, { fallback, logErrors });
}

// ============================================================================
// TIMEOUT
// ============================================================================

export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

/**
 * Execute with a timeout
 */
export async function withTimeout<T>(
  fn: () => Promise<T>,
  options: TimeoutOptions
): Promise<T> {
  const { timeout, message = `Operation timed out after ${timeout}ms` } = options;

  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(message));
    }, timeout);

    fn()
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

// ============================================================================
// COMBINED RESILIENCE PATTERNS
// ============================================================================

export interface ResilientOptions<T> extends RetryOptions, TimeoutOptions {
  fallback?: T | (() => T) | (() => Promise<T>);
  circuitBreaker?: CircuitBreaker;
}

/**
 * Execute with multiple resilience patterns combined
 */
export async function withResilience<T>(
  fn: () => Promise<T>,
  options: ResilientOptions<T>
): Promise<T> {
  const { fallback, circuitBreaker, timeout, ...retryOptions } = options;

  let operation = fn;

  // Wrap with timeout if specified
  if (timeout) {
    const originalFn = operation;
    operation = () => withTimeout(originalFn, { timeout, message: options.message });
  }

  // Wrap with circuit breaker if specified
  if (circuitBreaker) {
    const originalFn = operation;
    operation = () => circuitBreaker.execute(originalFn);
  }

  // Wrap with retry
  const retryableFn = () => withRetry(operation, retryOptions);

  // Wrap with fallback if specified
  if (fallback !== undefined) {
    return withFallback(retryableFn, { fallback, logErrors: true });
  }

  return retryableFn();
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

export interface BatchOptions {
  /** Maximum items per batch */
  batchSize?: number;
  /** Delay between batches in ms */
  batchDelay?: number;
  /** Continue processing remaining items on error */
  continueOnError?: boolean;
}

/**
 * Process items in batches with optional delay
 */
export async function processBatched<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  options: BatchOptions = {}
): Promise<{ results: R[]; errors: Array<{ item: T; error: unknown }> }> {
  const { batchSize = 10, batchDelay = 100, continueOnError = true } = options;

  const results: R[] = [];
  const errors: Array<{ item: T; error: unknown }> = [];

  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);

    const batchResults = await Promise.allSettled(batch.map(processor));

    for (let j = 0; j < batchResults.length; j++) {
      const result = batchResults[j];
      if (result.status === 'fulfilled') {
        results.push(result.value);
      } else {
        errors.push({ item: batch[j], error: result.reason });
        if (!continueOnError) {
          throw result.reason;
        }
      }
    }

    if (i + batchSize < items.length && batchDelay > 0) {
      await sleep(batchDelay);
    }
  }

  return { results, errors };
}

// ============================================================================
// ERROR CLASSIFICATION
// ============================================================================

export type ErrorType =
  | 'network'
  | 'timeout'
  | 'validation'
  | 'authentication'
  | 'authorization'
  | 'notFound'
  | 'conflict'
  | 'rateLimit'
  | 'server'
  | 'unknown';

/**
 * Classify an error for appropriate handling
 */
export function classifyError(error: unknown): ErrorType {
  if (error instanceof TimeoutError) {
    return 'timeout';
  }

  if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('network'))) {
    return 'network';
  }

  if (error instanceof Error) {
    const message = error.message.toLowerCase();

    if (message.includes('network') || message.includes('connection')) {
      return 'network';
    }
    if (message.includes('timeout')) {
      return 'timeout';
    }
    if (message.includes('validation') || message.includes('invalid')) {
      return 'validation';
    }
    if (message.includes('unauthorized') || message.includes('unauthenticated')) {
      return 'authentication';
    }
    if (message.includes('forbidden') || message.includes('permission')) {
      return 'authorization';
    }
    if (message.includes('not found') || message.includes('404')) {
      return 'notFound';
    }
    if (message.includes('conflict') || message.includes('409')) {
      return 'conflict';
    }
    if (message.includes('rate limit') || message.includes('429')) {
      return 'rateLimit';
    }
    if (message.includes('server') || message.includes('500')) {
      return 'server';
    }
  }

  return 'unknown';
}

/**
 * Check if an error is retryable
 */
export function isRetryableError(error: unknown): boolean {
  const type = classifyError(error);
  return ['network', 'timeout', 'rateLimit', 'server'].includes(type);
}

// ============================================================================
// HELPERS
// ============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Create a debounced version of a function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
      timeoutId = null;
    }, delay);
  };
}

/**
 * Create a throttled version of a function
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
  };
}

/**
 * Cache the result of an async function
 */
export function memoizeAsync<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  options: { ttl?: number; keyFn?: (...args: Parameters<T>) => string } = {}
): T {
  const { ttl = 60000, keyFn = (...args) => JSON.stringify(args) } = options;
  const cache = new Map<string, { value: unknown; expires: number }>();

  return (async (...args: Parameters<T>) => {
    const key = keyFn(...args);
    const cached = cache.get(key);

    if (cached && cached.expires > Date.now()) {
      return cached.value;
    }

    const value = await fn(...args);
    cache.set(key, { value, expires: Date.now() + ttl });
    return value;
  }) as T;
}

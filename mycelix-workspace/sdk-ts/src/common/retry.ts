// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common Retry Utilities
 *
 * Retry patterns and utilities for resilient SDK operations.
 *
 * @module @mycelix/sdk/common/retry
 */

import { SdkError } from './errors';

/**
 * Retry configuration options
 */
export interface RetryOptions {
  /** Maximum number of retry attempts (default: 3) */
  maxAttempts?: number;
  /** Alias for maxAttempts (for test compatibility) */
  maxRetries?: number;
  /** Initial delay in milliseconds (default: 1000) */
  initialDelayMs?: number;
  /** Backoff multiplier (default: 2) */
  backoffMultiplier?: number;
  /** Maximum delay between retries in milliseconds (default: 30000) */
  maxDelayMs?: number;
  /** Add random jitter to delays (default: true) */
  jitter?: boolean;
  /** Custom function to determine if error is retryable */
  isRetryable?: (error: unknown) => boolean;
  /** Callback when a retry occurs */
  onRetry?: (attempt: number, error: unknown, delayMs: number) => void;
}

const DEFAULT_OPTIONS: Required<Omit<RetryOptions, 'isRetryable' | 'onRetry'>> = {
  maxAttempts: 3,
  maxRetries: 3,
  initialDelayMs: 1000,
  backoffMultiplier: 2,
  maxDelayMs: 30000,
  jitter: true,
};

/**
 * Default retryable error check
 */
function defaultIsRetryable(error: unknown): boolean {
  if (error instanceof SdkError) {
    return error.isRetryable();
  }

  // Network-related errors
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('connection') ||
      message.includes('econnreset') ||
      message.includes('econnrefused') ||
      message.includes('socket hang up')
    );
  }

  return false;
}

/**
 * Calculate delay for a given attempt with optional jitter
 */
function calculateDelay(
  attempt: number,
  options: Required<Omit<RetryOptions, 'isRetryable' | 'onRetry'>>
): number {
  let delay = options.initialDelayMs * Math.pow(options.backoffMultiplier, attempt - 1);
  delay = Math.min(delay, options.maxDelayMs);

  if (options.jitter) {
    // Add 0-20% random jitter
    delay = delay * (1 + Math.random() * 0.2);
  }

  return Math.round(delay);
}

/**
 * Execute an operation with automatic retry on failure
 *
 * @param operation - Async function to execute
 * @param options - Retry configuration
 * @returns Result of the operation
 * @throws Last error if all retries are exhausted
 *
 * @example
 * ```typescript
 * const result = await withRetry(
 *   async () => await api.fetchData(),
 *   {
 *     maxAttempts: 5,
 *     initialDelayMs: 500,
 *     onRetry: (attempt, error) => console.log(`Retry ${attempt}:`, error),
 *   }
 * );
 * ```
 */
export async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  // Support maxRetries as alias for maxAttempts
  // maxRetries = retries after initial (so +1 for total attempts)
  // maxAttempts = total attempts including initial
  const maxAttempts = options.maxAttempts ??
    (options.maxRetries !== undefined ? options.maxRetries + 1 : DEFAULT_OPTIONS.maxAttempts);
  const opts = { ...DEFAULT_OPTIONS, ...options, maxAttempts };
  const isRetryable = options.isRetryable ?? defaultIsRetryable;

  let lastError: unknown;
  let attempt = 0;

  while (attempt < opts.maxAttempts) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      attempt++;

      // Check if we should retry
      if (attempt >= opts.maxAttempts || !isRetryable(error)) {
        throw error;
      }

      // Calculate delay
      const delay = calculateDelay(attempt, opts);

      // Notify callback
      if (options.onRetry) {
        options.onRetry(attempt, error, delay);
      }

      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  // This should be unreachable, but TypeScript needs it
  throw lastError;
}

/**
 * Create a retry wrapper for a function
 *
 * @param fn - Function to wrap
 * @param options - Retry configuration
 * @returns Wrapped function with automatic retry
 *
 * @example
 * ```typescript
 * const fetchWithRetry = createRetryWrapper(
 *   async (url: string) => await fetch(url),
 *   { maxAttempts: 3 }
 * );
 *
 * const response = await fetchWithRetry('https://api.example.com/data');
 * ```
 */
export function createRetryWrapper<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  options: RetryOptions = {}
): (...args: TArgs) => Promise<TResult> {
  return (...args: TArgs) => withRetry(() => fn(...args), options);
}

/**
 * Retry policy for configurable retry behavior
 */
export class RetryPolicy {
  private readonly options: Required<Omit<RetryOptions, 'isRetryable' | 'onRetry'>>;
  private readonly isRetryable: (error: unknown) => boolean;
  private readonly onRetry?: (attempt: number, error: unknown, delayMs: number) => void;

  constructor(options: RetryOptions = {}) {
    // Support maxRetries as alias for maxAttempts
    // maxRetries = retries after initial (so +1 for total attempts)
    // maxAttempts = total attempts including initial
    const maxAttempts = options.maxAttempts ??
      (options.maxRetries !== undefined ? options.maxRetries + 1 : DEFAULT_OPTIONS.maxAttempts);
    this.options = { ...DEFAULT_OPTIONS, ...options, maxAttempts };
    this.isRetryable = options.isRetryable ?? defaultIsRetryable;
    this.onRetry = options.onRetry;
  }

  /**
   * Execute an operation with this policy's retry configuration
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    return withRetry(operation, {
      ...this.options,
      isRetryable: this.isRetryable,
      onRetry: this.onRetry,
    });
  }

  /**
   * Wrap a function with this policy's retry configuration
   */
  wrap<TArgs extends unknown[], TResult>(
    fn: (...args: TArgs) => Promise<TResult>
  ): (...args: TArgs) => Promise<TResult> {
    return (...args: TArgs) => this.execute(() => fn(...args));
  }

  /**
   * Create a new policy with modified options
   */
  with(options: RetryOptions): RetryPolicy {
    return new RetryPolicy({ ...this.options, ...options });
  }
}

/**
 * Pre-configured retry policies
 */
export const RetryPolicies = {
  /** No retries */
  none: new RetryPolicy({ maxAttempts: 1 }),

  /** Light retry - 2 attempts, fast backoff */
  light: new RetryPolicy({
    maxAttempts: 2,
    initialDelayMs: 500,
    backoffMultiplier: 1.5,
    maxDelayMs: 2000,
  }),

  /** Standard retry - 3 attempts, moderate backoff */
  standard: new RetryPolicy({
    maxAttempts: 3,
    initialDelayMs: 1000,
    backoffMultiplier: 2,
    maxDelayMs: 10000,
  }),

  /** Aggressive retry - 5 attempts, slower backoff */
  aggressive: new RetryPolicy({
    maxAttempts: 5,
    initialDelayMs: 2000,
    backoffMultiplier: 2,
    maxDelayMs: 60000,
  }),

  /** Network-optimized - good for unstable connections */
  network: new RetryPolicy({
    maxAttempts: 4,
    initialDelayMs: 500,
    backoffMultiplier: 2.5,
    maxDelayMs: 30000,
    jitter: true,
  }),
};

/**
 * Execute with timeout
 *
 * @param operation - Async function to execute
 * @param timeoutMs - Timeout in milliseconds
 * @param timeoutMessage - Optional custom timeout error message
 * @returns Result of the operation
 * @throws Error if operation times out
 */
export async function withTimeout<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  timeoutMessage = 'Operation timed out'
): Promise<T> {
  return Promise.race([
    operation(),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new SdkError('TIMEOUT', timeoutMessage)), timeoutMs)
    ),
  ]);
}

/**
 * Execute with both retry and timeout
 *
 * @param operation - Async function to execute
 * @param options - Retry options
 * @param timeoutMs - Timeout per attempt in milliseconds
 * @returns Result of the operation
 */
export async function withRetryAndTimeout<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {},
  timeoutMs: number
): Promise<T> {
  return withRetry(
    () => withTimeout(operation, timeoutMs),
    {
      ...options,
      isRetryable: (error) => {
        // Timeouts are retryable by default
        if (error instanceof SdkError && error.code === 'TIMEOUT') {
          return true;
        }
        return options.isRetryable?.(error) ?? defaultIsRetryable(error);
      },
    }
  );
}

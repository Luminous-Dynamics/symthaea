/**
 * Core Retry Logic
 *
 * Exponential backoff retry utilities for resilient SDK operations.
 *
 * @module @mycelix/sdk/core/retry
 */

import { SdkError, TimeoutError, wrapError } from './errors.js';

/**
 * Retry configuration
 */
export interface RetryConfig {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries: number;

  /** Initial delay between retries in milliseconds (default: 1000) */
  baseDelay: number;

  /** Maximum delay between retries in milliseconds (default: 30000) */
  maxDelay: number;

  /** Backoff multiplier (default: 2) */
  backoffMultiplier?: number;

  /** Add random jitter to delays (default: true) */
  jitter?: boolean;

  /** Custom function to determine if error is retryable */
  isRetryable?: (error: unknown) => boolean;

  /** Callback when a retry occurs */
  onRetry?: (attempt: number, error: unknown, delayMs: number) => void;
}

/**
 * Default retry configuration
 */
export const DEFAULT_RETRY_CONFIG: Required<RetryConfig> = {
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: true,
  isRetryable: defaultIsRetryable,
  onRetry: () => {},
};

/**
 * Result of a retry operation
 */
export interface RetryResult<T> {
  /** Whether the operation succeeded */
  success: boolean;

  /** Result value if successful */
  value?: T;

  /** Error if operation failed */
  error?: SdkError;

  /** Number of attempts made */
  attempts: number;

  /** Total time spent in milliseconds */
  totalTimeMs: number;
}

/**
 * Default check for retryable errors
 */
function defaultIsRetryable(error: unknown): boolean {
  if (error instanceof SdkError) {
    return error.isRetryable();
  }

  if (error instanceof Error) {
    const msg = error.message.toLowerCase();
    return (
      msg.includes('network') ||
      msg.includes('timeout') ||
      msg.includes('connection') ||
      msg.includes('econnreset') ||
      msg.includes('econnrefused') ||
      msg.includes('socket hang up')
    );
  }

  return false;
}

/**
 * Calculate delay for a given attempt with exponential backoff and optional jitter
 */
function calculateDelay(
  attempt: number,
  config: Required<RetryConfig>
): number {
  // Exponential backoff: baseDelay * (multiplier ^ attempt)
  let delay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt);

  // Cap at max delay
  delay = Math.min(delay, config.maxDelay);

  // Add jitter (0-25% random variation)
  if (config.jitter) {
    const jitterFactor = 1 + Math.random() * 0.25;
    delay = Math.round(delay * jitterFactor);
  }

  return delay;
}

/**
 * Wait for a specified number of milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Execute an async operation with automatic retry on failure
 *
 * Implements exponential backoff with configurable jitter and retry conditions.
 *
 * @param fn - Async function to execute
 * @param config - Retry configuration
 * @returns Promise resolving to the function result
 * @throws SdkError if all retries are exhausted
 *
 * @example
 * ```typescript
 * const result = await withRetry(
 *   async () => await api.fetchData(),
 *   {
 *     maxRetries: 5,
 *     baseDelay: 500,
 *     onRetry: (attempt, error) => console.log(`Retry ${attempt}:`, error),
 *   }
 * );
 * ```
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const opts: Required<RetryConfig> = { ...DEFAULT_RETRY_CONFIG, ...config };

  let lastError: unknown;
  let attempt = 0;

  while (attempt <= opts.maxRetries) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      attempt++;

      // Check if we should retry
      if (attempt > opts.maxRetries || !opts.isRetryable(error)) {
        break;
      }

      // Calculate delay
      const delay = calculateDelay(attempt - 1, opts);

      // Notify callback
      opts.onRetry(attempt, error, delay);

      // Wait before retrying
      await sleep(delay);
    }
  }

  // All retries exhausted
  throw wrapError(lastError, {
    message: `Operation failed after ${attempt} attempts: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`,
  });
}

/**
 * Execute an async operation with retry and get detailed result
 *
 * Unlike `withRetry`, this function never throws. Instead, it returns
 * a result object indicating success or failure.
 *
 * @param fn - Async function to execute
 * @param config - Retry configuration
 * @returns RetryResult with success/failure details
 *
 * @example
 * ```typescript
 * const result = await tryWithRetry(async () => await api.fetchData());
 *
 * if (result.success) {
 *   console.log('Data:', result.value);
 * } else {
 *   console.error(`Failed after ${result.attempts} attempts:`, result.error);
 * }
 * ```
 */
export async function tryWithRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<RetryResult<T>> {
  const startTime = Date.now();
  const opts: Required<RetryConfig> = { ...DEFAULT_RETRY_CONFIG, ...config };

  let lastError: unknown;
  let attempt = 0;

  while (attempt <= opts.maxRetries) {
    try {
      const value = await fn();
      return {
        success: true,
        value,
        attempts: attempt + 1,
        totalTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      lastError = error;
      attempt++;

      if (attempt > opts.maxRetries || !opts.isRetryable(error)) {
        break;
      }

      const delay = calculateDelay(attempt - 1, opts);
      opts.onRetry(attempt, error, delay);
      await sleep(delay);
    }
  }

  return {
    success: false,
    error: wrapError(lastError),
    attempts: attempt,
    totalTimeMs: Date.now() - startTime,
  };
}

/**
 * Execute an async operation with timeout
 *
 * @param fn - Async function to execute
 * @param timeoutMs - Timeout in milliseconds
 * @param operationName - Name for error messages
 * @returns Promise resolving to the function result
 * @throws TimeoutError if operation exceeds timeout
 *
 * @example
 * ```typescript
 * const result = await withTimeout(
 *   async () => await api.fetchData(),
 *   5000,
 *   'fetchData'
 * );
 * ```
 */
export async function withTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number,
  operationName: string = 'operation'
): Promise<T> {
  return Promise.race([
    fn(),
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new TimeoutError(operationName, timeoutMs)),
        timeoutMs
      )
    ),
  ]);
}

/**
 * Execute an async operation with both retry and timeout
 *
 * The timeout applies to each individual attempt, not the total time.
 *
 * @param fn - Async function to execute
 * @param config - Retry configuration
 * @param timeoutMs - Timeout per attempt in milliseconds
 * @returns Promise resolving to the function result
 *
 * @example
 * ```typescript
 * const result = await withRetryAndTimeout(
 *   async () => await api.fetchData(),
 *   { maxRetries: 3 },
 *   5000
 * );
 * ```
 */
export async function withRetryAndTimeout<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {},
  timeoutMs: number
): Promise<T> {
  return withRetry(
    () => withTimeout(fn, timeoutMs, 'operation'),
    {
      ...config,
      isRetryable: (error) => {
        // Timeouts are retryable
        if (error instanceof TimeoutError) {
          return true;
        }
        // Fall back to configured or default check
        return config.isRetryable?.(error) ?? defaultIsRetryable(error);
      },
    }
  );
}

/**
 * Create a retry-wrapped version of a function
 *
 * @param fn - Function to wrap
 * @param config - Retry configuration
 * @returns Wrapped function with automatic retry
 *
 * @example
 * ```typescript
 * const fetchWithRetry = createRetryWrapper(
 *   async (url: string) => await fetch(url),
 *   { maxRetries: 3 }
 * );
 *
 * const response = await fetchWithRetry('https://api.example.com/data');
 * ```
 */
export function createRetryWrapper<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  config: Partial<RetryConfig> = {}
): (...args: TArgs) => Promise<TResult> {
  return (...args: TArgs) => withRetry(() => fn(...args), config);
}

/**
 * Pre-configured retry policies
 */
export const RetryPolicies = {
  /** No retries - fail immediately */
  none: (): Partial<RetryConfig> => ({
    maxRetries: 0,
  }),

  /** Light retry - 2 retries, fast backoff */
  light: (): Partial<RetryConfig> => ({
    maxRetries: 2,
    baseDelay: 500,
    maxDelay: 2000,
    backoffMultiplier: 1.5,
  }),

  /** Standard retry - 3 retries, moderate backoff */
  standard: (): Partial<RetryConfig> => ({
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    backoffMultiplier: 2,
  }),

  /** Aggressive retry - 5 retries, slower backoff */
  aggressive: (): Partial<RetryConfig> => ({
    maxRetries: 5,
    baseDelay: 2000,
    maxDelay: 60000,
    backoffMultiplier: 2,
  }),

  /** Network-optimized - good for unstable connections */
  network: (): Partial<RetryConfig> => ({
    maxRetries: 4,
    baseDelay: 500,
    maxDelay: 30000,
    backoffMultiplier: 2.5,
    jitter: true,
  }),
};

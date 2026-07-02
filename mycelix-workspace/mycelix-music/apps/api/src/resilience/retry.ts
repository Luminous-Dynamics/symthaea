// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Retry with Exponential Backoff
 *
 * Automatic retry of transient failures with configurable backoff.
 * Includes jitter to prevent thundering herd.
 */

import { getLogger } from '../logging';
import { getMetrics } from '../metrics';

const logger = getLogger();

/**
 * Retry configuration
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  maxAttempts: number;
  /** Initial delay in milliseconds */
  initialDelay: number;
  /** Maximum delay between retries */
  maxDelay: number;
  /** Backoff multiplier */
  multiplier: number;
  /** Add random jitter (0-1) */
  jitter: number;
  /** Timeout per attempt in ms */
  timeout?: number;
  /** Custom retry condition */
  retryCondition?: (error: Error, attempt: number) => boolean;
  /** Called before each retry */
  onRetry?: (error: Error, attempt: number, delay: number) => void;
}

/**
 * Default configuration
 */
const defaultConfig: RetryConfig = {
  maxAttempts: 3,
  initialDelay: 1000,
  maxDelay: 30000,
  multiplier: 2,
  jitter: 0.1,
};

/**
 * Retry result
 */
export interface RetryResult<T> {
  success: boolean;
  result?: T;
  error?: Error;
  attempts: number;
  totalTime: number;
}

/**
 * Calculate delay with exponential backoff and jitter
 */
function calculateDelay(
  attempt: number,
  config: RetryConfig
): number {
  // Exponential backoff
  let delay = config.initialDelay * Math.pow(config.multiplier, attempt - 1);

  // Cap at max delay
  delay = Math.min(delay, config.maxDelay);

  // Add jitter
  if (config.jitter > 0) {
    const jitterRange = delay * config.jitter;
    delay = delay + (Math.random() * 2 - 1) * jitterRange;
  }

  return Math.round(delay);
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Check if error is retryable
 */
function isRetryableError(error: Error): boolean {
  const retryableCodes = [
    'ECONNRESET',
    'ECONNREFUSED',
    'ETIMEDOUT',
    'ENOTFOUND',
    'EAI_AGAIN',
    'EPIPE',
    'EHOSTUNREACH',
  ];

  // Check error code
  if ((error as any).code && retryableCodes.includes((error as any).code)) {
    return true;
  }

  // Check HTTP status codes (if available)
  const status = (error as any).status || (error as any).statusCode;
  if (status) {
    // Retry on 5xx and specific 4xx
    if (status >= 500 && status < 600) return true;
    if (status === 429) return true; // Rate limited
    if (status === 408) return true; // Request timeout
  }

  // Check error message patterns
  const retryablePatterns = [
    /timeout/i,
    /temporarily unavailable/i,
    /connection refused/i,
    /socket hang up/i,
    /ECONNRESET/i,
    /network error/i,
    /service unavailable/i,
  ];

  return retryablePatterns.some(pattern => pattern.test(error.message));
}

/**
 * Retry a function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const cfg: RetryConfig = { ...defaultConfig, ...config };
  const startTime = Date.now();
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= cfg.maxAttempts; attempt++) {
    try {
      // Execute with optional timeout
      let result: T;

      if (cfg.timeout) {
        result = await Promise.race([
          fn(),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Operation timeout')), cfg.timeout)
          ),
        ]);
      } else {
        result = await fn();
      }

      // Success
      if (attempt > 1) {
        getMetrics().incCounter('retry_success_total', { attempts: String(attempt) });
      }

      return result;
    } catch (error) {
      lastError = error as Error;

      // Check if we should retry
      const shouldRetry = cfg.retryCondition
        ? cfg.retryCondition(lastError, attempt)
        : isRetryableError(lastError);

      if (!shouldRetry || attempt >= cfg.maxAttempts) {
        getMetrics().incCounter('retry_failed_total', { attempts: String(attempt) });
        throw lastError;
      }

      // Calculate delay
      const delay = calculateDelay(attempt, cfg);

      // Callback
      if (cfg.onRetry) {
        cfg.onRetry(lastError, attempt, delay);
      }

      logger.debug(`Retry attempt ${attempt}/${cfg.maxAttempts}`, {
        error: lastError.message,
        delay,
        nextAttempt: attempt + 1,
      });

      getMetrics().incCounter('retry_attempts_total', {});

      // Wait before retry
      await sleep(delay);
    }
  }

  // Should never reach here, but just in case
  throw lastError || new Error('Retry failed');
}

/**
 * Retry with result wrapper
 */
export async function retryWithResult<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<RetryResult<T>> {
  const startTime = Date.now();
  let attempts = 0;

  try {
    const onRetry = config.onRetry;
    const result = await retry(fn, {
      ...config,
      onRetry: (error, attempt, delay) => {
        attempts = attempt;
        if (onRetry) onRetry(error, attempt, delay);
      },
    });

    return {
      success: true,
      result,
      attempts: attempts + 1,
      totalTime: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: error as Error,
      attempts: attempts + 1,
      totalTime: Date.now() - startTime,
    };
  }
}

/**
 * Create a retryable version of a function
 */
export function withRetry<T, Args extends unknown[]>(
  fn: (...args: Args) => Promise<T>,
  config: Partial<RetryConfig> = {}
): (...args: Args) => Promise<T> {
  return (...args: Args) => retry(() => fn(...args), config);
}

/**
 * Decorator for retryable methods
 */
export function Retryable(config: Partial<RetryConfig> = {}) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: unknown[]) {
      return retry(() => originalMethod.apply(this, args), config);
    };

    return descriptor;
  };
}

/**
 * Pre-configured retry strategies
 */
export const retryStrategies = {
  /** Quick retries for fast-failing operations */
  fast: {
    maxAttempts: 3,
    initialDelay: 100,
    maxDelay: 1000,
    multiplier: 2,
    jitter: 0.1,
  },

  /** Standard retries */
  standard: {
    maxAttempts: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    multiplier: 2,
    jitter: 0.2,
  },

  /** Patient retries for slow services */
  patient: {
    maxAttempts: 5,
    initialDelay: 2000,
    maxDelay: 60000,
    multiplier: 2,
    jitter: 0.25,
  },

  /** Aggressive retries for critical operations */
  aggressive: {
    maxAttempts: 10,
    initialDelay: 500,
    maxDelay: 30000,
    multiplier: 1.5,
    jitter: 0.3,
  },

  /** Database connection retries */
  database: {
    maxAttempts: 5,
    initialDelay: 1000,
    maxDelay: 30000,
    multiplier: 2,
    jitter: 0.1,
    retryCondition: (error: Error) => {
      const dbErrors = ['ECONNREFUSED', 'ETIMEDOUT', 'connection terminated'];
      return dbErrors.some(e =>
        error.message.includes(e) || (error as any).code === e
      );
    },
  },

  /** HTTP request retries */
  http: {
    maxAttempts: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    multiplier: 2,
    jitter: 0.2,
    retryCondition: (error: Error) => {
      const status = (error as any).status || (error as any).statusCode;
      if (status >= 500) return true;
      if (status === 429) return true;
      return isRetryableError(error);
    },
  },
};

/**
 * Retry fetch requests
 */
export async function retryFetch(
  url: string,
  options: RequestInit = {},
  retryConfig: Partial<RetryConfig> = {}
): Promise<Response> {
  return retry(async () => {
    const response = await fetch(url, options);

    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`) as any;
      error.status = response.status;
      error.statusText = response.statusText;
      throw error;
    }

    return response;
  }, {
    ...retryStrategies.http,
    ...retryConfig,
  });
}

/**
 * Wait for a condition with retries
 */
export async function waitFor<T>(
  condition: () => Promise<T | null | undefined>,
  config: Partial<RetryConfig> & { pollInterval?: number } = {}
): Promise<T> {
  const pollInterval = config.pollInterval || 1000;
  const maxAttempts = config.maxAttempts || 30;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const result = await condition();
    if (result !== null && result !== undefined) {
      return result;
    }
    await sleep(pollInterval);
  }

  throw new Error('Condition not met within timeout');
}

export default {
  retry,
  retryWithResult,
  withRetry,
  Retryable,
  retryStrategies,
  retryFetch,
  waitFor,
  isRetryableError,
};

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Production Resilience Patterns
 *
 * Provides circuit breakers, retry logic, timeouts, and rate limiting
 * for robust production deployments.
 */

// ============================================================================
// Circuit Breaker
// ============================================================================

export enum CircuitState {
  CLOSED = 'closed', // Normal operation
  OPEN = 'open', // Failing, rejecting requests
  HALF_OPEN = 'half_open', // Testing if service recovered
}

export interface CircuitBreakerOptions {
  /** Number of failures before opening circuit */
  failureThreshold: number;
  /** Time in ms before attempting recovery */
  recoveryTimeout: number;
  /** Number of successful calls to close circuit */
  successThreshold: number;
  /** Optional name for logging */
  name?: string;
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures = 0;
  private successes = 0;
  private lastFailureTime = 0;
  private readonly options: Required<CircuitBreakerOptions>;

  constructor(options: Partial<CircuitBreakerOptions> = {}) {
    this.options = {
      failureThreshold: options.failureThreshold ?? 5,
      recoveryTimeout: options.recoveryTimeout ?? 30000,
      successThreshold: options.successThreshold ?? 2,
      name: options.name ?? 'default',
    };
  }

  get currentState(): CircuitState {
    return this.state;
  }

  get stats() {
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      lastFailureTime: this.lastFailureTime,
    };
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() - this.lastFailureTime >= this.options.recoveryTimeout) {
        this.state = CircuitState.HALF_OPEN;
        this.successes = 0;
      } else {
        throw new CircuitOpenError(
          `Circuit breaker '${this.options.name}' is open`,
          this.options.recoveryTimeout - (Date.now() - this.lastFailureTime)
        );
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
    this.failures = 0;
    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      if (this.successes >= this.options.successThreshold) {
        this.state = CircuitState.CLOSED;
        this.successes = 0;
      }
    }
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    this.successes = 0;

    if (this.failures >= this.options.failureThreshold) {
      this.state = CircuitState.OPEN;
    }
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failures = 0;
    this.successes = 0;
    this.lastFailureTime = 0;
  }
}

export class CircuitOpenError extends Error {
  constructor(
    message: string,
    public readonly retryAfterMs: number
  ) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

// ============================================================================
// Retry with Exponential Backoff
// ============================================================================

export interface RetryOptions {
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Initial delay in ms */
  initialDelay: number;
  /** Maximum delay in ms */
  maxDelay: number;
  /** Backoff multiplier */
  backoffFactor: number;
  /** Add randomness to prevent thundering herd */
  jitter: boolean;
  /** Predicate to determine if error is retryable */
  isRetryable?: (error: unknown) => boolean;
  /** Callback on each retry */
  onRetry?: (error: unknown, attempt: number, delay: number) => void;
}

const defaultRetryOptions: RetryOptions = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 30000,
  backoffFactor: 2,
  jitter: true,
  isRetryable: () => true,
};

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {}
): Promise<T> {
  const opts = { ...defaultRetryOptions, ...options };
  let lastError: unknown;
  let delay = opts.initialDelay;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === opts.maxRetries) {
        break;
      }

      if (opts.isRetryable && !opts.isRetryable(error)) {
        break;
      }

      // Calculate delay with optional jitter
      let currentDelay = Math.min(delay, opts.maxDelay);
      if (opts.jitter) {
        currentDelay = currentDelay * (0.5 + Math.random());
      }

      opts.onRetry?.(error, attempt + 1, currentDelay);

      await sleep(currentDelay);
      delay *= opts.backoffFactor;
    }
  }

  throw lastError;
}

// ============================================================================
// Timeout
// ============================================================================

export class TimeoutError extends Error {
  constructor(
    message: string,
    public readonly timeoutMs: number
  ) {
    super(message);
    this.name = 'TimeoutError';
  }
}

export async function withTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number,
  message?: string
): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout>;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new TimeoutError(message ?? `Operation timed out after ${timeoutMs}ms`, timeoutMs));
    }, timeoutMs);
  });

  try {
    const result = await Promise.race([fn(), timeoutPromise]);
    clearTimeout(timeoutId!);
    return result;
  } catch (error) {
    clearTimeout(timeoutId!);
    throw error;
  }
}

// ============================================================================
// Rate Limiter (Token Bucket)
// ============================================================================

export interface RateLimiterOptions {
  /** Maximum tokens in bucket */
  maxTokens: number;
  /** Tokens added per interval */
  refillRate: number;
  /** Refill interval in ms */
  refillInterval: number;
}

export class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  private readonly options: RateLimiterOptions;

  constructor(options: Partial<RateLimiterOptions> = {}) {
    this.options = {
      maxTokens: options.maxTokens ?? 100,
      refillRate: options.refillRate ?? 10,
      refillInterval: options.refillInterval ?? 1000,
    };
    this.tokens = this.options.maxTokens;
    this.lastRefill = Date.now();
  }

  get availableTokens(): number {
    this.refill();
    return this.tokens;
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    const intervals = Math.floor(elapsed / this.options.refillInterval);

    if (intervals > 0) {
      this.tokens = Math.min(
        this.options.maxTokens,
        this.tokens + intervals * this.options.refillRate
      );
      this.lastRefill = now;
    }
  }

  tryAcquire(tokens = 1): boolean {
    this.refill();

    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }

    return false;
  }

  async acquire(tokens = 1): Promise<void> {
    while (!this.tryAcquire(tokens)) {
      await sleep(this.options.refillInterval);
    }
  }

  async execute<T>(fn: () => Promise<T>, tokens = 1): Promise<T> {
    await this.acquire(tokens);
    return fn();
  }
}

export class RateLimitExceededError extends Error {
  constructor(
    message: string,
    public readonly retryAfterMs: number
  ) {
    super(message);
    this.name = 'RateLimitExceededError';
  }
}

// ============================================================================
// Bulkhead (Concurrency Limiter)
// ============================================================================

export interface BulkheadOptions {
  /** Maximum concurrent executions */
  maxConcurrent: number;
  /** Maximum queue size (0 = no queue) */
  maxQueue: number;
}

export class Bulkhead {
  private running = 0;
  private queue: Array<() => void> = [];
  private readonly options: BulkheadOptions;

  constructor(options: Partial<BulkheadOptions> = {}) {
    this.options = {
      maxConcurrent: options.maxConcurrent ?? 10,
      maxQueue: options.maxQueue ?? 100,
    };
  }

  get stats() {
    return {
      running: this.running,
      queued: this.queue.length,
      available: this.options.maxConcurrent - this.running,
    };
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();

    try {
      return await fn();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    if (this.running < this.options.maxConcurrent) {
      this.running++;
      return Promise.resolve();
    }

    if (this.queue.length >= this.options.maxQueue) {
      return Promise.reject(
        new BulkheadFullError(
          `Bulkhead queue full (${this.options.maxQueue} waiting)`,
          this.running,
          this.queue.length
        )
      );
    }

    return new Promise((resolve) => {
      this.queue.push(() => {
        this.running++;
        resolve();
      });
    });
  }

  private release(): void {
    this.running--;

    const next = this.queue.shift();
    if (next) {
      next();
    }
  }
}

export class BulkheadFullError extends Error {
  constructor(
    message: string,
    public readonly running: number,
    public readonly queued: number
  ) {
    super(message);
    this.name = 'BulkheadFullError';
  }
}

// ============================================================================
// Combined Resilience Policy
// ============================================================================

export interface ResiliencePolicyOptions {
  circuitBreaker?: Partial<CircuitBreakerOptions>;
  retry?: Partial<RetryOptions>;
  timeout?: number;
  rateLimiter?: Partial<RateLimiterOptions>;
  bulkhead?: Partial<BulkheadOptions>;
}

export class ResiliencePolicy {
  private readonly circuitBreaker?: CircuitBreaker;
  private readonly retryOptions?: Partial<RetryOptions>;
  private readonly timeoutMs?: number;
  private readonly rateLimiter?: RateLimiter;
  private readonly bulkhead?: Bulkhead;

  constructor(options: ResiliencePolicyOptions = {}) {
    if (options.circuitBreaker) {
      this.circuitBreaker = new CircuitBreaker(options.circuitBreaker);
    }
    this.retryOptions = options.retry;
    this.timeoutMs = options.timeout;
    if (options.rateLimiter) {
      this.rateLimiter = new RateLimiter(options.rateLimiter);
    }
    if (options.bulkhead) {
      this.bulkhead = new Bulkhead(options.bulkhead);
    }
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    let operation = fn;

    // Apply timeout
    if (this.timeoutMs) {
      const timeout = this.timeoutMs;
      const originalOp = operation;
      operation = () => withTimeout(originalOp, timeout);
    }

    // Apply retry
    if (this.retryOptions) {
      const retryOpts = this.retryOptions;
      const originalOp = operation;
      operation = () => withRetry(originalOp, retryOpts);
    }

    // Apply circuit breaker
    if (this.circuitBreaker) {
      const cb = this.circuitBreaker;
      const originalOp = operation;
      operation = () => cb.execute(originalOp);
    }

    // Apply rate limiter
    if (this.rateLimiter) {
      const rl = this.rateLimiter;
      const originalOp = operation;
      operation = () => rl.execute(originalOp);
    }

    // Apply bulkhead
    if (this.bulkhead) {
      const bh = this.bulkhead;
      const originalOp = operation;
      operation = () => bh.execute(originalOp);
    }

    return operation();
  }

  get stats() {
    return {
      circuitBreaker: this.circuitBreaker?.stats,
      rateLimiter: this.rateLimiter ? { available: this.rateLimiter.availableTokens } : undefined,
      bulkhead: this.bulkhead?.stats,
    };
  }
}

// ============================================================================
// Utilities
// ============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================================
// Pre-configured Policies
// ============================================================================

/** Default policy for Holochain operations */
export const holochainPolicy = new ResiliencePolicy({
  circuitBreaker: {
    failureThreshold: 5,
    recoveryTimeout: 30000,
    successThreshold: 2,
    name: 'holochain',
  },
  retry: {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    backoffFactor: 2,
    jitter: true,
  },
  timeout: 30000,
  bulkhead: {
    maxConcurrent: 20,
    maxQueue: 100,
  },
});

/** Aggressive retry policy for critical operations */
export const criticalPolicy = new ResiliencePolicy({
  retry: {
    maxRetries: 5,
    initialDelay: 500,
    maxDelay: 30000,
    backoffFactor: 2,
    jitter: true,
  },
  timeout: 60000,
});

/** Fast-fail policy for non-critical operations */
export const fastFailPolicy = new ResiliencePolicy({
  circuitBreaker: {
    failureThreshold: 3,
    recoveryTimeout: 10000,
    successThreshold: 1,
    name: 'fast-fail',
  },
  timeout: 5000,
});

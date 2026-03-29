// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Circuit Breaker Pattern Implementation
 *
 * Prevents cascading failures by temporarily disabling operations
 * that are likely to fail, allowing the system to recover.
 *
 * @module @mycelix/sdk/common/circuit-breaker
 */

import { SdkError } from './errors';

/**
 * Circuit breaker states
 */
export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
  /** Number of failures before opening circuit (default: 5) */
  failureThreshold?: number;
  /** Time in ms before attempting to close circuit (default: 30000) */
  resetTimeoutMs?: number;
  /** Number of successful calls in half-open state to close circuit (default: 2) */
  successThreshold?: number;
  /** Time window in ms for counting failures (default: 60000) */
  failureWindowMs?: number;
  /** Custom function to determine if error should be counted */
  isFailure?: (error: unknown) => boolean;
  /** Callback when state changes */
  onStateChange?: (from: CircuitState, to: CircuitState) => void;
  /** Callback on each failure */
  onFailure?: (error: unknown, failureCount: number) => void;
  /** Name for logging/debugging */
  name?: string;
}

const DEFAULT_CONFIG: Required<Omit<CircuitBreakerConfig, 'isFailure' | 'onStateChange' | 'onFailure' | 'name'>> = {
  failureThreshold: 5,
  resetTimeoutMs: 30000,
  successThreshold: 2,
  failureWindowMs: 60000,
};

/**
 * Circuit breaker error thrown when circuit is open
 */
export class CircuitBreakerOpenError extends SdkError {
  constructor(
    public readonly circuitName: string,
    public readonly openSince: number,
    public readonly resetAt: number
  ) {
    super(
      'CIRCUIT_OPEN',
      `Circuit breaker "${circuitName}" is open. Will attempt reset at ${new Date(resetAt).toISOString()}`
    );
    this.name = 'CircuitBreakerOpenError';
  }
}

/**
 * Failure record for tracking
 */
interface FailureRecord {
  timestamp: number;
  error: unknown;
}

/**
 * Circuit breaker statistics
 */
export interface CircuitBreakerStats {
  state: CircuitState;
  failureCount: number;
  successCount: number;
  totalCalls: number;
  totalFailures: number;
  totalSuccesses: number;
  lastFailure?: number;
  lastSuccess?: number;
  openedAt?: number;
  halfOpenAt?: number;
  closedAt?: number;
}

/**
 * Circuit Breaker
 *
 * Implements the circuit breaker pattern for resilient service calls.
 *
 * States:
 * - Closed: Normal operation, requests pass through
 * - Open: Requests fail immediately without attempting the operation
 * - Half-Open: Limited requests are allowed through to test recovery
 *
 * @example
 * ```typescript
 * const breaker = new CircuitBreaker({
 *   failureThreshold: 3,
 *   resetTimeoutMs: 10000,
 *   name: 'api-service',
 *   onStateChange: (from, to) => console.log(`Circuit ${from} -> ${to}`),
 * });
 *
 * try {
 *   const result = await breaker.execute(() => apiService.call());
 * } catch (error) {
 *   if (error instanceof CircuitBreakerOpenError) {
 *     // Circuit is open, use fallback
 *     return fallbackValue;
 *   }
 *   throw error;
 * }
 * ```
 */
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures: FailureRecord[] = [];
  private successCount = 0;
  private openedAt?: number;
  private totalCalls = 0;
  private totalFailures = 0;
  private totalSuccesses = 0;
  private lastFailure?: number;
  private lastSuccess?: number;

  private readonly config: Required<Omit<CircuitBreakerConfig, 'isFailure' | 'onStateChange' | 'onFailure' | 'name'>>;
  private readonly isFailure: (error: unknown) => boolean;
  private readonly onStateChange?: (from: CircuitState, to: CircuitState) => void;
  private readonly onFailure?: (error: unknown, failureCount: number) => void;
  private readonly name: string;

  constructor(config: CircuitBreakerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.isFailure = config.isFailure ?? (() => true);
    this.onStateChange = config.onStateChange;
    this.onFailure = config.onFailure;
    this.name = config.name ?? 'default';
  }

  /**
   * Execute an operation with circuit breaker protection
   *
   * @param operation - Async function to execute
   * @returns Result of the operation
   * @throws CircuitBreakerOpenError if circuit is open
   * @throws Original error if operation fails
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    this.totalCalls++;

    // Clean up old failures outside the window
    this.cleanupOldFailures();

    // Check if circuit should transition from open to half-open
    if (this.state === 'open') {
      const now = Date.now();
      if (this.openedAt && now >= this.openedAt + this.config.resetTimeoutMs) {
        this.transitionTo('half-open');
      } else {
        throw new CircuitBreakerOpenError(
          this.name,
          this.openedAt!,
          this.openedAt! + this.config.resetTimeoutMs
        );
      }
    }

    try {
      const result = await operation();
      this.recordSuccess();
      return result;
    } catch (error) {
      if (this.isFailure(error)) {
        this.recordFailure(error);
      }
      throw error;
    }
  }

  /**
   * Execute with a fallback value when circuit is open
   *
   * @param operation - Async function to execute
   * @param fallback - Fallback value or function when circuit is open
   * @returns Result of operation or fallback
   */
  async executeWithFallback<T>(
    operation: () => Promise<T>,
    fallback: T | (() => T | Promise<T>)
  ): Promise<T> {
    try {
      return await this.execute(operation);
    } catch (error) {
      if (error instanceof CircuitBreakerOpenError) {
        return typeof fallback === 'function' ? await (fallback as () => T | Promise<T>)() : fallback;
      }
      throw error;
    }
  }

  /**
   * Get current circuit breaker statistics
   */
  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failureCount: this.failures.length,
      successCount: this.successCount,
      totalCalls: this.totalCalls,
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
      lastFailure: this.lastFailure,
      lastSuccess: this.lastSuccess,
      openedAt: this.openedAt,
      halfOpenAt: this.state === 'half-open' ? Date.now() : undefined,
      closedAt: this.state === 'closed' ? Date.now() : undefined,
    };
  }

  /**
   * Get current state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Check if circuit is currently open
   */
  isOpen(): boolean {
    this.cleanupOldFailures();

    // Check for automatic transition to half-open
    if (this.state === 'open' && this.openedAt) {
      if (Date.now() >= this.openedAt + this.config.resetTimeoutMs) {
        this.transitionTo('half-open');
      }
    }

    return this.state === 'open';
  }

  /**
   * Manually reset the circuit breaker
   */
  reset(): void {
    this.failures = [];
    this.successCount = 0;
    this.openedAt = undefined;
    this.transitionTo('closed');
  }

  /**
   * Manually trip the circuit breaker
   */
  trip(): void {
    this.transitionTo('open');
    this.openedAt = Date.now();
  }

  private recordSuccess(): void {
    this.totalSuccesses++;
    this.lastSuccess = Date.now();

    if (this.state === 'half-open') {
      this.successCount++;
      if (this.successCount >= this.config.successThreshold) {
        this.transitionTo('closed');
        this.failures = [];
        this.successCount = 0;
      }
    }
  }

  private recordFailure(error: unknown): void {
    const now = Date.now();
    this.totalFailures++;
    this.lastFailure = now;
    this.failures.push({ timestamp: now, error });

    if (this.onFailure) {
      this.onFailure(error, this.failures.length);
    }

    if (this.state === 'half-open') {
      // Any failure in half-open immediately opens the circuit
      this.transitionTo('open');
      this.openedAt = now;
      this.successCount = 0;
    } else if (this.state === 'closed') {
      // Check if failure threshold exceeded
      if (this.failures.length >= this.config.failureThreshold) {
        this.transitionTo('open');
        this.openedAt = now;
      }
    }
  }

  private cleanupOldFailures(): void {
    const cutoff = Date.now() - this.config.failureWindowMs;
    this.failures = this.failures.filter(f => f.timestamp > cutoff);
  }

  private transitionTo(newState: CircuitState): void {
    if (this.state !== newState) {
      const oldState = this.state;
      this.state = newState;

      if (this.onStateChange) {
        this.onStateChange(oldState, newState);
      }
    }
  }
}

/**
 * Create a circuit breaker with pre-configured settings
 */
export const CircuitBreakers = {
  /**
   * Standard circuit breaker - good default
   */
  standard: (name?: string) => new CircuitBreaker({
    name,
    failureThreshold: 5,
    resetTimeoutMs: 30000,
    successThreshold: 2,
  }),

  /**
   * Sensitive circuit breaker - trips quickly
   */
  sensitive: (name?: string) => new CircuitBreaker({
    name,
    failureThreshold: 3,
    resetTimeoutMs: 60000,
    successThreshold: 3,
  }),

  /**
   * Resilient circuit breaker - tolerates more failures
   */
  resilient: (name?: string) => new CircuitBreaker({
    name,
    failureThreshold: 10,
    resetTimeoutMs: 15000,
    successThreshold: 1,
  }),

  /**
   * Aggressive circuit breaker - very fast response
   */
  aggressive: (name?: string) => new CircuitBreaker({
    name,
    failureThreshold: 2,
    resetTimeoutMs: 10000,
    successThreshold: 3,
    failureWindowMs: 30000,
  }),
};

/**
 * Wrap a function with circuit breaker protection
 *
 * @param fn - Function to wrap
 * @param breaker - Circuit breaker instance
 * @returns Wrapped function
 *
 * @example
 * ```typescript
 * const breaker = CircuitBreakers.standard('my-service');
 * const protectedFetch = withCircuitBreaker(
 *   (url: string) => fetch(url).then(r => r.json()),
 *   breaker
 * );
 *
 * const data = await protectedFetch('https://api.example.com/data');
 * ```
 */
export function withCircuitBreaker<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  breaker: CircuitBreaker
): (...args: TArgs) => Promise<TResult> {
  return (...args: TArgs) => breaker.execute(() => fn(...args));
}

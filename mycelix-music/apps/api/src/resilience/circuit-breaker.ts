// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Circuit Breaker Pattern
 *
 * Prevents cascade failures when external services are unavailable.
 * States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
 */

import { getMetrics } from '../metrics';
import { getLogger } from '../logging';

/**
 * Circuit breaker states
 */
export enum CircuitState {
  CLOSED = 'CLOSED',     // Normal operation
  OPEN = 'OPEN',         // Failing, reject requests
  HALF_OPEN = 'HALF_OPEN', // Testing if service recovered
}

/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
  /** Name for identification */
  name: string;
  /** Failure threshold before opening (default: 5) */
  failureThreshold: number;
  /** Success threshold to close from half-open (default: 3) */
  successThreshold: number;
  /** Time in ms before attempting recovery (default: 30000) */
  timeout: number;
  /** Time window for failure counting in ms (default: 60000) */
  failureWindow: number;
  /** Optional fallback function */
  fallback?: <T>() => T | Promise<T>;
  /** Monitor state changes */
  onStateChange?: (from: CircuitState, to: CircuitState) => void;
}

/**
 * Failure record
 */
interface FailureRecord {
  timestamp: number;
  error: string;
}

/**
 * Circuit breaker statistics
 */
export interface CircuitStats {
  name: string;
  state: CircuitState;
  failures: number;
  successes: number;
  totalRequests: number;
  lastFailure?: Date;
  lastSuccess?: Date;
  lastStateChange: Date;
}

/**
 * Circuit Breaker implementation
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: FailureRecord[] = [];
  private successCount = 0;
  private totalRequests = 0;
  private lastFailure?: Date;
  private lastSuccess?: Date;
  private lastStateChange: Date = new Date();
  private openedAt?: number;
  private config: Required<CircuitBreakerConfig>;
  private logger = getLogger();

  constructor(config: CircuitBreakerConfig) {
    this.config = {
      failureThreshold: 5,
      successThreshold: 3,
      timeout: 30000,
      failureWindow: 60000,
      fallback: undefined,
      onStateChange: undefined,
      ...config,
    };

    // Register metrics
    const metrics = getMetrics();
    metrics.createGauge('circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open, 2=half-open)', ['name']);
    metrics.createCounter('circuit_breaker_failures_total', 'Circuit breaker failures', ['name']);
    metrics.createCounter('circuit_breaker_successes_total', 'Circuit breaker successes', ['name']);
    metrics.createCounter('circuit_breaker_rejections_total', 'Circuit breaker rejections', ['name']);
  }

  /**
   * Execute a function with circuit breaker protection
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    this.totalRequests++;

    // Check if circuit is open
    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptRecovery()) {
        this.transitionTo(CircuitState.HALF_OPEN);
      } else {
        return this.handleRejection();
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure(error as Error);
      throw error;
    }
  }

  /**
   * Handle successful execution
   */
  private onSuccess(): void {
    this.lastSuccess = new Date();
    this.successCount++;

    getMetrics().incCounter('circuit_breaker_successes_total', { name: this.config.name });

    if (this.state === CircuitState.HALF_OPEN) {
      if (this.successCount >= this.config.successThreshold) {
        this.transitionTo(CircuitState.CLOSED);
      }
    } else if (this.state === CircuitState.CLOSED) {
      // Reset failure count on success in closed state
      this.clearOldFailures();
    }
  }

  /**
   * Handle failed execution
   */
  private onFailure(error: Error): void {
    this.lastFailure = new Date();
    this.failures.push({
      timestamp: Date.now(),
      error: error.message,
    });

    getMetrics().incCounter('circuit_breaker_failures_total', { name: this.config.name });

    this.logger.warn(`Circuit breaker ${this.config.name} recorded failure`, {
      error: error.message,
      state: this.state,
      failures: this.getRecentFailureCount(),
    });

    if (this.state === CircuitState.HALF_OPEN) {
      // Any failure in half-open goes back to open
      this.transitionTo(CircuitState.OPEN);
    } else if (this.state === CircuitState.CLOSED) {
      this.clearOldFailures();
      if (this.getRecentFailureCount() >= this.config.failureThreshold) {
        this.transitionTo(CircuitState.OPEN);
      }
    }
  }

  /**
   * Handle rejection when circuit is open
   */
  private async handleRejection<T>(): Promise<T> {
    getMetrics().incCounter('circuit_breaker_rejections_total', { name: this.config.name });

    this.logger.debug(`Circuit breaker ${this.config.name} rejected request`, {
      state: this.state,
      openedAt: this.openedAt,
      timeout: this.config.timeout,
    });

    if (this.config.fallback) {
      return this.config.fallback();
    }

    throw new CircuitOpenError(
      `Circuit breaker ${this.config.name} is OPEN`,
      this.config.name,
      this.getRemainingTimeout()
    );
  }

  /**
   * Transition to a new state
   */
  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;
    this.lastStateChange = new Date();

    // Update metrics
    const stateValue = newState === CircuitState.CLOSED ? 0 :
                       newState === CircuitState.OPEN ? 1 : 2;
    getMetrics().setGauge('circuit_breaker_state', stateValue, { name: this.config.name });

    this.logger.info(`Circuit breaker ${this.config.name} state changed`, {
      from: oldState,
      to: newState,
    });

    if (newState === CircuitState.OPEN) {
      this.openedAt = Date.now();
      this.successCount = 0;
    } else if (newState === CircuitState.CLOSED) {
      this.failures = [];
      this.successCount = 0;
      this.openedAt = undefined;
    } else if (newState === CircuitState.HALF_OPEN) {
      this.successCount = 0;
    }

    if (this.config.onStateChange) {
      this.config.onStateChange(oldState, newState);
    }
  }

  /**
   * Check if we should attempt recovery
   */
  private shouldAttemptRecovery(): boolean {
    if (!this.openedAt) return true;
    return Date.now() - this.openedAt >= this.config.timeout;
  }

  /**
   * Get remaining timeout
   */
  private getRemainingTimeout(): number {
    if (!this.openedAt) return 0;
    const elapsed = Date.now() - this.openedAt;
    return Math.max(0, this.config.timeout - elapsed);
  }

  /**
   * Clear old failures outside the window
   */
  private clearOldFailures(): void {
    const cutoff = Date.now() - this.config.failureWindow;
    this.failures = this.failures.filter(f => f.timestamp >= cutoff);
  }

  /**
   * Get recent failure count
   */
  private getRecentFailureCount(): number {
    this.clearOldFailures();
    return this.failures.length;
  }

  /**
   * Get current state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get statistics
   */
  getStats(): CircuitStats {
    return {
      name: this.config.name,
      state: this.state,
      failures: this.getRecentFailureCount(),
      successes: this.successCount,
      totalRequests: this.totalRequests,
      lastFailure: this.lastFailure,
      lastSuccess: this.lastSuccess,
      lastStateChange: this.lastStateChange,
    };
  }

  /**
   * Force circuit open (for testing or manual intervention)
   */
  forceOpen(): void {
    this.transitionTo(CircuitState.OPEN);
  }

  /**
   * Force circuit closed (for testing or manual intervention)
   */
  forceClosed(): void {
    this.transitionTo(CircuitState.CLOSED);
  }

  /**
   * Reset the circuit breaker
   */
  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failures = [];
    this.successCount = 0;
    this.totalRequests = 0;
    this.lastFailure = undefined;
    this.lastSuccess = undefined;
    this.lastStateChange = new Date();
    this.openedAt = undefined;
  }
}

/**
 * Error thrown when circuit is open
 */
export class CircuitOpenError extends Error {
  constructor(
    message: string,
    public readonly circuitName: string,
    public readonly retryAfter: number
  ) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

/**
 * Circuit breaker registry
 */
class CircuitBreakerRegistry {
  private breakers: Map<string, CircuitBreaker> = new Map();

  /**
   * Get or create a circuit breaker
   */
  get(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    let breaker = this.breakers.get(name);
    if (!breaker) {
      breaker = new CircuitBreaker({ name, ...config } as CircuitBreakerConfig);
      this.breakers.set(name, breaker);
    }
    return breaker;
  }

  /**
   * Get all circuit breakers
   */
  getAll(): CircuitBreaker[] {
    return Array.from(this.breakers.values());
  }

  /**
   * Get all statistics
   */
  getAllStats(): CircuitStats[] {
    return this.getAll().map(b => b.getStats());
  }

  /**
   * Reset all breakers
   */
  resetAll(): void {
    for (const breaker of this.breakers.values()) {
      breaker.reset();
    }
  }

  /**
   * Clear registry
   */
  clear(): void {
    this.breakers.clear();
  }
}

/**
 * Global registry
 */
const registry = new CircuitBreakerRegistry();

export function getCircuitBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
  return registry.get(name, config);
}

export function getAllCircuitBreakers(): CircuitBreaker[] {
  return registry.getAll();
}

export function getCircuitBreakerStats(): CircuitStats[] {
  return registry.getAllStats();
}

export function resetAllCircuitBreakers(): void {
  registry.resetAll();
}

/**
 * Decorator for circuit breaker protection
 */
export function withCircuitBreaker(name: string, config?: Partial<CircuitBreakerConfig>) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;
    const breaker = getCircuitBreaker(name, config);

    descriptor.value = async function (...args: any[]) {
      return breaker.execute(() => originalMethod.apply(this, args));
    };

    return descriptor;
  };
}

/**
 * Pre-configured circuit breakers for common services
 */
export const circuitBreakers = {
  ipfs: () => getCircuitBreaker('ipfs', {
    failureThreshold: 3,
    timeout: 60000,
    failureWindow: 120000,
  }),

  blockchain: () => getCircuitBreaker('blockchain', {
    failureThreshold: 5,
    timeout: 30000,
    failureWindow: 60000,
  }),

  external: (name: string) => getCircuitBreaker(`external:${name}`, {
    failureThreshold: 5,
    timeout: 30000,
    failureWindow: 60000,
  }),
};

export default CircuitBreaker;

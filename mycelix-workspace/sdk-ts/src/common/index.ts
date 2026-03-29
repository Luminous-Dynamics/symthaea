// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common SDK Utilities
 *
 * Shared utilities, error handling, and base classes for all Mycelix SDKs.
 *
 * @module @mycelix/sdk/common
 */

// Error handling
export {
  SdkError,
  ConnectionError,
  ZomeError,
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  createDomainError,
  withErrorHandling,
} from './errors';
export type { BaseSdkErrorCode } from './errors';

// Base client utilities
export {
  BaseZomeClient,
  createTimestamp,
  generateId,
} from './base-client';
export type {
  BaseClientConfig,
  ConnectionOptions,
  UnifiedClientBase,
} from './base-client';

// Retry utilities
export {
  withRetry,
  withTimeout,
  withRetryAndTimeout,
  createRetryWrapper,
  RetryPolicy,
  RetryPolicies,
} from './retry';
export type { RetryOptions } from './retry';

// Circuit breaker
export {
  CircuitBreaker,
  CircuitBreakerOpenError,
  CircuitBreakers,
  withCircuitBreaker,
} from './circuit-breaker';
export type {
  CircuitState,
  CircuitBreakerConfig,
  CircuitBreakerStats,
} from './circuit-breaker';

// Cache utilities
export {
  Cache,
  NamespacedCache,
  Caches,
  memoize,
} from './cache';
export type { CacheOptions } from './cache';

// Batch utilities
export {
  RequestBatcher,
  BatchExecutor,
  createBatchedFunction,
  debounce,
  throttle,
} from './batch';
export type { BatchOptions } from './batch';

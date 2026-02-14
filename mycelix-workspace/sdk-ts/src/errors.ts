/**
 * Mycelix SDK Custom Error Types
 *
 * Provides structured error handling with error codes,
 * context information, and recovery suggestions.
 */

// ============================================================================
// Error Codes
// ============================================================================

export enum ErrorCode {
  // General errors (1xxx)
  UNKNOWN = 1000,
  INVALID_ARGUMENT = 1001,
  NOT_FOUND = 1002,
  ALREADY_EXISTS = 1003,
  TIMEOUT = 1004,
  CANCELLED = 1005,

  // MATL errors (2xxx)
  MATL_INVALID_SCORE = 2001,
  MATL_INVALID_QUALITY = 2002,
  MATL_AGENT_NOT_FOUND = 2003,
  MATL_THRESHOLD_OUT_OF_RANGE = 2004,
  MATL_INSUFFICIENT_OBSERVATIONS = 2005,

  // Epistemic errors (3xxx)
  EPISTEMIC_INVALID_LEVEL = 3001,
  EPISTEMIC_INVALID_CODE = 3002,
  EPISTEMIC_CLAIM_EXPIRED = 3003,
  EPISTEMIC_INSUFFICIENT_EVIDENCE = 3004,
  EPISTEMIC_STANDARD_NOT_MET = 3005,

  // Federated Learning errors (4xxx)
  FL_NOT_ENOUGH_PARTICIPANTS = 4001,
  FL_PARTICIPANT_NOT_FOUND = 4002,
  FL_ROUND_NOT_STARTED = 4003,
  FL_INVALID_GRADIENT_SHAPE = 4004,
  FL_VERSION_MISMATCH = 4005,
  FL_AGGREGATION_FAILED = 4006,

  // Bridge errors (5xxx)
  BRIDGE_HAPP_NOT_FOUND = 5001,
  BRIDGE_CONNECTION_FAILED = 5002,
  BRIDGE_INVALID_WEIGHT = 5003,
  BRIDGE_AGGREGATION_FAILED = 5004,

  // Network/Connection errors (6xxx)
  CONNECTION_FAILED = 6001,
  CONNECTION_TIMEOUT = 6002,
  CONNECTION_REFUSED = 6003,
  AUTHENTICATION_FAILED = 6004,
  AUTHORIZATION_FAILED = 6005,
  NETWORK_ERROR = 6006,

  // Recovery errors (7xxx)
  RECOVERY_FAILED = 7001,
  CIRCUIT_OPEN = 7002,

  // Byzantine fault tolerance errors (8xxx)
  BYZANTINE_THRESHOLD_EXCEEDED = 8001,

  // Storage errors (9xxx)
  STORAGE_ERROR = 9000,
  STORAGE_IMMUTABLE = 9001,
  ACCESS_DENIED = 9002,
  INVARIANT_VIOLATION = 9003,
  STORAGE_DUPLICATE_KEY = 9004,
  STORAGE_NOT_FOUND = 9005,
}

// ============================================================================
// Base Error Class
// ============================================================================

export interface MycelixErrorContext {
  [key: string]: unknown;
}

export interface RecoverySuggestion {
  action: string;
  description: string;
}

export class MycelixError extends Error {
  public readonly code: ErrorCode;
  public readonly context: MycelixErrorContext;
  public readonly timestamp: Date;
  public readonly recoverySuggestions: RecoverySuggestion[];
  public readonly cause?: Error;

  constructor(
    message: string,
    code: ErrorCode = ErrorCode.UNKNOWN,
    context: MycelixErrorContext = {},
    cause?: Error
  ) {
    super(message);
    this.name = 'MycelixError';
    this.code = code;
    this.context = context;
    this.timestamp = new Date();
    this.cause = cause;
    this.recoverySuggestions = this.getSuggestions();

    // Capture stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, MycelixError);
    }
  }

  private getSuggestions(): RecoverySuggestion[] {
    const suggestions: Record<ErrorCode, RecoverySuggestion[]> = {
      [ErrorCode.INVALID_ARGUMENT]: [
        { action: 'validate', description: 'Check input parameters match expected types and ranges' },
      ],
      [ErrorCode.NOT_FOUND]: [
        { action: 'create', description: 'Create the resource if it should exist' },
        { action: 'verify', description: 'Verify the identifier is correct' },
      ],
      [ErrorCode.TIMEOUT]: [
        { action: 'retry', description: 'Retry the operation with exponential backoff' },
        { action: 'increase_timeout', description: 'Increase the timeout duration' },
      ],
      [ErrorCode.MATL_INVALID_SCORE]: [
        { action: 'clamp', description: 'Ensure score is between 0 and 1' },
      ],
      [ErrorCode.FL_NOT_ENOUGH_PARTICIPANTS]: [
        { action: 'wait', description: 'Wait for more participants to join' },
        { action: 'reduce_minimum', description: 'Reduce the minimum participant requirement' },
      ],
      [ErrorCode.EPISTEMIC_CLAIM_EXPIRED]: [
        { action: 'renew', description: 'Create a new claim with updated expiration' },
      ],
      [ErrorCode.CONNECTION_FAILED]: [
        { action: 'retry', description: 'Check network connectivity and retry' },
        { action: 'check_endpoint', description: 'Verify the endpoint URL is correct' },
      ],
    } as Record<ErrorCode, RecoverySuggestion[]>;

    return suggestions[this.code] || [];
  }

  toJSON(): object {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      codeString: ErrorCode[this.code],
      context: this.context,
      timestamp: this.timestamp.toISOString(),
      recoverySuggestions: this.recoverySuggestions,
      stack: this.stack,
      cause: this.cause?.message,
    };
  }

  toString(): string {
    let result = `[${ErrorCode[this.code]}] ${this.message}`;

    if (Object.keys(this.context).length > 0) {
      result += `\nContext: ${JSON.stringify(this.context, null, 2)}`;
    }

    if (this.recoverySuggestions.length > 0) {
      result += '\nSuggestions:';
      for (const suggestion of this.recoverySuggestions) {
        result += `\n  - ${suggestion.action}: ${suggestion.description}`;
      }
    }

    return result;
  }
}

// ============================================================================
// Specialized Error Classes
// ============================================================================

export class ValidationError extends MycelixError {
  public readonly field: string;
  public readonly constraint: string;

  constructor(
    field: string,
    constraint: string,
    value: unknown,
    cause?: Error
  ) {
    super(
      `Validation failed for '${field}': ${constraint}`,
      ErrorCode.INVALID_ARGUMENT,
      { field, constraint, value },
      cause
    );
    this.name = 'ValidationError';
    this.field = field;
    this.constraint = constraint;
  }
}

export class MATLError extends MycelixError {
  constructor(
    message: string,
    code: ErrorCode,
    context: MycelixErrorContext = {},
    cause?: Error
  ) {
    super(message, code, context, cause);
    this.name = 'MATLError';
  }
}

export class EpistemicError extends MycelixError {
  constructor(
    message: string,
    code: ErrorCode,
    context: MycelixErrorContext = {},
    cause?: Error
  ) {
    super(message, code, context, cause);
    this.name = 'EpistemicError';
  }
}

export class FederatedLearningError extends MycelixError {
  constructor(
    message: string,
    code: ErrorCode,
    context: MycelixErrorContext = {},
    cause?: Error
  ) {
    super(message, code, context, cause);
    this.name = 'FederatedLearningError';
  }
}

export class BridgeError extends MycelixError {
  constructor(
    message: string,
    code: ErrorCode,
    context: MycelixErrorContext = {},
    cause?: Error
  ) {
    super(message, code, context, cause);
    this.name = 'BridgeError';
  }
}

export class ConnectionError extends MycelixError {
  public readonly endpoint?: string;
  public readonly retryable: boolean;

  constructor(
    message: string,
    code: ErrorCode,
    endpoint?: string,
    retryable: boolean = true,
    cause?: Error
  ) {
    super(message, code, { endpoint, retryable }, cause);
    this.name = 'ConnectionError';
    this.endpoint = endpoint;
    this.retryable = retryable;
  }
}

// ============================================================================
// Validation Utilities
// ============================================================================

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

export class Validator {
  private errors: ValidationError[] = [];

  /**
   * Validate that a value is defined
   */
  required(field: string, value: unknown): this {
    if (value === undefined || value === null) {
      this.errors.push(new ValidationError(field, 'is required', value));
    }
    return this;
  }

  /**
   * Validate that a number is in range [min, max]
   */
  inRange(field: string, value: number, min: number, max: number): this {
    if (typeof value !== 'number' || isNaN(value)) {
      this.errors.push(new ValidationError(field, 'must be a number', value));
    } else if (value < min || value > max) {
      this.errors.push(
        new ValidationError(field, `must be between ${min} and ${max}`, value)
      );
    }
    return this;
  }

  /**
   * Validate that a number is positive
   */
  positive(field: string, value: number): this {
    if (typeof value !== 'number' || value <= 0) {
      this.errors.push(new ValidationError(field, 'must be positive', value));
    }
    return this;
  }

  /**
   * Validate that a number is non-negative
   */
  nonNegative(field: string, value: number): this {
    if (typeof value !== 'number' || value < 0) {
      this.errors.push(new ValidationError(field, 'must be non-negative', value));
    }
    return this;
  }

  /**
   * Validate that a string is not empty
   */
  notEmpty(field: string, value: string): this {
    if (typeof value !== 'string' || value.trim() === '') {
      this.errors.push(new ValidationError(field, 'must not be empty', value));
    }
    return this;
  }

  /**
   * Validate that a string matches a pattern
   */
  pattern(field: string, value: string, regex: RegExp, description?: string): this {
    if (typeof value !== 'string' || !regex.test(value)) {
      this.errors.push(
        new ValidationError(
          field,
          description || `must match pattern ${regex}`,
          value
        )
      );
    }
    return this;
  }

  /**
   * Validate that an array has minimum length
   */
  minLength(field: string, value: unknown[], min: number): this {
    if (!Array.isArray(value) || value.length < min) {
      this.errors.push(
        new ValidationError(field, `must have at least ${min} items`, value)
      );
    }
    return this;
  }

  /**
   * Validate with a custom function
   */
  custom(field: string, value: unknown, fn: (v: unknown) => boolean, message: string): this {
    if (!fn(value)) {
      this.errors.push(new ValidationError(field, message, value));
    }
    return this;
  }

  /**
   * Get the validation result
   */
  result(): ValidationResult {
    return {
      valid: this.errors.length === 0,
      errors: [...this.errors],
    };
  }

  /**
   * Throw if validation failed
   */
  throwIfInvalid(): void {
    if (this.errors.length > 0) {
      const messages = this.errors.map(e => e.message).join('; ');
      throw new MycelixError(
        `Validation failed: ${messages}`,
        ErrorCode.INVALID_ARGUMENT,
        { errors: this.errors.map(e => e.toJSON()) }
      );
    }
  }

  /**
   * Reset the validator for reuse
   */
  reset(): this {
    this.errors = [];
    return this;
  }
}

/**
 * Create a new validator instance
 */
export function validate(): Validator {
  return new Validator();
}

// ============================================================================
// Error Handling Utilities
// ============================================================================

/**
 * Wrap a function with error handling
 */
export function withErrorHandling<T extends (...args: any[]) => any>(
  fn: T,
  context: string
): T {
  return ((...args: Parameters<T>): ReturnType<T> => {
    try {
      const result = fn(...args);

      // Handle async functions
      if (result instanceof Promise) {
        return result.catch((error: unknown) => {
          if (error instanceof MycelixError) {
            throw error;
          }
          throw new MycelixError(
            `${context}: ${error instanceof Error ? error.message : String(error)}`,
            ErrorCode.UNKNOWN,
            { originalError: String(error) },
            error instanceof Error ? error : undefined
          );
        }) as ReturnType<T>;
      }

      return result;
    } catch (error) {
      if (error instanceof MycelixError) {
        throw error;
      }
      throw new MycelixError(
        `${context}: ${error instanceof Error ? error.message : String(error)}`,
        ErrorCode.UNKNOWN,
        { originalError: String(error) },
        error instanceof Error ? error : undefined
      );
    }
  }) as T;
}

/**
 * Retry a function with exponential backoff
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffFactor?: number;
    retryOn?: (error: unknown) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 100,
    maxDelay = 5000,
    backoffFactor = 2,
    retryOn = () => true,
  } = options;

  let lastError: unknown;
  let delay = initialDelay;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries || !retryOn(error)) {
        break;
      }

      await new Promise(resolve => setTimeout(resolve, delay));
      delay = Math.min(delay * backoffFactor, maxDelay);
    }
  }

  if (lastError instanceof MycelixError) {
    throw lastError;
  }

  throw new MycelixError(
    `Operation failed after ${maxRetries + 1} attempts`,
    ErrorCode.TIMEOUT,
    { attempts: maxRetries + 1, lastError: String(lastError) },
    lastError instanceof Error ? lastError : undefined
  );
}

/**
 * Assert a condition, throwing a ValidationError if false
 */
export function assert(
  condition: boolean,
  field: string,
  message: string
): asserts condition {
  if (!condition) {
    throw new ValidationError(field, message, undefined);
  }
}

/**
 * Assert a value is not null/undefined
 */
export function assertDefined<T>(
  value: T | null | undefined,
  field: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new ValidationError(field, 'must be defined', value);
  }
}

// ============================================================================
// Recovery Strategies
// ============================================================================

/**
 * Result of a recovery attempt
 */
export interface RecoveryResult<T> {
  success: boolean;
  value?: T;
  error?: Error;
  attempts: number;
  strategyUsed: string;
  duration: number;
}

/**
 * Configuration for recovery strategies
 */
export interface RecoveryConfig {
  /** Maximum total time to spend on recovery (ms) */
  maxDuration?: number;
  /** Whether to log recovery attempts */
  logAttempts?: boolean;
  /** Custom logger function */
  logger?: (message: string, context: Record<string, unknown>) => void;
}

/**
 * A recovery strategy that can handle errors
 */
export interface RecoveryStrategy<T> {
  name: string;
  /** Returns true if this strategy can handle the error */
  canHandle: (error: unknown) => boolean;
  /** Attempt to recover from the error */
  recover: (
    error: unknown,
    operation: () => Promise<T>,
    attemptNumber: number
  ) => Promise<T>;
}

/**
 * Creates an exponential backoff recovery strategy
 */
export function exponentialBackoffStrategy<T>(options: {
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  backoffFactor?: number;
  jitter?: boolean;
  retryableErrors?: ErrorCode[];
} = {}): RecoveryStrategy<T> {
  const {
    maxRetries = 3,
    initialDelayMs = 100,
    maxDelayMs = 10000,
    backoffFactor = 2,
    jitter = true,
    retryableErrors = [
      ErrorCode.TIMEOUT,
      ErrorCode.CONNECTION_FAILED,
      ErrorCode.NETWORK_ERROR,
    ],
  } = options;

  return {
    name: 'exponentialBackoff',
    canHandle: (error) => {
      if (error instanceof MycelixError) {
        return retryableErrors.includes(error.code);
      }
      return true; // Retry unknown errors by default
    },
    recover: async (error, operation, attemptNumber) => {
      if (attemptNumber > maxRetries) {
        throw error;
      }

      let delay = initialDelayMs * Math.pow(backoffFactor, attemptNumber - 1);
      delay = Math.min(delay, maxDelayMs);

      if (jitter) {
        delay = delay * (0.5 + Math.random() * 0.5);
      }

      await new Promise((resolve) => setTimeout(resolve, delay));
      return operation();
    },
  };
}

/**
 * Creates a fallback chain strategy that tries alternative operations
 */
export function fallbackChainStrategy<T>(
  fallbacks: Array<() => Promise<T>>,
  options: {
    name?: string;
  } = {}
): RecoveryStrategy<T> {
  let fallbackIndex = 0;

  return {
    name: options.name ?? 'fallbackChain',
    canHandle: () => fallbackIndex < fallbacks.length,
    recover: async () => {
      const fallback = fallbacks[fallbackIndex];
      fallbackIndex++;
      return fallback();
    },
  };
}

/**
 * Circuit breaker state
 */
export type CircuitState = 'closed' | 'open' | 'half-open';

/**
 * Creates a circuit breaker strategy to prevent cascading failures
 */
export function circuitBreakerStrategy<T>(options: {
  failureThreshold?: number;
  resetTimeoutMs?: number;
  halfOpenRequests?: number;
} = {}): RecoveryStrategy<T> & { getState: () => CircuitState; reset: () => void } {
  const {
    failureThreshold = 5,
    resetTimeoutMs = 30000,
    halfOpenRequests = 1,
  } = options;

  let state: CircuitState = 'closed';
  let failures = 0;
  let lastFailureTime = 0;
  let halfOpenAttempts = 0;

  const strategy: RecoveryStrategy<T> & { getState: () => CircuitState; reset: () => void } = {
    name: 'circuitBreaker',
    canHandle: () => {
      const now = Date.now();

      if (state === 'open') {
        if (now - lastFailureTime >= resetTimeoutMs) {
          state = 'half-open';
          halfOpenAttempts = 0;
          return true;
        }
        return false; // Circuit is open, don't retry
      }

      if (state === 'half-open') {
        return halfOpenAttempts < halfOpenRequests;
      }

      return true;
    },
    recover: async (_error, operation) => {
      if (state === 'open') {
        throw new MycelixError(
          'Circuit breaker is open',
          ErrorCode.CIRCUIT_OPEN,
          { failures, resetTimeoutMs, timeSinceLastFailure: Date.now() - lastFailureTime }
        );
      }

      if (state === 'half-open') {
        halfOpenAttempts++;
      }

      try {
        const result = await operation();

        // Success - reset circuit
        if (state === 'half-open') {
          state = 'closed';
          failures = 0;
        }

        return result;
      } catch (err) {
        failures++;
        lastFailureTime = Date.now();

        if (failures >= failureThreshold) {
          state = 'open';
        }

        throw err;
      }
    },
    getState: () => state,
    reset: () => {
      state = 'closed';
      failures = 0;
      lastFailureTime = 0;
      halfOpenAttempts = 0;
    },
  };

  return strategy;
}

/**
 * Creates a grace period strategy that allows temporary failures
 */
export function gracePeriodStrategy<T>(options: {
  gracePeriodMs?: number;
  defaultValue?: T;
  errorCodes?: ErrorCode[];
} = {}): RecoveryStrategy<T> {
  const {
    gracePeriodMs = 5000,
    defaultValue,
    errorCodes = [ErrorCode.NOT_FOUND, ErrorCode.BYZANTINE_THRESHOLD_EXCEEDED],
  } = options;

  let firstErrorTime: number | null = null;

  return {
    name: 'gracePeriod',
    canHandle: (error) => {
      if (!(error instanceof MycelixError)) return false;
      return errorCodes.includes(error.code);
    },
    recover: async (error, operation) => {
      const now = Date.now();

      if (firstErrorTime === null) {
        firstErrorTime = now;
      }

      if (now - firstErrorTime <= gracePeriodMs) {
        if (defaultValue !== undefined) {
          return defaultValue;
        }
        // Wait a bit and retry
        await new Promise((resolve) => setTimeout(resolve, 100));
        return operation();
      }

      // Grace period expired, propagate error
      firstErrorTime = null;
      throw error;
    },
  };
}

/**
 * Combines multiple recovery strategies
 */
export function combineStrategies<T>(
  strategies: RecoveryStrategy<T>[]
): RecoveryStrategy<T> {
  return {
    name: 'combined(' + strategies.map((s) => s.name).join(', ') + ')',
    canHandle: (error) => strategies.some((s) => s.canHandle(error)),
    recover: async (error, operation, attemptNumber) => {
      for (const strategy of strategies) {
        if (strategy.canHandle(error)) {
          return strategy.recover(error, operation, attemptNumber);
        }
      }
      throw error;
    },
  };
}

/**
 * Execute an operation with recovery strategies.
 * Attempts recovery using the provided strategies when errors occur.
 *
 * @example
 * ```typescript
 * const result = await withRecovery(
 *   async () => fetchData(),
 *   [
 *     exponentialBackoffStrategy({ maxRetries: 3 }),
 *     fallbackChainStrategy([fetchFromCache, fetchFromBackup]),
 *   ],
 *   { maxDuration: 30000 }
 * );
 * ```
 */
export async function withRecovery<T>(
  operation: () => Promise<T>,
  strategies: RecoveryStrategy<T>[],
  config: RecoveryConfig = {}
): Promise<RecoveryResult<T>> {
  const { maxDuration = 60000, logAttempts = false, logger = console.log } = config;

  const startTime = Date.now();
  let attempts = 0;
  let lastError: Error | undefined;
  let strategyUsed = 'none';

  const log = (message: string, context: Record<string, unknown> = {}) => {
    if (logAttempts) {
      logger(message, { ...context, attempts, elapsed: Date.now() - startTime });
    }
  };

  while (Date.now() - startTime < maxDuration) {
    attempts++;

    try {
      log('Attempting operation', { attempt: attempts });
      const value = await operation();

      return {
        success: true,
        value,
        attempts,
        strategyUsed,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      log('Operation failed', { error: lastError.message });

      // Find a strategy that can handle this error
      let recovered = false;
      for (const strategy of strategies) {
        if (strategy.canHandle(error)) {
          strategyUsed = strategy.name;
          log('Applying recovery strategy', { strategy: strategy.name });

          try {
            const value = await strategy.recover(error, operation, attempts);
            return {
              success: true,
              value,
              attempts,
              strategyUsed,
              duration: Date.now() - startTime,
            };
          } catch (recoveryError) {
            lastError = recoveryError instanceof Error
              ? recoveryError
              : new Error(String(recoveryError));
            recovered = true;
            break; // Strategy failed, try operation again
          }
        }
      }

      if (!recovered) {
        // No strategy could handle this error
        break;
      }
    }
  }

  return {
    success: false,
    error: lastError,
    attempts,
    strategyUsed,
    duration: Date.now() - startTime,
  };
}

/**
 * Execute an operation with recovery, throwing on failure.
 * Convenience wrapper around withRecovery that throws if recovery fails.
 */
export async function withRecoveryOrThrow<T>(
  operation: () => Promise<T>,
  strategies: RecoveryStrategy<T>[],
  config: RecoveryConfig = {}
): Promise<T> {
  const result = await withRecovery(operation, strategies, config);

  if (!result.success) {
    throw result.error ?? new MycelixError(
      `Operation failed after ${result.attempts} attempts`,
      ErrorCode.RECOVERY_FAILED,
      {
        attempts: result.attempts,
        duration: result.duration,
        strategyUsed: result.strategyUsed,
      }
    );
  }

  return result.value as T;
}

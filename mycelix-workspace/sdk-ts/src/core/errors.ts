/**
 * Core SDK Error Handling
 *
 * Standardized error types and codes for the unified Mycelix client.
 * Provides consistent error handling across all hApp clients.
 *
 * @module @mycelix/sdk/core/errors
 */

/**
 * Standardized SDK error codes
 *
 * Provides a consistent set of error codes across all Mycelix SDK operations.
 */
export enum SdkErrorCode {
  // Connection errors
  NETWORK_ERROR = 'NETWORK_ERROR',
  CONNECTION_FAILED = 'CONNECTION_FAILED',
  CONNECTION_CLOSED = 'CONNECTION_CLOSED',
  TIMEOUT = 'TIMEOUT',

  // Validation errors
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  INVALID_INPUT = 'INVALID_INPUT',
  INVALID_RESPONSE = 'INVALID_RESPONSE',
  SCHEMA_MISMATCH = 'SCHEMA_MISMATCH',

  // Resource errors
  NOT_FOUND = 'NOT_FOUND',
  ALREADY_EXISTS = 'ALREADY_EXISTS',
  CONFLICT = 'CONFLICT',

  // Authorization errors
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN',
  CAPABILITY_DENIED = 'CAPABILITY_DENIED',

  // Zome errors
  ZOME_CALL_FAILED = 'ZOME_CALL_FAILED',
  ZOME_NOT_FOUND = 'ZOME_NOT_FOUND',
  FUNCTION_NOT_FOUND = 'FUNCTION_NOT_FOUND',

  // Client errors
  CLIENT_NOT_CONNECTED = 'CLIENT_NOT_CONNECTED',
  CLIENT_ALREADY_CONNECTED = 'CLIENT_ALREADY_CONNECTED',
  INITIALIZATION_FAILED = 'INITIALIZATION_FAILED',

  // Internal errors
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

/**
 * Determine if an error code is retryable
 */
export function isRetryableCode(code: SdkErrorCode): boolean {
  const retryableCodes: SdkErrorCode[] = [
    SdkErrorCode.NETWORK_ERROR,
    SdkErrorCode.CONNECTION_FAILED,
    SdkErrorCode.CONNECTION_CLOSED,
    SdkErrorCode.TIMEOUT,
    SdkErrorCode.ZOME_CALL_FAILED,
  ];
  return retryableCodes.includes(code);
}

/**
 * Additional error details
 */
export interface SdkErrorDetails {
  /** Original error that caused this error */
  cause?: unknown;
  /** Zome name if applicable */
  zomeName?: string;
  /** Function name if applicable */
  fnName?: string;
  /** Payload that caused the error */
  payload?: unknown;
  /** URL or endpoint involved */
  url?: string;
  /** HTTP status code if applicable */
  statusCode?: number;
  /** Additional context-specific data */
  [key: string]: unknown;
}

/**
 * Standardized SDK Error
 *
 * Base error class for all Mycelix SDK errors. Provides consistent
 * structure for error handling, logging, and debugging.
 *
 * @example
 * ```typescript
 * throw new SdkError(
 *   SdkErrorCode.VALIDATION_ERROR,
 *   'Invalid DID format',
 *   { field: 'did', value: invalidDid }
 * );
 * ```
 */
export class SdkError extends Error {
  /** Error code for programmatic handling */
  public readonly code: SdkErrorCode;

  /** Additional error details */
  public readonly details: SdkErrorDetails;

  /** Timestamp when error occurred */
  public readonly timestamp: number;

  /** Stack trace from original error if available */
  public readonly causeStack?: string;

  constructor(
    code: SdkErrorCode,
    message: string,
    details: SdkErrorDetails = {}
  ) {
    super(message);
    this.name = 'SdkError';
    this.code = code;
    this.details = details;
    this.timestamp = Date.now();

    // Preserve cause stack trace
    if (details.cause instanceof Error) {
      this.causeStack = details.cause.stack;
    }

    // Maintain proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Check if this error is retryable
   */
  isRetryable(): boolean {
    return isRetryableCode(this.code);
  }

  /**
   * Check if this error matches a specific code
   */
  isCode(code: SdkErrorCode): boolean {
    return this.code === code;
  }

  /**
   * Check if this is a connection-related error
   */
  isConnectionError(): boolean {
    return [
      SdkErrorCode.NETWORK_ERROR,
      SdkErrorCode.CONNECTION_FAILED,
      SdkErrorCode.CONNECTION_CLOSED,
      SdkErrorCode.TIMEOUT,
    ].includes(this.code);
  }

  /**
   * Check if this is an authorization-related error
   */
  isAuthError(): boolean {
    return [
      SdkErrorCode.UNAUTHORIZED,
      SdkErrorCode.FORBIDDEN,
      SdkErrorCode.CAPABILITY_DENIED,
    ].includes(this.code);
  }

  /**
   * Convert to JSON for logging/serialization
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      details: this.sanitizeDetails(),
      timestamp: this.timestamp,
      stack: this.stack,
      causeStack: this.causeStack,
    };
  }

  /**
   * Get a formatted string representation
   */
  toString(): string {
    return `[${this.code}] ${this.message}`;
  }

  /**
   * Sanitize details for safe serialization
   */
  private sanitizeDetails(): Record<string, unknown> {
    const sanitized: Record<string, unknown> = { ...this.details };

    // Remove cause object (already captured in causeStack)
    delete sanitized.cause;

    // Truncate large payloads
    if (sanitized.payload && JSON.stringify(sanitized.payload).length > 1000) {
      sanitized.payload = '[truncated]';
    }

    return sanitized;
  }
}

/**
 * Network error for connectivity issues
 */
export class NetworkError extends SdkError {
  constructor(message: string, details: SdkErrorDetails = {}) {
    super(SdkErrorCode.NETWORK_ERROR, message, details);
    this.name = 'NetworkError';
  }
}

/**
 * Validation error for invalid inputs
 */
export class ValidationError extends SdkError {
  constructor(message: string, details: SdkErrorDetails = {}) {
    super(SdkErrorCode.VALIDATION_ERROR, message, details);
    this.name = 'ValidationError';
  }
}

/**
 * Not found error for missing resources
 */
export class NotFoundError extends SdkError {
  constructor(
    resourceType: string,
    resourceId: string,
    details: SdkErrorDetails = {}
  ) {
    super(
      SdkErrorCode.NOT_FOUND,
      `${resourceType} not found: ${resourceId}`,
      { resourceType, resourceId, ...details }
    );
    this.name = 'NotFoundError';
  }
}

/**
 * Timeout error for operations that exceed time limits
 */
export class TimeoutError extends SdkError {
  constructor(
    operation: string,
    timeoutMs: number,
    details: SdkErrorDetails = {}
  ) {
    super(
      SdkErrorCode.TIMEOUT,
      `Operation '${operation}' timed out after ${timeoutMs}ms`,
      { operation, timeoutMs, ...details }
    );
    this.name = 'TimeoutError';
  }
}

/**
 * Unauthorized error for permission issues
 */
export class UnauthorizedError extends SdkError {
  constructor(message: string, details: SdkErrorDetails = {}) {
    super(SdkErrorCode.UNAUTHORIZED, message, details);
    this.name = 'UnauthorizedError';
  }
}

/**
 * Zome call error for Holochain zome failures
 */
export class ZomeCallError extends SdkError {
  constructor(
    zomeName: string,
    fnName: string,
    message: string,
    details: SdkErrorDetails = {}
  ) {
    super(
      SdkErrorCode.ZOME_CALL_FAILED,
      `${zomeName}.${fnName}: ${message}`,
      { zomeName, fnName, ...details }
    );
    this.name = 'ZomeCallError';
  }
}

/**
 * Wrap an error as an SdkError
 *
 * Converts any error into a standardized SdkError while preserving
 * the original error information.
 *
 * @param error - Original error
 * @param context - Additional context to add
 * @returns SdkError instance
 */
export function wrapError(
  error: unknown,
  context: { code?: SdkErrorCode; message?: string } = {}
): SdkError {
  if (error instanceof SdkError) {
    return error;
  }

  const message =
    context.message ??
    (error instanceof Error ? error.message : String(error));

  const code = context.code ?? detectErrorCode(error);

  return new SdkError(code, message, { cause: error });
}

/**
 * Attempt to detect the appropriate error code from an error
 */
function detectErrorCode(error: unknown): SdkErrorCode {
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();

    if (msg.includes('timeout') || msg.includes('timed out')) {
      return SdkErrorCode.TIMEOUT;
    }
    if (
      msg.includes('network') ||
      msg.includes('econnrefused') ||
      msg.includes('econnreset')
    ) {
      return SdkErrorCode.NETWORK_ERROR;
    }
    if (msg.includes('not found') || msg.includes('404')) {
      return SdkErrorCode.NOT_FOUND;
    }
    if (msg.includes('unauthorized') || msg.includes('401')) {
      return SdkErrorCode.UNAUTHORIZED;
    }
    if (msg.includes('forbidden') || msg.includes('403')) {
      return SdkErrorCode.FORBIDDEN;
    }
    if (msg.includes('validation') || msg.includes('invalid')) {
      return SdkErrorCode.VALIDATION_ERROR;
    }
  }

  return SdkErrorCode.UNKNOWN_ERROR;
}

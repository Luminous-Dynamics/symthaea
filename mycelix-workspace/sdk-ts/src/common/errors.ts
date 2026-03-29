// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common SDK Error Classes
 *
 * Base error handling for all Mycelix SDK domains.
 *
 * @module @mycelix/sdk/common/errors
 */

/**
 * Base error codes common to all SDKs
 */
export type BaseSdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'TIMEOUT'
  | 'NETWORK_ERROR'
  | 'UNKNOWN_ERROR';

/**
 * Base SDK error class
 *
 * All domain-specific errors should extend this class.
 */
export class SdkError extends Error {
  public readonly timestamp: number;

  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown,
    public readonly domain?: string
  ) {
    super(message);
    this.name = domain ? `${domain}SdkError` : 'SdkError';
    this.timestamp = Date.now();

    // Maintain proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Create a formatted error message
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      domain: this.domain,
      timestamp: this.timestamp,
      cause: this.cause instanceof Error ? this.cause.message : this.cause,
    };
  }

  /**
   * Check if error is of a specific code
   */
  isCode(code: string): boolean {
    return this.code === code;
  }

  /**
   * Check if error is a connection error
   */
  isConnectionError(): boolean {
    return this.code === 'CONNECTION_ERROR' || this.code === 'NETWORK_ERROR';
  }

  /**
   * Check if error is retryable
   */
  isRetryable(): boolean {
    return ['CONNECTION_ERROR', 'TIMEOUT', 'NETWORK_ERROR'].includes(this.code);
  }
}

/**
 * Connection error for WebSocket failures
 */
export class ConnectionError extends SdkError {
  constructor(message: string, cause?: unknown) {
    super('CONNECTION_ERROR', message, cause);
    this.name = 'ConnectionError';
  }
}

/**
 * Zome call error for Holochain zome failures
 */
export class ZomeError extends SdkError {
  constructor(
    public readonly zomeName: string,
    public readonly fnName: string,
    message: string,
    cause?: unknown
  ) {
    super('ZOME_ERROR', `${zomeName}.${fnName}: ${message}`, cause);
    this.name = 'ZomeError';
  }
}

/**
 * Validation error for invalid inputs
 */
export class ValidationError extends SdkError {
  constructor(
    public readonly field: string,
    message: string,
    public readonly value?: unknown
  ) {
    super('INVALID_INPUT', `${field}: ${message}`);
    this.name = 'ValidationError';
  }
}

/**
 * Not found error for missing resources
 */
export class NotFoundError extends SdkError {
  constructor(
    public readonly resourceType: string,
    public readonly resourceId: string
  ) {
    super('NOT_FOUND', `${resourceType} not found: ${resourceId}`);
    this.name = 'NotFoundError';
  }
}

/**
 * Unauthorized error for permission failures
 */
export class UnauthorizedError extends SdkError {
  constructor(message: string, public readonly requiredPermission?: string) {
    super('UNAUTHORIZED', message);
    this.name = 'UnauthorizedError';
  }
}

/**
 * Factory function to create domain-specific error classes
 */
export function createDomainError(domain: string) {
  return class DomainSdkError extends SdkError {
    constructor(code: string, message: string, cause?: unknown) {
      super(code, message, cause, domain);
    }
  };
}

/**
 * Helper to wrap async operations with error handling
 */
export async function withErrorHandling<T>(
  operation: () => Promise<T>,
  context: { domain?: string; operation?: string } = {}
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (error instanceof SdkError) {
      throw error;
    }
    throw new SdkError(
      'UNKNOWN_ERROR',
      `${context.operation || 'Operation'} failed: ${error instanceof Error ? error.message : String(error)}`,
      error,
      context.domain
    );
  }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Response Utilities
 *
 * Provides a consistent response envelope for all API responses.
 * Ensures predictable structure for clients consuming the API.
 *
 * Response Structure:
 * {
 *   success: boolean,
 *   data?: T,
 *   error?: { code: string, message: string, details?: unknown },
 *   meta?: { pagination?, timing?, version? }
 * }
 */

import { Response } from 'express';

/**
 * Standard error codes for API responses
 */
export const ErrorCodes = {
  // Client errors (4xx)
  BAD_REQUEST: 'BAD_REQUEST',
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  CONFLICT: 'CONFLICT',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  RATE_LIMITED: 'RATE_LIMITED',

  // Server errors (5xx)
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  DATABASE_ERROR: 'DATABASE_ERROR',
  EXTERNAL_SERVICE_ERROR: 'EXTERNAL_SERVICE_ERROR',
} as const;

export type ErrorCode = (typeof ErrorCodes)[keyof typeof ErrorCodes];

/**
 * Pagination metadata
 */
export interface PaginationMeta {
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
  nextCursor?: string;
}

/**
 * Response metadata
 */
export interface ResponseMeta {
  pagination?: PaginationMeta;
  timing?: {
    startedAt: string;
    duration: number;
  };
  version?: string;
  requestId?: string;
}

/**
 * Error details structure
 */
export interface ErrorDetails {
  code: ErrorCode;
  message: string;
  details?: unknown;
  stack?: string;
}

/**
 * Standard API response envelope
 */
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ErrorDetails;
  meta?: ResponseMeta;
}

/**
 * Response builder for chainable response construction
 */
export class ResponseBuilder<T = unknown> {
  private response: ApiResponse<T> = { success: true };
  private statusCode = 200;
  private startTime: number;

  constructor() {
    this.startTime = Date.now();
  }

  /**
   * Set response data
   */
  data(data: T): this {
    this.response.data = data;
    return this;
  }

  /**
   * Set pagination metadata
   */
  pagination(meta: PaginationMeta): this {
    this.response.meta = {
      ...this.response.meta,
      pagination: meta,
    };
    return this;
  }

  /**
   * Set custom metadata
   */
  meta(meta: Partial<ResponseMeta>): this {
    this.response.meta = {
      ...this.response.meta,
      ...meta,
    };
    return this;
  }

  /**
   * Set error details
   */
  error(code: ErrorCode, message: string, details?: unknown): this {
    this.response.success = false;
    this.response.error = {
      code,
      message,
      details,
    };
    delete this.response.data;
    return this;
  }

  /**
   * Set HTTP status code
   */
  status(code: number): this {
    this.statusCode = code;
    return this;
  }

  /**
   * Build and send response
   */
  send(res: Response): Response {
    // Add timing info
    this.response.meta = {
      ...this.response.meta,
      timing: {
        startedAt: new Date(this.startTime).toISOString(),
        duration: Date.now() - this.startTime,
      },
    };

    // Add request ID if available
    const requestId = res.getHeader('X-Request-ID');
    if (requestId) {
      this.response.meta.requestId = String(requestId);
    }

    // Add API version if available
    const version = res.getHeader('X-API-Version');
    if (version) {
      this.response.meta.version = String(version);
    }

    return res.status(this.statusCode).json(this.response);
  }

  /**
   * Get the built response object (for testing)
   */
  build(): ApiResponse<T> {
    return { ...this.response };
  }
}

/**
 * Create a new response builder
 */
export function response<T = unknown>(): ResponseBuilder<T> {
  return new ResponseBuilder<T>();
}

/**
 * Quick success response
 */
export function success<T>(res: Response, data: T, statusCode = 200): Response {
  return response<T>()
    .data(data)
    .status(statusCode)
    .send(res);
}

/**
 * Quick success response with pagination
 */
export function paginated<T>(
  res: Response,
  data: T[],
  pagination: PaginationMeta,
  statusCode = 200
): Response {
  return response<T[]>()
    .data(data)
    .pagination(pagination)
    .status(statusCode)
    .send(res);
}

/**
 * Quick created response (201)
 */
export function created<T>(res: Response, data: T): Response {
  return success(res, data, 201);
}

/**
 * Quick no content response (204)
 */
export function noContent(res: Response): Response {
  return res.status(204).send();
}

/**
 * Quick error response
 */
export function error(
  res: Response,
  code: ErrorCode,
  message: string,
  statusCode = 500,
  details?: unknown
): Response {
  return response()
    .error(code, message, details)
    .status(statusCode)
    .send(res);
}

/**
 * Common error responses
 */
export const errors = {
  badRequest: (res: Response, message = 'Bad request', details?: unknown) =>
    error(res, ErrorCodes.BAD_REQUEST, message, 400, details),

  unauthorized: (res: Response, message = 'Unauthorized') =>
    error(res, ErrorCodes.UNAUTHORIZED, message, 401),

  forbidden: (res: Response, message = 'Forbidden') =>
    error(res, ErrorCodes.FORBIDDEN, message, 403),

  notFound: (res: Response, resource = 'Resource') =>
    error(res, ErrorCodes.NOT_FOUND, `${resource} not found`, 404),

  conflict: (res: Response, message = 'Resource already exists') =>
    error(res, ErrorCodes.CONFLICT, message, 409),

  validationError: (res: Response, details: unknown) =>
    error(res, ErrorCodes.VALIDATION_ERROR, 'Validation failed', 422, details),

  rateLimited: (res: Response, retryAfter?: number) => {
    if (retryAfter) {
      res.setHeader('Retry-After', retryAfter);
    }
    return error(res, ErrorCodes.RATE_LIMITED, 'Too many requests', 429);
  },

  internalError: (res: Response, message = 'Internal server error') =>
    error(res, ErrorCodes.INTERNAL_ERROR, message, 500),

  serviceUnavailable: (res: Response, message = 'Service unavailable') =>
    error(res, ErrorCodes.SERVICE_UNAVAILABLE, message, 503),
};

/**
 * Type guard for checking if response is an error
 */
export function isErrorResponse(response: ApiResponse): response is ApiResponse & { error: ErrorDetails } {
  return !response.success && response.error !== undefined;
}

/**
 * Type guard for checking if response has pagination
 */
export function isPaginatedResponse<T>(
  response: ApiResponse<T[]>
): response is ApiResponse<T[]> & { meta: { pagination: PaginationMeta } } {
  return response.meta?.pagination !== undefined;
}

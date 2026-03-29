// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Error Handling Middleware
 *
 * Provides centralized error handling with:
 * - Consistent error response format
 * - Error classification and appropriate status codes
 * - Logging integration
 * - Stack trace hiding in production
 */

import { Request, Response, NextFunction, ErrorRequestHandler } from 'express';
import { RepositoryError } from '../repositories';
import { errors, ErrorCodes, error } from '../utils/response';

/**
 * Custom application error with status code
 */
export class AppError extends Error {
  public readonly statusCode: number;
  public readonly code: string;
  public readonly details?: unknown;
  public readonly isOperational: boolean;

  constructor(
    message: string,
    statusCode = 500,
    code = ErrorCodes.INTERNAL_ERROR,
    details?: unknown
  ) {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
    this.isOperational = true; // Distinguishes operational errors from bugs

    Error.captureStackTrace(this, this.constructor);
  }

  static badRequest(message: string, details?: unknown): AppError {
    return new AppError(message, 400, ErrorCodes.BAD_REQUEST, details);
  }

  static unauthorized(message = 'Unauthorized'): AppError {
    return new AppError(message, 401, ErrorCodes.UNAUTHORIZED);
  }

  static forbidden(message = 'Forbidden'): AppError {
    return new AppError(message, 403, ErrorCodes.FORBIDDEN);
  }

  static notFound(resource = 'Resource'): AppError {
    return new AppError(`${resource} not found`, 404, ErrorCodes.NOT_FOUND);
  }

  static conflict(message: string): AppError {
    return new AppError(message, 409, ErrorCodes.CONFLICT);
  }

  static validationError(details: unknown): AppError {
    return new AppError('Validation failed', 422, ErrorCodes.VALIDATION_ERROR, details);
  }
}

/**
 * Async handler wrapper to catch async errors
 */
export function asyncHandler<T>(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<T>
) {
  return (req: Request, res: Response, next: NextFunction): void => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

/**
 * Map known error types to appropriate responses
 */
function classifyError(err: Error): {
  statusCode: number;
  code: string;
  message: string;
  details?: unknown;
} {
  // Application errors
  if (err instanceof AppError) {
    return {
      statusCode: err.statusCode,
      code: err.code,
      message: err.message,
      details: err.details,
    };
  }

  // Repository/database errors
  if (err instanceof RepositoryError) {
    // Don't expose database details in production
    const message = process.env.NODE_ENV === 'production'
      ? 'Database operation failed'
      : err.message;

    return {
      statusCode: 500,
      code: ErrorCodes.DATABASE_ERROR,
      message,
      details: process.env.NODE_ENV === 'development' ? err.context : undefined,
    };
  }

  // PostgreSQL specific errors
  if ('code' in err && typeof (err as any).code === 'string') {
    const pgCode = (err as any).code;

    // Unique violation
    if (pgCode === '23505') {
      return {
        statusCode: 409,
        code: ErrorCodes.CONFLICT,
        message: 'Resource already exists',
      };
    }

    // Foreign key violation
    if (pgCode === '23503') {
      return {
        statusCode: 400,
        code: ErrorCodes.BAD_REQUEST,
        message: 'Referenced resource does not exist',
      };
    }

    // Check violation
    if (pgCode === '23514') {
      return {
        statusCode: 400,
        code: ErrorCodes.VALIDATION_ERROR,
        message: 'Data validation failed',
      };
    }
  }

  // Validation errors from libraries like Joi/Zod
  if (err.name === 'ValidationError' || err.name === 'ZodError') {
    return {
      statusCode: 422,
      code: ErrorCodes.VALIDATION_ERROR,
      message: 'Validation failed',
      details: 'details' in err ? (err as any).details : undefined,
    };
  }

  // JSON parsing errors
  if (err instanceof SyntaxError && 'body' in err) {
    return {
      statusCode: 400,
      code: ErrorCodes.BAD_REQUEST,
      message: 'Invalid JSON in request body',
    };
  }

  // Default to internal error
  return {
    statusCode: 500,
    code: ErrorCodes.INTERNAL_ERROR,
    message: process.env.NODE_ENV === 'production'
      ? 'An unexpected error occurred'
      : err.message,
  };
}

/**
 * Main error handling middleware
 */
export const errorHandler: ErrorRequestHandler = (
  err: Error,
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  // Log the error
  const requestInfo = {
    method: req.method,
    path: req.path,
    query: req.query,
    ip: req.ip,
    userAgent: req.get('user-agent'),
    requestId: req.headers['x-request-id'],
  };

  // Classify the error
  const classified = classifyError(err);

  // Log appropriately based on severity
  if (classified.statusCode >= 500) {
    console.error('[ERROR]', {
      ...requestInfo,
      error: {
        name: err.name,
        message: err.message,
        stack: err.stack,
        ...classified,
      },
    });
  } else if (process.env.NODE_ENV === 'development') {
    console.warn('[WARN]', {
      ...requestInfo,
      error: classified,
    });
  }

  // Send error response
  error(
    res,
    classified.code as any,
    classified.message,
    classified.statusCode,
    classified.details
  );
};

/**
 * 404 Not Found handler for unmatched routes
 */
export function notFoundHandler(req: Request, res: Response): void {
  errors.notFound(res, `Endpoint ${req.method} ${req.path}`);
}

/**
 * Unhandled rejection handler
 */
export function setupUnhandledRejectionHandler(): void {
  process.on('unhandledRejection', (reason: unknown) => {
    console.error('[UNHANDLED_REJECTION]', reason);
    // In production, we might want to gracefully shutdown
    // process.exit(1);
  });

  process.on('uncaughtException', (error: Error) => {
    console.error('[UNCAUGHT_EXCEPTION]', error);
    // Graceful shutdown
    process.exit(1);
  });
}

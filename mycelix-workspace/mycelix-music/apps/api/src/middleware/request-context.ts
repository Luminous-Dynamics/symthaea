// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Request Context Middleware
 *
 * Provides request-scoped context including:
 * - Unique request ID
 * - Request timing
 * - User information (when authenticated)
 * - Tracing correlation IDs
 */

import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

/**
 * Request context interface
 */
export interface RequestContext {
  requestId: string;
  startTime: number;
  ip: string;
  userAgent: string;
  path: string;
  method: string;
  userId?: string;
  walletAddress?: string;
  traceId?: string;
  spanId?: string;
}

// Extend Express Request
declare global {
  namespace Express {
    interface Request {
      context: RequestContext;
    }
  }
}

/**
 * Extract client IP from request
 */
function getClientIp(req: Request): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket.remoteAddress || 'unknown';
}

/**
 * Request context middleware
 */
export function requestContextMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Generate or use existing request ID
  const requestId =
    (req.headers['x-request-id'] as string) ||
    (req.headers['x-correlation-id'] as string) ||
    randomUUID();

  // Extract tracing IDs if present (from OpenTelemetry)
  const traceId = req.headers['x-trace-id'] as string;
  const spanId = req.headers['x-span-id'] as string;

  // Build context
  const context: RequestContext = {
    requestId,
    startTime: Date.now(),
    ip: getClientIp(req),
    userAgent: req.get('user-agent') || 'unknown',
    path: req.path,
    method: req.method,
    traceId,
    spanId,
  };

  // Attach to request
  req.context = context;

  // Set response headers
  res.setHeader('X-Request-ID', requestId);

  // Log request start
  if (process.env.NODE_ENV !== 'test') {
    console.log(`[${requestId}] ${req.method} ${req.path} - Started`);
  }

  // Log request completion
  res.on('finish', () => {
    const duration = Date.now() - context.startTime;
    const level = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info';

    if (process.env.NODE_ENV !== 'test') {
      console[level === 'error' ? 'error' : level === 'warn' ? 'warn' : 'log'](
        `[${requestId}] ${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`
      );
    }
  });

  next();
}

/**
 * Middleware to attach authenticated user to context
 */
export function attachUserToContext(userId: string, walletAddress?: string) {
  return (req: Request, _res: Response, next: NextFunction): void => {
    if (req.context) {
      req.context.userId = userId;
      req.context.walletAddress = walletAddress;
    }
    next();
  };
}

/**
 * Get current request context (for use in async operations)
 */
export function getRequestContext(req: Request): RequestContext | undefined {
  return req.context;
}

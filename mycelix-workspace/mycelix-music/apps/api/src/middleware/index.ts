// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Middleware Index
 *
 * Central export for all middleware and middleware pipeline setup.
 */

import { Express, json, urlencoded } from 'express';
import compression from 'compression';
import helmet from 'helmet';
import cors from 'cors';

// Export individual middleware
export { requestContextMiddleware, attachUserToContext, getRequestContext } from './request-context';
export { errorHandler, notFoundHandler, AppError, asyncHandler, setupUnhandledRejectionHandler } from './error-handler';
export { cacheMiddleware, noCache, cacheHeaders } from './cache';
export { auditMiddleware, audit, AuditHelpers, AuditEventType, AuditSeverity, setAuditLogger } from './audit-logger';
export { validate, validateBody, validateQuery, validateParams, validateRequest, typedHandler } from './validate';
export { rateLimit, RateLimitPresets, createRateLimiters, MemoryRateLimiter, RedisRateLimiter } from './rate-limit';
export { requestLogger, createLogger, redactSensitive, redactHeaders } from './request-logger';

// Re-export types
export type { RequestContext } from './request-context';
export type { CacheMiddlewareOptions } from './cache';
export type { AuditEvent, AuditLogger } from './audit-logger';
export type { ValidateOptions, TypedRequestHandler, ValidatedBody, ValidatedQuery, ValidatedParams } from './validate';
export type { RateLimitConfig, RateLimitResult, RateLimiter } from './rate-limit';
export type { RequestLoggerConfig } from './request-logger';

/**
 * Middleware pipeline configuration
 */
export interface MiddlewarePipelineOptions {
  cors?: cors.CorsOptions;
  compression?: compression.CompressionOptions;
  bodyLimit?: string;
  trustProxy?: boolean;
  enableAuditLog?: boolean;
  enableHelmet?: boolean;
}

/**
 * Configure standard middleware pipeline
 */
export function setupMiddlewarePipeline(
  app: Express,
  options: MiddlewarePipelineOptions = {}
): void {
  // Trust proxy (for correct IP when behind load balancer)
  if (options.trustProxy !== false) {
    app.set('trust proxy', 1);
  }

  // Security headers
  if (options.enableHelmet !== false) {
    app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", 'data:', 'https:'],
          connectSrc: ["'self'", 'wss:', 'https:'],
        },
      },
      crossOriginEmbedderPolicy: false, // Needed for some IPFS content
    }));
  }

  // CORS
  const defaultCorsOrigin = (
    origin: string | undefined,
    callback: (err: Error | null, allow?: boolean) => void,
  ): void => {
    // Non-browser clients (no Origin header) are allowed by default.
    if (!origin) return callback(null, true);

    if (
      origin.startsWith('http://localhost') ||
      origin.startsWith('https://localhost') ||
      origin.startsWith('http://127.0.0.1') ||
      origin.startsWith('https://127.0.0.1') ||
      origin.startsWith('http://[::1]') ||
      origin.startsWith('https://[::1]') ||
      origin === 'null' // file:// contexts
    ) {
      return callback(null, true);
    }

    return callback(new Error('CORS: Origin not allowed'));
  };

  const corsDefaults = {
    origin: process.env.CORS_ORIGIN?.split(',').map(s => s.trim()).filter(Boolean) || '',
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID', 'Accept-Version'],
    exposedHeaders: ['X-Request-ID', 'X-Total-Count', 'X-API-Version'],
  };

  const corsConfig = options.cors || corsDefaults;
  const origin = corsConfig.origin;
  const normalizedOrigin =
    origin === '' || (Array.isArray(origin) && origin.length === 0)
      ? defaultCorsOrigin
      : origin;

  if (normalizedOrigin === '*') {
    // Keep the escape hatch, but make it visible in logs.
    console.warn('WARNING: CORS_ORIGIN="*" enables permissive cross-origin access (insecure).');
  }

  app.use(cors({
    ...corsConfig,
    origin: normalizedOrigin,
  }));

  // Compression
  app.use(compression(options.compression || {
    level: 6,
    threshold: 1024, // Only compress responses > 1KB
    filter: (req, res) => {
      // Don't compress if client doesn't accept it
      if (req.headers['x-no-compression']) {
        return false;
      }
      return compression.filter(req, res);
    },
  }));

  // Body parsing
  app.use(json({ limit: options.bodyLimit || '10mb' }));
  app.use(urlencoded({ extended: true, limit: options.bodyLimit || '10mb' }));

  // Request context (must be early in pipeline)
  const { requestContextMiddleware } = require('./request-context');
  app.use(requestContextMiddleware);

  // Audit logging
  if (options.enableAuditLog) {
    const { auditMiddleware } = require('./audit-logger');
    app.use(auditMiddleware({ logAllRequests: process.env.NODE_ENV === 'development' }));
  }
}

/**
 * Setup error handling (must be called after routes)
 */
export function setupErrorHandling(app: Express): void {
  const { notFoundHandler, errorHandler } = require('./error-handler');

  // 404 handler
  app.use(notFoundHandler);

  // Error handler
  app.use(errorHandler);
}

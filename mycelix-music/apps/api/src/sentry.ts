// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Sentry Error Tracking Integration
 *
 * Environment variables:
 *   SENTRY_DSN - Sentry DSN (required to enable)
 *   SENTRY_ENVIRONMENT - Environment name (default: NODE_ENV)
 *   SENTRY_RELEASE - Release version (default: npm_package_version)
 *   SENTRY_TRACES_SAMPLE_RATE - Trace sampling rate 0-1 (default: 0.1)
 *   SENTRY_PROFILES_SAMPLE_RATE - Profile sampling rate 0-1 (default: 0.1)
 */

import * as Sentry from '@sentry/node';
import { Express, Request, Response, NextFunction } from 'express';

let initialized = false;

export function initSentry(): void {
  const dsn = process.env.SENTRY_DSN;

  if (!dsn) {
    console.log('[sentry] SENTRY_DSN not set, error tracking disabled');
    return;
  }

  const environment = process.env.SENTRY_ENVIRONMENT || process.env.NODE_ENV || 'development';
  const release = process.env.SENTRY_RELEASE || process.env.npm_package_version || '0.1.0';
  const tracesSampleRate = parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || '0.1');
  const profilesSampleRate = parseFloat(process.env.SENTRY_PROFILES_SAMPLE_RATE || '0.1');

  Sentry.init({
    dsn,
    environment,
    release: `mycelix-api@${release}`,
    tracesSampleRate,
    profilesSampleRate,

    // Filter out health check errors
    beforeSend(event, hint) {
      const error = hint.originalException;

      // Don't report certain expected errors
      if (error instanceof Error) {
        if (error.message.includes('CORS blocked')) {
          return null;
        }
        if (error.message.includes('rate limit')) {
          return null;
        }
      }

      return event;
    },

    // Filter out health check transactions
    beforeSendTransaction(event) {
      const url = event.request?.url || '';
      if (url.includes('/health') || url.includes('/metrics')) {
        return null;
      }
      return event;
    },

    integrations: [
      // Capture unhandled promise rejections
      Sentry.captureConsoleIntegration(),
      // HTTP integration
      Sentry.httpIntegration(),
      // Express integration
      Sentry.expressIntegration(),
      // Postgres integration
      Sentry.postgresIntegration(),
      // Redis integration (v4)
      Sentry.redisIntegration(),
    ],

    // Additional context
    initialScope: {
      tags: {
        service: 'mycelix-api',
      },
    },
  });

  initialized = true;
  console.log(`[sentry] Initialized for ${environment} (release: ${release})`);
}

/**
 * Express request handler - add to app before routes
 */
export function sentryRequestHandler() {
  if (!initialized) {
    return (_req: Request, _res: Response, next: NextFunction) => next();
  }
  return Sentry.Handlers.requestHandler({
    // Include request data in error reports
    include: {
      user: true,
      ip: true,
      request: ['method', 'url', 'query_string', 'headers'],
    },
  });
}

/**
 * Express tracing handler - add after request handler
 */
export function sentryTracingHandler() {
  if (!initialized) {
    return (_req: Request, _res: Response, next: NextFunction) => next();
  }
  return Sentry.Handlers.tracingHandler();
}

/**
 * Express error handler - add after all routes
 */
export function sentryErrorHandler() {
  if (!initialized) {
    return (err: Error, _req: Request, _res: Response, next: NextFunction) => next(err);
  }
  return Sentry.Handlers.errorHandler({
    shouldHandleError(error: any) {
      // Report 4xx and 5xx errors
      if (error.status >= 400) {
        return true;
      }
      return true;
    },
  });
}

/**
 * Setup Sentry for Express app
 */
export function setupSentryForExpress(app: Express): void {
  if (!initialized) return;

  // Request handler creates a unique ID for each request
  app.use(sentryRequestHandler());

  // Tracing handler for performance monitoring
  app.use(sentryTracingHandler());
}

/**
 * Add Sentry error handler to Express app (call after all routes)
 */
export function addSentryErrorHandler(app: Express): void {
  if (!initialized) return;

  app.use(sentryErrorHandler());
}

/**
 * Capture an exception manually
 */
export function captureException(error: Error, context?: Record<string, any>): string | undefined {
  if (!initialized) {
    console.error('[sentry] Not initialized, error not reported:', error);
    return undefined;
  }

  return Sentry.captureException(error, {
    extra: context,
  });
}

/**
 * Capture a message
 */
export function captureMessage(message: string, level: 'info' | 'warning' | 'error' = 'info'): string | undefined {
  if (!initialized) return undefined;

  return Sentry.captureMessage(message, level);
}

/**
 * Set user context
 */
export function setUser(user: { id?: string; email?: string; address?: string }): void {
  if (!initialized) return;

  Sentry.setUser({
    id: user.id || user.address,
    email: user.email,
    username: user.address,
  });
}

/**
 * Add breadcrumb for debugging
 */
export function addBreadcrumb(breadcrumb: {
  category: string;
  message: string;
  level?: 'debug' | 'info' | 'warning' | 'error';
  data?: Record<string, any>;
}): void {
  if (!initialized) return;

  Sentry.addBreadcrumb({
    category: breadcrumb.category,
    message: breadcrumb.message,
    level: breadcrumb.level || 'info',
    data: breadcrumb.data,
    timestamp: Date.now() / 1000,
  });
}

/**
 * Create a transaction for custom performance tracking
 */
export function startTransaction(name: string, op: string) {
  if (!initialized) return null;

  return Sentry.startInactiveSpan({
    name,
    op,
  });
}

/**
 * Flush pending events before shutdown
 */
export async function flushSentry(timeout = 2000): Promise<boolean> {
  if (!initialized) return true;

  return Sentry.close(timeout);
}

export { Sentry };

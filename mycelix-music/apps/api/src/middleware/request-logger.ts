// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Request/Response Logging Middleware
 *
 * Structured logging with:
 * - Sensitive data redaction
 * - Request/response body logging (configurable)
 * - Performance metrics
 * - Correlation IDs
 */

import { Request, Response, NextFunction } from 'express';

/**
 * Fields to redact from logs
 */
const SENSITIVE_FIELDS = new Set([
  'password',
  'token',
  'secret',
  'api_key',
  'apiKey',
  'authorization',
  'cookie',
  'session',
  'private_key',
  'privateKey',
  'mnemonic',
  'seed',
  'credit_card',
  'creditCard',
  'cvv',
  'ssn',
]);

/**
 * Headers to redact
 */
const SENSITIVE_HEADERS = new Set([
  'authorization',
  'cookie',
  'x-api-key',
  'x-auth-token',
]);

/**
 * Redaction placeholder
 */
const REDACTED = '[REDACTED]';

/**
 * Logger configuration
 */
export interface RequestLoggerConfig {
  /** Log request body */
  logBody?: boolean;
  /** Log response body */
  logResponseBody?: boolean;
  /** Maximum body size to log (bytes) */
  maxBodySize?: number;
  /** Routes to skip logging */
  skipRoutes?: string[];
  /** Custom redaction fields */
  sensitiveFields?: string[];
  /** Output format */
  format?: 'json' | 'pretty';
}

const DEFAULT_CONFIG: RequestLoggerConfig = {
  logBody: true,
  logResponseBody: false,
  maxBodySize: 10000,
  skipRoutes: ['/health', '/metrics', '/favicon.ico'],
  format: process.env.NODE_ENV === 'production' ? 'json' : 'pretty',
};

/**
 * Deep clone and redact sensitive fields
 */
function redactSensitive(obj: unknown, sensitiveFields: Set<string>): unknown {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => redactSensitive(item, sensitiveFields));
  }

  const redacted: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    const lowerKey = key.toLowerCase();

    if (sensitiveFields.has(lowerKey)) {
      redacted[key] = REDACTED;
    } else if (typeof value === 'object' && value !== null) {
      redacted[key] = redactSensitive(value, sensitiveFields);
    } else {
      redacted[key] = value;
    }
  }

  return redacted;
}

/**
 * Redact headers
 */
function redactHeaders(headers: Record<string, unknown>): Record<string, unknown> {
  const redacted: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(headers)) {
    if (SENSITIVE_HEADERS.has(key.toLowerCase())) {
      redacted[key] = REDACTED;
    } else {
      redacted[key] = value;
    }
  }

  return redacted;
}

/**
 * Truncate body if too large
 */
function truncateBody(body: unknown, maxSize: number): unknown {
  if (body === null || body === undefined) {
    return body;
  }

  const str = typeof body === 'string' ? body : JSON.stringify(body);

  if (str.length > maxSize) {
    return `[TRUNCATED: ${str.length} bytes, showing first ${maxSize}]${str.slice(0, maxSize)}...`;
  }

  return body;
}

/**
 * Log entry structure
 */
interface LogEntry {
  timestamp: string;
  requestId: string;
  method: string;
  path: string;
  query?: Record<string, unknown>;
  headers?: Record<string, unknown>;
  body?: unknown;
  statusCode?: number;
  responseBody?: unknown;
  duration?: number;
  ip: string;
  userAgent: string;
  error?: string;
}

/**
 * Format log entry
 */
function formatLogEntry(entry: LogEntry, format: 'json' | 'pretty'): string {
  if (format === 'json') {
    return JSON.stringify(entry);
  }

  // Pretty format for development
  const lines = [
    `[${entry.timestamp}] ${entry.method} ${entry.path}`,
    `  Request ID: ${entry.requestId}`,
    `  IP: ${entry.ip}`,
    `  User-Agent: ${entry.userAgent}`,
  ];

  if (entry.query && Object.keys(entry.query).length > 0) {
    lines.push(`  Query: ${JSON.stringify(entry.query)}`);
  }

  if (entry.body) {
    lines.push(`  Body: ${JSON.stringify(entry.body)}`);
  }

  if (entry.statusCode) {
    lines.push(`  Status: ${entry.statusCode}`);
  }

  if (entry.duration !== undefined) {
    lines.push(`  Duration: ${entry.duration}ms`);
  }

  if (entry.error) {
    lines.push(`  Error: ${entry.error}`);
  }

  return lines.join('\n');
}

/**
 * Create request logger middleware
 */
export function requestLogger(config: RequestLoggerConfig = {}): (
  req: Request,
  res: Response,
  next: NextFunction
) => void {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };
  const sensitiveFields = new Set([
    ...SENSITIVE_FIELDS,
    ...(mergedConfig.sensitiveFields || []),
  ]);

  return (req: Request, res: Response, next: NextFunction): void => {
    // Skip certain routes
    if (mergedConfig.skipRoutes?.some((route) => req.path.startsWith(route))) {
      return next();
    }

    const startTime = Date.now();
    const requestId = req.context?.requestId || req.headers['x-request-id'] || 'unknown';

    // Capture original response methods
    const originalJson = res.json.bind(res);
    const originalSend = res.send.bind(res);
    let responseBody: unknown;

    // Intercept response if logging response body
    if (mergedConfig.logResponseBody) {
      res.json = function (body: unknown): Response {
        responseBody = body;
        return originalJson(body);
      };

      res.send = function (body: unknown): Response {
        if (typeof body === 'string') {
          try {
            responseBody = JSON.parse(body);
          } catch {
            responseBody = body;
          }
        } else {
          responseBody = body;
        }
        return originalSend(body);
      };
    }

    // Log on response finish
    res.on('finish', () => {
      const duration = Date.now() - startTime;

      const entry: LogEntry = {
        timestamp: new Date().toISOString(),
        requestId: String(requestId),
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        ip: req.context?.ip || req.ip || 'unknown',
        userAgent: req.context?.userAgent || req.get('user-agent') || 'unknown',
      };

      // Add query params
      if (Object.keys(req.query).length > 0) {
        entry.query = redactSensitive(req.query, sensitiveFields) as Record<string, unknown>;
      }

      // Add request body
      if (mergedConfig.logBody && req.body && Object.keys(req.body).length > 0) {
        const redactedBody = redactSensitive(req.body, sensitiveFields);
        entry.body = truncateBody(redactedBody, mergedConfig.maxBodySize!);
      }

      // Add response body
      if (mergedConfig.logResponseBody && responseBody) {
        const redactedResponse = redactSensitive(responseBody, sensitiveFields);
        entry.responseBody = truncateBody(redactedResponse, mergedConfig.maxBodySize!);
      }

      // Log appropriately based on status code
      const logLine = formatLogEntry(entry, mergedConfig.format!);

      if (res.statusCode >= 500) {
        console.error(logLine);
      } else if (res.statusCode >= 400) {
        console.warn(logLine);
      } else {
        console.log(logLine);
      }
    });

    next();
  };
}

/**
 * Create a structured logger
 */
export function createLogger(context?: Record<string, unknown>) {
  const baseContext = context || {};

  return {
    info: (message: string, data?: Record<string, unknown>) => {
      console.log(JSON.stringify({
        level: 'info',
        timestamp: new Date().toISOString(),
        message,
        ...baseContext,
        ...data,
      }));
    },

    warn: (message: string, data?: Record<string, unknown>) => {
      console.warn(JSON.stringify({
        level: 'warn',
        timestamp: new Date().toISOString(),
        message,
        ...baseContext,
        ...data,
      }));
    },

    error: (message: string, error?: Error, data?: Record<string, unknown>) => {
      console.error(JSON.stringify({
        level: 'error',
        timestamp: new Date().toISOString(),
        message,
        error: error ? {
          name: error.name,
          message: error.message,
          stack: error.stack,
        } : undefined,
        ...baseContext,
        ...data,
      }));
    },

    debug: (message: string, data?: Record<string, unknown>) => {
      if (process.env.LOG_LEVEL === 'debug') {
        console.log(JSON.stringify({
          level: 'debug',
          timestamp: new Date().toISOString(),
          message,
          ...baseContext,
          ...data,
        }));
      }
    },
  };
}

/**
 * Redact sensitive data from an object (exported for use elsewhere)
 */
export { redactSensitive, redactHeaders };

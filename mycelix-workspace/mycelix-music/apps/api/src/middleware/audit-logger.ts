// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audit Logging Middleware
 *
 * Provides comprehensive audit logging for:
 * - Security-sensitive operations
 * - Data access patterns
 * - User actions
 * - Compliance requirements
 */

import { Request, Response, NextFunction } from 'express';

/**
 * Audit event types
 */
export enum AuditEventType {
  // Authentication events
  AUTH_LOGIN = 'auth.login',
  AUTH_LOGOUT = 'auth.logout',
  AUTH_FAILED = 'auth.failed',
  AUTH_TOKEN_REFRESH = 'auth.token_refresh',

  // Resource access
  RESOURCE_READ = 'resource.read',
  RESOURCE_CREATE = 'resource.create',
  RESOURCE_UPDATE = 'resource.update',
  RESOURCE_DELETE = 'resource.delete',

  // Sensitive operations
  SONG_REGISTERED = 'song.registered',
  PAYMENT_PROCESSED = 'payment.processed',
  PAYMENT_FAILED = 'payment.failed',
  CLAIM_CREATED = 'claim.created',

  // Admin operations
  ADMIN_ACCESS = 'admin.access',
  CONFIG_CHANGED = 'config.changed',
  USER_MODIFIED = 'user.modified',

  // Security events
  RATE_LIMITED = 'security.rate_limited',
  INVALID_INPUT = 'security.invalid_input',
  UNAUTHORIZED_ACCESS = 'security.unauthorized',
  SUSPICIOUS_ACTIVITY = 'security.suspicious',
}

/**
 * Audit event severity
 */
export enum AuditSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

/**
 * Audit event structure
 */
export interface AuditEvent {
  timestamp: string;
  eventType: AuditEventType;
  severity: AuditSeverity;
  requestId?: string;
  userId?: string;
  walletAddress?: string;
  ip: string;
  userAgent: string;
  method: string;
  path: string;
  statusCode?: number;
  resourceType?: string;
  resourceId?: string;
  action?: string;
  details?: Record<string, unknown>;
  duration?: number;
}

/**
 * Audit logger interface
 */
export interface AuditLogger {
  log(event: AuditEvent): Promise<void>;
}

/**
 * Console audit logger (for development)
 */
export class ConsoleAuditLogger implements AuditLogger {
  async log(event: AuditEvent): Promise<void> {
    const levelMap = {
      [AuditSeverity.INFO]: 'log',
      [AuditSeverity.WARNING]: 'warn',
      [AuditSeverity.ERROR]: 'error',
      [AuditSeverity.CRITICAL]: 'error',
    } as const;

    const level = levelMap[event.severity];
    console[level]('[AUDIT]', JSON.stringify(event, null, 2));
  }
}

/**
 * JSON file audit logger (for production without external service)
 */
export class JsonAuditLogger implements AuditLogger {
  async log(event: AuditEvent): Promise<void> {
    // In production, this would write to a file or send to an external service
    // For now, just output structured JSON
    process.stdout.write(JSON.stringify({ type: 'audit', ...event }) + '\n');
  }
}

/**
 * Global audit logger instance
 */
let auditLogger: AuditLogger = new ConsoleAuditLogger();

/**
 * Set the global audit logger
 */
export function setAuditLogger(logger: AuditLogger): void {
  auditLogger = logger;
}

/**
 * Get the current audit logger
 */
export function getAuditLogger(): AuditLogger {
  return auditLogger;
}

/**
 * Log an audit event
 */
export async function audit(
  eventType: AuditEventType,
  severity: AuditSeverity,
  req: Request,
  details?: Record<string, unknown>
): Promise<void> {
  const event: AuditEvent = {
    timestamp: new Date().toISOString(),
    eventType,
    severity,
    requestId: req.context?.requestId,
    userId: req.context?.userId,
    walletAddress: req.context?.walletAddress,
    ip: req.context?.ip || req.ip || 'unknown',
    userAgent: req.context?.userAgent || req.get('user-agent') || 'unknown',
    method: req.method,
    path: req.path,
    details,
  };

  try {
    await auditLogger.log(event);
  } catch (error) {
    // Don't let audit failures break the application
    console.error('Audit logging failed:', error);
  }
}

/**
 * Audit middleware for automatic request logging
 */
export function auditMiddleware(options: {
  logAllRequests?: boolean;
  sensitiveRoutes?: string[];
  excludeRoutes?: string[];
} = {}) {
  const sensitiveRoutes = options.sensitiveRoutes || [
    '/api/songs',      // Song operations
    '/api/payments',   // Payment operations
    '/api/claims',     // Claim operations
    '/api/admin',      // Admin operations
    '/api/auth',       // Auth operations
  ];

  const excludeRoutes = options.excludeRoutes || [
    '/health',
    '/metrics',
    '/favicon.ico',
  ];

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Skip excluded routes
    if (excludeRoutes.some(route => req.path.startsWith(route))) {
      return next();
    }

    const startTime = Date.now();
    const isSensitive = sensitiveRoutes.some(route => req.path.startsWith(route));

    // Log response
    res.on('finish', async () => {
      // Skip if not logging all requests and not sensitive
      if (!options.logAllRequests && !isSensitive) {
        return;
      }

      // Determine event type based on method
      let eventType: AuditEventType;
      switch (req.method) {
        case 'POST':
          eventType = AuditEventType.RESOURCE_CREATE;
          break;
        case 'PUT':
        case 'PATCH':
          eventType = AuditEventType.RESOURCE_UPDATE;
          break;
        case 'DELETE':
          eventType = AuditEventType.RESOURCE_DELETE;
          break;
        default:
          eventType = AuditEventType.RESOURCE_READ;
      }

      // Determine severity based on status code
      let severity: AuditSeverity;
      if (res.statusCode >= 500) {
        severity = AuditSeverity.ERROR;
      } else if (res.statusCode >= 400) {
        severity = AuditSeverity.WARNING;
      } else {
        severity = AuditSeverity.INFO;
      }

      // Special handling for auth failures
      if (res.statusCode === 401) {
        eventType = AuditEventType.AUTH_FAILED;
        severity = AuditSeverity.WARNING;
      }

      // Special handling for rate limiting
      if (res.statusCode === 429) {
        eventType = AuditEventType.RATE_LIMITED;
        severity = AuditSeverity.WARNING;
      }

      const event: AuditEvent = {
        timestamp: new Date().toISOString(),
        eventType,
        severity,
        requestId: req.context?.requestId,
        userId: req.context?.userId,
        walletAddress: req.context?.walletAddress,
        ip: req.context?.ip || req.ip || 'unknown',
        userAgent: req.context?.userAgent || req.get('user-agent') || 'unknown',
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration: Date.now() - startTime,
        details: {
          query: Object.keys(req.query).length > 0 ? req.query : undefined,
        },
      };

      try {
        await auditLogger.log(event);
      } catch (error) {
        console.error('Audit logging failed:', error);
      }
    });

    next();
  };
}

/**
 * Helper to audit specific events
 */
export const AuditHelpers = {
  songRegistered: (req: Request, songId: string, songTitle: string) =>
    audit(AuditEventType.SONG_REGISTERED, AuditSeverity.INFO, req, { songId, songTitle }),

  paymentProcessed: (req: Request, amount: string, songId: string) =>
    audit(AuditEventType.PAYMENT_PROCESSED, AuditSeverity.INFO, req, { amount, songId }),

  paymentFailed: (req: Request, error: string, songId: string) =>
    audit(AuditEventType.PAYMENT_FAILED, AuditSeverity.ERROR, req, { error, songId }),

  claimCreated: (req: Request, claimStreamId: string, songId: string) =>
    audit(AuditEventType.CLAIM_CREATED, AuditSeverity.INFO, req, { claimStreamId, songId }),

  suspiciousActivity: (req: Request, reason: string, details?: Record<string, unknown>) =>
    audit(AuditEventType.SUSPICIOUS_ACTIVITY, AuditSeverity.WARNING, req, { reason, ...details }),
};

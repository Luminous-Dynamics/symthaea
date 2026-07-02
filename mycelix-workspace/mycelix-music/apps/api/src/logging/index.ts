// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Structured Logging System
 *
 * Provides structured JSON logging with:
 * - Correlation IDs for request tracing
 * - Log levels with filtering
 * - Context propagation
 * - Sensitive data redaction
 * - Multiple output formats
 */

import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';
import { AsyncLocalStorage } from 'async_hooks';

/**
 * Log levels
 */
export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
}

/**
 * Log entry structure
 */
export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  correlationId?: string;
  requestId?: string;
  service: string;
  environment: string;
  context?: Record<string, unknown>;
  error?: {
    name: string;
    message: string;
    stack?: string;
    code?: string;
  };
  duration?: number;
  [key: string]: unknown;
}

/**
 * Logger context
 */
export interface LogContext {
  correlationId: string;
  requestId?: string;
  walletAddress?: string;
  userId?: string;
  path?: string;
  method?: string;
  [key: string]: unknown;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  level: LogLevel;
  service: string;
  environment: string;
  format: 'json' | 'pretty';
  redactKeys: string[];
  includeTimestamp: boolean;
  includeStack: boolean;
}

/**
 * Default configuration
 */
const defaultConfig: LoggerConfig = {
  level: LogLevel.INFO,
  service: 'mycelix-api',
  environment: process.env.NODE_ENV || 'development',
  format: process.env.NODE_ENV === 'production' ? 'json' : 'pretty',
  redactKeys: ['password', 'secret', 'token', 'apiKey', 'authorization', 'signature'],
  includeTimestamp: true,
  includeStack: process.env.NODE_ENV !== 'production',
};

/**
 * Async context storage for correlation IDs
 */
const asyncLocalStorage = new AsyncLocalStorage<LogContext>();

/**
 * Get current context
 */
export function getLogContext(): LogContext | undefined {
  return asyncLocalStorage.getStore();
}

/**
 * Run with context
 */
export function runWithContext<T>(context: LogContext, fn: () => T): T {
  return asyncLocalStorage.run(context, fn);
}

/**
 * Logger class
 */
export class Logger {
  private config: LoggerConfig;
  private static instance: Logger | null = null;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  /**
   * Get singleton instance
   */
  static getInstance(config?: Partial<LoggerConfig>): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger(config);
    }
    return Logger.instance;
  }

  /**
   * Reset singleton (for testing)
   */
  static reset(): void {
    Logger.instance = null;
  }

  /**
   * Create child logger with additional context
   */
  child(context: Record<string, unknown>): ChildLogger {
    return new ChildLogger(this, context);
  }

  /**
   * Check if level is enabled
   */
  isLevelEnabled(level: LogLevel): boolean {
    return level >= this.config.level;
  }

  /**
   * Core log method
   */
  private log(level: LogLevel, message: string, data?: Record<string, unknown>): void {
    if (!this.isLevelEnabled(level)) return;

    const context = getLogContext();
    const entry = this.buildEntry(level, message, data, context);

    this.output(entry);
  }

  /**
   * Build log entry
   */
  private buildEntry(
    level: LogLevel,
    message: string,
    data?: Record<string, unknown>,
    context?: LogContext
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: LogLevel[level],
      message,
      service: this.config.service,
      environment: this.config.environment,
    };

    // Add correlation/request IDs from context
    if (context) {
      entry.correlationId = context.correlationId;
      entry.requestId = context.requestId;

      // Add safe context fields
      if (context.walletAddress) entry.walletAddress = context.walletAddress;
      if (context.path) entry.path = context.path;
      if (context.method) entry.method = context.method;
    }

    // Add extra data
    if (data) {
      const redacted = this.redactSensitive(data);
      Object.assign(entry, redacted);
    }

    return entry;
  }

  /**
   * Redact sensitive data
   */
  private redactSensitive(data: Record<string, unknown>): Record<string, unknown> {
    const redacted: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(data)) {
      const lowerKey = key.toLowerCase();

      if (this.config.redactKeys.some(k => lowerKey.includes(k.toLowerCase()))) {
        redacted[key] = '[REDACTED]';
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        redacted[key] = this.redactSensitive(value as Record<string, unknown>);
      } else {
        redacted[key] = value;
      }
    }

    return redacted;
  }

  /**
   * Output log entry
   */
  private output(entry: LogEntry): void {
    const output = this.config.format === 'json'
      ? JSON.stringify(entry)
      : this.formatPretty(entry);

    // Use appropriate console method based on level
    switch (entry.level) {
      case 'ERROR':
      case 'FATAL':
        console.error(output);
        break;
      case 'WARN':
        console.warn(output);
        break;
      case 'DEBUG':
      case 'TRACE':
        console.debug(output);
        break;
      default:
        console.log(output);
    }
  }

  /**
   * Format entry for pretty printing
   */
  private formatPretty(entry: LogEntry): string {
    const colors = {
      TRACE: '\x1b[90m',
      DEBUG: '\x1b[36m',
      INFO: '\x1b[32m',
      WARN: '\x1b[33m',
      ERROR: '\x1b[31m',
      FATAL: '\x1b[35m',
      reset: '\x1b[0m',
    };

    const color = colors[entry.level as keyof typeof colors] || colors.reset;
    const time = entry.timestamp.split('T')[1].slice(0, 12);

    let output = `${colors.TRACE}${time}${colors.reset} ${color}${entry.level.padEnd(5)}${colors.reset}`;

    if (entry.correlationId) {
      output += ` ${colors.TRACE}[${entry.correlationId.slice(0, 8)}]${colors.reset}`;
    }

    output += ` ${entry.message}`;

    // Add extra fields
    const extraKeys = Object.keys(entry).filter(
      k => !['timestamp', 'level', 'message', 'service', 'environment', 'correlationId', 'requestId'].includes(k)
    );

    if (extraKeys.length > 0) {
      const extra: Record<string, unknown> = {};
      for (const key of extraKeys) {
        extra[key] = entry[key];
      }
      output += ` ${colors.TRACE}${JSON.stringify(extra)}${colors.reset}`;
    }

    if (entry.error?.stack && this.config.includeStack) {
      output += `\n${colors.TRACE}${entry.error.stack}${colors.reset}`;
    }

    return output;
  }

  // Convenience methods
  trace(message: string, data?: Record<string, unknown>): void {
    this.log(LogLevel.TRACE, message, data);
  }

  debug(message: string, data?: Record<string, unknown>): void {
    this.log(LogLevel.DEBUG, message, data);
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.log(LogLevel.INFO, message, data);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.log(LogLevel.WARN, message, data);
  }

  error(message: string, error?: Error | Record<string, unknown>): void {
    let data: Record<string, unknown> = {};

    if (error instanceof Error) {
      data.error = {
        name: error.name,
        message: error.message,
        stack: this.config.includeStack ? error.stack : undefined,
        code: (error as any).code,
      };
    } else if (error) {
      data = error;
    }

    this.log(LogLevel.ERROR, message, data);
  }

  fatal(message: string, error?: Error | Record<string, unknown>): void {
    let data: Record<string, unknown> = {};

    if (error instanceof Error) {
      data.error = {
        name: error.name,
        message: error.message,
        stack: error.stack,
        code: (error as any).code,
      };
    } else if (error) {
      data = error;
    }

    this.log(LogLevel.FATAL, message, data);
  }
}

/**
 * Child logger with additional context
 */
class ChildLogger {
  constructor(
    private parent: Logger,
    private context: Record<string, unknown>
  ) {}

  trace(message: string, data?: Record<string, unknown>): void {
    this.parent.trace(message, { ...this.context, ...data });
  }

  debug(message: string, data?: Record<string, unknown>): void {
    this.parent.debug(message, { ...this.context, ...data });
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.parent.info(message, { ...this.context, ...data });
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.parent.warn(message, { ...this.context, ...data });
  }

  error(message: string, error?: Error | Record<string, unknown>): void {
    if (error instanceof Error) {
      this.parent.error(message, error);
    } else {
      this.parent.error(message, { ...this.context, ...error });
    }
  }

  fatal(message: string, error?: Error | Record<string, unknown>): void {
    if (error instanceof Error) {
      this.parent.fatal(message, error);
    } else {
      this.parent.fatal(message, { ...this.context, ...error });
    }
  }
}

/**
 * Express middleware for correlation ID and logging context
 */
export function correlationMiddleware() {
  const logger = Logger.getInstance();

  return (req: Request, res: Response, next: NextFunction): void => {
    // Get or generate correlation ID
    const correlationId = (req.headers['x-correlation-id'] as string) || randomUUID();
    const requestId = randomUUID();

    // Set response headers
    res.setHeader('X-Correlation-ID', correlationId);
    res.setHeader('X-Request-ID', requestId);

    // Create context
    const context: LogContext = {
      correlationId,
      requestId,
      path: req.path,
      method: req.method,
    };

    // Add wallet address if authenticated
    if (req.auth?.address) {
      context.walletAddress = req.auth.address;
    }

    // Run request handling with context
    runWithContext(context, () => {
      // Attach to request for easy access
      (req as any).correlationId = correlationId;
      (req as any).requestId = requestId;
      (req as any).log = logger.child({
        correlationId,
        requestId,
        path: req.path,
        method: req.method,
      });

      // Log request start
      logger.info('Request started', {
        path: req.path,
        method: req.method,
        userAgent: req.headers['user-agent'],
        ip: req.ip,
      });

      // Log response on finish
      const startTime = Date.now();

      res.on('finish', () => {
        const duration = Date.now() - startTime;
        const level = res.statusCode >= 500 ? 'error' :
                      res.statusCode >= 400 ? 'warn' : 'info';

        logger[level]('Request completed', {
          statusCode: res.statusCode,
          duration,
          contentLength: res.get('Content-Length'),
        });
      });

      next();
    });
  };
}

/**
 * Create logger instance
 */
export function createLogger(config?: Partial<LoggerConfig>): Logger {
  return new Logger(config);
}

/**
 * Get default logger
 */
export function getLogger(): Logger {
  return Logger.getInstance();
}

/**
 * Convenience exports
 */
export const log = {
  trace: (message: string, data?: Record<string, unknown>) => Logger.getInstance().trace(message, data),
  debug: (message: string, data?: Record<string, unknown>) => Logger.getInstance().debug(message, data),
  info: (message: string, data?: Record<string, unknown>) => Logger.getInstance().info(message, data),
  warn: (message: string, data?: Record<string, unknown>) => Logger.getInstance().warn(message, data),
  error: (message: string, error?: Error | Record<string, unknown>) => Logger.getInstance().error(message, error),
  fatal: (message: string, error?: Error | Record<string, unknown>) => Logger.getInstance().fatal(message, error),
};

export default Logger;

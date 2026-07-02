// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Structured Logging Service
 * JSON-based logging with context, levels, and transport support
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';

// Log levels
export enum LogLevel {
  TRACE = 0,
  DEBUG = 10,
  INFO = 20,
  WARN = 30,
  ERROR = 40,
  FATAL = 50,
}

// Log entry
export interface LogEntry {
  timestamp: string;
  level: string;
  levelNum: number;
  message: string;
  context?: Record<string, unknown>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  request?: {
    id?: string;
    method: string;
    path: string;
    ip?: string;
    userAgent?: string;
  };
  user?: {
    id?: string;
    address?: string;
  };
  duration?: number;
  service?: string;
  environment?: string;
  hostname?: string;
  pid?: number;
}

// Transport interface
export interface LogTransport {
  log(entry: LogEntry): void;
}

// Console transport
class ConsoleTransport implements LogTransport {
  private colors = {
    TRACE: '\x1b[90m',
    DEBUG: '\x1b[36m',
    INFO: '\x1b[32m',
    WARN: '\x1b[33m',
    ERROR: '\x1b[31m',
    FATAL: '\x1b[35m',
    reset: '\x1b[0m',
  };

  log(entry: LogEntry): void {
    if (process.env.NODE_ENV === 'production') {
      // JSON output in production
      console.log(JSON.stringify(entry));
    } else {
      // Pretty print in development
      const color = this.colors[entry.level as keyof typeof this.colors] || '';
      const reset = this.colors.reset;

      let output = `${color}[${entry.level}]${reset} ${entry.timestamp} - ${entry.message}`;

      if (entry.context && Object.keys(entry.context).length > 0) {
        output += ` ${JSON.stringify(entry.context)}`;
      }

      if (entry.duration !== undefined) {
        output += ` (${entry.duration}ms)`;
      }

      if (entry.error) {
        output += `\n${color}Error: ${entry.error.message}${reset}`;
        if (entry.error.stack) {
          output += `\n${entry.error.stack}`;
        }
      }

      console.log(output);
    }
  }
}

// File transport (for log aggregation)
class FileTransport implements LogTransport {
  private buffer: LogEntry[] = [];
  private flushInterval: NodeJS.Timeout | null = null;

  constructor(
    private filePath: string,
    private maxBufferSize = 100,
    private flushIntervalMs = 5000
  ) {
    this.startFlushInterval();
  }

  private startFlushInterval(): void {
    this.flushInterval = setInterval(() => {
      this.flush();
    }, this.flushIntervalMs);
  }

  log(entry: LogEntry): void {
    this.buffer.push(entry);
    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  private flush(): void {
    if (this.buffer.length === 0) return;

    const entries = this.buffer;
    this.buffer = [];

    // In production, use fs.appendFile or a proper file writer
    // For now, just log that we would write
    if (process.env.NODE_ENV === 'development') {
      console.log(`[FileTransport] Would write ${entries.length} entries to ${this.filePath}`);
    }
  }

  destroy(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flush();
  }
}

// HTTP transport (for log aggregation services)
class HttpTransport implements LogTransport {
  private buffer: LogEntry[] = [];
  private flushInterval: NodeJS.Timeout | null = null;

  constructor(
    private endpoint: string,
    private headers: Record<string, string> = {},
    private maxBufferSize = 50,
    private flushIntervalMs = 10000
  ) {
    this.startFlushInterval();
  }

  private startFlushInterval(): void {
    this.flushInterval = setInterval(() => {
      this.flush();
    }, this.flushIntervalMs);
  }

  log(entry: LogEntry): void {
    this.buffer.push(entry);
    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const entries = this.buffer;
    this.buffer = [];

    try {
      await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this.headers,
        },
        body: JSON.stringify({ entries }),
      });
    } catch (error) {
      // Don't log errors from logging (infinite loop)
      console.error('[HttpTransport] Failed to send logs:', error);
      // Re-add entries to buffer
      this.buffer.unshift(...entries);
    }
  }

  destroy(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flush();
  }
}

// Logger configuration
export interface LoggerConfig {
  service?: string;
  environment?: string;
  minLevel?: LogLevel;
  transports?: LogTransport[];
  defaultContext?: Record<string, unknown>;
  redactKeys?: string[];
}

/**
 * Logger class
 */
export class Logger {
  private service: string;
  private environment: string;
  private minLevel: LogLevel;
  private transports: LogTransport[];
  private defaultContext: Record<string, unknown>;
  private redactKeys: Set<string>;
  private hostname: string;
  private pid: number;

  constructor(config: LoggerConfig = {}) {
    this.service = config.service || process.env.SERVICE_NAME || 'mycelix-api';
    this.environment = config.environment || process.env.NODE_ENV || 'development';
    this.minLevel = config.minLevel ?? (this.environment === 'production' ? LogLevel.INFO : LogLevel.DEBUG);
    this.transports = config.transports || [new ConsoleTransport()];
    this.defaultContext = config.defaultContext || {};
    this.redactKeys = new Set(config.redactKeys || [
      'password',
      'secret',
      'token',
      'authorization',
      'cookie',
      'privateKey',
      'mnemonic',
    ]);
    this.hostname = process.env.HOSTNAME || require('os').hostname();
    this.pid = process.pid;
  }

  private createEntry(
    level: string,
    levelNum: number,
    message: string,
    context?: Record<string, unknown>,
    error?: Error
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      levelNum,
      message,
      service: this.service,
      environment: this.environment,
      hostname: this.hostname,
      pid: this.pid,
    };

    if (context || Object.keys(this.defaultContext).length > 0) {
      entry.context = this.redact({ ...this.defaultContext, ...context });
    }

    if (error) {
      entry.error = {
        name: error.name,
        message: error.message,
        stack: error.stack,
      };
    }

    return entry;
  }

  private redact(obj: Record<string, unknown>): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(obj)) {
      if (this.redactKeys.has(key.toLowerCase())) {
        result[key] = '[REDACTED]';
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        result[key] = this.redact(value as Record<string, unknown>);
      } else {
        result[key] = value;
      }
    }

    return result;
  }

  private log(level: string, levelNum: number, message: string, context?: Record<string, unknown>, error?: Error): void {
    if (levelNum < this.minLevel) return;

    const entry = this.createEntry(level, levelNum, message, context, error);

    for (const transport of this.transports) {
      try {
        transport.log(entry);
      } catch (e) {
        console.error('Transport error:', e);
      }
    }
  }

  trace(message: string, context?: Record<string, unknown>): void {
    this.log('TRACE', LogLevel.TRACE, message, context);
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.log('DEBUG', LogLevel.DEBUG, message, context);
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.log('INFO', LogLevel.INFO, message, context);
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.log('WARN', LogLevel.WARN, message, context);
  }

  error(message: string, error?: Error | Record<string, unknown>, context?: Record<string, unknown>): void {
    if (error instanceof Error) {
      this.log('ERROR', LogLevel.ERROR, message, context, error);
    } else {
      this.log('ERROR', LogLevel.ERROR, message, { ...error, ...context });
    }
  }

  fatal(message: string, error?: Error, context?: Record<string, unknown>): void {
    this.log('FATAL', LogLevel.FATAL, message, context, error);
  }

  /**
   * Create a child logger with additional context
   */
  child(context: Record<string, unknown>): Logger {
    const childLogger = new Logger({
      service: this.service,
      environment: this.environment,
      minLevel: this.minLevel,
      transports: this.transports,
      defaultContext: { ...this.defaultContext, ...context },
    });
    return childLogger;
  }

  /**
   * Time a function and log the duration
   */
  async time<T>(
    label: string,
    fn: () => Promise<T>,
    context?: Record<string, unknown>
  ): Promise<T> {
    const start = Date.now();
    try {
      const result = await fn();
      const duration = Date.now() - start;
      this.info(`${label} completed`, { ...context, duration });
      return result;
    } catch (error) {
      const duration = Date.now() - start;
      this.error(`${label} failed`, error as Error, { ...context, duration });
      throw error;
    }
  }
}

// Singleton instance
export const logger = new Logger();

/**
 * Request logging middleware
 */
export function requestLoggingMiddleware(): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = Date.now();
    const requestId = (req.headers['x-request-id'] as string) || `req_${Date.now().toString(36)}`;

    // Create request-scoped logger
    const reqLogger = logger.child({
      requestId,
      method: req.method,
      path: req.path,
      ip: req.ip || req.socket.remoteAddress,
      userAgent: req.headers['user-agent'],
    });

    // Attach to request
    (req as any).logger = reqLogger;

    // Log request
    reqLogger.info('Request received');

    // Log response
    res.on('finish', () => {
      const duration = Date.now() - startTime;
      const level = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info';

      reqLogger[level]('Request completed', {
        statusCode: res.statusCode,
        duration,
      });
    });

    next();
  };
}

// Export transports
export { ConsoleTransport, FileTransport, HttpTransport };

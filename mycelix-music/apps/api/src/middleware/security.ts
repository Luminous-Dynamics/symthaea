// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Comprehensive Security Middleware
 * Combines multiple security measures into a single configuration
 */

import { Request, Response, NextFunction, RequestHandler, Application } from 'express';
import helmet from 'helmet';
import cors from 'cors';

// Security configuration
export interface SecurityConfig {
  // CORS
  cors?: {
    origins: string[];
    methods?: string[];
    allowedHeaders?: string[];
    exposedHeaders?: string[];
    credentials?: boolean;
    maxAge?: number;
  };
  // Rate limiting
  rateLimit?: {
    windowMs: number;
    max: number;
    message?: string;
    keyGenerator?: (req: Request) => string;
  };
  // Request size limits
  requestLimits?: {
    json?: string;
    urlencoded?: string;
    raw?: string;
  };
  // Content Security Policy
  csp?: {
    directives?: Record<string, string[]>;
    reportOnly?: boolean;
    reportUri?: string;
  };
  // Additional headers
  additionalHeaders?: Record<string, string>;
  // IP filtering
  ipWhitelist?: string[];
  ipBlacklist?: string[];
  // Request validation
  validation?: {
    maxParamLength?: number;
    maxQueryParams?: number;
    allowedContentTypes?: string[];
  };
}

// Default configuration
const DEFAULT_CONFIG: SecurityConfig = {
  cors: {
    origins: ['http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: [
      'Content-Type',
      'Authorization',
      'X-Requested-With',
      'Accept',
      'Origin',
      'X-Wallet-Signature',
      'X-Wallet-Payload',
      'X-Session-Token',
      'X-CSRF-Token',
    ],
    exposedHeaders: [
      'X-RateLimit-Limit',
      'X-RateLimit-Remaining',
      'X-RateLimit-Reset',
      'X-Session-Token',
    ],
    credentials: true,
    maxAge: 86400,
  },
  rateLimit: {
    windowMs: 60000,
    max: 100,
    message: 'Too many requests, please try again later.',
  },
  requestLimits: {
    json: '10mb',
    urlencoded: '10mb',
    raw: '50mb',
  },
  validation: {
    maxParamLength: 500,
    maxQueryParams: 50,
    allowedContentTypes: [
      'application/json',
      'application/x-www-form-urlencoded',
      'multipart/form-data',
    ],
  },
};

/**
 * Apply all security middleware to Express app
 */
export function applySecurity(app: Application, config: SecurityConfig = {}): void {
  const cfg = mergeConfig(DEFAULT_CONFIG, config);

  // Trust proxy (for accurate IP detection behind load balancer)
  app.set('trust proxy', 1);

  // Helmet for security headers
  app.use(createHelmetMiddleware(cfg));

  // CORS
  if (cfg.cors) {
    app.use(createCorsMiddleware(cfg.cors));
  }

  // Request validation
  app.use(requestValidation(cfg.validation));

  // IP filtering
  if (cfg.ipWhitelist?.length || cfg.ipBlacklist?.length) {
    app.use(ipFilter(cfg.ipWhitelist, cfg.ipBlacklist));
  }

  // Additional security headers
  app.use(additionalSecurityHeaders(cfg.additionalHeaders));

  // Request ID for tracing
  app.use(requestId());

  // Security logging
  app.use(securityLogging());
}

/**
 * Create Helmet middleware with custom configuration
 */
function createHelmetMiddleware(config: SecurityConfig): RequestHandler {
  return helmet({
    contentSecurityPolicy: config.csp?.directives
      ? {
          directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
            imgSrc: ["'self'", 'data:', 'blob:', 'https:'],
            fontSrc: ["'self'", 'https://fonts.gstatic.com'],
            connectSrc: ["'self'", 'wss:', 'https:'],
            mediaSrc: ["'self'", 'blob:', 'https:'],
            objectSrc: ["'none'"],
            frameAncestors: ["'none'"],
            ...config.csp.directives,
          },
          reportOnly: config.csp.reportOnly,
        }
      : false,
    crossOriginEmbedderPolicy: false, // Required for audio streaming
    crossOriginResourcePolicy: { policy: 'cross-origin' },
    frameguard: { action: 'deny' },
    hidePoweredBy: true,
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true,
    },
    noSniff: true,
    referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
    xssFilter: true,
  });
}

/**
 * Create CORS middleware
 */
function createCorsMiddleware(config: NonNullable<SecurityConfig['cors']>): RequestHandler {
  return cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (mobile apps, Postman, etc.)
      if (!origin) {
        return callback(null, true);
      }

      // Check against whitelist
      if (config.origins.includes('*') || config.origins.includes(origin)) {
        return callback(null, true);
      }

      // Check for wildcard subdomains
      const isAllowed = config.origins.some((allowed) => {
        if (allowed.startsWith('*.')) {
          const domain = allowed.slice(2);
          return origin.endsWith(domain);
        }
        return false;
      });

      if (isAllowed) {
        return callback(null, true);
      }

      callback(new Error('Not allowed by CORS'));
    },
    methods: config.methods,
    allowedHeaders: config.allowedHeaders,
    exposedHeaders: config.exposedHeaders,
    credentials: config.credentials,
    maxAge: config.maxAge,
  });
}

/**
 * Request validation middleware
 */
function requestValidation(config?: SecurityConfig['validation']): RequestHandler {
  const cfg = config || DEFAULT_CONFIG.validation!;

  return (req: Request, res: Response, next: NextFunction): void => {
    // Check URL path length
    if (req.path.length > (cfg.maxParamLength || 500)) {
      res.status(414).json({
        error: 'URI Too Long',
        message: 'Request path exceeds maximum length',
      });
      return;
    }

    // Check query parameter count
    const queryCount = Object.keys(req.query).length;
    if (queryCount > (cfg.maxQueryParams || 50)) {
      res.status(400).json({
        error: 'Bad Request',
        message: 'Too many query parameters',
      });
      return;
    }

    // Validate content type for POST/PUT/PATCH
    if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
      const contentType = req.headers['content-type']?.split(';')[0];
      if (contentType && cfg.allowedContentTypes) {
        const isAllowed = cfg.allowedContentTypes.some((allowed) =>
          contentType.includes(allowed)
        );
        if (!isAllowed) {
          res.status(415).json({
            error: 'Unsupported Media Type',
            message: `Content-Type '${contentType}' is not supported`,
          });
          return;
        }
      }
    }

    // Check for suspicious patterns
    const suspiciousPatterns = [
      /\.\.\//g, // Path traversal
      /<script/gi, // XSS
      /javascript:/gi, // XSS
      /on\w+\s*=/gi, // Event handlers
    ];

    const urlDecoded = decodeURIComponent(req.url);
    for (const pattern of suspiciousPatterns) {
      if (pattern.test(urlDecoded)) {
        res.status(400).json({
          error: 'Bad Request',
          message: 'Suspicious pattern detected in request',
        });
        return;
      }
    }

    next();
  };
}

/**
 * IP filtering middleware
 */
function ipFilter(whitelist?: string[], blacklist?: string[]): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const ip = getClientIp(req);

    // Check blacklist first
    if (blacklist?.length && blacklist.includes(ip)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Access denied',
      });
      return;
    }

    // Check whitelist if configured
    if (whitelist?.length && !whitelist.includes(ip)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Access denied',
      });
      return;
    }

    next();
  };
}

/**
 * Additional security headers
 */
function additionalSecurityHeaders(headers?: Record<string, string>): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    // Standard security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');

    // Prevent caching of sensitive responses
    if (req.path.includes('/auth') || req.path.includes('/user')) {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
      res.setHeader('Pragma', 'no-cache');
    }

    // Custom headers
    if (headers) {
      Object.entries(headers).forEach(([key, value]) => {
        res.setHeader(key, value);
      });
    }

    next();
  };
}

/**
 * Request ID middleware for tracing
 */
function requestId(): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const id =
      (req.headers['x-request-id'] as string) ||
      `req_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

    req.headers['x-request-id'] = id;
    res.setHeader('X-Request-ID', id);

    next();
  };
}

/**
 * Security logging middleware
 */
function securityLogging(): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = Date.now();
    const requestId = req.headers['x-request-id'] as string;

    // Log request
    const logData = {
      requestId,
      method: req.method,
      path: req.path,
      ip: getClientIp(req),
      userAgent: req.headers['user-agent'],
      timestamp: new Date().toISOString(),
    };

    // Log on response finish
    res.on('finish', () => {
      const duration = Date.now() - startTime;
      const securityLog = {
        ...logData,
        statusCode: res.statusCode,
        duration,
        ...(res.statusCode >= 400 && {
          warning: 'Error response',
          level: res.statusCode >= 500 ? 'error' : 'warn',
        }),
      };

      // Log security-relevant events
      if (res.statusCode === 401 || res.statusCode === 403) {
        console.warn('[Security] Auth failure:', JSON.stringify(securityLog));
      } else if (res.statusCode === 429) {
        console.warn('[Security] Rate limit:', JSON.stringify(securityLog));
      }
    });

    next();
  };
}

/**
 * Get client IP from request
 */
function getClientIp(req: Request): string {
  return (
    (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() ||
    (req.headers['x-real-ip'] as string) ||
    req.socket.remoteAddress ||
    'unknown'
  );
}

/**
 * Deep merge configurations
 */
function mergeConfig(defaults: SecurityConfig, overrides: SecurityConfig): SecurityConfig {
  return {
    ...defaults,
    ...overrides,
    cors: overrides.cors ? { ...defaults.cors, ...overrides.cors } : defaults.cors,
    rateLimit: overrides.rateLimit
      ? { ...defaults.rateLimit, ...overrides.rateLimit }
      : defaults.rateLimit,
    requestLimits: overrides.requestLimits
      ? { ...defaults.requestLimits, ...overrides.requestLimits }
      : defaults.requestLimits,
    validation: overrides.validation
      ? { ...defaults.validation, ...overrides.validation }
      : defaults.validation,
  };
}

/**
 * Error handler for security-related errors
 */
export function securityErrorHandler() {
  return (err: Error, req: Request, res: Response, next: NextFunction): void => {
    // CORS error
    if (err.message === 'Not allowed by CORS') {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Origin not allowed',
        code: 'CORS_BLOCKED',
      });
      return;
    }

    // Pass to next error handler
    next(err);
  };
}

export default { applySecurity, securityErrorHandler };

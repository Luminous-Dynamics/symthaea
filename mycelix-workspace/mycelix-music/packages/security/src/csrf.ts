// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CSRF Protection
 * Cross-Site Request Forgery protection middleware
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import { createHash, randomBytes } from 'crypto';

// CSRF token configuration
export interface CsrfOptions {
  /** Cookie name for CSRF token */
  cookieName?: string;
  /** Header name for CSRF token */
  headerName?: string;
  /** Token expiry in seconds */
  expirySeconds?: number;
  /** Skip CSRF for these methods */
  ignoreMethods?: string[];
  /** Skip CSRF for these paths (regex patterns) */
  ignorePaths?: RegExp[];
  /** Secret for signing tokens */
  secret?: string;
  /** Cookie options */
  cookie?: {
    httpOnly?: boolean;
    secure?: boolean;
    sameSite?: 'strict' | 'lax' | 'none';
    path?: string;
    domain?: string;
  };
}

interface TokenData {
  token: string;
  timestamp: number;
  signature: string;
}

const DEFAULT_OPTIONS: Required<CsrfOptions> = {
  cookieName: '_csrf',
  headerName: 'x-csrf-token',
  expirySeconds: 3600, // 1 hour
  ignoreMethods: ['GET', 'HEAD', 'OPTIONS'],
  ignorePaths: [],
  secret: process.env.CSRF_SECRET || 'change-me-in-production',
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
  },
};

/**
 * Create CSRF protection middleware
 */
export function csrfProtection(options: CsrfOptions = {}): RequestHandler {
  const config = { ...DEFAULT_OPTIONS, ...options };

  return (req: Request, res: Response, next: NextFunction): void => {
    // Skip for ignored methods
    if (config.ignoreMethods.includes(req.method)) {
      return next();
    }

    // Skip for ignored paths
    const shouldSkip = config.ignorePaths.some((pattern) => pattern.test(req.path));
    if (shouldSkip) {
      return next();
    }

    // Get token from cookie
    const cookieToken = req.cookies?.[config.cookieName];

    // Get token from header or body
    const submittedToken =
      req.headers[config.headerName.toLowerCase()] as string ||
      req.body?._csrf;

    // Validate both tokens exist
    if (!cookieToken || !submittedToken) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token missing',
        code: 'CSRF_MISSING',
      });
      return;
    }

    // Parse and validate cookie token
    let tokenData: TokenData;
    try {
      tokenData = JSON.parse(
        Buffer.from(cookieToken, 'base64').toString('utf8')
      );
    } catch {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Invalid CSRF token format',
        code: 'CSRF_INVALID',
      });
      return;
    }

    // Validate token hasn't expired
    const now = Date.now();
    if (now - tokenData.timestamp > config.expirySeconds * 1000) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token expired',
        code: 'CSRF_EXPIRED',
      });
      return;
    }

    // Validate signature
    const expectedSignature = signToken(tokenData.token, tokenData.timestamp, config.secret);
    if (tokenData.signature !== expectedSignature) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token signature invalid',
        code: 'CSRF_SIGNATURE',
      });
      return;
    }

    // Validate submitted token matches cookie token
    if (!timingSafeEqual(submittedToken, tokenData.token)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token mismatch',
        code: 'CSRF_MISMATCH',
      });
      return;
    }

    next();
  };
}

/**
 * Generate a new CSRF token and set cookie
 */
export function generateCsrfToken(
  res: Response,
  options: CsrfOptions = {}
): string {
  const config = { ...DEFAULT_OPTIONS, ...options };

  const token = randomBytes(32).toString('hex');
  const timestamp = Date.now();
  const signature = signToken(token, timestamp, config.secret);

  const tokenData: TokenData = { token, timestamp, signature };
  const cookieValue = Buffer.from(JSON.stringify(tokenData)).toString('base64');

  res.cookie(config.cookieName, cookieValue, {
    httpOnly: config.cookie.httpOnly,
    secure: config.cookie.secure,
    sameSite: config.cookie.sameSite,
    path: config.cookie.path,
    domain: config.cookie.domain,
    maxAge: config.expirySeconds * 1000,
  });

  return token;
}

/**
 * CSRF token endpoint handler
 */
export function csrfTokenHandler(options: CsrfOptions = {}): RequestHandler {
  return (req: Request, res: Response): void => {
    const token = generateCsrfToken(res, options);
    res.json({ csrfToken: token });
  };
}

/**
 * Double Submit Cookie pattern middleware
 * Alternative to synchronizer token pattern
 */
export function doubleSubmitCookie(options: CsrfOptions = {}): RequestHandler {
  const config = { ...DEFAULT_OPTIONS, ...options };

  return (req: Request, res: Response, next: NextFunction): void => {
    // Skip for ignored methods
    if (config.ignoreMethods.includes(req.method)) {
      return next();
    }

    const cookieToken = req.cookies?.[config.cookieName];
    const headerToken = req.headers[config.headerName.toLowerCase()] as string;

    if (!cookieToken || !headerToken) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token missing',
        code: 'CSRF_MISSING',
      });
      return;
    }

    // Compare tokens using timing-safe comparison
    if (!timingSafeEqual(cookieToken, headerToken)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'CSRF token mismatch',
        code: 'CSRF_MISMATCH',
      });
      return;
    }

    next();
  };
}

// Helper: Sign token
function signToken(token: string, timestamp: number, secret: string): string {
  return createHash('sha256')
    .update(`${token}:${timestamp}:${secret}`)
    .digest('hex');
}

// Helper: Timing-safe string comparison
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

/**
 * Express middleware to attach CSRF helper to response
 */
export function csrfHelpers(options: CsrfOptions = {}): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    // Attach method to generate tokens
    (res as any).csrfToken = () => generateCsrfToken(res, options);
    next();
  };
}

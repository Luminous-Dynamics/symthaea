// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Security Headers
 *
 * Helmet configuration and CORS middleware.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import helmet from 'helmet';

/**
 * Create security headers middleware using Helmet
 */
export function securityHeaders(): RequestHandler {
  return helmet({
    // Content Security Policy
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"], // Required for Next.js
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'blob:', 'https:'],
        fontSrc: ["'self'", 'https://fonts.gstatic.com'],
        connectSrc: [
          "'self'",
          'wss:', // WebSocket connections
          'https://api.mycelix.io',
          'https://*.infura.io', // Web3 providers
          'https://*.alchemy.com',
        ],
        mediaSrc: ["'self'", 'blob:', 'https:'],
        objectSrc: ["'none'"],
        frameAncestors: ["'none'"],
        upgradeInsecureRequests: [],
      },
    },
    // Disable embedding in iframes
    frameguard: { action: 'deny' },
    // Hide X-Powered-By header
    hidePoweredBy: true,
    // Strict Transport Security
    hsts: {
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true,
    },
    // Prevent MIME type sniffing
    noSniff: true,
    // Referrer Policy
    referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
    // XSS Protection
    xssFilter: true,
  });
}

export interface CorsOptions {
  origins?: string[];
  methods?: string[];
  allowedHeaders?: string[];
  exposedHeaders?: string[];
  credentials?: boolean;
  maxAge?: number;
}

/**
 * Create CORS middleware
 */
export function corsMiddleware(options: CorsOptions = {}): RequestHandler {
  const {
    origins = ['http://localhost:3000'],
    methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders = [
      'Content-Type',
      'Authorization',
      'X-Requested-With',
      'Accept',
      'Origin',
    ],
    exposedHeaders = [
      'X-RateLimit-Limit',
      'X-RateLimit-Remaining',
      'X-RateLimit-Reset',
    ],
    credentials = true,
    maxAge = 86400, // 24 hours
  } = options;

  return (req: Request, res: Response, next: NextFunction): void => {
    const origin = req.headers.origin;

    // Check if origin is allowed
    if (origin) {
      const isAllowed = origins.includes('*') || origins.includes(origin);
      if (isAllowed) {
        res.setHeader('Access-Control-Allow-Origin', origin);
      }
    }

    // Set CORS headers
    res.setHeader('Access-Control-Allow-Methods', methods.join(', '));
    res.setHeader('Access-Control-Allow-Headers', allowedHeaders.join(', '));
    res.setHeader('Access-Control-Expose-Headers', exposedHeaders.join(', '));
    res.setHeader('Access-Control-Max-Age', maxAge.toString());

    if (credentials) {
      res.setHeader('Access-Control-Allow-Credentials', 'true');
    }

    // Handle preflight requests
    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }

    next();
  };
}

/**
 * Additional security headers for API responses
 */
export function apiSecurityHeaders(): RequestHandler {
  return (req: Request, res: Response, next: NextFunction): void => {
    // Prevent caching of sensitive data
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'DENY');

    // Prevent content type sniffing
    res.setHeader('X-Content-Type-Options', 'nosniff');

    // Enable XSS filter
    res.setHeader('X-XSS-Protection', '1; mode=block');

    next();
  };
}

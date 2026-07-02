// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Middleware
 *
 * Express middleware for automatic response caching.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import { CacheClient } from './client';

export interface CacheMiddlewareOptions {
  cache: CacheClient;
  ttl?: number;
  keyGenerator?: (req: Request) => string;
  condition?: (req: Request) => boolean;
  statusCodes?: number[];
  varyBy?: string[]; // Headers to vary by
}

/**
 * Create cache middleware for GET requests
 */
export function cacheMiddleware(options: CacheMiddlewareOptions): RequestHandler {
  const {
    cache,
    ttl = 300,
    keyGenerator = defaultKeyGenerator,
    condition = () => true,
    statusCodes = [200],
    varyBy = [],
  } = options;

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Only cache GET requests
    if (req.method !== 'GET') {
      return next();
    }

    // Check condition
    if (!condition(req)) {
      return next();
    }

    const cacheKey = keyGenerator(req);

    // Try to get from cache
    const cached = await cache.get<{
      status: number;
      headers: Record<string, string>;
      body: unknown;
    }>(cacheKey);

    if (cached) {
      // Set headers
      for (const [key, value] of Object.entries(cached.headers)) {
        res.setHeader(key, value);
      }
      res.setHeader('X-Cache', 'HIT');
      res.status(cached.status).json(cached.body);
      return;
    }

    // Store original json method
    const originalJson = res.json.bind(res);

    // Override json to cache the response
    res.json = function (body: unknown): Response {
      // Only cache allowed status codes
      if (statusCodes.includes(res.statusCode)) {
        const headers: Record<string, string> = {};

        // Store relevant headers
        const headersToCache = ['content-type', 'etag', ...varyBy];
        for (const header of headersToCache) {
          const value = res.getHeader(header);
          if (value) {
            headers[header] = String(value);
          }
        }

        cache.set(cacheKey, {
          status: res.statusCode,
          headers,
          body,
        }, ttl).catch(console.error);
      }

      res.setHeader('X-Cache', 'MISS');
      return originalJson(body);
    };

    next();
  };
}

function defaultKeyGenerator(req: Request): string {
  const parts = [
    req.method,
    req.originalUrl || req.url,
  ];

  // Include user ID if authenticated (for user-specific caching)
  const userId = (req as any).user?.id;
  if (userId) {
    parts.push(`user:${userId}`);
  }

  return parts.join(':');
}

/**
 * Create cache control middleware for setting cache headers
 */
export function cacheControl(options: {
  maxAge?: number;
  sMaxAge?: number;
  staleWhileRevalidate?: number;
  staleIfError?: number;
  private?: boolean;
  noStore?: boolean;
  mustRevalidate?: boolean;
}): RequestHandler {
  const directives: string[] = [];

  if (options.noStore) {
    directives.push('no-store');
  } else {
    if (options.private) {
      directives.push('private');
    } else {
      directives.push('public');
    }

    if (options.maxAge !== undefined) {
      directives.push(`max-age=${options.maxAge}`);
    }

    if (options.sMaxAge !== undefined) {
      directives.push(`s-maxage=${options.sMaxAge}`);
    }

    if (options.staleWhileRevalidate !== undefined) {
      directives.push(`stale-while-revalidate=${options.staleWhileRevalidate}`);
    }

    if (options.staleIfError !== undefined) {
      directives.push(`stale-if-error=${options.staleIfError}`);
    }

    if (options.mustRevalidate) {
      directives.push('must-revalidate');
    }
  }

  const headerValue = directives.join(', ');

  return (req: Request, res: Response, next: NextFunction): void => {
    res.setHeader('Cache-Control', headerValue);
    next();
  };
}

/**
 * Middleware to invalidate cache on mutations
 */
export function invalidateCache(
  cache: CacheClient,
  patterns: string[] | ((req: Request) => string[])
): RequestHandler {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Store original end method
    const originalEnd = res.end.bind(res);

    // Override to invalidate cache after successful mutation
    res.end = function (...args: any[]): Response {
      if (res.statusCode >= 200 && res.statusCode < 300) {
        const patternsToInvalidate = typeof patterns === 'function'
          ? patterns(req)
          : patterns;

        Promise.all(
          patternsToInvalidate.map(p => cache.invalidatePattern(p))
        ).catch(console.error);
      }

      return originalEnd(...args);
    };

    next();
  };
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Middleware
 *
 * HTTP caching middleware for Express routes with:
 * - Response caching
 * - Cache-Control headers
 * - ETag support
 * - Conditional requests (If-None-Match)
 */

import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import { CacheService } from '../services/cache.service';

/**
 * Cache middleware options
 */
export interface CacheMiddlewareOptions {
  ttl: number;              // Cache TTL in seconds
  key?: (req: Request) => string;  // Custom key generator
  tags?: string[];          // Cache tags
  private?: boolean;        // Cache-Control: private
  staleWhileRevalidate?: number;  // SWR window in seconds
}

/**
 * Generate cache key from request
 */
function defaultKeyGenerator(req: Request): string {
  const parts = [
    req.method,
    req.originalUrl || req.url,
    req.headers['accept-version'] || 'v1',
  ];
  return parts.join(':');
}

/**
 * Generate ETag from content
 */
function generateETag(content: string): string {
  const hash = createHash('md5').update(content).digest('hex');
  return `"${hash}"`;
}

/**
 * Create cache middleware
 */
export function cacheMiddleware(
  cacheService: CacheService,
  options: CacheMiddlewareOptions
) {
  const keyFn = options.key || defaultKeyGenerator;

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Only cache GET requests
    if (req.method !== 'GET') {
      return next();
    }

    const cacheKey = keyFn(req);

    try {
      // Check cache
      const cached = await cacheService.get<{
        body: string;
        headers: Record<string, string>;
        etag: string;
      }>(cacheKey);

      if (cached) {
        // Check If-None-Match header
        const ifNoneMatch = req.headers['if-none-match'];
        if (ifNoneMatch === cached.etag) {
          res.status(304).end();
          return;
        }

        // Set cached headers
        for (const [key, value] of Object.entries(cached.headers)) {
          res.setHeader(key, value);
        }
        res.setHeader('ETag', cached.etag);
        res.setHeader('X-Cache', 'HIT');

        res.send(cached.body);
        return;
      }

      // Cache miss - intercept response
      res.setHeader('X-Cache', 'MISS');

      const originalJson = res.json.bind(res);
      res.json = function (body: unknown): Response {
        // Only cache successful responses
        if (res.statusCode >= 200 && res.statusCode < 300) {
          const bodyStr = JSON.stringify(body);
          const etag = generateETag(bodyStr);

          // Build Cache-Control header
          const directives = [
            options.private ? 'private' : 'public',
            `max-age=${options.ttl}`,
          ];
          if (options.staleWhileRevalidate) {
            directives.push(`stale-while-revalidate=${options.staleWhileRevalidate}`);
          }

          const cacheControl = directives.join(', ');

          // Set headers on response
          res.setHeader('Cache-Control', cacheControl);
          res.setHeader('ETag', etag);

          // Store in cache
          cacheService.set(
            cacheKey,
            {
              body: bodyStr,
              headers: { 'Content-Type': 'application/json', 'Cache-Control': cacheControl },
              etag,
            },
            { ttl: options.ttl, tags: options.tags }
          ).catch(err => {
            console.error('Cache write error:', err);
          });
        }

        return originalJson(body);
      };

      next();
    } catch (error) {
      // On cache error, proceed without caching
      console.error('Cache middleware error:', error);
      next();
    }
  };
}

/**
 * No-cache middleware for routes that shouldn't be cached
 */
export function noCache(req: Request, res: Response, next: NextFunction): void {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  next();
}

/**
 * Set cache headers without storing in cache service
 */
export function cacheHeaders(options: {
  maxAge: number;
  private?: boolean;
  staleWhileRevalidate?: number;
}) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const directives = [
      options.private ? 'private' : 'public',
      `max-age=${options.maxAge}`,
    ];
    if (options.staleWhileRevalidate) {
      directives.push(`stale-while-revalidate=${options.staleWhileRevalidate}`);
    }

    res.setHeader('Cache-Control', directives.join(', '));
    next();
  };
}

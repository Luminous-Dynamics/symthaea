// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ETag and Conditional Requests Middleware
 *
 * Implements HTTP caching with:
 * - ETag generation and validation
 * - If-None-Match support (GET/HEAD)
 * - If-Match support (PUT/PATCH/DELETE)
 * - If-Modified-Since support
 * - Last-Modified headers
 */

import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import { getMetrics } from '../metrics';
import { getLogger } from '../logging';

const logger = getLogger();

/**
 * ETag configuration
 */
export interface ETagConfig {
  /** Use weak ETags (W/"...") */
  weak: boolean;
  /** Hash algorithm */
  algorithm: 'md5' | 'sha1' | 'sha256';
  /** Include headers in ETag calculation */
  includeHeaders?: string[];
}

/**
 * Default configuration
 */
const defaultConfig: ETagConfig = {
  weak: true,
  algorithm: 'md5',
};

/**
 * Generate ETag from content
 */
export function generateETag(
  content: string | Buffer | object,
  config: ETagConfig = defaultConfig
): string {
  let data: string;

  if (typeof content === 'object' && !(content instanceof Buffer)) {
    data = JSON.stringify(content);
  } else if (content instanceof Buffer) {
    data = content.toString('utf-8');
  } else {
    data = content;
  }

  const hash = createHash(config.algorithm)
    .update(data)
    .digest('hex')
    .slice(0, 27); // Shorten for readability

  return config.weak ? `W/"${hash}"` : `"${hash}"`;
}

/**
 * Generate ETag from version/timestamp
 */
export function generateVersionETag(
  id: string,
  updatedAt: Date | string,
  version?: number
): string {
  const timestamp = updatedAt instanceof Date
    ? updatedAt.getTime()
    : new Date(updatedAt).getTime();

  const content = version !== undefined
    ? `${id}:${version}:${timestamp}`
    : `${id}:${timestamp}`;

  return generateETag(content);
}

/**
 * Parse If-None-Match header
 */
function parseIfNoneMatch(header: string): string[] {
  if (header === '*') return ['*'];

  return header
    .split(',')
    .map(tag => tag.trim())
    .filter(Boolean);
}

/**
 * Check if ETag matches
 */
function etagMatches(etag: string, matches: string[]): boolean {
  if (matches.includes('*')) return true;

  // Normalize for comparison (remove weak prefix)
  const normalizedEtag = etag.replace(/^W\//, '');

  return matches.some(match => {
    const normalizedMatch = match.replace(/^W\//, '');
    return normalizedEtag === normalizedMatch;
  });
}

/**
 * Parse If-Modified-Since header
 */
function parseIfModifiedSince(header: string): Date | null {
  try {
    const date = new Date(header);
    return isNaN(date.getTime()) ? null : date;
  } catch {
    return null;
  }
}

/**
 * Conditional GET middleware
 *
 * Handles If-None-Match and If-Modified-Since for GET requests.
 * Returns 304 Not Modified if conditions are met.
 */
export function conditionalGet(config: Partial<ETagConfig> = {}) {
  const cfg = { ...defaultConfig, ...config };
  const metrics = getMetrics();

  return (req: Request, res: Response, next: NextFunction): void => {
    // Only apply to GET and HEAD
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return next();
    }

    // Store original json method
    const originalJson = res.json.bind(res);

    // Override json to add ETag
    res.json = function (body: unknown): Response {
      // Generate ETag from response body
      const etag = generateETag(body as object, cfg);
      res.setHeader('ETag', etag);

      // Check If-None-Match
      const ifNoneMatch = req.headers['if-none-match'] as string;
      if (ifNoneMatch) {
        const matches = parseIfNoneMatch(ifNoneMatch);
        if (etagMatches(etag, matches)) {
          metrics.incCounter('conditional_304_total', { type: 'etag' });
          logger.debug('Returning 304 Not Modified (ETag match)', { etag });
          res.status(304).end();
          return res;
        }
      }

      // Check If-Modified-Since (if Last-Modified is set)
      const lastModified = res.getHeader('Last-Modified') as string;
      if (lastModified) {
        const ifModifiedSince = req.headers['if-modified-since'] as string;
        if (ifModifiedSince) {
          const modifiedSince = parseIfModifiedSince(ifModifiedSince);
          const lastMod = parseIfModifiedSince(lastModified);

          if (modifiedSince && lastMod && lastMod <= modifiedSince) {
            metrics.incCounter('conditional_304_total', { type: 'modified' });
            logger.debug('Returning 304 Not Modified (Last-Modified)', {
              lastModified,
              ifModifiedSince,
            });
            res.status(304).end();
            return res;
          }
        }
      }

      return originalJson(body);
    };

    next();
  };
}

/**
 * Conditional update middleware
 *
 * Handles If-Match for PUT/PATCH/DELETE requests.
 * Returns 412 Precondition Failed if conditions are not met.
 */
export function conditionalUpdate() {
  return (req: Request, res: Response, next: NextFunction): void => {
    // Only apply to mutation methods
    if (!['PUT', 'PATCH', 'DELETE'].includes(req.method)) {
      return next();
    }

    const ifMatch = req.headers['if-match'] as string;
    if (!ifMatch) {
      return next();
    }

    // Store the If-Match value for later validation
    (req as any).ifMatch = parseIfNoneMatch(ifMatch);

    next();
  };
}

/**
 * Validate If-Match precondition
 * Call this after fetching the current resource
 */
export function validateIfMatch(
  req: Request,
  res: Response,
  currentETag: string
): boolean {
  const ifMatch = (req as any).ifMatch as string[] | undefined;

  if (!ifMatch) {
    return true; // No precondition
  }

  if (etagMatches(currentETag, ifMatch)) {
    return true;
  }

  // Precondition failed
  res.status(412).json({
    success: false,
    error: {
      code: 'PRECONDITION_FAILED',
      message: 'Resource has been modified. Please refresh and try again.',
      currentETag,
    },
  });

  return false;
}

/**
 * Set Last-Modified header
 */
export function setLastModified(res: Response, date: Date | string): void {
  const d = date instanceof Date ? date : new Date(date);
  res.setHeader('Last-Modified', d.toUTCString());
}

/**
 * Set cache control headers
 */
export interface CacheControlOptions {
  /** Max age in seconds */
  maxAge?: number;
  /** Shared max age (CDN) */
  sMaxAge?: number;
  /** Private (user-specific) */
  private?: boolean;
  /** No cache (must revalidate) */
  noCache?: boolean;
  /** No store */
  noStore?: boolean;
  /** Must revalidate */
  mustRevalidate?: boolean;
  /** Stale while revalidate */
  staleWhileRevalidate?: number;
  /** Stale if error */
  staleIfError?: number;
}

export function setCacheControl(res: Response, options: CacheControlOptions): void {
  const directives: string[] = [];

  if (options.noStore) {
    directives.push('no-store');
  } else {
    if (options.private) {
      directives.push('private');
    } else {
      directives.push('public');
    }

    if (options.noCache) {
      directives.push('no-cache');
    }

    if (options.maxAge !== undefined) {
      directives.push(`max-age=${options.maxAge}`);
    }

    if (options.sMaxAge !== undefined) {
      directives.push(`s-maxage=${options.sMaxAge}`);
    }

    if (options.mustRevalidate) {
      directives.push('must-revalidate');
    }

    if (options.staleWhileRevalidate !== undefined) {
      directives.push(`stale-while-revalidate=${options.staleWhileRevalidate}`);
    }

    if (options.staleIfError !== undefined) {
      directives.push(`stale-if-error=${options.staleIfError}`);
    }
  }

  res.setHeader('Cache-Control', directives.join(', '));
}

/**
 * Cache control presets
 */
export const cachePresets = {
  /** No caching at all */
  noCache: (): CacheControlOptions => ({
    noStore: true,
    noCache: true,
  }),

  /** Private, short cache with revalidation */
  private: (maxAge = 60): CacheControlOptions => ({
    private: true,
    maxAge,
    mustRevalidate: true,
  }),

  /** Public, medium cache */
  public: (maxAge = 300): CacheControlOptions => ({
    maxAge,
    sMaxAge: maxAge * 2,
    staleWhileRevalidate: 60,
  }),

  /** Long cache for immutable content */
  immutable: (maxAge = 31536000): CacheControlOptions => ({
    maxAge,
    sMaxAge: maxAge,
  }),

  /** API response (short cache, revalidate) */
  api: (maxAge = 10): CacheControlOptions => ({
    private: true,
    maxAge,
    mustRevalidate: true,
    staleWhileRevalidate: 30,
    staleIfError: 300,
  }),
};

/**
 * Vary header helper
 */
export function setVary(res: Response, headers: string[]): void {
  const existing = res.getHeader('Vary') as string | undefined;
  const all = existing ? existing.split(',').map(h => h.trim()) : [];

  for (const header of headers) {
    if (!all.includes(header)) {
      all.push(header);
    }
  }

  res.setHeader('Vary', all.join(', '));
}

export default {
  generateETag,
  generateVersionETag,
  conditionalGet,
  conditionalUpdate,
  validateIfMatch,
  setLastModified,
  setCacheControl,
  cachePresets,
  setVary,
};

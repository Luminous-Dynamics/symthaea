// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Rate Limiting Middleware
 *
 * Production-grade rate limiting with:
 * - Sliding window algorithm
 * - Redis support for distributed systems
 * - IP and user-based limiting
 * - Configurable limits per route
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import Redis from 'ioredis';
import { errors } from '../utils/response';

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  /** Maximum requests allowed in the window */
  max: number;
  /** Time window in seconds */
  windowSeconds: number;
  /** Key prefix for Redis */
  keyPrefix?: string;
  /** Custom key generator */
  keyGenerator?: (req: Request) => string;
  /** Skip rate limiting for certain requests */
  skip?: (req: Request) => boolean;
  /** Custom response when rate limited */
  onRateLimited?: (req: Request, res: Response) => void;
}

/**
 * Rate limit result
 */
export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  total: number;
}

/**
 * Rate limiter interface
 */
export interface RateLimiter {
  check(key: string): Promise<RateLimitResult>;
  reset(key: string): Promise<void>;
}

/**
 * In-memory rate limiter (for development/testing)
 */
export class MemoryRateLimiter implements RateLimiter {
  private store = new Map<string, { count: number; resetAt: number }>();
  private cleanupInterval: NodeJS.Timeout;

  constructor(
    private readonly max: number,
    private readonly windowMs: number
  ) {
    // Cleanup expired entries every minute
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
  }

  async check(key: string): Promise<RateLimitResult> {
    const now = Date.now();
    const entry = this.store.get(key);

    // If no entry or expired, create new window
    if (!entry || now > entry.resetAt) {
      const resetAt = now + this.windowMs;
      this.store.set(key, { count: 1, resetAt });
      return {
        allowed: true,
        remaining: this.max - 1,
        resetAt,
        total: this.max,
      };
    }

    // Increment count
    entry.count++;
    const allowed = entry.count <= this.max;

    return {
      allowed,
      remaining: Math.max(0, this.max - entry.count),
      resetAt: entry.resetAt,
      total: this.max,
    };
  }

  async reset(key: string): Promise<void> {
    this.store.delete(key);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.store.entries()) {
      if (now > entry.resetAt) {
        this.store.delete(key);
      }
    }
  }

  destroy(): void {
    clearInterval(this.cleanupInterval);
    this.store.clear();
  }
}

/**
 * Redis-based rate limiter (for production)
 */
export class RedisRateLimiter implements RateLimiter {
  constructor(
    private readonly client: Redis,
    private readonly max: number,
    private readonly windowSeconds: number,
    private readonly keyPrefix = 'ratelimit:'
  ) {}

  async check(key: string): Promise<RateLimitResult> {
    const redisKey = `${this.keyPrefix}${key}`;
    const now = Math.floor(Date.now() / 1000);
    const windowStart = now - this.windowSeconds;

    // Use Redis transaction for atomic operations
    const results = await this.client
      .multi()
      // Remove expired entries
      .zremrangebyscore(redisKey, '-inf', windowStart)
      // Add current request
      .zadd(redisKey, now.toString(), `${now}:${Math.random()}`)
      // Count requests in window
      .zcard(redisKey)
      // Set expiry
      .expire(redisKey, this.windowSeconds)
      .exec();

    const count = (results?.[2]?.[1] as number) || 0;
    const allowed = count <= this.max;
    const resetAt = (now + this.windowSeconds) * 1000;

    return {
      allowed,
      remaining: Math.max(0, this.max - count),
      resetAt,
      total: this.max,
    };
  }

  async reset(key: string): Promise<void> {
    const redisKey = `${this.keyPrefix}${key}`;
    await this.client.del(redisKey);
  }
}

/**
 * Default key generator (by IP)
 */
function defaultKeyGenerator(req: Request): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return req.ip || req.socket.remoteAddress || 'unknown';
}

/**
 * Create rate limiting middleware
 */
export function rateLimit(
  limiter: RateLimiter,
  config: Partial<RateLimitConfig> = {}
): RequestHandler {
  const keyGenerator = config.keyGenerator || defaultKeyGenerator;
  const keyPrefix = config.keyPrefix || '';

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Check if should skip
    if (config.skip?.(req)) {
      return next();
    }

    const key = keyPrefix + keyGenerator(req);

    try {
      const result = await limiter.check(key);

      // Set rate limit headers
      res.setHeader('X-RateLimit-Limit', result.total);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      res.setHeader('X-RateLimit-Reset', Math.ceil(result.resetAt / 1000));

      if (!result.allowed) {
        const retryAfter = Math.ceil((result.resetAt - Date.now()) / 1000);
        res.setHeader('Retry-After', retryAfter);

        if (config.onRateLimited) {
          config.onRateLimited(req, res);
        } else {
          errors.rateLimited(res, retryAfter);
        }
        return;
      }

      next();
    } catch (error) {
      // On error, allow the request (fail open)
      console.error('Rate limiter error:', error);
      next();
    }
  };
}

/**
 * Preset rate limit configurations
 */
export const RateLimitPresets = {
  /** Standard API rate limit */
  standard: (limiter: RateLimiter) =>
    rateLimit(limiter, { keyPrefix: 'std:' }),

  /** Strict rate limit for sensitive operations */
  strict: (limiter: RateLimiter) =>
    rateLimit(limiter, { keyPrefix: 'strict:' }),

  /** Auth endpoint rate limit (prevent brute force) */
  auth: (limiter: RateLimiter) =>
    rateLimit(limiter, {
      keyPrefix: 'auth:',
      keyGenerator: (req) => {
        // Rate limit by IP + attempted username/address
        const ip = defaultKeyGenerator(req);
        const identifier = req.body?.address || req.body?.username || '';
        return `${ip}:${identifier}`;
      },
    }),

  /** Upload rate limit */
  upload: (limiter: RateLimiter) =>
    rateLimit(limiter, { keyPrefix: 'upload:' }),

  /** Search rate limit (prevent abuse) */
  search: (limiter: RateLimiter) =>
    rateLimit(limiter, { keyPrefix: 'search:' }),
};

/**
 * Create rate limiters for different tiers
 */
export function createRateLimiters(redis?: Redis): {
  standard: RateLimiter;
  strict: RateLimiter;
  auth: RateLimiter;
} {
  if (redis) {
    return {
      standard: new RedisRateLimiter(redis, 100, 60),     // 100 req/min
      strict: new RedisRateLimiter(redis, 10, 60),        // 10 req/min
      auth: new RedisRateLimiter(redis, 5, 300),          // 5 req/5min
    };
  }

  // Memory fallback for development
  return {
    standard: new MemoryRateLimiter(100, 60000),
    strict: new MemoryRateLimiter(10, 60000),
    auth: new MemoryRateLimiter(5, 300000),
  };
}

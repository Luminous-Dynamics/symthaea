// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Rate Limiter
 *
 * Redis-backed rate limiting with multiple strategies.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import {
  RateLimiterRedis,
  RateLimiterMemory,
  RateLimiterRes,
  IRateLimiterOptions,
} from 'rate-limiter-flexible';
import Redis from 'ioredis';

export interface RateLimiterOptions {
  redis?: Redis;
  keyPrefix?: string;
  points?: number; // Number of requests
  duration?: number; // Per seconds
  blockDuration?: number; // Block for seconds after limit exceeded
  inmemoryFallback?: boolean;
}

export interface RateLimitInfo {
  limit: number;
  current: number;
  remaining: number;
  resetAt: Date;
}

/**
 * Create a rate limiter middleware
 */
export function createRateLimiter(options: RateLimiterOptions = {}): RequestHandler {
  const {
    redis,
    keyPrefix = 'rl',
    points = 100,
    duration = 60,
    blockDuration = 0,
    inmemoryFallback = true,
  } = options;

  const limiterOptions: IRateLimiterOptions = {
    keyPrefix,
    points,
    duration,
    blockDuration,
  };

  let limiter: RateLimiterRedis | RateLimiterMemory;

  if (redis) {
    limiter = new RateLimiterRedis({
      ...limiterOptions,
      storeClient: redis,
      insuranceLimiter: inmemoryFallback
        ? new RateLimiterMemory(limiterOptions)
        : undefined,
    });
  } else {
    limiter = new RateLimiterMemory(limiterOptions);
  }

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const key = getKey(req);

    try {
      const result = await limiter.consume(key);
      setRateLimitHeaders(res, result, points);
      next();
    } catch (error) {
      if (error instanceof RateLimiterRes) {
        setRateLimitHeaders(res, error, points);
        res.status(429).json({
          error: 'Too Many Requests',
          message: 'Rate limit exceeded. Please try again later.',
          retryAfter: Math.ceil(error.msBeforeNext / 1000),
        });
      } else {
        // On error, allow the request (fail open)
        next();
      }
    }
  };
}

/**
 * Create tiered rate limiters with different limits per tier
 */
export interface TieredRateLimiterOptions {
  redis?: Redis;
  tiers: {
    anonymous: { points: number; duration: number };
    authenticated: { points: number; duration: number };
    premium: { points: number; duration: number };
  };
}

export function createTieredRateLimiter(options: TieredRateLimiterOptions) {
  const { redis, tiers } = options;

  const limiters = {
    anonymous: createLimiter(redis, 'rl:anon', tiers.anonymous),
    authenticated: createLimiter(redis, 'rl:auth', tiers.authenticated),
    premium: createLimiter(redis, 'rl:premium', tiers.premium),
  };

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const key = getKey(req);
    const tier = getTier(req);
    const limiter = limiters[tier];
    const points = tiers[tier].points;

    try {
      const result = await limiter.consume(key);
      setRateLimitHeaders(res, result, points);
      next();
    } catch (error) {
      if (error instanceof RateLimiterRes) {
        setRateLimitHeaders(res, error, points);
        res.status(429).json({
          error: 'Too Many Requests',
          message: 'Rate limit exceeded. Please try again later.',
          retryAfter: Math.ceil(error.msBeforeNext / 1000),
        });
      } else {
        next();
      }
    }
  };
}

/**
 * Create a sliding window rate limiter for more even distribution
 */
export function createSlidingWindowLimiter(options: RateLimiterOptions = {}): RequestHandler {
  const { redis, keyPrefix = 'rl:sw', points = 100, duration = 60 } = options;

  // Use a combination of fixed window + sliding log for efficiency
  const limiter = redis
    ? new RateLimiterRedis({
        keyPrefix,
        points,
        duration,
        storeClient: redis,
        execEvenly: true, // Distribute points evenly
      })
    : new RateLimiterMemory({
        keyPrefix,
        points,
        duration,
        execEvenly: true,
      });

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const key = getKey(req);

    try {
      const result = await limiter.consume(key);
      setRateLimitHeaders(res, result, points);
      next();
    } catch (error) {
      if (error instanceof RateLimiterRes) {
        setRateLimitHeaders(res, error, points);
        res.status(429).json({
          error: 'Too Many Requests',
          message: 'Rate limit exceeded.',
          retryAfter: Math.ceil(error.msBeforeNext / 1000),
        });
      } else {
        next();
      }
    }
  };
}

// Helper functions

function createLimiter(
  redis: Redis | undefined,
  keyPrefix: string,
  config: { points: number; duration: number }
): RateLimiterRedis | RateLimiterMemory {
  if (redis) {
    return new RateLimiterRedis({
      keyPrefix,
      points: config.points,
      duration: config.duration,
      storeClient: redis,
    });
  }
  return new RateLimiterMemory({
    keyPrefix,
    points: config.points,
    duration: config.duration,
  });
}

function getKey(req: Request): string {
  // Use user ID if authenticated, otherwise use IP
  const userId = (req as any).user?.id || (req as any).user?.address;
  if (userId) return `user:${userId}`;

  // Get real IP behind proxy
  const ip =
    req.headers['x-forwarded-for']?.toString().split(',')[0]?.trim() ||
    req.headers['x-real-ip']?.toString() ||
    req.socket.remoteAddress ||
    'unknown';

  return `ip:${ip}`;
}

function getTier(req: Request): 'anonymous' | 'authenticated' | 'premium' {
  const user = (req as any).user;
  if (!user) return 'anonymous';
  if (user.subscription === 'premium' || user.role === 'admin') return 'premium';
  return 'authenticated';
}

function setRateLimitHeaders(res: Response, result: RateLimiterRes, limit: number): void {
  res.set({
    'X-RateLimit-Limit': limit.toString(),
    'X-RateLimit-Remaining': Math.max(0, result.remainingPoints).toString(),
    'X-RateLimit-Reset': new Date(Date.now() + result.msBeforeNext).toISOString(),
    'Retry-After': Math.ceil(result.msBeforeNext / 1000).toString(),
  });
}

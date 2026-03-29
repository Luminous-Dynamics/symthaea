// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * IP Blocklist
 *
 * Block malicious IPs and rate limit abusers.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import Redis from 'ioredis';

export interface IpBlocklistOptions {
  redis?: Redis;
  blockDuration?: number; // Seconds to block
  checkHeaders?: boolean; // Check X-Forwarded-For
}

/**
 * Create an IP blocklist middleware
 */
export function ipBlocklist(options: IpBlocklistOptions = {}): {
  middleware: RequestHandler;
  blockIp: (ip: string, reason?: string, duration?: number) => Promise<void>;
  unblockIp: (ip: string) => Promise<void>;
  isBlocked: (ip: string) => Promise<boolean>;
} {
  const { redis, blockDuration = 3600, checkHeaders = true } = options;

  // In-memory fallback
  const blocklist = new Map<string, { reason?: string; until: number }>();

  const getIp = (req: Request): string => {
    if (checkHeaders) {
      const forwarded = req.headers['x-forwarded-for'];
      if (forwarded) {
        return Array.isArray(forwarded)
          ? forwarded[0].split(',')[0].trim()
          : forwarded.split(',')[0].trim();
      }

      const realIp = req.headers['x-real-ip'];
      if (realIp) {
        return Array.isArray(realIp) ? realIp[0] : realIp;
      }
    }

    return req.socket.remoteAddress || 'unknown';
  };

  const blockIp = async (
    ip: string,
    reason?: string,
    duration: number = blockDuration
  ): Promise<void> => {
    if (redis) {
      const key = `blocklist:${ip}`;
      await redis.setex(key, duration, reason || 'blocked');
    } else {
      blocklist.set(ip, {
        reason,
        until: Date.now() + duration * 1000,
      });
    }
  };

  const unblockIp = async (ip: string): Promise<void> => {
    if (redis) {
      await redis.del(`blocklist:${ip}`);
    } else {
      blocklist.delete(ip);
    }
  };

  const isBlocked = async (ip: string): Promise<boolean> => {
    if (redis) {
      const result = await redis.get(`blocklist:${ip}`);
      return result !== null;
    } else {
      const entry = blocklist.get(ip);
      if (!entry) return false;
      if (entry.until < Date.now()) {
        blocklist.delete(ip);
        return false;
      }
      return true;
    }
  };

  const middleware: RequestHandler = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    const ip = getIp(req);

    if (await isBlocked(ip)) {
      res.status(403).json({
        error: 'Forbidden',
        message: 'Your IP address has been blocked',
      });
      return;
    }

    next();
  };

  return { middleware, blockIp, unblockIp, isBlocked };
}

/**
 * Auto-block IPs that exceed rate limits multiple times
 */
export function autoBlocker(options: {
  redis?: Redis;
  maxViolations?: number; // Block after this many rate limit violations
  violationWindow?: number; // Window in seconds
  blockDuration?: number; // Block duration in seconds
}): {
  recordViolation: (ip: string) => Promise<boolean>;
  middleware: RequestHandler;
} {
  const {
    redis,
    maxViolations = 10,
    violationWindow = 3600,
    blockDuration = 86400,
  } = options;

  const violations = new Map<string, { count: number; firstAt: number }>();
  const blocked = new Map<string, number>();

  const recordViolation = async (ip: string): Promise<boolean> => {
    const now = Date.now();

    if (redis) {
      const key = `violations:${ip}`;
      const count = await redis.incr(key);

      if (count === 1) {
        await redis.expire(key, violationWindow);
      }

      if (count >= maxViolations) {
        await redis.setex(`blocklist:${ip}`, blockDuration, 'auto-blocked');
        await redis.del(key);
        return true;
      }

      return false;
    } else {
      let entry = violations.get(ip);

      if (!entry || now - entry.firstAt > violationWindow * 1000) {
        entry = { count: 0, firstAt: now };
      }

      entry.count++;
      violations.set(ip, entry);

      if (entry.count >= maxViolations) {
        blocked.set(ip, now + blockDuration * 1000);
        violations.delete(ip);
        return true;
      }

      return false;
    }
  };

  const middleware: RequestHandler = async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    const ip =
      req.headers['x-forwarded-for']?.toString().split(',')[0]?.trim() ||
      req.socket.remoteAddress ||
      'unknown';

    // Check if auto-blocked
    if (redis) {
      const isBlocked = await redis.get(`blocklist:${ip}`);
      if (isBlocked === 'auto-blocked') {
        res.status(403).json({
          error: 'Forbidden',
          message: 'Your IP has been temporarily blocked due to suspicious activity',
        });
        return;
      }
    } else {
      const blockedUntil = blocked.get(ip);
      if (blockedUntil) {
        if (Date.now() < blockedUntil) {
          res.status(403).json({
            error: 'Forbidden',
            message: 'Your IP has been temporarily blocked due to suspicious activity',
          });
          return;
        }
        blocked.delete(ip);
      }
    }

    next();
  };

  return { recordViolation, middleware };
}

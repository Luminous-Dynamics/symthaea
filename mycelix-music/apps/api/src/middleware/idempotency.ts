// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Idempotency Keys Middleware
 *
 * Ensures mutations can be safely retried without duplicate effects.
 * Critical for payment operations, blockchain transactions, and play recording.
 */

import { Request, Response, NextFunction } from 'express';
import { createHash } from 'crypto';
import Redis from 'ioredis';
import { getLogger } from '../logging';

/**
 * Idempotency record stored in cache
 */
interface IdempotencyRecord {
  /** Request fingerprint for validation */
  fingerprint: string;
  /** HTTP status code of original response */
  statusCode: number;
  /** Response headers */
  headers: Record<string, string>;
  /** Response body */
  body: unknown;
  /** When the request was processed */
  processedAt: string;
  /** Request path */
  path: string;
  /** Request method */
  method: string;
}

/**
 * In-flight request tracking
 */
interface InFlightRequest {
  promise: Promise<void>;
  resolve: () => void;
}

/**
 * Idempotency configuration
 */
export interface IdempotencyConfig {
  /** Header name for idempotency key */
  headerName: string;
  /** TTL for idempotency records in seconds */
  ttl: number;
  /** Methods that require idempotency */
  methods: string[];
  /** Paths that require idempotency (regex patterns) */
  paths?: RegExp[];
  /** Generate fingerprint from request */
  fingerprint?: (req: Request) => string;
}

/**
 * Default configuration
 */
const defaultConfig: IdempotencyConfig = {
  headerName: 'Idempotency-Key',
  ttl: 86400, // 24 hours
  methods: ['POST', 'PUT', 'PATCH', 'DELETE'],
  fingerprint: (req: Request) => {
    const parts = [
      req.method,
      req.path,
      JSON.stringify(req.body || {}),
    ];
    return createHash('sha256').update(parts.join('|')).digest('hex');
  },
};

/**
 * Memory-based idempotency store (for development/testing)
 */
class MemoryIdempotencyStore {
  private records: Map<string, { record: IdempotencyRecord; expiresAt: number }> = new Map();
  private inFlight: Map<string, InFlightRequest> = new Map();

  async get(key: string): Promise<IdempotencyRecord | null> {
    const entry = this.records.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.records.delete(key);
      return null;
    }
    return entry.record;
  }

  async set(key: string, record: IdempotencyRecord, ttlSeconds: number): Promise<void> {
    this.records.set(key, {
      record,
      expiresAt: Date.now() + ttlSeconds * 1000,
    });
  }

  async isInFlight(key: string): Promise<boolean> {
    return this.inFlight.has(key);
  }

  async setInFlight(key: string): Promise<void> {
    let resolve: () => void;
    const promise = new Promise<void>(r => { resolve = r; });
    this.inFlight.set(key, { promise, resolve: resolve! });
  }

  async waitForInFlight(key: string): Promise<void> {
    const inFlight = this.inFlight.get(key);
    if (inFlight) {
      await inFlight.promise;
    }
  }

  async clearInFlight(key: string): Promise<void> {
    const inFlight = this.inFlight.get(key);
    if (inFlight) {
      inFlight.resolve();
      this.inFlight.delete(key);
    }
  }

  clear(): void {
    this.records.clear();
    this.inFlight.clear();
  }
}

/**
 * Redis-based idempotency store (for production)
 */
class RedisIdempotencyStore {
  constructor(private redis: Redis) {}

  async get(key: string): Promise<IdempotencyRecord | null> {
    const data = await this.redis.get(`idempotency:${key}`);
    if (!data) return null;
    return JSON.parse(data);
  }

  async set(key: string, record: IdempotencyRecord, ttlSeconds: number): Promise<void> {
    await this.redis.setex(`idempotency:${key}`, ttlSeconds, JSON.stringify(record));
  }

  async isInFlight(key: string): Promise<boolean> {
    const result = await this.redis.exists(`idempotency:inflight:${key}`);
    return result === 1;
  }

  async setInFlight(key: string): Promise<void> {
    // Set with short TTL to prevent deadlocks
    await this.redis.setex(`idempotency:inflight:${key}`, 30, '1');
  }

  async waitForInFlight(key: string, maxWait = 5000): Promise<void> {
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      const inFlight = await this.isInFlight(key);
      if (!inFlight) return;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  async clearInFlight(key: string): Promise<void> {
    await this.redis.del(`idempotency:inflight:${key}`);
  }
}

/**
 * Idempotency store interface
 */
type IdempotencyStore = MemoryIdempotencyStore | RedisIdempotencyStore;

/**
 * Global store instance
 */
let store: IdempotencyStore = new MemoryIdempotencyStore();

/**
 * Initialize with Redis
 */
export function initIdempotencyStore(redis?: Redis): void {
  if (redis) {
    store = new RedisIdempotencyStore(redis);
  } else {
    store = new MemoryIdempotencyStore();
  }
}

/**
 * Create idempotency middleware
 */
export function idempotencyMiddleware(config: Partial<IdempotencyConfig> = {}) {
  const cfg = { ...defaultConfig, ...config };
  const logger = getLogger();

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // Only apply to configured methods
    if (!cfg.methods.includes(req.method)) {
      return next();
    }

    // Check path patterns if configured
    if (cfg.paths && cfg.paths.length > 0) {
      const matchesPath = cfg.paths.some(pattern => pattern.test(req.path));
      if (!matchesPath) {
        return next();
      }
    }

    // Get idempotency key from header
    const idempotencyKey = req.headers[cfg.headerName.toLowerCase()] as string;

    // If no key provided, proceed without idempotency
    if (!idempotencyKey) {
      // For critical endpoints, we might want to require the key
      return next();
    }

    // Validate key format
    if (!/^[a-zA-Z0-9\-_]{8,64}$/.test(idempotencyKey)) {
      res.status(400).json({
        success: false,
        error: {
          code: 'INVALID_IDEMPOTENCY_KEY',
          message: 'Idempotency key must be 8-64 alphanumeric characters',
        },
      });
      return;
    }

    // Create composite key with wallet address if authenticated
    const walletAddress = req.auth?.address || 'anonymous';
    const compositeKey = `${walletAddress}:${idempotencyKey}`;

    // Check for existing record
    const existing = await store.get(compositeKey);

    if (existing) {
      // Validate request fingerprint matches
      const currentFingerprint = cfg.fingerprint!(req);
      if (existing.fingerprint !== currentFingerprint) {
        logger.warn('Idempotency key reused with different request', {
          key: idempotencyKey,
          originalPath: existing.path,
          currentPath: req.path,
        });

        res.status(422).json({
          success: false,
          error: {
            code: 'IDEMPOTENCY_KEY_MISMATCH',
            message: 'Idempotency key was used for a different request',
          },
        });
        return;
      }

      // Return cached response
      logger.debug('Returning cached idempotent response', {
        key: idempotencyKey,
        originalProcessedAt: existing.processedAt,
      });

      res.setHeader('Idempotent-Replayed', 'true');
      res.setHeader('Original-Request-Time', existing.processedAt);

      for (const [name, value] of Object.entries(existing.headers)) {
        if (!['content-length', 'transfer-encoding'].includes(name.toLowerCase())) {
          res.setHeader(name, value);
        }
      }

      res.status(existing.statusCode).json(existing.body);
      return;
    }

    // Check if request is in flight
    if (await store.isInFlight(compositeKey)) {
      // Wait for the in-flight request to complete
      logger.debug('Waiting for in-flight idempotent request', { key: idempotencyKey });
      await store.waitForInFlight(compositeKey);

      // Try to get the result
      const result = await store.get(compositeKey);
      if (result) {
        res.setHeader('Idempotent-Replayed', 'true');
        res.status(result.statusCode).json(result.body);
        return;
      }
    }

    // Mark as in flight
    await store.setInFlight(compositeKey);

    // Capture original response methods
    const originalJson = res.json.bind(res);
    const originalSend = res.send.bind(res);
    const fingerprint = cfg.fingerprint!(req);

    // Override json method to capture response
    res.json = function (body: unknown) {
      // Store the idempotency record
      const record: IdempotencyRecord = {
        fingerprint,
        statusCode: res.statusCode,
        headers: Object.fromEntries(
          Object.entries(res.getHeaders())
            .filter(([_, v]) => typeof v === 'string')
            .map(([k, v]) => [k, v as string])
        ),
        body,
        processedAt: new Date().toISOString(),
        path: req.path,
        method: req.method,
      };

      // Only store successful responses (2xx)
      if (res.statusCode >= 200 && res.statusCode < 300) {
        store.set(compositeKey, record, cfg.ttl).catch(err => {
          logger.error('Failed to store idempotency record', err);
        });
      }

      // Clear in-flight status
      store.clearInFlight(compositeKey).catch(() => {});

      return originalJson(body);
    };

    // Handle errors - clear in-flight on error
    res.on('finish', () => {
      if (res.statusCode >= 400) {
        store.clearInFlight(compositeKey).catch(() => {});
      }
    });

    next();
  };
}

/**
 * Require idempotency key for endpoint
 */
export function requireIdempotencyKey(req: Request, res: Response, next: NextFunction): void {
  const key = req.headers['idempotency-key'];

  if (!key) {
    res.status(400).json({
      success: false,
      error: {
        code: 'IDEMPOTENCY_KEY_REQUIRED',
        message: 'This endpoint requires an Idempotency-Key header',
      },
    });
    return;
  }

  next();
}

/**
 * Generate a new idempotency key
 */
export function generateIdempotencyKey(): string {
  return createHash('sha256')
    .update(`${Date.now()}-${Math.random()}`)
    .digest('hex')
    .slice(0, 32);
}

/**
 * Clear idempotency store (for testing)
 */
export function clearIdempotencyStore(): void {
  if (store instanceof MemoryIdempotencyStore) {
    store.clear();
  }
}

export default idempotencyMiddleware;

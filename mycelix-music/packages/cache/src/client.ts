// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Client
 *
 * Redis caching client with TTL and serialization.
 */

import Redis from 'ioredis';

export interface CacheOptions {
  redis: Redis;
  prefix?: string;
  defaultTTL?: number; // seconds
  serializer?: {
    serialize: (value: unknown) => string;
    deserialize: <T>(value: string) => T;
  };
}

export interface CacheClient {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttl?: number): Promise<void>;
  del(key: string | string[]): Promise<void>;
  exists(key: string): Promise<boolean>;
  expire(key: string, ttl: number): Promise<void>;
  ttl(key: string): Promise<number>;
  keys(pattern: string): Promise<string[]>;
  invalidatePattern(pattern: string): Promise<number>;
  getOrSet<T>(key: string, factory: () => Promise<T>, ttl?: number): Promise<T>;
  mget<T>(...keys: string[]): Promise<(T | null)[]>;
  mset<T>(entries: Array<{ key: string; value: T; ttl?: number }>): Promise<void>;
  incr(key: string, by?: number): Promise<number>;
  decr(key: string, by?: number): Promise<number>;
}

export function createCacheClient(options: CacheOptions): CacheClient {
  const {
    redis,
    prefix = 'cache:',
    defaultTTL = 3600,
    serializer = {
      serialize: (v) => JSON.stringify(v),
      deserialize: <T>(v: string) => JSON.parse(v) as T,
    },
  } = options;

  const prefixKey = (key: string) => `${prefix}${key}`;

  return {
    async get<T>(key: string): Promise<T | null> {
      const value = await redis.get(prefixKey(key));
      if (!value) return null;
      try {
        return serializer.deserialize<T>(value);
      } catch {
        return null;
      }
    },

    async set<T>(key: string, value: T, ttl: number = defaultTTL): Promise<void> {
      const serialized = serializer.serialize(value);
      if (ttl > 0) {
        await redis.setex(prefixKey(key), ttl, serialized);
      } else {
        await redis.set(prefixKey(key), serialized);
      }
    },

    async del(key: string | string[]): Promise<void> {
      const keys = Array.isArray(key) ? key : [key];
      if (keys.length > 0) {
        await redis.del(...keys.map(prefixKey));
      }
    },

    async exists(key: string): Promise<boolean> {
      const count = await redis.exists(prefixKey(key));
      return count > 0;
    },

    async expire(key: string, ttl: number): Promise<void> {
      await redis.expire(prefixKey(key), ttl);
    },

    async ttl(key: string): Promise<number> {
      return redis.ttl(prefixKey(key));
    },

    async keys(pattern: string): Promise<string[]> {
      const prefixedPattern = prefixKey(pattern);
      const keys = await redis.keys(prefixedPattern);
      return keys.map(k => k.slice(prefix.length));
    },

    async invalidatePattern(pattern: string): Promise<number> {
      const keys = await redis.keys(prefixKey(pattern));
      if (keys.length === 0) return 0;
      return redis.del(...keys);
    },

    async getOrSet<T>(
      key: string,
      factory: () => Promise<T>,
      ttl: number = defaultTTL
    ): Promise<T> {
      const existing = await this.get<T>(key);
      if (existing !== null) return existing;

      const value = await factory();
      await this.set(key, value, ttl);
      return value;
    },

    async mget<T>(...keys: string[]): Promise<(T | null)[]> {
      if (keys.length === 0) return [];
      const values = await redis.mget(...keys.map(prefixKey));
      return values.map(v => {
        if (!v) return null;
        try {
          return serializer.deserialize<T>(v);
        } catch {
          return null;
        }
      });
    },

    async mset<T>(entries: Array<{ key: string; value: T; ttl?: number }>): Promise<void> {
      const pipeline = redis.pipeline();
      for (const entry of entries) {
        const serialized = serializer.serialize(entry.value);
        const ttl = entry.ttl ?? defaultTTL;
        if (ttl > 0) {
          pipeline.setex(prefixKey(entry.key), ttl, serialized);
        } else {
          pipeline.set(prefixKey(entry.key), serialized);
        }
      }
      await pipeline.exec();
    },

    async incr(key: string, by: number = 1): Promise<number> {
      if (by === 1) {
        return redis.incr(prefixKey(key));
      }
      return redis.incrby(prefixKey(key), by);
    },

    async decr(key: string, by: number = 1): Promise<number> {
      if (by === 1) {
        return redis.decr(prefixKey(key));
      }
      return redis.decrby(prefixKey(key), by);
    },
  };
}

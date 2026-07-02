// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Decorators
 *
 * TypeScript decorators for method-level caching.
 */

import { CacheClient } from './client';

// Global cache client reference
let globalCacheClient: CacheClient | null = null;

export function setCacheClient(client: CacheClient): void {
  globalCacheClient = client;
}

export interface CacheDecoratorOptions {
  key: string | ((...args: any[]) => string);
  ttl?: number;
  condition?: (...args: any[]) => boolean;
}

export type CacheDecorator = (
  options: CacheDecoratorOptions
) => MethodDecorator;

/**
 * Method decorator for caching return values
 *
 * Usage:
 * @cached({ key: (id) => `song:${id}`, ttl: 3600 })
 * async getSong(id: string): Promise<Song> { ... }
 */
export function cached(options: CacheDecoratorOptions): MethodDecorator {
  return function (
    target: any,
    propertyKey: string | symbol,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]): Promise<any> {
      if (!globalCacheClient) {
        return originalMethod.apply(this, args);
      }

      // Check condition
      if (options.condition && !options.condition(...args)) {
        return originalMethod.apply(this, args);
      }

      // Generate cache key
      const cacheKey = typeof options.key === 'function'
        ? options.key(...args)
        : options.key;

      // Try to get from cache
      const cached = await globalCacheClient.get(cacheKey);
      if (cached !== null) {
        return cached;
      }

      // Execute method
      const result = await originalMethod.apply(this, args);

      // Store in cache
      if (result !== undefined && result !== null) {
        await globalCacheClient.set(cacheKey, result, options.ttl);
      }

      return result;
    };

    return descriptor;
  };
}

export interface InvalidateOptions {
  patterns: string[] | ((...args: any[]) => string[]);
}

/**
 * Method decorator for cache invalidation
 *
 * Usage:
 * @invalidate({ patterns: (id) => [`song:${id}`, `song:${id}:*`] })
 * async updateSong(id: string, data: UpdateData): Promise<Song> { ... }
 */
export function invalidate(options: InvalidateOptions): MethodDecorator {
  return function (
    target: any,
    propertyKey: string | symbol,
    descriptor: PropertyDescriptor
  ): PropertyDescriptor {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]): Promise<any> {
      // Execute method first
      const result = await originalMethod.apply(this, args);

      // Invalidate cache
      if (globalCacheClient) {
        const patterns = typeof options.patterns === 'function'
          ? options.patterns(...args)
          : options.patterns;

        await Promise.all(
          patterns.map(pattern => globalCacheClient!.invalidatePattern(pattern))
        );
      }

      return result;
    };

    return descriptor;
  };
}

/**
 * Create a cacheable wrapper function
 *
 * Usage:
 * const getCachedSong = cacheable(
 *   (id: string) => songRepository.findById(id),
 *   { key: (id) => `song:${id}`, ttl: 3600 }
 * );
 */
export function cacheable<T, Args extends any[]>(
  fn: (...args: Args) => Promise<T>,
  options: CacheDecoratorOptions
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    if (!globalCacheClient) {
      return fn(...args);
    }

    // Check condition
    if (options.condition && !options.condition(...args)) {
      return fn(...args);
    }

    // Generate cache key
    const cacheKey = typeof options.key === 'function'
      ? options.key(...args)
      : options.key;

    // Use getOrSet for atomic operation
    return globalCacheClient.getOrSet(
      cacheKey,
      () => fn(...args),
      options.ttl
    );
  };
}

/**
 * Create a batch caching wrapper
 *
 * Useful for fetching multiple items with cache support
 */
export function batchCacheable<T, K extends string>(
  fetchMany: (ids: K[]) => Promise<Map<K, T>>,
  options: {
    keyPrefix: string;
    ttl?: number;
  }
): (ids: K[]) => Promise<Map<K, T>> {
  return async (ids: K[]): Promise<Map<K, T>> => {
    if (!globalCacheClient || ids.length === 0) {
      return fetchMany(ids);
    }

    const result = new Map<K, T>();
    const missingIds: K[] = [];

    // Check cache for each ID
    const cacheKeys = ids.map(id => `${options.keyPrefix}:${id}`);
    const cached = await globalCacheClient.mget<T>(...cacheKeys);

    for (let i = 0; i < ids.length; i++) {
      const cachedValue = cached[i];
      if (cachedValue !== null) {
        result.set(ids[i], cachedValue);
      } else {
        missingIds.push(ids[i]);
      }
    }

    // Fetch missing items
    if (missingIds.length > 0) {
      const fetched = await fetchMany(missingIds);

      // Cache and add to result
      const toCache: Array<{ key: string; value: T; ttl?: number }> = [];

      for (const [id, value] of fetched) {
        result.set(id, value);
        toCache.push({
          key: `${options.keyPrefix}:${id}`,
          value,
          ttl: options.ttl,
        });
      }

      if (toCache.length > 0) {
        await globalCacheClient.mset(toCache);
      }
    }

    return result;
  };
}

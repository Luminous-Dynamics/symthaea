// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common Cache Utilities
 *
 * Caching patterns for SDK operations with TTL, LRU eviction, and cache invalidation.
 *
 * @module @mycelix/sdk/common/cache
 */

/**
 * Cache entry with metadata
 */
interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  accessedAt: number;
}

/**
 * Cache configuration options
 */
export interface CacheOptions {
  /** Time-to-live in milliseconds (default: 5 minutes) */
  ttlMs?: number;
  /** Maximum number of entries (default: 1000) */
  maxSize?: number;
  /** Enable LRU eviction when max size is reached (default: true) */
  lruEviction?: boolean;
  /** Callback when entry expires */
  onExpire?: (key: string, value: unknown) => void;
  /** Callback when entry is evicted */
  onEvict?: (key: string, value: unknown) => void;
}

const DEFAULT_OPTIONS: Required<Omit<CacheOptions, 'onExpire' | 'onEvict'>> = {
  ttlMs: 5 * 60 * 1000, // 5 minutes
  maxSize: 1000,
  lruEviction: true,
};

/**
 * In-memory cache with TTL and LRU eviction
 *
 * @example
 * ```typescript
 * const cache = new Cache<User>({ ttlMs: 60000, maxSize: 100 });
 *
 * // Set a value
 * cache.set('user:123', { id: '123', name: 'Alice' });
 *
 * // Get a value
 * const user = cache.get('user:123');
 *
 * // Get or fetch
 * const user = await cache.getOrFetch('user:456', async () => {
 *   return await fetchUser('456');
 * });
 * ```
 */
export class Cache<T> {
  private entries = new Map<string, CacheEntry<T>>();
  private readonly options: Required<Omit<CacheOptions, 'onExpire' | 'onEvict'>>;
  private readonly onExpire?: CacheOptions['onExpire'];
  private readonly onEvict?: CacheOptions['onEvict'];

  constructor(options: CacheOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.onExpire = options.onExpire;
    this.onEvict = options.onEvict;
  }

  /**
   * Get a value from the cache
   */
  get(key: string): T | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;

    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.delete(key);
      this.onExpire?.(key, entry.value);
      return undefined;
    }

    // Update access time for LRU
    entry.accessedAt = Date.now();
    return entry.value;
  }

  /**
   * Set a value in the cache
   */
  set(key: string, value: T, ttlMs?: number): void {
    // Evict if at capacity
    if (this.entries.size >= this.options.maxSize && !this.entries.has(key)) {
      this.evictOne();
    }

    const now = Date.now();
    this.entries.set(key, {
      value,
      expiresAt: now + (ttlMs ?? this.options.ttlMs),
      accessedAt: now,
    });
  }

  /**
   * Get a value or fetch it if not cached
   */
  async getOrFetch(
    key: string,
    fetcher: () => Promise<T>,
    ttlMs?: number
  ): Promise<T> {
    const cached = this.get(key);
    if (cached !== undefined) {
      return cached;
    }

    const value = await fetcher();
    this.set(key, value, ttlMs);
    return value;
  }

  /**
   * Delete a value from the cache
   */
  delete(key: string): boolean {
    return this.entries.delete(key);
  }

  /**
   * Clear all entries from the cache
   */
  clear(): void {
    this.entries.clear();
  }

  /**
   * Check if a key exists and is not expired
   */
  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    hitRate: number;
  } {
    return {
      size: this.entries.size,
      maxSize: this.options.maxSize,
      hitRate: 0, // Would need hit/miss tracking
    };
  }

  /**
   * Remove expired entries
   */
  prune(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [key, entry] of this.entries) {
      if (now > entry.expiresAt) {
        this.entries.delete(key);
        this.onExpire?.(key, entry.value);
        pruned++;
      }
    }

    return pruned;
  }

  /**
   * Get all keys in the cache
   */
  keys(): string[] {
    return Array.from(this.entries.keys());
  }

  private evictOne(): void {
    if (!this.options.lruEviction) {
      // FIFO eviction - remove oldest entry
      const firstKey = this.entries.keys().next().value;
      if (firstKey) {
        const entry = this.entries.get(firstKey);
        this.entries.delete(firstKey);
        this.onEvict?.(firstKey, entry?.value);
      }
      return;
    }

    // LRU eviction - find least recently accessed
    let lruKey: string | undefined;
    let lruTime = Infinity;

    for (const [key, entry] of this.entries) {
      if (entry.accessedAt < lruTime) {
        lruTime = entry.accessedAt;
        lruKey = key;
      }
    }

    if (lruKey) {
      const entry = this.entries.get(lruKey);
      this.entries.delete(lruKey);
      this.onEvict?.(lruKey, entry?.value);
    }
  }
}

/**
 * Create a memoized version of an async function with caching
 *
 * @example
 * ```typescript
 * const fetchUser = memoize(
 *   async (id: string) => await api.getUser(id),
 *   (id) => `user:${id}`,
 *   { ttlMs: 60000 }
 * );
 *
 * // First call fetches, subsequent calls use cache
 * const user1 = await fetchUser('123');
 * const user2 = await fetchUser('123'); // Cached
 * ```
 */
export function memoize<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  keyFn: (...args: TArgs) => string,
  options: CacheOptions = {}
): (...args: TArgs) => Promise<TResult> {
  const cache = new Cache<TResult>(options);

  return async (...args: TArgs): Promise<TResult> => {
    const key = keyFn(...args);
    return cache.getOrFetch(key, () => fn(...args));
  };
}

/**
 * Cache with namespace support for organizing entries
 *
 * @example
 * ```typescript
 * const cache = new NamespacedCache<any>({ ttlMs: 60000 });
 *
 * cache.set('users', '123', userData);
 * cache.set('posts', '456', postData);
 *
 * cache.clearNamespace('users'); // Only clears user entries
 * ```
 */
export class NamespacedCache<T> {
  private cache: Cache<T>;

  constructor(options: CacheOptions = {}) {
    this.cache = new Cache<T>(options);
  }

  private makeKey(namespace: string, key: string): string {
    return `${namespace}:${key}`;
  }

  get(namespace: string, key: string): T | undefined {
    return this.cache.get(this.makeKey(namespace, key));
  }

  set(namespace: string, key: string, value: T, ttlMs?: number): void {
    this.cache.set(this.makeKey(namespace, key), value, ttlMs);
  }

  async getOrFetch(
    namespace: string,
    key: string,
    fetcher: () => Promise<T>,
    ttlMs?: number
  ): Promise<T> {
    return this.cache.getOrFetch(this.makeKey(namespace, key), fetcher, ttlMs);
  }

  delete(namespace: string, key: string): boolean {
    return this.cache.delete(this.makeKey(namespace, key));
  }

  clearNamespace(namespace: string): number {
    const prefix = `${namespace}:`;
    let cleared = 0;

    for (const key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        this.cache.delete(key);
        cleared++;
      }
    }

    return cleared;
  }

  clear(): void {
    this.cache.clear();
  }

  getStats(): ReturnType<Cache<T>['getStats']> {
    return this.cache.getStats();
  }
}

/**
 * Pre-configured cache instances for common use cases
 */
export const Caches = {
  /** Short-lived cache for frequently accessed data */
  short: () => new Cache({ ttlMs: 30 * 1000, maxSize: 500 }),

  /** Standard cache for most use cases */
  standard: () => new Cache({ ttlMs: 5 * 60 * 1000, maxSize: 1000 }),

  /** Long-lived cache for rarely changing data */
  long: () => new Cache({ ttlMs: 30 * 60 * 1000, maxSize: 2000 }),

  /** Session-length cache (1 hour) */
  session: () => new Cache({ ttlMs: 60 * 60 * 1000, maxSize: 5000 }),
};

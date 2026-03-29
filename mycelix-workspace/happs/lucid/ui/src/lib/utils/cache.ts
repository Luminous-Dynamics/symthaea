// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Caching Utilities
 *
 * Provides intelligent caching for expensive operations with:
 * - TTL (time-to-live) support
 * - LRU (least recently used) eviction
 * - Stale-while-revalidate pattern
 * - Request deduplication
 */

// ============================================================================
// TYPES
// ============================================================================

export interface CacheOptions {
  /** Time-to-live in milliseconds */
  ttl?: number;
  /** Maximum number of entries */
  maxSize?: number;
  /** Serve stale data while revalidating */
  staleWhileRevalidate?: boolean;
  /** Grace period for stale data in ms */
  staleGracePeriod?: number;
}

interface CacheEntry<T> {
  value: T;
  expires: number;
  lastAccessed: number;
}

// ============================================================================
// LRU CACHE
// ============================================================================

/**
 * LRU Cache with TTL support
 */
export class LRUCache<K, V> {
  private cache = new Map<K, CacheEntry<V>>();
  private ttl: number;
  private maxSize: number;

  constructor(options: CacheOptions = {}) {
    this.ttl = options.ttl ?? 5 * 60 * 1000; // 5 minutes default
    this.maxSize = options.maxSize ?? 100;
  }

  get(key: K): V | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // Check if expired
    if (Date.now() > entry.expires) {
      this.cache.delete(key);
      return undefined;
    }

    // Update last accessed time (LRU)
    entry.lastAccessed = Date.now();
    return entry.value;
  }

  set(key: K, value: V, ttl?: number): void {
    // Evict if at capacity
    if (this.cache.size >= this.maxSize) {
      this.evictLRU();
    }

    const now = Date.now();
    this.cache.set(key, {
      value,
      expires: now + (ttl ?? this.ttl),
      lastAccessed: now,
    });
  }

  has(key: K): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;
    if (Date.now() > entry.expires) {
      this.cache.delete(key);
      return false;
    }
    return true;
  }

  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }

  private evictLRU(): void {
    let oldest: K | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldest = key;
      }
    }

    if (oldest !== null) {
      this.cache.delete(oldest);
    }
  }

  // Clean up expired entries
  prune(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [key, entry] of this.cache) {
      if (now > entry.expires) {
        this.cache.delete(key);
        pruned++;
      }
    }

    return pruned;
  }
}

// ============================================================================
// ASYNC CACHE WITH DEDUPLICATION
// ============================================================================

interface PendingRequest<T> {
  promise: Promise<T>;
  timestamp: number;
}

/**
 * Async cache that deduplicates concurrent requests
 */
export class AsyncCache<K, V> {
  private cache: LRUCache<K, V>;
  private pending = new Map<K, PendingRequest<V>>();
  private staleWhileRevalidate: boolean;
  private staleGracePeriod: number;
  private staleCache = new Map<K, { value: V; staleAt: number }>();

  constructor(options: CacheOptions = {}) {
    this.cache = new LRUCache<K, V>(options);
    this.staleWhileRevalidate = options.staleWhileRevalidate ?? false;
    this.staleGracePeriod = options.staleGracePeriod ?? 60000; // 1 minute
  }

  /**
   * Get a value, fetching if not cached
   */
  async get(key: K, fetcher: () => Promise<V>): Promise<V> {
    // Check cache first
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      return cached;
    }

    // Check for pending request (deduplication)
    const pending = this.pending.get(key);
    if (pending) {
      return pending.promise;
    }

    // Check stale cache for stale-while-revalidate
    if (this.staleWhileRevalidate) {
      const stale = this.staleCache.get(key);
      if (stale && Date.now() < stale.staleAt + this.staleGracePeriod) {
        // Return stale data immediately, revalidate in background
        this.revalidateInBackground(key, fetcher);
        return stale.value;
      }
    }

    // Fetch fresh data
    return this.fetchAndCache(key, fetcher);
  }

  /**
   * Invalidate a cached value
   */
  invalidate(key: K): void {
    this.cache.delete(key);
    this.staleCache.delete(key);
  }

  /**
   * Invalidate all cached values
   */
  invalidateAll(): void {
    this.cache.clear();
    this.staleCache.clear();
  }

  /**
   * Prefetch a value into the cache
   */
  async prefetch(key: K, fetcher: () => Promise<V>): Promise<void> {
    if (!this.cache.has(key)) {
      await this.fetchAndCache(key, fetcher);
    }
  }

  private async fetchAndCache(key: K, fetcher: () => Promise<V>): Promise<V> {
    const promise = fetcher();
    this.pending.set(key, { promise, timestamp: Date.now() });

    try {
      const value = await promise;
      this.cache.set(key, value);

      // Store for stale-while-revalidate
      if (this.staleWhileRevalidate) {
        this.staleCache.set(key, { value, staleAt: Date.now() });
      }

      return value;
    } finally {
      this.pending.delete(key);
    }
  }

  private revalidateInBackground(key: K, fetcher: () => Promise<V>): void {
    // Don't revalidate if already pending
    if (this.pending.has(key)) return;

    this.fetchAndCache(key, fetcher).catch((error) => {
      console.warn('Background revalidation failed:', error);
    });
  }
}

// ============================================================================
// MEMOIZATION HELPERS
// ============================================================================

/**
 * Create a memoized version of an async function
 */
export function memoize<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  options: CacheOptions & { keyFn?: (...args: Parameters<T>) => string } = {}
): T {
  const cache = new AsyncCache<string, Awaited<ReturnType<T>>>(options);
  const keyFn = options.keyFn ?? ((...args) => JSON.stringify(args));

  return (async (...args: Parameters<T>) => {
    const key = keyFn(...args);
    return cache.get(key, () => fn(...args) as Promise<Awaited<ReturnType<T>>>);
  }) as T;
}

/**
 * Create a memoized version of a sync function
 */
export function memoizeSync<A extends unknown[], R>(
  fn: (...args: A) => R,
  options: CacheOptions & { keyFn?: (...args: A) => string } = {}
): (...args: A) => R {
  const cache = new LRUCache<string, R>(options);
  const keyFn = options.keyFn ?? ((...args: A) => JSON.stringify(args));

  return (...args: A): R => {
    const key = keyFn(...args);
    const cached = cache.get(key);
    if (cached !== undefined) return cached;

    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

// ============================================================================
// GLOBAL CACHES FOR COMMON OPERATIONS
// ============================================================================

// Cache for embedding computations (expensive)
export const embeddingCache = new AsyncCache<string, number[]>({
  ttl: 30 * 60 * 1000, // 30 minutes
  maxSize: 500,
  staleWhileRevalidate: true,
});

// Cache for pattern detection results
export const patternCache = new AsyncCache<string, unknown>({
  ttl: 5 * 60 * 1000, // 5 minutes
  maxSize: 50,
  staleWhileRevalidate: true,
});

// Cache for consensus data
export const consensusCache = new AsyncCache<string, unknown>({
  ttl: 2 * 60 * 1000, // 2 minutes
  maxSize: 100,
});

// Cache for relationship data
export const relationshipCache = new AsyncCache<string, unknown>({
  ttl: 10 * 60 * 1000, // 10 minutes
  maxSize: 200,
});

// ============================================================================
// CACHE STATISTICS
// ============================================================================

interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
}

class CacheStatsTracker {
  private hits = 0;
  private misses = 0;

  recordHit(): void {
    this.hits++;
  }

  recordMiss(): void {
    this.misses++;
  }

  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  reset(): void {
    this.hits = 0;
    this.misses = 0;
  }
}

export const cacheStats = new CacheStatsTracker();

// ============================================================================
// BATCH LOADER
// ============================================================================

/**
 * Batch multiple requests into a single call
 * Useful for GraphQL-style batching
 */
export class BatchLoader<K, V> {
  private batch: Map<K, { resolve: (v: V) => void; reject: (e: Error) => void }[]> = new Map();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private batchFn: (keys: K[]) => Promise<Map<K, V>>;
  private delay: number;

  constructor(batchFn: (keys: K[]) => Promise<Map<K, V>>, delay = 10) {
    this.batchFn = batchFn;
    this.delay = delay;
  }

  async load(key: K): Promise<V> {
    return new Promise<V>((resolve, reject) => {
      if (!this.batch.has(key)) {
        this.batch.set(key, []);
      }
      this.batch.get(key)!.push({ resolve, reject });
      this.scheduleBatch();
    });
  }

  private scheduleBatch(): void {
    if (this.timer) return;

    this.timer = setTimeout(async () => {
      this.timer = null;
      const currentBatch = this.batch;
      this.batch = new Map();

      const keys = Array.from(currentBatch.keys());
      if (keys.length === 0) return;

      try {
        const results = await this.batchFn(keys);

        for (const [key, handlers] of currentBatch) {
          const value = results.get(key);
          for (const handler of handlers) {
            if (value !== undefined) {
              handler.resolve(value);
            } else {
              handler.reject(new Error(`No result for key: ${key}`));
            }
          }
        }
      } catch (error) {
        for (const handlers of currentBatch.values()) {
          for (const handler of handlers) {
            handler.reject(error as Error);
          }
        }
      }
    }, this.delay);
  }
}

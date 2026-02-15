/**
 * Cache Utilities Tests
 *
 * Tests for LRU cache, async cache with deduplication,
 * memoization, and batch loading.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  LRUCache,
  AsyncCache,
  memoize,
  memoizeSync,
  BatchLoader,
  embeddingCache,
  patternCache,
} from '../utils/cache';

describe('LRUCache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should store and retrieve values', () => {
    const cache = new LRUCache<string, number>();
    cache.set('a', 1);
    cache.set('b', 2);

    expect(cache.get('a')).toBe(1);
    expect(cache.get('b')).toBe(2);
  });

  it('should return undefined for missing keys', () => {
    const cache = new LRUCache<string, number>();
    expect(cache.get('nonexistent')).toBeUndefined();
  });

  it('should respect TTL', () => {
    const cache = new LRUCache<string, number>({ ttl: 1000 });
    cache.set('key', 42);

    expect(cache.get('key')).toBe(42);

    vi.advanceTimersByTime(1001);

    expect(cache.get('key')).toBeUndefined();
  });

  it('should evict LRU entries when at capacity', () => {
    const cache = new LRUCache<string, number>({ maxSize: 3, ttl: 60000 });

    cache.set('a', 1);
    vi.advanceTimersByTime(10);
    cache.set('b', 2);
    vi.advanceTimersByTime(10);
    cache.set('c', 3);
    vi.advanceTimersByTime(10);

    // Access 'a' and 'c' to make 'b' least recently used
    cache.get('a');
    vi.advanceTimersByTime(10);
    cache.get('c');
    vi.advanceTimersByTime(10);

    // Add new item, should evict 'b'
    cache.set('d', 4);

    expect(cache.get('a')).toBe(1);
    expect(cache.get('b')).toBeUndefined(); // Evicted
    expect(cache.get('c')).toBe(3);
    expect(cache.get('d')).toBe(4);
  });

  it('should check if key exists', () => {
    const cache = new LRUCache<string, number>({ ttl: 1000 });
    cache.set('key', 42);

    expect(cache.has('key')).toBe(true);
    expect(cache.has('other')).toBe(false);

    vi.advanceTimersByTime(1001);

    expect(cache.has('key')).toBe(false);
  });

  it('should delete entries', () => {
    const cache = new LRUCache<string, number>();
    cache.set('key', 42);

    expect(cache.delete('key')).toBe(true);
    expect(cache.get('key')).toBeUndefined();
    expect(cache.delete('key')).toBe(false);
  });

  it('should clear all entries', () => {
    const cache = new LRUCache<string, number>();
    cache.set('a', 1);
    cache.set('b', 2);

    cache.clear();

    expect(cache.size).toBe(0);
    expect(cache.get('a')).toBeUndefined();
  });

  it('should prune expired entries', () => {
    const cache = new LRUCache<string, number>({ ttl: 1000 });
    cache.set('a', 1);
    cache.set('b', 2);

    vi.advanceTimersByTime(500);
    cache.set('c', 3); // Not expired yet

    vi.advanceTimersByTime(600); // 'a' and 'b' now expired

    const pruned = cache.prune();

    expect(pruned).toBe(2);
    expect(cache.get('a')).toBeUndefined();
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('c')).toBe(3);
  });
});

describe('AsyncCache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should cache async results', async () => {
    const cache = new AsyncCache<string, number>();
    const fetcher = vi.fn().mockResolvedValue(42);

    const result1 = await cache.get('key', fetcher);
    const result2 = await cache.get('key', fetcher);

    expect(result1).toBe(42);
    expect(result2).toBe(42);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('should deduplicate concurrent requests', async () => {
    const cache = new AsyncCache<string, number>();
    let resolvePromise: (value: number) => void;
    const fetcher = vi.fn().mockImplementation(
      () => new Promise<number>((resolve) => {
        resolvePromise = resolve;
      })
    );

    // Start two concurrent requests
    const promise1 = cache.get('key', fetcher);
    const promise2 = cache.get('key', fetcher);

    // Resolve the single underlying request
    resolvePromise!(42);

    const [result1, result2] = await Promise.all([promise1, promise2]);

    expect(result1).toBe(42);
    expect(result2).toBe(42);
    expect(fetcher).toHaveBeenCalledTimes(1); // Only one actual fetch
  });

  it('should invalidate cached values', async () => {
    const cache = new AsyncCache<string, number>();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(1)
      .mockResolvedValueOnce(2);

    const result1 = await cache.get('key', fetcher);
    cache.invalidate('key');
    const result2 = await cache.get('key', fetcher);

    expect(result1).toBe(1);
    expect(result2).toBe(2);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('should support stale-while-revalidate', async () => {
    const cache = new AsyncCache<string, number>({
      ttl: 1000,
      staleWhileRevalidate: true,
      staleGracePeriod: 5000,
    });

    const fetcher = vi.fn()
      .mockResolvedValueOnce(1)
      .mockResolvedValueOnce(2);

    // Initial fetch
    const result1 = await cache.get('key', fetcher);
    expect(result1).toBe(1);

    // Expire the cache but still in grace period
    vi.advanceTimersByTime(1500);

    // Should return stale value immediately
    const result2 = await cache.get('key', fetcher);
    expect(result2).toBe(1); // Stale value

    // Background revalidation should have started
    await vi.advanceTimersByTimeAsync(100);

    // Now should have fresh value
    const result3 = await cache.get('key', fetcher);
    expect(result3).toBe(2);
  });
});

describe('memoize', () => {
  it('should memoize async function results', async () => {
    const fn = vi.fn().mockResolvedValue(42);
    const memoized = memoize(fn, { ttl: 60000 });

    const result1 = await memoized('arg');
    const result2 = await memoized('arg');

    expect(result1).toBe(42);
    expect(result2).toBe(42);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('should use custom key function', async () => {
    const fn = vi.fn().mockImplementation((a, b) => Promise.resolve(a + b));
    const memoized = memoize(fn, {
      keyFn: (a, b) => `${a}-${b}`,
    });

    await memoized(1, 2);
    await memoized(1, 2);
    await memoized(2, 1);

    expect(fn).toHaveBeenCalledTimes(2);
  });
});

describe('memoizeSync', () => {
  it('should memoize sync function results', () => {
    const fn = vi.fn().mockReturnValue(42);
    const memoized = memoizeSync(fn);

    const result1 = memoized('arg');
    const result2 = memoized('arg');

    expect(result1).toBe(42);
    expect(result2).toBe(42);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});

describe('BatchLoader', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should batch multiple requests', async () => {
    const batchFn = vi.fn().mockImplementation((keys: string[]) => {
      const results = new Map<string, number>();
      keys.forEach((key, i) => results.set(key, i + 1));
      return Promise.resolve(results);
    });

    const loader = new BatchLoader(batchFn, 10);

    const promise1 = loader.load('a');
    const promise2 = loader.load('b');
    const promise3 = loader.load('c');

    vi.advanceTimersByTime(10);

    const results = await Promise.all([promise1, promise2, promise3]);

    expect(results).toEqual([1, 2, 3]);
    expect(batchFn).toHaveBeenCalledTimes(1);
    expect(batchFn).toHaveBeenCalledWith(['a', 'b', 'c']);
  });

  it('should handle missing keys', async () => {
    const batchFn = vi.fn().mockImplementation((keys: string[]) => {
      const results = new Map<string, number>();
      // Only add some keys
      if (keys.includes('a')) results.set('a', 1);
      return Promise.resolve(results);
    });

    const loader = new BatchLoader(batchFn, 10);

    const promise1 = loader.load('a');
    const promise2 = loader.load('b');

    vi.advanceTimersByTime(10);

    await expect(promise1).resolves.toBe(1);
    await expect(promise2).rejects.toThrow('No result for key');
  });

  it('should handle batch errors', async () => {
    const batchFn = vi.fn().mockRejectedValue(new Error('Batch failed'));

    const loader = new BatchLoader(batchFn, 10);

    const promise1 = loader.load('a');
    const promise2 = loader.load('b');

    vi.advanceTimersByTime(10);

    await expect(promise1).rejects.toThrow('Batch failed');
    await expect(promise2).rejects.toThrow('Batch failed');
  });
});

describe('Global caches', () => {
  it('should have embedding cache configured', () => {
    expect(embeddingCache).toBeDefined();
  });

  it('should have pattern cache configured', () => {
    expect(patternCache).toBeDefined();
  });
});

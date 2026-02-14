/**
 * UESS Storage Backend Tests
 *
 * Tests for memory and local storage backends.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MemoryBackend, createMemoryBackend } from '../../src/storage/backends/memory.js';
import { LocalBackend, createLocalBackend } from '../../src/storage/backends/local.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../src/epistemic/types.js';
import type { StorageMetadata } from '../../src/storage/types.js';

describe('MemoryBackend', () => {
  let backend: MemoryBackend;

  beforeEach(() => {
    backend = createMemoryBackend({ gcIntervalMs: 0 }); // Disable auto-GC for tests
  });

  afterEach(() => {
    backend.dispose();
  });

  describe('basic operations', () => {
    it('stores and retrieves data', async () => {
      const data = { name: 'Alice', age: 30 };
      const metadata = createTestMetadata();

      await backend.set('user:alice', data, metadata);
      const result = await backend.get('user:alice');

      expect(result).not.toBeNull();
      expect(result!.data).toEqual(data);
      expect(result!.metadata.cid).toMatch(/^cid:/);
    });

    it('returns null for non-existent keys', async () => {
      const result = await backend.get('nonexistent');

      expect(result).toBeNull();
    });

    it('checks key existence', async () => {
      const data = { test: true };
      const metadata = createTestMetadata();

      expect(await backend.has('test:key')).toBe(false);

      await backend.set('test:key', data, metadata);

      expect(await backend.has('test:key')).toBe(true);
    });

    it('deletes data', async () => {
      const data = { test: true };
      const metadata = createTestMetadata();

      await backend.set('test:key', data, metadata);
      expect(await backend.has('test:key')).toBe(true);

      const deleted = await backend.delete('test:key');

      expect(deleted).toBe(true);
      expect(await backend.has('test:key')).toBe(false);
    });

    it('returns false when deleting non-existent key', async () => {
      const deleted = await backend.delete('nonexistent');

      expect(deleted).toBe(false);
    });
  });

  describe('TTL and expiration', () => {
    it('expires data after TTL', async () => {
      const shortTtlBackend = createMemoryBackend({
        defaultTtlMs: 50,
        gcIntervalMs: 0,
      });

      const data = { test: true };
      const metadata = createTestMetadata();

      await shortTtlBackend.set('expiring:key', data, metadata);

      // Should exist immediately
      expect(await shortTtlBackend.has('expiring:key')).toBe(true);

      // Wait for expiration
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Should be expired now
      expect(await shortTtlBackend.has('expiring:key')).toBe(false);

      shortTtlBackend.dispose();
    });

    it('respects custom expiresAt in metadata', async () => {
      const data = { test: true };
      const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
        ...createTestMetadata(),
        expiresAt: Date.now() + 50, // Expire in 50ms
      };

      await backend.set('custom:expiry', data, metadata);

      expect(await backend.has('custom:expiry')).toBe(true);

      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(await backend.has('custom:expiry')).toBe(false);
    });
  });

  describe('garbage collection', () => {
    it('removes expired entries during GC', async () => {
      const shortTtlBackend = createMemoryBackend({
        defaultTtlMs: 50,
        gcIntervalMs: 0,
      });

      await shortTtlBackend.set('key1', { a: 1 }, createTestMetadata());
      await shortTtlBackend.set('key2', { b: 2 }, createTestMetadata());

      await new Promise((resolve) => setTimeout(resolve, 100));

      const removed = shortTtlBackend.gc();

      expect(removed).toBe(2);

      shortTtlBackend.dispose();
    });
  });

  describe('keys()', () => {
    it('lists all keys', async () => {
      await backend.set('key1', { a: 1 }, createTestMetadata());
      await backend.set('key2', { b: 2 }, createTestMetadata());
      await backend.set('key3', { c: 3 }, createTestMetadata());

      const keys = await backend.keys();

      expect(keys).toHaveLength(3);
      expect(keys).toContain('key1');
      expect(keys).toContain('key2');
      expect(keys).toContain('key3');
    });

    it('filters keys by pattern', async () => {
      await backend.set('user:alice', { name: 'Alice' }, createTestMetadata());
      await backend.set('user:bob', { name: 'Bob' }, createTestMetadata());
      await backend.set('session:123', { token: 'xyz' }, createTestMetadata());

      const userKeys = await backend.keys('user:*');

      expect(userKeys).toHaveLength(2);
      expect(userKeys).toContain('user:alice');
      expect(userKeys).toContain('user:bob');
      expect(userKeys).not.toContain('session:123');
    });

    it('excludes tombstoned entries', async () => {
      const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
        ...createTestMetadata(),
        tombstone: true,
      };

      await backend.set('tombstoned', { deleted: true }, metadata);
      await backend.set('active', { alive: true }, createTestMetadata());

      const keys = await backend.keys();

      expect(keys).toHaveLength(1);
      expect(keys).toContain('active');
      expect(keys).not.toContain('tombstoned');
    });
  });

  describe('stats()', () => {
    it('returns correct statistics', async () => {
      await backend.set('key1', { data: 'a'.repeat(100) }, createTestMetadata());
      await backend.set('key2', { data: 'b'.repeat(200) }, createTestMetadata());

      const stats = await backend.stats();

      expect(stats.itemCount).toBe(2);
      expect(stats.totalSizeBytes).toBeGreaterThan(0);
      expect(stats.oldestItem).toBeDefined();
      expect(stats.newestItem).toBeDefined();
    });

    it('returns empty stats for empty backend', async () => {
      const stats = await backend.stats();

      expect(stats.itemCount).toBe(0);
      expect(stats.totalSizeBytes).toBe(0);
      expect(stats.oldestItem).toBeUndefined();
      expect(stats.newestItem).toBeUndefined();
    });
  });

  describe('clear()', () => {
    it('removes all data', async () => {
      await backend.set('key1', { a: 1 }, createTestMetadata());
      await backend.set('key2', { b: 2 }, createTestMetadata());

      await backend.clear();

      expect(await backend.has('key1')).toBe(false);
      expect(await backend.has('key2')).toBe(false);
      expect((await backend.stats()).itemCount).toBe(0);
    });
  });

  describe('CID computation', () => {
    it('generates consistent CIDs for same data', async () => {
      const data = { test: 'consistent' };

      const meta1 = await backend.set('key1', data, createTestMetadata());
      const meta2 = await backend.set('key2', data, createTestMetadata());

      expect(meta1.cid).toBe(meta2.cid);
    });

    it('generates different CIDs for different data', async () => {
      const meta1 = await backend.set('key1', { a: 1 }, createTestMetadata());
      const meta2 = await backend.set('key2', { b: 2 }, createTestMetadata());

      expect(meta1.cid).not.toBe(meta2.cid);
    });
  });
});

describe('LocalBackend', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createLocalBackend({
      namespace: 'test',
      gcIntervalMs: 0,
    });
  });

  afterEach(() => {
    backend.dispose();
  });

  describe('basic operations', () => {
    it('stores and retrieves data', async () => {
      const data = { name: 'Bob', score: 100 };
      const metadata = createTestMetadata();

      await backend.set('game:bob', data, metadata);
      const result = await backend.get('game:bob');

      expect(result).not.toBeNull();
      expect(result!.data).toEqual(data);
    });

    it('uses namespace for key isolation', async () => {
      const backend1 = createLocalBackend({ namespace: 'ns1', gcIntervalMs: 0 });
      const backend2 = createLocalBackend({ namespace: 'ns2', gcIntervalMs: 0 });

      await backend1.set('key', { ns: 1 }, createTestMetadata());
      await backend2.set('key', { ns: 2 }, createTestMetadata());

      const result1 = await backend1.get('key');
      const result2 = await backend2.get('key');

      expect(result1!.data).toEqual({ ns: 1 });
      expect(result2!.data).toEqual({ ns: 2 });

      backend1.dispose();
      backend2.dispose();
    });

    it('clears only keys in namespace', async () => {
      const backend1 = createLocalBackend({ namespace: 'ns1', gcIntervalMs: 0 });
      const backend2 = createLocalBackend({ namespace: 'ns2', gcIntervalMs: 0 });

      await backend1.set('key', { ns: 1 }, createTestMetadata());
      await backend2.set('key', { ns: 2 }, createTestMetadata());

      await backend1.clear();

      expect(await backend1.has('key')).toBe(false);
      expect(await backend2.has('key')).toBe(true);

      backend1.dispose();
      backend2.dispose();
    });
  });

  describe('TTL and expiration', () => {
    it('expires data after TTL', async () => {
      const shortTtlBackend = createLocalBackend({
        namespace: 'short-ttl',
        defaultTtlMs: 50,
        gcIntervalMs: 0,
      });

      await shortTtlBackend.set('expiring', { test: true }, createTestMetadata());

      expect(await shortTtlBackend.has('expiring')).toBe(true);

      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(await shortTtlBackend.has('expiring')).toBe(false);

      shortTtlBackend.dispose();
    });
  });

  describe('keys()', () => {
    it('lists keys in namespace', async () => {
      await backend.set('a', { a: 1 }, createTestMetadata());
      await backend.set('b', { b: 2 }, createTestMetadata());

      const keys = await backend.keys();

      expect(keys).toContain('a');
      expect(keys).toContain('b');
    });
  });

  describe('stats()', () => {
    it('returns namespace-scoped statistics', async () => {
      // Clear any persisted data from previous test runs
      await backend.clear();

      await backend.set('key1', { data: 'test' }, createTestMetadata());

      const stats = await backend.stats();

      expect(stats.itemCount).toBe(1);
      expect(stats.totalSizeBytes).toBeGreaterThan(0);
    });
  });

  describe('garbage collection', () => {
    it('removes expired entries', async () => {
      const shortTtlBackend = createLocalBackend({
        namespace: 'gc-test',
        defaultTtlMs: 50,
        gcIntervalMs: 0,
      });

      await shortTtlBackend.set('key1', { a: 1 }, createTestMetadata());

      await new Promise((resolve) => setTimeout(resolve, 100));

      const removed = await shortTtlBackend.gc();

      expect(removed).toBe(1);

      shortTtlBackend.dispose();
    });
  });
});

// Helper function to create test metadata
function createTestMetadata(): Omit<StorageMetadata, 'cid' | 'sizeBytes'> {
  return {
    classification: {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    },
    schema: { id: 'test', version: '1.0.0' },
    storedAt: Date.now(),
    version: 1,
    createdBy: 'test-agent',
    tombstone: false,
  };
}

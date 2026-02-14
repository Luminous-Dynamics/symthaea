/**
 * Local Backend Storage Tests
 *
 * Tests the LocalBackend for UESS M1 temporal storage:
 * get/set/delete, TTL expiration, namespace isolation,
 * garbage collection, pattern matching, statistics, and persistence lifecycle.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  LocalBackend,
  createLocalBackend,
  createEphemeralLocalBackend,
} from '../../src/storage/backends/local.js';
import type { StorageMetadata } from '../../src/storage/types.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../src/epistemic/types.js';

// Helper to create minimal metadata
function makeMetadata(overrides: Partial<StorageMetadata> = {}): Omit<StorageMetadata, 'cid' | 'sizeBytes'> {
  return {
    classification: {
      empirical: EmpiricalLevel.E0,
      normative: NormativeLevel.N0,
      materiality: MaterialityLevel.M0,
    },
    schema: { id: 'test-schema', version: '1.0.0' },
    storedAt: Date.now(),
    version: 1,
    createdBy: 'test-agent',
    tombstone: false,
    ...overrides,
  } as Omit<StorageMetadata, 'cid' | 'sizeBytes'>;
}

// ============================================================================
// Basic Operations
// ============================================================================

describe('LocalBackend - Basic Operations', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({ gcIntervalMs: 0 });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should set and get data', async () => {
    const meta = makeMetadata();
    await backend.set('key1', { hello: 'world' }, meta);

    const result = await backend.get<{ hello: string }>('key1');
    expect(result).not.toBeNull();
    expect(result!.data.hello).toBe('world');
    expect(result!.metadata.cid).toBeTruthy();
    expect(result!.metadata.sizeBytes).toBeGreaterThan(0);
  });

  it('should return null for missing key', async () => {
    const result = await backend.get('nonexistent');
    expect(result).toBeNull();
  });

  it('should check if key exists', async () => {
    await backend.set('exists', 'yes', makeMetadata());

    expect(await backend.has('exists')).toBe(true);
    expect(await backend.has('missing')).toBe(false);
  });

  it('should delete data', async () => {
    await backend.set('to-delete', 'value', makeMetadata());
    expect(await backend.has('to-delete')).toBe(true);

    const deleted = await backend.delete('to-delete');
    expect(deleted).toBe(true);
    expect(await backend.has('to-delete')).toBe(false);
  });

  it('should return false when deleting nonexistent key', async () => {
    const deleted = await backend.delete('nonexistent');
    expect(deleted).toBe(false);
  });

  it('should overwrite existing key', async () => {
    await backend.set('key1', 'first', makeMetadata());
    await backend.set('key1', 'second', makeMetadata());

    const result = await backend.get<string>('key1');
    expect(result!.data).toBe('second');
  });

  it('should store various data types', async () => {
    const meta = makeMetadata();

    await backend.set('string', 'hello', meta);
    await backend.set('number', 42, meta);
    await backend.set('array', [1, 2, 3], meta);
    await backend.set('nested', { a: { b: { c: true } } }, meta);

    expect((await backend.get<string>('string'))!.data).toBe('hello');
    expect((await backend.get<number>('number'))!.data).toBe(42);
    expect((await backend.get<number[]>('array'))!.data).toEqual([1, 2, 3]);
    expect((await backend.get<any>('nested'))!.data.a.b.c).toBe(true);
  });
});

// ============================================================================
// TTL & Expiration
// ============================================================================

describe('LocalBackend - TTL & Expiration', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({
      defaultTtlMs: 100, // 100ms default TTL for quick tests
      gcIntervalMs: 0,
    });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should expire entries after default TTL', async () => {
    await backend.set('ephemeral', 'data', makeMetadata());

    // Should exist immediately
    expect(await backend.has('ephemeral')).toBe(true);

    // Wait for TTL
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should be expired
    expect(await backend.get('ephemeral')).toBeNull();
    expect(await backend.has('ephemeral')).toBe(false);
  });

  it('should use custom expiresAt from metadata', async () => {
    const meta = makeMetadata({ expiresAt: Date.now() + 50 });
    await backend.set('custom-ttl', 'data', meta);

    expect(await backend.has('custom-ttl')).toBe(true);

    await new Promise(resolve => setTimeout(resolve, 80));

    expect(await backend.has('custom-ttl')).toBe(false);
  });
});

// ============================================================================
// Tombstones
// ============================================================================

describe('LocalBackend - Tombstones', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({ gcIntervalMs: 0, defaultTtlMs: 60000 });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should hide tombstoned entries', async () => {
    await backend.set('tombstoned', 'data', makeMetadata({ tombstone: true }));

    expect(await backend.get('tombstoned')).toBeNull();
    expect(await backend.has('tombstoned')).toBe(false);
  });
});

// ============================================================================
// Namespace Isolation
// ============================================================================

describe('LocalBackend - Namespace Isolation', () => {
  let ns1: LocalBackend;
  let ns2: LocalBackend;

  beforeEach(() => {
    ns1 = createEphemeralLocalBackend({ namespace: 'ns1', gcIntervalMs: 0, defaultTtlMs: 60000 });
    ns2 = createEphemeralLocalBackend({ namespace: 'ns2', gcIntervalMs: 0, defaultTtlMs: 60000 });
  });

  afterEach(async () => {
    await ns1.dispose();
    await ns2.dispose();
  });

  it('should isolate data between namespaces', async () => {
    await ns1.set('shared-key', 'ns1-value', makeMetadata());
    await ns2.set('shared-key', 'ns2-value', makeMetadata());

    expect((await ns1.get<string>('shared-key'))!.data).toBe('ns1-value');
    expect((await ns2.get<string>('shared-key'))!.data).toBe('ns2-value');
  });

  it('should clear only own namespace', async () => {
    await ns1.set('key1', 'a', makeMetadata());
    await ns2.set('key2', 'b', makeMetadata());

    await ns1.clear();

    expect(await ns1.has('key1')).toBe(false);
    expect(await ns2.has('key2')).toBe(true);
  });
});

// ============================================================================
// Key Listing & Pattern Matching
// ============================================================================

describe('LocalBackend - Key Listing', () => {
  let backend: LocalBackend;

  beforeEach(async () => {
    backend = createEphemeralLocalBackend({ gcIntervalMs: 0, defaultTtlMs: 60000 });
    const meta = makeMetadata();
    await backend.set('users/alice', 'a', meta);
    await backend.set('users/bob', 'b', meta);
    await backend.set('posts/1', 'p1', meta);
    await backend.set('posts/2', 'p2', meta);
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should list all keys without pattern', async () => {
    const keys = await backend.keys();
    expect(keys).toHaveLength(4);
  });

  it('should filter keys with wildcard pattern', async () => {
    const userKeys = await backend.keys('users/*');
    expect(userKeys).toHaveLength(2);
    expect(userKeys).toContain('users/alice');
    expect(userKeys).toContain('users/bob');
  });

  it('should match exact key pattern', async () => {
    const keys = await backend.keys('posts/1');
    expect(keys).toHaveLength(1);
    expect(keys[0]).toBe('posts/1');
  });

  it('should exclude expired keys from listing', async () => {
    const shortBackend = createEphemeralLocalBackend({ defaultTtlMs: 50, gcIntervalMs: 0 });
    await shortBackend.set('temp', 'data', makeMetadata());

    expect((await shortBackend.keys())).toHaveLength(1);

    await new Promise(resolve => setTimeout(resolve, 80));

    expect((await shortBackend.keys())).toHaveLength(0);
    await shortBackend.dispose();
  });

  it('should exclude tombstoned keys from listing', async () => {
    await backend.set('dead', 'data', makeMetadata({ tombstone: true }));

    const keys = await backend.keys();
    expect(keys).not.toContain('dead');
  });
});

// ============================================================================
// Garbage Collection
// ============================================================================

describe('LocalBackend - Garbage Collection', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({ defaultTtlMs: 50, gcIntervalMs: 0 });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should remove expired entries on gc', async () => {
    await backend.set('temp1', 'a', makeMetadata());
    await backend.set('temp2', 'b', makeMetadata());

    await new Promise(resolve => setTimeout(resolve, 80));

    const removed = await backend.gc();
    expect(removed).toBe(2);
  });

  it('should not remove non-expired entries', async () => {
    const longBackend = createEphemeralLocalBackend({ defaultTtlMs: 60000, gcIntervalMs: 0 });
    await longBackend.set('persist', 'data', makeMetadata());

    const removed = await longBackend.gc();
    expect(removed).toBe(0);
    expect(await longBackend.has('persist')).toBe(true);
    await longBackend.dispose();
  });
});

// ============================================================================
// Statistics
// ============================================================================

describe('LocalBackend - Statistics', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({ gcIntervalMs: 0, defaultTtlMs: 60000 });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should report correct stats', async () => {
    const meta = makeMetadata();
    await backend.set('a', 'hello', meta);
    await backend.set('b', 'world', meta);

    const stats = await backend.stats();
    expect(stats.itemCount).toBe(2);
    expect(stats.totalSizeBytes).toBeGreaterThan(0);
    expect(stats.oldestItem).toBeDefined();
    expect(stats.newestItem).toBeDefined();
    expect(stats.oldestItem!).toBeLessThanOrEqual(stats.newestItem!);
  });

  it('should report zero stats for empty store', async () => {
    const stats = await backend.stats();
    expect(stats.itemCount).toBe(0);
    expect(stats.totalSizeBytes).toBe(0);
    expect(stats.oldestItem).toBeUndefined();
    expect(stats.newestItem).toBeUndefined();
  });

  it('should exclude expired items from stats', async () => {
    const shortBackend = createEphemeralLocalBackend({ defaultTtlMs: 30, gcIntervalMs: 0 });
    await shortBackend.set('temp', 'data', makeMetadata());

    await new Promise(resolve => setTimeout(resolve, 50));

    const stats = await shortBackend.stats();
    expect(stats.itemCount).toBe(0);
    await shortBackend.dispose();
  });
});

// ============================================================================
// Clear
// ============================================================================

describe('LocalBackend - Clear', () => {
  let backend: LocalBackend;

  beforeEach(() => {
    backend = createEphemeralLocalBackend({ gcIntervalMs: 0, defaultTtlMs: 60000 });
  });

  afterEach(async () => {
    await backend.dispose();
  });

  it('should remove all entries in namespace', async () => {
    const meta = makeMetadata();
    await backend.set('a', 1, meta);
    await backend.set('b', 2, meta);
    await backend.set('c', 3, meta);

    await backend.clear();

    expect((await backend.keys())).toHaveLength(0);
    expect(await backend.has('a')).toBe(false);
  });
});

// ============================================================================
// Flush & Dispose
// ============================================================================

describe('LocalBackend - Lifecycle', () => {
  it('should flush pending persistence', async () => {
    const backend = createEphemeralLocalBackend({ gcIntervalMs: 0, defaultTtlMs: 60000 });
    await backend.set('key', 'value', makeMetadata());
    await backend.flush();
    // Should not throw
    await backend.dispose();
  });

  it('should initialize only once', async () => {
    const backend = createEphemeralLocalBackend({ gcIntervalMs: 0 });
    // Multiple calls to init should be safe
    await (backend as any).init();
    await (backend as any).init();
    await backend.dispose();
  });
});

// ============================================================================
// Factory Functions
// ============================================================================

describe('LocalBackend - Factories', () => {
  it('createLocalBackend creates with defaults', () => {
    const backend = createLocalBackend({ disablePersistence: true, gcIntervalMs: 0 });
    expect(backend).toBeInstanceOf(LocalBackend);
  });

  it('createEphemeralLocalBackend creates without persistence', () => {
    const backend = createEphemeralLocalBackend({ gcIntervalMs: 0 });
    expect(backend).toBeInstanceOf(LocalBackend);
  });
});

// ============================================================================
// Custom Persistence
// ============================================================================

describe('LocalBackend - Custom Persistence', () => {
  it('should use custom persist/load functions', async () => {
    let stored: Map<string, unknown> | null = null;

    const backend = new LocalBackend({
      persist: async (data) => { stored = data; },
      load: async () => stored as any,
      gcIntervalMs: 0,
      defaultTtlMs: 60000,
    });

    await backend.set('key', 'value', makeMetadata());
    await backend.flush();

    expect(stored).not.toBeNull();
    expect(stored!.size).toBeGreaterThan(0);

    await backend.dispose();
  });
});

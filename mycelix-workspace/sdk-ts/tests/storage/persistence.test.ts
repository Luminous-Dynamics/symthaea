// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for UESS persistence functionality
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  LocalBackend,
  createLocalBackend,
  createEphemeralLocalBackend,
  IPFSBackend,
  InMemoryPersistence,
  isBrowser,
  isNode,
} from '../../src/storage/backends/index.js';

describe('Persistence Utilities', () => {
  describe('Environment Detection', () => {
    it('should correctly detect Node.js environment', () => {
      // We're running in Node.js for tests
      expect(isNode()).toBe(true);
    });

    it('should correctly detect not-browser environment', () => {
      // We're not in a browser
      expect(isBrowser()).toBe(false);
    });
  });

  describe('InMemoryPersistence', () => {
    let persistence: InMemoryPersistence<Map<string, unknown>>;

    beforeEach(() => {
      persistence = new InMemoryPersistence<Map<string, unknown>>();
    });

    it('should save and load a Map', async () => {
      const data = new Map<string, unknown>([
        ['key1', { value: 'test1' }],
        ['key2', { value: 'test2' }],
      ]);

      await persistence.save('test-key', data);
      const loaded = await persistence.load('test-key');

      expect(loaded).toBeInstanceOf(Map);
      expect(loaded?.size).toBe(2);
      expect(loaded?.get('key1')).toEqual({ value: 'test1' });
    });

    it('should return null for non-existent keys', async () => {
      const loaded = await persistence.load('non-existent');
      expect(loaded).toBeNull();
    });

    it('should delete data', async () => {
      const data = new Map<string, unknown>([['key', 'value']]);
      await persistence.save('test-key', data);

      expect(await persistence.exists('test-key')).toBe(true);

      await persistence.delete('test-key');

      expect(await persistence.exists('test-key')).toBe(false);
    });
  });
});

describe('LocalBackend with Persistence', () => {
  let backend: LocalBackend;

  afterEach(async () => {
    if (backend) {
      await backend.dispose();
    }
  });

  it('should create ephemeral backend without persistence', async () => {
    backend = createEphemeralLocalBackend({ namespace: 'test-ephemeral' });
    await backend.init();

    // Should work normally
    await backend.set('key', { test: true }, {
      classification: { empirical: 0, normative: 0, materiality: 1 },
      schema: { id: 'test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      tombstone: false,
      createdBy: 'test',
    });

    const result = await backend.get('key');
    expect(result?.data).toEqual({ test: true });
  });

  it('should handle persistence with forceAdapter=memory', async () => {
    // Use in-memory adapter for testing
    backend = createLocalBackend({
      namespace: 'test-memory',
      forceAdapter: 'memory',
    });

    await backend.init();

    await backend.set('persist-test', { data: 'should persist' }, {
      classification: { empirical: 0, normative: 0, materiality: 1 },
      schema: { id: 'test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      tombstone: false,
      createdBy: 'test',
    });

    // Verify data exists
    const result = await backend.get<{ data: string }>('persist-test');
    expect(result?.data.data).toBe('should persist');

    // Flush to ensure persistence
    await backend.flush();
  });

  it('should garbage collect expired entries', async () => {
    backend = createLocalBackend({
      namespace: 'test-gc',
      forceAdapter: 'memory',
      defaultTtlMs: 50, // Very short TTL
      gcIntervalMs: 0, // Disable auto-GC
    });

    await backend.init();

    await backend.set('gc-test', { data: 'will expire' }, {
      classification: { empirical: 0, normative: 0, materiality: 1 },
      schema: { id: 'test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      tombstone: false,
      createdBy: 'test',
      // Note: expiresAt is computed from defaultTtlMs
    });

    // Should exist immediately
    expect(await backend.has('gc-test')).toBe(true);

    // Wait for expiration
    await new Promise(resolve => setTimeout(resolve, 100));

    // Should be gone now (checked on access)
    expect(await backend.has('gc-test')).toBe(false);
  });
});

describe('IPFSBackend Index Persistence', () => {
  it('should create backend with index persistence disabled', () => {
    const backend = new IPFSBackend({
      disableIndexPersistence: true,
    });

    // Should not throw
    expect(backend).toBeInstanceOf(IPFSBackend);
  });

  // Note: Full IPFS tests require a running IPFS node
  // These are integration tests that would run with CONDUCTOR_AVAILABLE=true
});

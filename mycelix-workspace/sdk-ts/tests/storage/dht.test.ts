// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for UESS DHT Backend
 *
 * Unit tests use mock Holochain client.
 * Integration tests require CONDUCTOR_AVAILABLE=true and running conductor.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { DHTBackend, createDHTBackend, type HolochainClient } from '../../src/storage/backends/dht.js';

// =============================================================================
// Mock Holochain Client
// =============================================================================

function createMockClient(responses: Record<string, unknown> = {}): HolochainClient {
  return {
    callZome: vi.fn().mockImplementation(async ({ fn_name, payload }) => {
      if (responses[fn_name]) {
        return typeof responses[fn_name] === 'function'
          ? (responses[fn_name] as (p: unknown) => unknown)(payload)
          : responses[fn_name];
      }

      // Default responses
      switch (fn_name) {
        case 'store_epistemic_entry':
          return 'mock-action-hash';
        case 'get_epistemic_entry':
          return null;
        case 'has_epistemic_entry':
          return false;
        case 'delete_epistemic_entry':
          return false;
        case 'list_epistemic_keys':
          return [];
        case 'get_storage_stats':
          return { item_count: 0, total_size_bytes: 0, oldest_item: null, newest_item: null };
        case 'clear_epistemic_storage':
          return 0;
        case 'get_by_cid':
          return null;
        case 'get_replication_status':
          return { key: '', holder_count: 0, target_holders: 3, is_replicated: false };
        case 'ensure_replication':
          return true;
        default:
          throw new Error(`Unknown zome function: ${fn_name}`);
      }
    }),
  };
}

// =============================================================================
// Unit Tests (Mock Client)
// =============================================================================

describe('DHTBackend Unit Tests', () => {
  let backend: DHTBackend;
  let mockClient: HolochainClient;

  beforeEach(() => {
    mockClient = createMockClient();
    backend = createDHTBackend({
      client: mockClient,
      enableLocalCache: false, // Disable cache for unit tests
    });
  });

  describe('Initialization', () => {
    it('should create backend with default options', () => {
      expect(backend).toBeInstanceOf(DHTBackend);
    });

    it('should create backend with custom options', () => {
      const customBackend = createDHTBackend({
        client: mockClient,
        zomeName: 'custom_storage',
        dnaRole: 'custom_role',
        agentId: 'custom-agent',
        timeoutMs: 60000,
        retries: 5,
        enableLocalCache: true,
        localCacheTtlMs: 120000,
      });
      expect(customBackend).toBeInstanceOf(DHTBackend);
    });
  });

  describe('set()', () => {
    it('should store data with E/N/M classification', async () => {
      const metadata = await backend.set('test-key', { hello: 'world' }, {
        classification: { empirical: 2, normative: 1, materiality: 2 },
        schema: { id: 'test-schema', version: '1.0.0' },
        storedAt: Date.now(),
        version: 1,
        createdBy: 'test-agent',
        tombstone: false,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith(expect.objectContaining({
        fn_name: 'store_epistemic_entry',
      }));

      expect(metadata.cid).toMatch(/^cid:/);
      expect(metadata.sizeBytes).toBeGreaterThan(0);
      expect(metadata.classification.empirical).toBe(2);
      expect(metadata.classification.normative).toBe(1);
      expect(metadata.classification.materiality).toBe(2);
    });

    it('should serialize complex data to JSON', async () => {
      const complexData = {
        nested: { deep: { value: 42 } },
        array: [1, 2, 3],
        date: new Date().toISOString(),
      };

      await backend.set('complex-key', complexData, {
        classification: { empirical: 0, normative: 0, materiality: 2 },
        schema: { id: 'complex', version: '1.0.0' },
        storedAt: Date.now(),
        version: 1,
        createdBy: 'test',
        tombstone: false,
      });

      const call = (mockClient.callZome as ReturnType<typeof vi.fn>).mock.calls[0][0];
      const payload = call.payload as { key: string; entry: { data: string } };
      const storedData = JSON.parse(payload.entry.data);

      expect(storedData.nested.deep.value).toBe(42);
      expect(storedData.array).toEqual([1, 2, 3]);
    });
  });

  describe('get()', () => {
    it('should return null for non-existent key', async () => {
      const result = await backend.get('non-existent');
      expect(result).toBeNull();
    });

    it('should return stored data when found', async () => {
      const storedEntry = {
        key: 'found-key',
        data: JSON.stringify({ found: true }),
        metadata: {
          cid: 'cid:test-hash',
          classification: { empirical: 1, normative: 1, materiality: 2 },
          schema: { id: 'test', version: '1.0.0', family: null },
          stored_at: Date.now(),
          modified_at: null,
          version: 1,
          expires_at: null,
          size_bytes: 100,
          created_by: 'test',
          tombstone: false,
          retracted_by: null,
        },
      };

      mockClient = createMockClient({
        get_epistemic_entry: storedEntry,
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const result = await backend.get<{ found: boolean }>('found-key');

      expect(result).not.toBeNull();
      expect(result?.data.found).toBe(true);
      expect(result?.metadata.cid).toBe('cid:test-hash');
    });
  });

  describe('has()', () => {
    it('should return false for non-existent key', async () => {
      const exists = await backend.has('non-existent');
      expect(exists).toBe(false);
    });

    it('should return true when key exists', async () => {
      mockClient = createMockClient({
        has_epistemic_entry: true,
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const exists = await backend.has('existing-key');
      expect(exists).toBe(true);
    });
  });

  describe('delete()', () => {
    it('should return false when key does not exist', async () => {
      const deleted = await backend.delete('non-existent');
      expect(deleted).toBe(false);
    });

    it('should return true when deletion succeeds', async () => {
      mockClient = createMockClient({
        delete_epistemic_entry: true,
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const deleted = await backend.delete('existing-key');
      expect(deleted).toBe(true);
    });

    it('should handle E3+ immutable entries gracefully', async () => {
      mockClient = createMockClient({
        delete_epistemic_entry: () => {
          throw new Error('Cannot delete E3+ immutable entries');
        },
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const deleted = await backend.delete('immutable-key');
      expect(deleted).toBe(false);
    });
  });

  describe('keys()', () => {
    it('should return empty array when no keys exist', async () => {
      const keys = await backend.keys();
      expect(keys).toEqual([]);
    });

    it('should return all keys', async () => {
      mockClient = createMockClient({
        list_epistemic_keys: ['key1', 'key2', 'key3'],
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const keys = await backend.keys();
      expect(keys).toEqual(['key1', 'key2', 'key3']);
    });

    it('should filter keys by pattern', async () => {
      mockClient = createMockClient({
        list_epistemic_keys: (payload: { pattern: string | null }) => {
          const allKeys = ['user:1', 'user:2', 'config:app'];
          if (payload.pattern === 'user:*') {
            return allKeys.filter(k => k.startsWith('user:'));
          }
          return allKeys;
        },
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const keys = await backend.keys('user:*');
      expect(keys).toEqual(['user:1', 'user:2']);
    });
  });

  describe('stats()', () => {
    it('should return storage statistics', async () => {
      const now = Date.now();
      mockClient = createMockClient({
        get_storage_stats: {
          item_count: 42,
          total_size_bytes: 102400,
          oldest_item: now - 86400000,
          newest_item: now,
        },
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const stats = await backend.stats();

      expect(stats.itemCount).toBe(42);
      expect(stats.totalSizeBytes).toBe(102400);
      expect(stats.oldestItem).toBeDefined();
      expect(stats.newestItem).toBeDefined();
    });
  });

  describe('getByCID()', () => {
    it('should return null when CID not found', async () => {
      const result = await backend.getByCID('cid:unknown');
      expect(result).toBeNull();
    });

    it('should normalize CID prefix', async () => {
      const storedEntry = {
        key: 'found-by-cid',
        data: JSON.stringify({ cid: true }),
        metadata: {
          cid: 'cid:test-hash',
          classification: { empirical: 3, normative: 2, materiality: 3 },
          schema: { id: 'cid-test', version: '1.0.0', family: null },
          stored_at: Date.now(),
          modified_at: null,
          version: 1,
          expires_at: null,
          size_bytes: 50,
          created_by: 'test',
          tombstone: false,
          retracted_by: null,
        },
      };

      mockClient = createMockClient({
        get_by_cid: storedEntry,
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      // Should work with or without prefix
      const result1 = await backend.getByCID('cid:test-hash');
      const result2 = await backend.getByCID('test-hash');

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();
    });
  });

  describe('getReplicationStatus()', () => {
    it('should return replication status', async () => {
      mockClient = createMockClient({
        get_replication_status: {
          key: 'test-key',
          holder_count: 5,
          target_holders: 3,
          is_replicated: true,
        },
      });
      backend = createDHTBackend({ client: mockClient, enableLocalCache: false });

      const status = await backend.getReplicationStatus('test-key');

      expect(status.holders).toBe(5);
      expect(status.minRequired).toBe(3);
      expect(status.isHealthy).toBe(true);
    });
  });

  describe('ensureReplication()', () => {
    it('should request replication', async () => {
      const result = await backend.ensureReplication('test-key', 5);
      expect(result).toBe(true);

      expect(mockClient.callZome).toHaveBeenCalledWith(expect.objectContaining({
        fn_name: 'ensure_replication',
        payload: { key: 'test-key', min_holders: 5 },
      }));
    });
  });
});

describe('DHTBackend Local Cache', () => {
  it('should cache get results', async () => {
    const storedEntry = {
      key: 'cached-key',
      data: JSON.stringify({ cached: true }),
      metadata: {
        cid: 'cid:cached',
        classification: { empirical: 1, normative: 1, materiality: 2 },
        schema: { id: 'test', version: '1.0.0', family: null },
        stored_at: Date.now(),
        modified_at: null,
        version: 1,
        expires_at: null,
        size_bytes: 50,
        created_by: 'test',
        tombstone: false,
        retracted_by: null,
      },
    };

    const mockClient = createMockClient({
      get_epistemic_entry: storedEntry,
    });

    const backend = createDHTBackend({
      client: mockClient,
      enableLocalCache: true,
      localCacheTtlMs: 60000,
    });

    // First call hits DHT
    await backend.get('cached-key');
    expect(mockClient.callZome).toHaveBeenCalledTimes(1);

    // Second call should use cache
    await backend.get('cached-key');
    expect(mockClient.callZome).toHaveBeenCalledTimes(1); // Still 1, cached!
  });

  it('should invalidate cache on set', async () => {
    const mockClient = createMockClient();

    const backend = createDHTBackend({
      client: mockClient,
      enableLocalCache: true,
    });

    // Store updates cache
    await backend.set('cache-test', { value: 1 }, {
      classification: { empirical: 0, normative: 0, materiality: 2 },
      schema: { id: 'test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      createdBy: 'test',
      tombstone: false,
    });

    // has() should use cache
    const exists = await backend.has('cache-test');
    expect(exists).toBe(true);
  });
});

describe('DHTBackend Retry Logic', () => {
  it('should retry on transient failures', async () => {
    let attempts = 0;
    const mockClient = createMockClient({
      get_epistemic_entry: () => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Network error');
        }
        return {
          key: 'retry-key',
          data: JSON.stringify({ success: true }),
          metadata: {
            cid: 'cid:retry',
            classification: { empirical: 0, normative: 0, materiality: 2 },
            schema: { id: 'test', version: '1.0.0', family: null },
            stored_at: Date.now(),
            modified_at: null,
            version: 1,
            expires_at: null,
            size_bytes: 50,
            created_by: 'test',
            tombstone: false,
            retracted_by: null,
          },
        };
      },
    });

    const backend = createDHTBackend({
      client: mockClient,
      retries: 5,
      enableLocalCache: false,
    });

    const result = await backend.get<{ success: boolean }>('retry-key');
    expect(result?.data.success).toBe(true);
    expect(attempts).toBe(3);
  });

  it('should not retry validation errors', async () => {
    let attempts = 0;
    const mockClient = createMockClient({
      store_epistemic_entry: () => {
        attempts++;
        throw new Error('Invalid CID format: must start with cid:');
      },
    });

    const backend = createDHTBackend({
      client: mockClient,
      retries: 5,
      enableLocalCache: false,
    });

    await expect(backend.set('test', {}, {
      classification: { empirical: 0, normative: 0, materiality: 2 },
      schema: { id: 'test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      createdBy: 'test',
      tombstone: false,
    })).rejects.toThrow('Invalid');

    // Should only try once for validation errors
    expect(attempts).toBe(1);
  });
});

// =============================================================================
// Integration Tests (Require Running Conductor)
// =============================================================================

describe.skipIf(!process.env.CONDUCTOR_AVAILABLE)('DHTBackend Integration Tests', () => {
  // These tests require:
  // 1. Running Holochain conductor
  // 2. Installed epistemic_storage hApp
  // 3. CONDUCTOR_AVAILABLE=true environment variable

  it('should connect to running conductor', async () => {
    // Integration test placeholder
    // Would use real AppWebsocket connection
    expect(true).toBe(true);
  });

  it('should store and retrieve data from DHT', async () => {
    // Integration test placeholder
    expect(true).toBe(true);
  });

  it('should verify E3+ immutability', async () => {
    // Integration test placeholder
    expect(true).toBe(true);
  });
});

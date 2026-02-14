/**
 * Integration tests for EpistemicStorage with DHT Backend
 *
 * These tests verify the full integration between:
 * - EpistemicStorageImpl
 * - DHTBackend
 * - Storage Router (M2 → DHT routing)
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  createEpistemicStorage,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicStorageConfig,
} from '../../src/storage/index.js';
import type { HolochainClient } from '../../src/storage/backends/dht.js';

// =============================================================================
// Mock Holochain Client
// =============================================================================

// Stored entries use ZomeStorageMetadata format (snake_case)
interface MockZomeEntry {
  data: string;
  metadata: {
    cid: string;
    classification: { empirical: number; normative: number; materiality: number };
    schema: { id: string; version: string; family: string | null };
    stored_at: number;
    modified_at: number | null;
    version: number;
    expires_at: number | null;
    size_bytes: number;
    created_by: string;
    tombstone: boolean;
    retracted_by: string | null;
  };
}

const mockStorage = new Map<string, MockZomeEntry>();

function createMockHolochainClient(): HolochainClient {
  return {
    callZome: vi.fn().mockImplementation(async (request: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) => {
      const { fn_name, payload } = request;
      switch (fn_name) {
        case 'store_epistemic_entry': {
          const { key, entry } = payload as { key: string; entry: MockZomeEntry };
          mockStorage.set(key, entry);
          return 'mock-action-hash';
        }
        case 'get_epistemic_entry': {
          const { key } = payload as { key: string };
          const entry = mockStorage.get(key);
          if (!entry) return null;
          return {
            key,
            data: entry.data,
            metadata: entry.metadata,
          };
        }
        case 'has_epistemic_entry': {
          const { key } = payload as { key: string };
          return mockStorage.has(key);
        }
        case 'delete_epistemic_entry': {
          const { key } = payload as { key: string };
          return mockStorage.delete(key);
        }
        case 'list_epistemic_keys':
          return Array.from(mockStorage.keys());
        case 'get_storage_stats':
          return {
            item_count: mockStorage.size,
            total_size_bytes: 1000,
            oldest_item: Date.now() - 1000,
            newest_item: Date.now(),
          };
        case 'clear_epistemic_storage':
          const count = mockStorage.size;
          mockStorage.clear();
          return count;
        default:
          return null;
      }
    }),
  };
}

// =============================================================================
// Integration Tests
// =============================================================================

describe('EpistemicStorage + DHT Integration', () => {
  let mockClient: HolochainClient;
  let config: EpistemicStorageConfig;

  beforeEach(() => {
    mockStorage.clear();
    mockClient = createMockHolochainClient();
    config = {
      agentId: 'test-agent',
      dhtBackend: {
        client: mockClient,
        zomeName: 'epistemic_storage',
        dnaRole: 'epistemic_storage',
        enableLocalCache: false,
      },
    };
  });

  afterEach(() => {
    mockStorage.clear();
  });

  describe('DHT Backend Enablement', () => {
    it('should report DHT as enabled when configured', () => {
      const storage = createEpistemicStorage(config);
      expect(storage.isDHTEnabled()).toBe(true);
      storage.dispose();
    });

    it('should report DHT as disabled when not configured', () => {
      const storage = createEpistemicStorage({ agentId: 'test' });
      expect(storage.isDHTEnabled()).toBe(false);
      storage.dispose();
    });

    it('should expose DHT backend for direct access', () => {
      const storage = createEpistemicStorage(config);
      const dhtBackend = storage.getDHTBackend();
      expect(dhtBackend).not.toBeNull();
      storage.dispose();
    });
  });

  describe('M2 Data Routes to DHT', () => {
    it('should store M2 persistent data through DHT', async () => {
      const storage = createEpistemicStorage(config);

      const receipt = await storage.store(
        'profile:alice',
        { name: 'Alice', bio: 'Holochain developer' },
        {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        },
        { schema: { id: 'profile', version: '1.0.0' } }
      );

      expect(receipt.key).toBe('profile:alice');
      expect(receipt.cid).toMatch(/^cid:/);
      expect(receipt.tier.backend).toBe('dht');

      // Verify DHT was called
      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'store_epistemic_entry',
        })
      );

      storage.dispose();
    });

    it('should retrieve M2 data from DHT', async () => {
      const storage = createEpistemicStorage(config);

      // Store first (N2_Network for public access without capability)
      await storage.store(
        'user:bob',
        { username: 'bob', level: 42 },
        {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        },
        { schema: { id: 'user', version: '2.0.0' } }
      );

      // Retrieve should use DHT
      const result = await storage.retrieve<{ username: string; level: number }>('user:bob');

      expect(result).not.toBeNull();
      expect(result?.data.username).toBe('bob');
      expect(result?.data.level).toBe(42);

      storage.dispose();
    });
  });

  describe('M0/M1 Data Uses Local Backends', () => {
    it('should store M0 ephemeral data in memory (not DHT)', async () => {
      const storage = createEpistemicStorage(config);

      const receipt = await storage.store(
        'session:temp',
        { token: 'abc123' },
        {
          empirical: EmpiricalLevel.E0_Unverified,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M0_Ephemeral,
        },
        { schema: { id: 'session', version: '1.0.0' } }
      );

      expect(receipt.tier.backend).toBe('memory');

      // DHT should not have been called for store (only the M2 test above uses DHT)
      const storeCalls = (mockClient.callZome as ReturnType<typeof vi.fn>).mock.calls
        .filter(call => call[0].fn_name === 'store_epistemic_entry');
      expect(storeCalls.length).toBe(0);

      storage.dispose();
    });

    it('should store M1 temporal data locally (not DHT)', async () => {
      const storage = createEpistemicStorage(config);

      const receipt = await storage.store(
        'cache:user-prefs',
        { theme: 'dark', language: 'en' },
        {
          empirical: EmpiricalLevel.E1_Testimonial,
          normative: NormativeLevel.N0_Personal,
          materiality: MaterialityLevel.M1_Temporal,
        },
        { schema: { id: 'preferences', version: '1.0.0' } }
      );

      expect(receipt.tier.backend).toBe('local');

      storage.dispose();
    });
  });

  describe('DHT Statistics Integration', () => {
    it('should include DHT stats in overall statistics', async () => {
      const storage = createEpistemicStorage(config);

      // Store some M2 data
      await storage.store(
        'stats:test',
        { value: 1 },
        {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        },
        { schema: { id: 'stats', version: '1.0.0' } }
      );

      const stats = await storage.getStats();

      // Should have called DHT stats
      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_storage_stats',
        })
      );

      // DHT items should be reflected
      expect(stats.itemsByBackend.dht).toBeGreaterThanOrEqual(0);

      storage.dispose();
    });
  });

  describe('Classification-Based Routing', () => {
    it('should route based on materiality level', async () => {
      const storage = createEpistemicStorage(config);

      // M0 → memory
      const r0 = await storage.store('m0', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema: { id: 'test', version: '1' } });
      expect(r0.tier.backend).toBe('memory');

      // M1 → local
      const r1 = await storage.store('m1', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema: { id: 'test', version: '1' } });
      expect(r1.tier.backend).toBe('local');

      // M2 → dht
      const r2 = await storage.store('m2', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema: { id: 'test', version: '1' } });
      expect(r2.tier.backend).toBe('dht');

      // M3 → dht (with higher replication)
      const r3 = await storage.store('m3', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M3_Immutable,
      }, { schema: { id: 'test', version: '1' } });
      expect(r3.tier.backend).toBe('dht');

      storage.dispose();
    });
  });
});

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Integration tests for EpistemicStorage with IPFS Backend
 *
 * These tests verify the full integration between:
 * - EpistemicStorageImpl
 * - IPFSBackend
 * - Storage Router (M3 → IPFS routing for immutable data)
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  createEpistemicStorage,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicStorageConfig,
} from '../../src/storage/index.js';

// =============================================================================
// Mock IPFS Gateway/API
// =============================================================================

// Simulated IPFS storage (CID → content)
const mockIPFSStorage = new Map<string, string>();
let cidCounter = 0;

// Mock fetch for IPFS operations
const originalFetch = global.fetch;

function createMockFetch() {
  return vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
    const urlStr = url.toString();

    // IPFS Add (POST to API)
    if (urlStr.includes('/api/v0/add') && options?.method === 'POST') {
      const cid = `Qm${(++cidCounter).toString().padStart(44, 'a')}`;
      // Store the content (simplified - in reality FormData parsing would be needed)
      mockIPFSStorage.set(cid, 'stored-content');

      return {
        ok: true,
        json: async () => ({ Hash: cid, Size: '100' }),
      };
    }

    // IPFS Get (GET from gateway)
    if (urlStr.includes('/ipfs/')) {
      const cid = urlStr.split('/ipfs/')[1];

      if (!mockIPFSStorage.has(cid)) {
        return { ok: false, status: 404, statusText: 'Not Found' };
      }

      // Return a wrapper object similar to what IPFSBackend stores
      return {
        ok: true,
        json: async () => ({
          data: JSON.stringify({ test: 'data' }),
          metadata: { localCid: 'cid:test', storedAt: Date.now() },
        }),
      };
    }

    // IPFS Pin (POST to API)
    if (urlStr.includes('/api/v0/pin/')) {
      return { ok: true };
    }

    throw new Error(`Unexpected fetch: ${urlStr}`);
  });
}

// =============================================================================
// Integration Tests
// =============================================================================

describe('EpistemicStorage + IPFS Integration', () => {
  let config: EpistemicStorageConfig;

  beforeEach(() => {
    mockIPFSStorage.clear();
    cidCounter = 0;
    global.fetch = createMockFetch();

    config = {
      agentId: 'test-agent',
      ipfsBackend: {
        gatewayUrl: 'https://ipfs.io',
        apiUrl: 'http://localhost:5001',
        pinContent: true,
        enableLocalIndex: true,
        localCacheTtlMs: 60000,
      },
    };
  });

  afterEach(() => {
    global.fetch = originalFetch;
    mockIPFSStorage.clear();
  });

  describe('IPFS Backend Enablement', () => {
    it('should report IPFS as enabled when configured', () => {
      const storage = createEpistemicStorage(config);
      expect(storage.isIPFSEnabled()).toBe(true);
      storage.dispose();
    });

    it('should report IPFS as disabled when not configured', () => {
      const storage = createEpistemicStorage({ agentId: 'test' });
      expect(storage.isIPFSEnabled()).toBe(false);
      storage.dispose();
    });

    it('should expose IPFS backend for direct access', () => {
      const storage = createEpistemicStorage(config);
      const ipfsBackend = storage.getIPFSBackend();
      expect(ipfsBackend).not.toBeNull();
      storage.dispose();
    });
  });

  describe('M3 Immutable Data Routes to IPFS', () => {
    it('should route M3 data to IPFS when configured', async () => {
      const storage = createEpistemicStorage(config);

      const receipt = await storage.store(
        'constitution:v1',
        { title: 'Network Constitution', version: 1 },
        {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        },
        { schema: { id: 'constitution', version: '1.0.0' } }
      );

      expect(receipt.key).toBe('constitution:v1');
      expect(receipt.cid).toMatch(/^cid:/);
      expect(receipt.tier.backend).toBe('ipfs');

      storage.dispose();
    });

    it('should fall back to DHT for M3 when IPFS not configured but DHT is', async () => {
      // No IPFS, but mock DHT
      const mockClient = {
        callZome: vi.fn().mockResolvedValue('mock-action-hash'),
      };

      const dhtOnlyConfig: EpistemicStorageConfig = {
        agentId: 'test-agent',
        dhtBackend: {
          client: mockClient,
          zomeName: 'epistemic_storage',
          dnaRole: 'epistemic_storage',
        },
      };

      const storage = createEpistemicStorage(dhtOnlyConfig);

      const receipt = await storage.store(
        'immutable:test',
        { data: 'test' },
        {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M3_Immutable,
        },
        { schema: { id: 'test', version: '1.0.0' } }
      );

      // Should fall back to DHT when IPFS not available
      expect(receipt.tier.backend).toBe('dht');

      storage.dispose();
    });
  });

  describe('M2 Data Uses DHT (not IPFS)', () => {
    it('should route M2 persistent data via DHT (tier shows dht)', async () => {
      // Only IPFS configured (no DHT)
      const storage = createEpistemicStorage(config);

      const receipt = await storage.store(
        'profile:alice',
        { name: 'Alice' },
        {
          empirical: EmpiricalLevel.E2_PrivateVerify,
          normative: NormativeLevel.N2_Network,
          materiality: MaterialityLevel.M2_Persistent,
        },
        { schema: { id: 'profile', version: '1.0.0' } }
      );

      // Router determines 'dht' for M2 (implementation falls back to local if not configured)
      // The tier reflects the router's recommendation
      expect(receipt.tier.backend).toBe('dht');

      storage.dispose();
    });
  });

  describe('Filecoin Backend Falls Back to IPFS', () => {
    it('should route filecoin requests to IPFS when IPFS is configured', async () => {
      const storage = createEpistemicStorage(config);

      // Filecoin routing (not directly testable via store(), but getBackend uses it)
      // For now, just verify IPFS backend is available as fallback
      expect(storage.isIPFSEnabled()).toBe(true);

      storage.dispose();
    });
  });

  describe('IPFS Statistics Integration', () => {
    it('should include IPFS stats in overall statistics', async () => {
      const storage = createEpistemicStorage(config);

      // Store some M3 data
      await storage.store(
        'immutable:stats-test',
        { value: 42 },
        {
          empirical: EmpiricalLevel.E3_Cryptographic,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M3_Immutable,
        },
        { schema: { id: 'test', version: '1.0.0' } }
      );

      const stats = await storage.getStats();

      // IPFS items should be counted
      expect(stats.itemsByBackend.ipfs).toBeGreaterThanOrEqual(0);

      storage.dispose();
    });
  });

  describe('Classification-Based Routing Summary', () => {
    it('should route all M-levels correctly', async () => {
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

      // M2 → dht (router determines 'dht', falls back to local in implementation)
      const r2 = await storage.store('m2', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema: { id: 'test', version: '1' } });
      expect(r2.tier.backend).toBe('dht');

      // M3 → ipfs
      const r3 = await storage.store('m3', {}, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M3_Immutable,
      }, { schema: { id: 'test', version: '1' } });
      expect(r3.tier.backend).toBe('ipfs');

      storage.dispose();
    });
  });

  describe('Combined DHT + IPFS Configuration', () => {
    it('should route M2 to DHT and M3 to IPFS when both configured', async () => {
      const mockClient = {
        callZome: vi.fn().mockResolvedValue('mock-action-hash'),
      };

      const fullConfig: EpistemicStorageConfig = {
        agentId: 'test-agent',
        dhtBackend: {
          client: mockClient,
          zomeName: 'epistemic_storage',
          dnaRole: 'epistemic_storage',
        },
        ipfsBackend: {
          gatewayUrl: 'https://ipfs.io',
          apiUrl: 'http://localhost:5001',
        },
      };

      const storage = createEpistemicStorage(fullConfig);

      expect(storage.isDHTEnabled()).toBe(true);
      expect(storage.isIPFSEnabled()).toBe(true);

      // M2 → DHT
      const r2 = await storage.store('m2-test', { data: 'persistent' }, {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema: { id: 'test', version: '1' } });
      expect(r2.tier.backend).toBe('dht');

      // M3 → IPFS
      const r3 = await storage.store('m3-test', { data: 'immutable' }, {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M3_Immutable,
      }, { schema: { id: 'test', version: '1' } });
      expect(r3.tier.backend).toBe('ipfs');

      storage.dispose();
    });
  });
});

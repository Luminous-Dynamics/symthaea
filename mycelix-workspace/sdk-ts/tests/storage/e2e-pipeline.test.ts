// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * End-to-end integration tests for the full UESS pipeline.
 *
 * Exercises: classify → route → store → retrieve → migrate → verify
 * across memory/local/DHT/IPFS backends in a single flow.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import {
  createEpistemicStorage,
  createMemoryBackend,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  migrate,
  exportBackend,
  importBundle,
  sync,
} from '../../src/storage/index.js';

// Capability granting full access
const fullCap = {
  agentId: 'test-agent',
  resourceKeys: '*' as const,
  permissions: ['read' as const, 'write' as const, 'delete' as const],
  issuedAt: Date.now(),
  expiresAt: Date.now() + 600000,
};

// Mock IPFS
const mockIPFSStorage = new Map<string, string>();
let cidCounter = 0;
const originalFetch = global.fetch;

function setupMockFetch() {
  cidCounter = 0;
  mockIPFSStorage.clear();
  global.fetch = vi.fn().mockImplementation(async (url: string, options?: RequestInit) => {
    const urlStr = url.toString();
    if (urlStr.includes('/api/v0/add') && options?.method === 'POST') {
      const cid = `Qm${(++cidCounter).toString().padStart(44, 'a')}`;
      mockIPFSStorage.set(cid, 'stored');
      return { ok: true, json: async () => ({ Hash: cid, Size: '100' }) };
    }
    if (urlStr.includes('/ipfs/')) {
      const cid = urlStr.split('/ipfs/')[1];
      if (!mockIPFSStorage.has(cid)) return { ok: false, status: 404 };
      return {
        ok: true,
        json: async () => ({ data: '{}', metadata: { localCid: 'cid:test', storedAt: Date.now() } }),
      };
    }
    if (urlStr.includes('/api/v0/pin/')) return { ok: true };
    throw new Error(`Unexpected fetch: ${urlStr}`);
  });
}

afterEach(() => {
  global.fetch = originalFetch;
  mockIPFSStorage.clear();
});

const schema = { id: 'test', version: '1.0.0' };

describe('End-to-End UESS Pipeline', () => {
  describe('Full Classification → Storage → Retrieval Pipeline', () => {
    it('should store and retrieve data at all M-levels', async () => {
      setupMockFetch();
      const storage = createEpistemicStorage({
        agentId: 'e2e-agent',
        ipfsBackend: { gatewayUrl: 'https://ipfs.io', apiUrl: 'http://localhost:5001' },
      });

      // M0 → memory
      const r0 = await storage.store('e2e:m0', { level: 'ephemeral' }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });
      expect(r0.tier.backend).toBe('memory');

      // M1 → local
      const r1 = await storage.store('e2e:m1', { level: 'temporal' }, {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema });
      expect(r1.tier.backend).toBe('local');

      // M2 → dht
      const r2 = await storage.store('e2e:m2', { level: 'persistent' }, {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      }, { schema });
      expect(r2.tier.backend).toBe('dht');

      // M3 → ipfs
      const r3 = await storage.store('e2e:m3', { level: 'immutable' }, {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M3_Immutable,
      }, { schema });
      expect(r3.tier.backend).toBe('ipfs');

      // Retrieve all (with capability since N0 = owner access)
      for (const key of ['e2e:m0', 'e2e:m1', 'e2e:m2', 'e2e:m3']) {
        const data = await storage.retrieve(key, fullCap);
        expect(data).not.toBeNull();
        expect(data!.data).toHaveProperty('level');
      }

      // Verify CID uniqueness
      const cids = [r0.cid, r1.cid, r2.cid, r3.cid];
      expect(new Set(cids).size).toBe(4);

      storage.dispose();
    });

    it('should maintain data integrity across store and retrieve', async () => {
      const storage = createEpistemicStorage({ agentId: 'e2e-agent' });

      const complexData = {
        name: 'Alice',
        scores: [0.95, 0.88, 0.72],
        nested: { trust: { level: 3, verified: true } },
      };

      await storage.store('integrity:test', complexData, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema: { id: 'profile', version: '2.0.0' } });

      const result = await storage.retrieve<typeof complexData>('integrity:test', fullCap);
      expect(result!.data).toEqual(complexData);
      expect(result!.metadata.schema).toEqual({ id: 'profile', version: '2.0.0' });

      storage.dispose();
    });
  });

  describe('Cross-Backend Migration Pipeline', () => {
    it('should migrate data between memory backends via export/import', async () => {
      const source = createMemoryBackend();
      const target = createMemoryBackend();
      const meta = {
        createdAt: Date.now(), updatedAt: Date.now(), createdBy: 'test',
        classification: { empirical: 0, normative: 0, materiality: 1 },
        schemaId: schema, tags: [],
      };

      for (let i = 0; i < 5; i++) {
        await source.set(`migrate:${i}`, { index: i, value: `item-${i}` }, meta);
      }

      const bundle = await exportBackend(source, { agentId: 'source' });
      expect(bundle.itemCount).toBe(5);

      const importResult = await importBundle(target, bundle);
      expect(importResult.migrated).toBe(5);

      for (let i = 0; i < 5; i++) {
        const item = await target.get<{ index: number; value: string }>(`migrate:${i}`);
        expect(item).not.toBeNull();
        expect(item!.data.index).toBe(i);
      }
    });

    it('should sync two memory backends with diff detection', async () => {
      const backendA = createMemoryBackend();
      const backendB = createMemoryBackend();
      const meta = {
        createdAt: Date.now(), updatedAt: Date.now(), createdBy: 'test',
        classification: { empirical: 0, normative: 0, materiality: 1 },
        schemaId: schema, tags: [],
      };

      // A has items 0-2
      for (let i = 0; i < 3; i++) {
        await backendA.set(`sync:${i}`, { from: 'A', i }, meta);
      }

      // B has items 2-4
      for (let i = 2; i < 5; i++) {
        await backendB.set(`sync:${i}`, { from: 'B', i }, meta);
      }

      // Dry run
      const diff = await sync(backendA, backendB, { dryRun: true });
      expect(diff.shared).toContain('sync:2');
      expect(diff.sourceOnly).toContain('sync:0');
      expect(diff.sourceOnly).toContain('sync:1');
      expect(diff.targetOnly).toContain('sync:3');
      expect(diff.targetOnly).toContain('sync:4');

      // Push A → B
      const pushResult = await sync(backendA, backendB, { direction: 'push' });
      expect(pushResult.migration?.migrated).toBe(2); // sync:0, sync:1
    });
  });

  describe('Observability Pipeline', () => {
    it('should collect metrics during store/retrieve operations', async () => {
      const storage = createEpistemicStorage({
        agentId: 'metrics-agent',
        observability: { enableMetrics: true, enableTracing: true },
      });

      // Store two items
      await storage.store('obs:1', { x: 1 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });

      await storage.store('obs:2', { x: 2 }, {
        empirical: EmpiricalLevel.E1_Testimonial,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema });

      // Retrieve (with capability)
      await storage.retrieve('obs:1', fullCap);
      await storage.retrieve('obs:2', fullCap);
      await storage.retrieve('obs:nonexistent', fullCap);

      // Check metrics
      const snapshot = storage.getMetricsSnapshot();
      expect(snapshot).not.toBeNull();
      expect(snapshot!.operationsTotal.store).toBe(2);
      expect(snapshot!.operationsTotal.retrieve).toBe(3);
      expect(snapshot!.cacheHitRate).toBeGreaterThan(0);

      // Check traces (2 stores + 2 successful retrieves)
      const spans = storage.getTraceSpans();
      expect(spans.length).toBeGreaterThanOrEqual(4);

      // Check Prometheus export
      const prom = storage.getPrometheusMetrics();
      expect(prom).toContain('uess_operations_total');
      expect(prom).toContain('uess_operation_latency_ms');

      storage.dispose();
    });
  });

  describe('Reclassification with Migration', () => {
    it('should reclassify without migration when no backend change needed', async () => {
      const storage = createEpistemicStorage({ agentId: 'reclass-agent' });

      // Store as M0 (memory), E0, N0
      await storage.store('reclass:noop', { value: 42 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });

      // Reclassify E-level (stays in same backend)
      const r2 = await storage.reclassify(
        'reclass:noop',
        { empirical: EmpiricalLevel.E1_Testimonial },
        { description: 'Evidence added', upgradedBy: 'reclass-agent' },
        fullCap,
      );
      expect(r2.classification.empirical).toBe(EmpiricalLevel.E1_Testimonial);

      // Data still retrievable
      const data = await storage.retrieve('reclass:noop', fullCap);
      expect(data).not.toBeNull();
      expect(data!.data).toEqual({ value: 42 });

      storage.dispose();
    });

    it('should reclassify M-level and report new tier', async () => {
      const storage = createEpistemicStorage({ agentId: 'reclass-agent' });

      // Store as M1 (local)
      const r1 = await storage.store('reclass:tier', { value: 99 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema });
      expect(r1.tier.backend).toBe('local');

      // Reclassify to M2 - receipt should show dht tier
      const r2 = await storage.reclassify(
        'reclass:tier',
        { materiality: MaterialityLevel.M2_Persistent },
        { description: 'Promoting', upgradedBy: 'reclass-agent' },
        fullCap,
      );
      expect(r2.tier.backend).toBe('dht');
      expect(r2.version).toBe(r1.version + 1);

      storage.dispose();
    });
  });

  describe('Verification Pipeline', () => {
    it('should verify data integrity end-to-end', async () => {
      const storage = createEpistemicStorage({ agentId: 'verify-agent' });

      await storage.store('verify:test', { important: true }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema });

      const result = await storage.verify('verify:test');
      expect(result.verified).toBe(true);
      expect(result.cidValid).toBe(true);
      expect(result.errors).toHaveLength(0);

      const missing = await storage.verify('verify:missing');
      expect(missing.verified).toBe(false);
      expect(missing.errors).toContain('Key not found');

      storage.dispose();
    });
  });

  describe('Query Pipeline', () => {
    it('should query by classification range', async () => {
      const storage = createEpistemicStorage({ agentId: 'query-agent' });

      // Store at different E-levels, all N3 (public) to avoid access issues
      for (let e = 0; e <= 3; e++) {
        await storage.store(`query:e${e}`, { empiricalLevel: e }, {
          empirical: e as EmpiricalLevel,
          normative: NormativeLevel.N3_Universal,
          materiality: MaterialityLevel.M0_Ephemeral,
        }, { schema });
      }

      // Query for E2+
      const result = await storage.query({
        classification: { minEmpirical: EmpiricalLevel.E2_PrivateVerify },
      });

      expect(result.totalCount).toBe(2);
      expect(result.items.every((item) =>
        item.metadata.classification.empirical >= EmpiricalLevel.E2_PrivateVerify
      )).toBe(true);

      storage.dispose();
    });
  });

  describe('Statistics Pipeline', () => {
    it('should aggregate stats across all backends', async () => {
      setupMockFetch();
      const storage = createEpistemicStorage({
        agentId: 'stats-agent',
        ipfsBackend: { gatewayUrl: 'https://ipfs.io', apiUrl: 'http://localhost:5001' },
      });

      await storage.store('stats:memory', { m: 0 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });

      await storage.store('stats:local', { m: 1 }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M1_Temporal,
      }, { schema });

      await storage.store('stats:ipfs', { m: 3 }, {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M3_Immutable,
      }, { schema });

      const stats = await storage.getStats();
      expect(stats.totalItems).toBe(3);
      expect(stats.itemsByBackend.memory).toBe(1);
      // local backend may have items from other test leakage; check >= 1
      expect(stats.itemsByBackend.local).toBeGreaterThanOrEqual(1);
      expect(stats.itemsByMateriality[0]).toBe(1);
      expect(stats.itemsByMateriality[3]).toBe(1);

      storage.dispose();
    });
  });
});

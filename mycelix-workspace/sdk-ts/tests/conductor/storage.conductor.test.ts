// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Storage Conductor Integration Tests
 *
 * Tests the DHT backend against a live Holochain conductor.
 * These tests are skipped when no conductor is available.
 *
 * Start conductor with:
 *   docker compose -f docker/docker-compose.yml up -d
 *
 * Run tests:
 *   CONDUCTOR_AVAILABLE=1 npx vitest run tests/conductor/storage.conductor.test.ts
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  getConductorConfig,
  isConductorAvailable,
  retry,
  waitForSync,
  generateTestAgentId,
  type ConductorConfig,
} from './conductor-harness.js';
import { DHTBackend, createDHTBackend, type HolochainClient } from '../../src/storage/backends/dht.js';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../src/storage/index.js';

// =============================================================================
// Test Setup
// =============================================================================

let conductorAvailable = false;
let config: ConductorConfig;
let appClient: HolochainClient;
let backend: DHTBackend;
const testAgentId = generateTestAgentId();

// Unique prefix per run to avoid collisions
const runId = Date.now().toString(36);
const testKey = (name: string) => `test:${runId}:${name}`;

beforeAll(async () => {
  config = getConductorConfig();
  conductorAvailable = await isConductorAvailable(config);

  if (!conductorAvailable) {
    console.log('⚠️  Skipping storage conductor tests - no conductor available');
    console.log('   Start with: docker compose -f docker/docker-compose.yml up -d');
    return;
  }

  // Dynamic import to avoid libsodium ESM issues
  const { AppWebsocket } = await import('@holochain/client');
  appClient = await AppWebsocket.connect({ url: new URL(config.appUrl) }) as unknown as HolochainClient;

  backend = createDHTBackend({
    client: appClient,
    agentId: testAgentId,
    enableLocalCache: false, // Disable cache to test DHT directly
    retries: 3,
    timeoutMs: 15000,
  });
});

afterAll(async () => {
  if (backend) {
    backend.dispose();
  }
  if (appClient && 'client' in appClient) {
    await (appClient as any).client.close();
  }
});

// =============================================================================
// Basic CRUD Operations
// =============================================================================

describe('Storage: DHT Backend (conductor)', () => {
  const metadata = {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    createdBy: testAgentId,
    classification: {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M2_Persistent,
    },
    schema: { id: 'test', version: '1.0.0' },
    storedAt: Date.now(),
    version: 1,
    tags: [],
  };

  it.skipIf(!conductorAvailable)('should store and retrieve data', async () => {
    const key = testKey('basic-crud');
    const data = { name: 'Alice', score: 42 };

    const storedMeta = await retry(() => backend.set(key, data, metadata));
    expect(storedMeta.cid).toBeDefined();
    expect(storedMeta.sizeBytes).toBeGreaterThan(0);

    await waitForSync();

    const result = await retry(() => backend.get(key));
    expect(result).not.toBeNull();
    expect(result!.data).toEqual(data);
    expect(result!.metadata.createdBy).toBe(testAgentId);
  });

  it.skipIf(!conductorAvailable)('should report key existence', async () => {
    const key = testKey('has-check');
    const data = { exists: true };

    await retry(() => backend.set(key, data, metadata));
    await waitForSync();

    const exists = await retry(() => backend.has(key));
    expect(exists).toBe(true);

    const missing = await retry(() => backend.has(testKey('nonexistent')));
    expect(missing).toBe(false);
  });

  it.skipIf(!conductorAvailable)('should delete entries (tombstone)', async () => {
    const key = testKey('delete-me');
    await retry(() => backend.set(key, { temp: true }, metadata));
    await waitForSync();

    const deleted = await retry(() => backend.delete(key));
    expect(deleted).toBe(true);

    await waitForSync();

    const result = await retry(() => backend.get(key));
    expect(result).toBeNull();
  });

  it.skipIf(!conductorAvailable)('should list keys', async () => {
    const prefix = `list:${runId}`;
    for (let i = 0; i < 3; i++) {
      await retry(() => backend.set(`${prefix}:item-${i}`, { i }, metadata));
    }
    await waitForSync();

    const keys = await retry(() => backend.keys(`${prefix}:*`));
    expect(keys.length).toBeGreaterThanOrEqual(3);
    expect(keys.every((k) => k.startsWith(prefix))).toBe(true);
  });

  it.skipIf(!conductorAvailable)('should return stats', async () => {
    const stats = await retry(() => backend.stats());
    expect(stats.itemCount).toBeGreaterThanOrEqual(0);
    expect(stats.totalSizeBytes).toBeGreaterThanOrEqual(0);
  });
});

// =============================================================================
// Content-Addressed Retrieval
// =============================================================================

describe('Storage: DHT CID Lookup (conductor)', () => {
  const metadata = {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    createdBy: testAgentId,
    classification: {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    },
    schema: { id: 'cid-test', version: '1.0.0' },
    storedAt: Date.now(),
    version: 1,
    tags: [],
  };

  it.skipIf(!conductorAvailable)('should retrieve by CID', async () => {
    const key = testKey('cid-lookup');
    const data = { content: 'addressable', nonce: runId };

    const storedMeta = await retry(() => backend.set(key, data, metadata));
    const cid = storedMeta.cid;
    expect(cid).toMatch(/^cid:/);

    await waitForSync();

    const result = await retry(() => backend.getByCID(cid));
    expect(result).not.toBeNull();
    expect(result!.key).toBe(key);
    expect(result!.data).toEqual(data);
  });
});

// =============================================================================
// Classification Preservation
// =============================================================================

describe('Storage: DHT Classification (conductor)', () => {
  it.skipIf(!conductorAvailable)('should preserve E/N/M classification through DHT roundtrip', async () => {
    const key = testKey('classification');
    const data = { classified: true };
    const classification = {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N3_Universal,
      materiality: MaterialityLevel.M3_Immutable,
    };

    const meta = {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: testAgentId,
      classification,
      schema: { id: 'class-test', version: '2.0.0' },
      storedAt: Date.now(),
      version: 1,
      tags: [],
    };

    await retry(() => backend.set(key, data, meta));
    await waitForSync();

    const result = await retry(() => backend.get(key));
    expect(result).not.toBeNull();
    expect(result!.metadata.classification).toEqual(classification);
    expect(result!.metadata.schema.id).toBe('class-test');
    expect(result!.metadata.schema.version).toBe('2.0.0');
  });
});

// =============================================================================
// Replication
// =============================================================================

describe('Storage: DHT Replication (conductor)', () => {
  it.skipIf(!conductorAvailable)('should report replication status', async () => {
    const key = testKey('replication');
    const metadata = {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: testAgentId,
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      },
      schema: { id: 'repl-test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      tags: [],
    };

    await retry(() => backend.set(key, { replicated: true }, metadata));
    await waitForSync(1000);

    const status = await retry(() => backend.getReplicationStatus(key));
    expect(status.holders).toBeGreaterThanOrEqual(0);
    expect(status.minRequired).toBeGreaterThanOrEqual(1);
  });
});

// =============================================================================
// Cache Integration
// =============================================================================

describe('Storage: DHT Cache Behavior (conductor)', () => {
  it.skipIf(!conductorAvailable)('should serve from cache on second read', async () => {
    const cachedBackend = createDHTBackend({
      client: appClient,
      agentId: testAgentId,
      enableLocalCache: true,
      localCacheTtlMs: 5000,
      retries: 3,
    });

    const key = testKey('cache-test');
    const data = { cached: true };
    const metadata = {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      createdBy: testAgentId,
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M2_Persistent,
      },
      schema: { id: 'cache-test', version: '1.0.0' },
      storedAt: Date.now(),
      version: 1,
      tags: [],
    };

    await retry(() => cachedBackend.set(key, data, metadata));
    await waitForSync();

    // First read populates cache
    const first = await retry(() => cachedBackend.get(key));
    expect(first).not.toBeNull();

    // Second read should be instant (from cache)
    const start = performance.now();
    const second = await cachedBackend.get(key);
    const elapsed = performance.now() - start;

    expect(second).not.toBeNull();
    expect(second!.data).toEqual(data);
    // Cache read should be sub-millisecond (no network)
    expect(elapsed).toBeLessThan(5);

    cachedBackend.dispose();
  });
});

// =============================================================================
// Error Handling
// =============================================================================

describe('Storage: DHT Error Handling (conductor)', () => {
  it.skipIf(!conductorAvailable)('should handle get for nonexistent key gracefully', async () => {
    const result = await backend.get(testKey('does-not-exist'));
    expect(result).toBeNull();
  });

  it('should throw after dispose', async () => {
    const disposedBackend = createDHTBackend({
      client: { callZome: async () => null } as unknown as HolochainClient,
      agentId: 'disposed',
    });
    disposedBackend.dispose();

    await expect(() => disposedBackend.get('any')).rejects.toThrow('disposed');
  });
});

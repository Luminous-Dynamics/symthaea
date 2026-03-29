// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Storage Benchmarks
 *
 * Performance tests for:
 * - Store operations across backends
 * - Retrieve operations with/without cache
 * - Router classification
 * - Batch operations
 * - Migration throughput
 * - CRDT merge operations
 */

import { bench, describe } from 'vitest';
import {
  createEpistemicStorage,
  createMemoryBackend,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  StorageRouter,
  DEFAULT_ROUTER_CONFIG,
  migrate,
  exportBackend,
  importBundle,
  LWWRegister,
  ORSet,
  GCounter,
  PNCounter,
  createVectorClock,
  incrementClock,
  mergeClocks,
  batchStore,
  batchRetrieve,
} from '../src/storage/index.js';

// ============================================================================
// Router Classification Benchmarks
// ============================================================================

describe('Storage: Router Classification', () => {
  const router = new StorageRouter(DEFAULT_ROUTER_CONFIG);

  bench('route M0 (ephemeral)', () => {
    router.route({
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    });
  });

  bench('route M1 (temporal)', () => {
    router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M1_Temporal,
    });
  });

  bench('route M2 (persistent)', () => {
    router.route({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });
  });

  bench('route M3 (immutable)', () => {
    router.route({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N3_Universal,
      materiality: MaterialityLevel.M3_Immutable,
    });
  });

  bench('route 100 classifications', () => {
    for (let i = 0; i < 100; i++) {
      router.route({
        empirical: (i % 5) as EmpiricalLevel,
        normative: (i % 4) as NormativeLevel,
        materiality: (i % 4) as MaterialityLevel,
      });
    }
  });
});

// ============================================================================
// Store/Retrieve Benchmarks
// ============================================================================

describe('Storage: Store Operations', () => {
  const schema = { id: 'bench', version: '1.0.0' };

  bench('store M0 (memory)', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    await storage.store('bench:m0', { x: 1 }, {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    }, { schema });
    storage.dispose();
  });

  bench('store M1 (local)', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    await storage.store('bench:m1', { x: 1 }, {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M1_Temporal,
    }, { schema });
    storage.dispose();
  });

  bench('store 10 items (memory)', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    for (let i = 0; i < 10; i++) {
      await storage.store(`bench:${i}`, { i }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });
    }
    storage.dispose();
  });
});

describe('Storage: Retrieve Operations', () => {
  const schema = { id: 'bench', version: '1.0.0' };
  const cap = {
    agentId: 'bench',
    resourceKeys: '*' as const,
    permissions: ['read' as const, 'write' as const],
    issuedAt: Date.now(),
    expiresAt: Date.now() + 600000,
  };

  bench('retrieve (cache hit)', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    await storage.store('bench:cached', { x: 1 }, {
      empirical: EmpiricalLevel.E0_Unverified,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    }, { schema });
    // Second retrieve should hit cache
    await storage.retrieve('bench:cached', cap);
    await storage.retrieve('bench:cached', cap);
    storage.dispose();
  });

  bench('retrieve (miss)', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    await storage.retrieve('bench:nonexistent', cap);
    storage.dispose();
  });
});

// ============================================================================
// Backend Direct Benchmarks
// ============================================================================

describe('Storage: MemoryBackend Direct', () => {
  const meta = {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    createdBy: 'bench',
    classification: { empirical: 0, normative: 0, materiality: 0 },
    schemaId: { id: 'bench', version: '1.0.0' },
    tags: [],
  };

  bench('set + get', async () => {
    const backend = createMemoryBackend();
    await backend.set('k', { v: 1 }, meta);
    await backend.get('k');
  });

  bench('set 100 items', async () => {
    const backend = createMemoryBackend();
    for (let i = 0; i < 100; i++) {
      await backend.set(`k:${i}`, { v: i }, meta);
    }
  });

  bench('has (existing)', async () => {
    const backend = createMemoryBackend();
    await backend.set('k', { v: 1 }, meta);
    await backend.has('k');
  });

  bench('keys (100 items)', async () => {
    const backend = createMemoryBackend();
    for (let i = 0; i < 100; i++) {
      await backend.set(`k:${i}`, { v: i }, meta);
    }
    await backend.keys();
  });

  bench('keys with pattern (100 items)', async () => {
    const backend = createMemoryBackend();
    for (let i = 0; i < 100; i++) {
      await backend.set(`${i % 2 === 0 ? 'even' : 'odd'}:${i}`, { v: i }, meta);
    }
    await backend.keys('even:*');
  });

  bench('delete', async () => {
    const backend = createMemoryBackend();
    await backend.set('k', { v: 1 }, meta);
    await backend.delete('k');
  });
});

// ============================================================================
// Migration Benchmarks
// ============================================================================

describe('Storage: Migration', () => {
  const meta = {
    createdAt: Date.now(),
    updatedAt: Date.now(),
    createdBy: 'bench',
    classification: { empirical: 0, normative: 0, materiality: 0 },
    schemaId: { id: 'bench', version: '1.0.0' },
    tags: [],
  };

  bench('migrate 10 items', async () => {
    const source = createMemoryBackend();
    const target = createMemoryBackend();
    for (let i = 0; i < 10; i++) {
      await source.set(`m:${i}`, { v: i }, meta);
    }
    await migrate(source, target);
  });

  bench('migrate 100 items', async () => {
    const source = createMemoryBackend();
    const target = createMemoryBackend();
    for (let i = 0; i < 100; i++) {
      await source.set(`m:${i}`, { v: i }, meta);
    }
    await migrate(source, target);
  });

  bench('export 50 items', async () => {
    const source = createMemoryBackend();
    for (let i = 0; i < 50; i++) {
      await source.set(`e:${i}`, { v: i }, meta);
    }
    await exportBackend(source, { agentId: 'bench' });
  });

  bench('export + import roundtrip (50 items)', async () => {
    const source = createMemoryBackend();
    const target = createMemoryBackend();
    for (let i = 0; i < 50; i++) {
      await source.set(`r:${i}`, { v: i }, meta);
    }
    const bundle = await exportBackend(source, { agentId: 'bench' });
    await importBundle(target, bundle);
  });
});

// ============================================================================
// CRDT Benchmarks
// ============================================================================

describe('Storage: CRDT Operations', () => {
  bench('LWWRegister set + get', () => {
    const reg = new LWWRegister<string>('node-1');
    reg.set('hello', 'node-1');
    reg.get();
  });

  bench('LWWRegister merge (2 nodes)', () => {
    const a = new LWWRegister<string>('node-a');
    const b = new LWWRegister<string>('node-b');
    a.set('from-a', 'node-a');
    b.set('from-b', 'node-b');
    a.merge(b.getWithMetadata());
  });

  bench('ORSet add + has', () => {
    const set = new ORSet<string>('node-1');
    set.add('item');
    set.has('item');
  });

  bench('ORSet add 50 items', () => {
    const set = new ORSet<string>('node-1');
    for (let i = 0; i < 50; i++) {
      set.add(`item-${i}`);
    }
  });

  bench('ORSet merge (2 nodes, 10 items each)', () => {
    const a = new ORSet<string>('node-a');
    const b = new ORSet<string>('node-b');
    for (let i = 0; i < 10; i++) {
      a.add(`a-${i}`);
      b.add(`b-${i}`);
    }
    a.merge(b.getElements());
  });

  bench('GCounter increment + value', () => {
    const counter = new GCounter('node-1');
    counter.increment(5);
    counter.value();
  });

  bench('PNCounter increment + decrement + value', () => {
    const counter = new PNCounter('node-1');
    counter.increment(10);
    counter.decrement(3);
    counter.value();
  });

  bench('VectorClock operations', () => {
    let clock = createVectorClock();
    clock = incrementClock(clock, 'node-a');
    clock = incrementClock(clock, 'node-b');
    const other = incrementClock(createVectorClock(), 'node-c');
    mergeClocks(clock, other);
  });
});

// ============================================================================
// Batch Operations Benchmarks
// ============================================================================

describe('Storage: Batch Operations', () => {
  const schema = { id: 'bench', version: '1.0.0' };

  bench('batchStore 10 items', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    const items = Array.from({ length: 10 }, (_, i) => ({
      key: `batch:${i}`,
      data: { i },
      classification: {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N0_Personal,
        materiality: MaterialityLevel.M0_Ephemeral,
      },
      options: { schema },
    }));
    await batchStore(
      items,
      (key, data, classification, options) => storage.store(key, data, classification, options),
    );
    storage.dispose();
  });

  bench('batchRetrieve 10 items', async () => {
    const storage = createEpistemicStorage({ agentId: 'bench' });
    for (let i = 0; i < 10; i++) {
      await storage.store(`batch:${i}`, { i }, {
        empirical: EmpiricalLevel.E0_Unverified,
        normative: NormativeLevel.N3_Universal,
        materiality: MaterialityLevel.M0_Ephemeral,
      }, { schema });
    }
    const keys = Array.from({ length: 10 }, (_, i) => `batch:${i}`);
    await batchRetrieve(
      keys,
      (key) => storage.retrieve(key),
    );
    storage.dispose();
  });
});

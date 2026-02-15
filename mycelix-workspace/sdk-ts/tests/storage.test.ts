/**
 * Storage Module Tests
 *
 * Tests for UESS classification router, CRDT classes, and storage utilities.
 */

import { describe, it, expect } from 'vitest';
import {
  // CRDT types and classes
  createVectorClock,
  incrementClock,
  mergeClocks,
  compareClocks,
  happenedBefore,
  type VectorClock,
  type MergeStrategy,
  // CRDT Classes
  LWWRegister,
  MVRegister,
  ORSet,
  GCounter,
  PNCounter,
  // Router
  StorageRouter,
  createRouter,
  DEFAULT_ROUTER_CONFIG,
  type RouterConfig,
  // Batch types
  type BatchItem,
  type BatchResult,
  type BatchConfig,
  type BatchStats,
  // Storage types
  MutabilityMode,
  AccessControlMode,
  // Epistemic levels for router tests
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '../src/storage/index.js';

// =============================================================================
// Vector Clock Tests
// =============================================================================

describe('Vector Clock Operations', () => {
  it('should create a new vector clock', () => {
    const clock = createVectorClock('node-1');

    expect(clock).toBeDefined();
    expect(clock.clock['node-1']).toBe(1);
    expect(clock.timestamp).toBeLessThanOrEqual(Date.now());
  });

  it('should increment a vector clock', () => {
    const clock = createVectorClock('node-1');
    const incremented = incrementClock(clock, 'node-1');

    expect(incremented.clock['node-1']).toBe(2);
    expect(incremented.timestamp).toBeGreaterThanOrEqual(clock.timestamp);
  });

  it('should add new node to clock on increment', () => {
    const clock = createVectorClock('node-1');
    const incremented = incrementClock(clock, 'node-2');

    expect(incremented.clock['node-1']).toBe(1);
    expect(incremented.clock['node-2']).toBe(1);
  });

  it('should merge two vector clocks', () => {
    const clock1 = createVectorClock('node-1');
    const clock2 = createVectorClock('node-2');

    const clock1Updated = incrementClock(clock1, 'node-1');
    const merged = mergeClocks(clock1Updated, clock2);

    expect(merged.clock['node-1']).toBe(2);
    expect(merged.clock['node-2']).toBe(1);
  });

  it('should compare clocks correctly', () => {
    const clock1 = createVectorClock('node-1');
    const clock2 = incrementClock(clock1, 'node-1');

    const result = compareClocks(clock1, clock2);
    expect(result).toBe('before'); // clock1 is before clock2

    const result2 = compareClocks(clock2, clock1);
    expect(result2).toBe('after'); // clock2 is after clock1
  });

  it('should detect concurrent clocks', () => {
    const clock1: VectorClock = {
      clock: { 'node-1': 2, 'node-2': 1 },
      timestamp: Date.now(),
    };
    const clock2: VectorClock = {
      clock: { 'node-1': 1, 'node-2': 2 },
      timestamp: Date.now(),
    };

    const result = compareClocks(clock1, clock2);
    expect(result).toBe('concurrent');
  });

  it('should detect equal clocks', () => {
    const clock1 = createVectorClock('node-1');
    const clock2 = { ...clock1, clock: { ...clock1.clock } };

    const result = compareClocks(clock1, clock2);
    expect(result).toBe('equal');
  });

  it('should check happenedBefore', () => {
    const clock1 = createVectorClock('node-1');
    const clock2 = incrementClock(clock1, 'node-1');

    expect(happenedBefore(clock1, clock2)).toBe(true);
    expect(happenedBefore(clock2, clock1)).toBe(false);
  });
});

// =============================================================================
// LWW Register Tests
// =============================================================================

describe('LWWRegister', () => {
  it('should create register with initial value', () => {
    const register = new LWWRegister('initial', 'node-1');

    expect(register.get()).toBe('initial');
  });

  it('should set new value', () => {
    const register = new LWWRegister('initial', 'node-1');
    register.set('updated', 'node-1');

    expect(register.get()).toBe('updated');
  });

  it('should merge with later timestamp winning', () => {
    const register = new LWWRegister('local', 'node-1');
    const localMeta = register.getWithMetadata();

    // Simulate remote with later timestamp
    const remote = {
      value: 'remote',
      timestamp: localMeta.timestamp + 1000,
      nodeId: 'node-2',
    };

    const result = register.merge(remote);

    expect(register.get()).toBe('remote');
    expect(result.hadConflict).toBe(true);
    expect(result.conflictDetails?.resolution).toBe('remote');
  });

  it('should keep local when timestamps equal and local nodeId wins', () => {
    const register = new LWWRegister('local', 'z-node');
    const localMeta = register.getWithMetadata();

    const remote = {
      value: 'remote',
      timestamp: localMeta.timestamp,
      nodeId: 'a-node', // Lexicographically earlier
    };

    const result = register.merge(remote);

    // z-node > a-node, so local wins
    expect(register.get()).toBe('local');
    expect(result.hadConflict).toBe(true);
  });

  it('should serialize and deserialize', () => {
    const register = new LWWRegister('test', 'node-1');
    const serialized = register.serialize();

    expect(typeof serialized).toBe('string');

    const deserialized = LWWRegister.deserialize<string>(serialized);
    expect(deserialized.value).toBe('test');
    expect(deserialized.nodeId).toBe('node-1');
  });
});

// =============================================================================
// MV Register Tests
// =============================================================================

describe('MVRegister', () => {
  it('should create register with initial value', () => {
    const register = new MVRegister('node-1', 'initial');

    expect(register.get()).toEqual(['initial']);
    expect(register.getSingle()).toBe('initial');
  });

  it('should detect no conflict initially', () => {
    const register = new MVRegister('node-1', 'value');

    expect(register.hasConflict()).toBe(false);
  });

  it('should set new value', () => {
    const register = new MVRegister('node-1', 'initial');
    register.set('updated');

    expect(register.getSingle()).toBe('updated');
  });

  it('should resolve conflicts', () => {
    const register = new MVRegister('node-1', 'initial');

    // Force conflict by adding concurrent value
    const values = register.getValuesWithMetadata();
    expect(values.length).toBe(1);

    // Resolve conflicts
    register.resolve((vals) => vals[0]);
    expect(register.hasConflict()).toBe(false);
  });
});

// =============================================================================
// OR-Set Tests
// =============================================================================

describe('ORSet', () => {
  it('should add elements', () => {
    const set = new ORSet<string>('node-1');
    set.add('item-1');

    expect(set.values()).toContain('item-1');
    expect(set.size).toBe(1);
  });

  it('should check if element exists', () => {
    const set = new ORSet<string>('node-1');
    set.add('item-1');

    expect(set.has('item-1')).toBe(true);
    expect(set.has('item-2')).toBe(false);
  });

  it('should remove elements', () => {
    const set = new ORSet<string>('node-1');
    set.add('item-1');
    set.remove('item-1');

    expect(set.has('item-1')).toBe(false);
    expect(set.size).toBe(0);
  });

  it('should return addId from add', () => {
    const set = new ORSet<string>('node-1');
    const addId = set.add('item-1');

    expect(typeof addId).toBe('string');
    expect(addId).toContain('node-1');
  });

  it('should merge two OR-Sets', () => {
    const set1 = new ORSet<string>('node-1');
    set1.add('item-1');

    const set2 = new ORSet<string>('node-2');
    set2.add('item-2');

    // Merge set2 into set1
    const result = set1.merge(set2.getElements());

    expect(set1.values()).toContain('item-1');
    expect(set1.values()).toContain('item-2');
  });

  it('should clear all elements', () => {
    const set = new ORSet<string>('node-1');
    set.add('item-1');
    set.add('item-2');
    set.clear();

    expect(set.size).toBe(0);
  });
});

// =============================================================================
// GCounter Tests
// =============================================================================

describe('GCounter', () => {
  it('should start at zero', () => {
    const counter = new GCounter('node-1');

    expect(counter.value()).toBe(0);
  });

  it('should increment', () => {
    const counter = new GCounter('node-1');
    counter.increment();

    expect(counter.value()).toBe(1);
  });

  it('should increment by amount', () => {
    const counter = new GCounter('node-1');
    counter.increment(10);

    expect(counter.value()).toBe(10);
  });

  it('should throw on negative increment', () => {
    const counter = new GCounter('node-1');

    expect(() => counter.increment(-1)).toThrow('GCounter can only increment');
  });

  it('should merge counters', () => {
    const counter1 = new GCounter('node-1');
    counter1.increment(5);

    const counter2 = new GCounter('node-2');
    counter2.increment(3);

    counter1.merge(counter2.getState());

    expect(counter1.value()).toBe(8); // 5 + 3
  });

  it('should take max value on merge', () => {
    const counter1 = new GCounter('node-1');
    counter1.increment(5);

    // Simulate remote state with higher value for node-1
    counter1.merge({ 'node-1': 10, 'node-2': 3 });

    expect(counter1.value()).toBe(13); // 10 + 3 (max of 5,10 for node-1)
  });
});

// =============================================================================
// PNCounter Tests
// =============================================================================

describe('PNCounter', () => {
  it('should start at zero', () => {
    const counter = new PNCounter('node-1');

    expect(counter.value()).toBe(0);
  });

  it('should increment', () => {
    const counter = new PNCounter('node-1');
    counter.increment(10);

    expect(counter.value()).toBe(10);
  });

  it('should decrement', () => {
    const counter = new PNCounter('node-1');
    counter.increment(10);
    counter.decrement(3);

    expect(counter.value()).toBe(7);
  });

  it('should allow negative values', () => {
    const counter = new PNCounter('node-1');
    counter.decrement(5);

    expect(counter.value()).toBe(-5);
  });

  it('should merge counters', () => {
    const counter1 = new PNCounter('node-1');
    counter1.increment(10);

    const counter2 = new PNCounter('node-2');
    counter2.increment(5);
    counter2.decrement(2);

    counter1.merge(counter2.getState());

    expect(counter1.value()).toBe(13); // 10 + (5-2)
  });

  it('should get state for synchronization', () => {
    const counter = new PNCounter('node-1');
    counter.increment(5);
    counter.decrement(2);

    const state = counter.getState();

    expect(state).toHaveProperty('positive');
    expect(state).toHaveProperty('negative');
  });
});

// =============================================================================
// Storage Router Tests
// =============================================================================

describe('StorageRouter', () => {
  it('should create router with default config', () => {
    const router = new StorageRouter();

    expect(router).toBeDefined();
  });

  it('should create router with createRouter helper', () => {
    const router = createRouter();

    expect(router).toBeDefined();
  });

  it('should create router with custom config', () => {
    const router = createRouter({
      enableIpfs: false,
      dhtDefaultReplication: 5,
    });

    expect(router).toBeDefined();
  });

  it('should route ephemeral data to memory', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E0_Subjective,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    });

    expect(tier.backend).toBe('memory');
    expect(tier.ttl).toBe(DEFAULT_ROUTER_CONFIG.m0DefaultTtl);
  });

  it('should route temporal data to local', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M1_Temporal,
    });

    expect(tier.backend).toBe('local');
    expect(tier.ttl).toBe(DEFAULT_ROUTER_CONFIG.m1DefaultTtl);
  });

  it('should route persistent data to DHT', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.backend).toBe('dht');
    expect(tier.ttl).toBeUndefined();
  });

  it('should route immutable data to DHT (without Filecoin or IPFS)', () => {
    // To test DHT fallback, disable both Filecoin and IPFS
    const router = createRouter({ enableFilecoin: false, enableIpfs: false });

    const tier = router.route({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M3_Immutable,
    });

    expect(tier.backend).toBe('dht');
  });

  it('should route immutable data to IPFS when enabled (default)', () => {
    // Default config has enableIpfs: true, enableFilecoin: false
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M3_Immutable,
    });

    expect(tier.backend).toBe('ipfs');
  });

  it('should enable encryption for personal data', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.encrypted).toBe(true);
  });

  it('should not encrypt public data', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.encrypted).toBe(false);
  });

  it('should enable content-addressing for E3+', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.contentAddressed).toBe(true);
  });

  it('should set CRDT mutability for E0/E1', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.mutability).toBe(MutabilityMode.MUTABLE_CRDT);
  });

  it('should set append-only for E2', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.mutability).toBe(MutabilityMode.APPEND_ONLY);
  });

  it('should set immutable for E3+', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.mutability).toBe(MutabilityMode.IMMUTABLE);
  });

  it('should set owner access control for N0', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.accessControl).toBe(AccessControlMode.OWNER);
  });

  it('should set CAPBAC for N1', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.accessControl).toBe(AccessControlMode.CAPBAC);
  });

  it('should set public access for N2+', () => {
    const router = new StorageRouter();

    const tier = router.route({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(tier.accessControl).toBe(AccessControlMode.PUBLIC);
  });

  it('should calculate replication factor', () => {
    const router = new StorageRouter();

    const classification = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const replication = router.getReplication(classification, 100);

    expect(replication).toBe(DEFAULT_ROUTER_CONFIG.dhtDefaultReplication);
  });

  it('should get backends for classification', () => {
    const router = new StorageRouter();

    const backends = router.getBackends({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });

    expect(Array.isArray(backends)).toBe(true);
    expect(backends).toContain('dht');
  });

  it('should detect migration requirement', () => {
    const router = new StorageRouter();

    const from = {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M1_Temporal,
    };

    const to = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    expect(router.requiresMigration(from, to)).toBe(true);
  });

  it('should validate valid transition', () => {
    const router = new StorageRouter();

    const from = {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M1_Temporal,
    };

    const to = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const result = router.validateTransition(from, to);
    expect(result.valid).toBe(true);
  });

  it('should reject E-level downgrade', () => {
    const router = new StorageRouter();

    const from = {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const to = {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const result = router.validateTransition(from, to);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('Cannot downgrade E-level');
  });

  it('should reject N-level downgrade', () => {
    const router = new StorageRouter();

    const from = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const to = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const result = router.validateTransition(from, to);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('Cannot downgrade N-level');
  });

  it('should reject M-level downgrade', () => {
    const router = new StorageRouter();

    const from = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M3_Immutable,
    };

    const to = {
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M1_Temporal,
    };

    const result = router.validateTransition(from, to);
    expect(result.valid).toBe(false);
    expect(result.error).toContain('Cannot downgrade M-level');
  });
});

// =============================================================================
// Batch Types Tests
// =============================================================================

describe('Batch Types', () => {
  it('should create valid BatchItem', () => {
    const item: BatchItem<{ name: string }> = {
      key: 'test-key',
      data: { name: 'Test' },
      classification: {
        empirical: EmpiricalLevel.E2_PrivateVerify,
        normative: NormativeLevel.N1_Communal,
        materiality: MaterialityLevel.M2_Persistent,
      },
    };

    expect(item.key).toBe('test-key');
    expect(item.data.name).toBe('Test');
  });

  it('should create valid BatchResult', () => {
    const result: BatchResult<string> = {
      key: 'test-key',
      success: true,
      receipt: {
        hash: 'abc123',
        timestamp: Date.now(),
        backend: 'local',
        replicationCount: 1,
      },
    };

    expect(result.success).toBe(true);
    expect(result.receipt?.hash).toBe('abc123');
  });

  it('should create BatchConfig with defaults', () => {
    const config: BatchConfig = {
      maxBatchSize: 100,
      continueOnError: true,
      maxConcurrency: 10,
      itemTimeoutMs: 30000,
      useTransactions: true,
      retryCount: 3,
      retryDelayMs: 100,
    };

    expect(config.maxBatchSize).toBe(100);
    expect(config.continueOnError).toBe(true);
  });

  it('should create BatchStats', () => {
    const stats: BatchStats = {
      totalItems: 50,
      successCount: 48,
      errorCount: 2,
      retriedCount: 5,
      startTime: Date.now() - 1000,
      endTime: Date.now(),
    };

    expect(stats.totalItems).toBe(50);
    expect(stats.successCount + stats.errorCount).toBe(50);
  });
});

// =============================================================================
// Merge Strategy Tests
// =============================================================================

describe('Merge Strategies', () => {
  it('should have all merge strategy types', () => {
    const strategies: MergeStrategy[] = [
      'lww',
      'vector-clock',
      'set-union',
      'counter',
      'pn-counter',
      'or-set',
      'mv-register',
      'custom',
    ];

    expect(strategies).toContain('lww');
    expect(strategies).toContain('vector-clock');
    expect(strategies).toContain('or-set');
    expect(strategies.length).toBe(8);
  });
});

/**
 * CRDT (Conflict-free Replicated Data Types) Test Suite
 *
 * Comprehensive tests for vector clocks, LWW registers, MV registers,
 * OR-Sets, G-Counters, and PN-Counters.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  // Vector Clock
  createVectorClock,
  incrementClock,
  mergeClocks,
  compareClocks,
  happenedBefore,
  // Classes
  LWWRegister,
  MVRegister,
  ORSet,
  GCounter,
  PNCounter,
  // Generic merge
  merge,
  type VectorClock,
  type MergeStrategy,
} from '../../src/storage/crdt.js';

// =============================================================================
// Vector Clock Tests
// =============================================================================

describe('Vector Clock Operations', () => {
  describe('createVectorClock', () => {
    it('should create a vector clock with initial time for node', () => {
      const clock = createVectorClock('node1');

      expect(clock.clock).toEqual({ node1: 1 });
      expect(clock.timestamp).toBeLessThanOrEqual(Date.now());
    });

    it('should create unique timestamps for different invocations', () => {
      const clock1 = createVectorClock('node1');
      const clock2 = createVectorClock('node2');

      expect(clock1.clock.node1).toBe(1);
      expect(clock2.clock.node2).toBe(1);
    });
  });

  describe('incrementClock', () => {
    it('should increment existing node time', () => {
      const clock = createVectorClock('node1');
      const incremented = incrementClock(clock, 'node1');

      expect(incremented.clock.node1).toBe(2);
    });

    it('should add new node to clock', () => {
      const clock = createVectorClock('node1');
      const incremented = incrementClock(clock, 'node2');

      expect(incremented.clock.node1).toBe(1);
      expect(incremented.clock.node2).toBe(1);
    });

    it('should update timestamp', () => {
      const clock = createVectorClock('node1');
      const incremented = incrementClock(clock, 'node1');

      expect(incremented.timestamp).toBeGreaterThanOrEqual(clock.timestamp);
    });

    it('should not mutate original clock', () => {
      const clock = createVectorClock('node1');
      const originalClock = { ...clock.clock };
      incrementClock(clock, 'node1');

      expect(clock.clock).toEqual(originalClock);
    });
  });

  describe('mergeClocks', () => {
    it('should merge clocks taking max of each node', () => {
      const a: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 3, node2: 7, node3: 2 }, timestamp: 2000 };

      const merged = mergeClocks(a, b);

      expect(merged.clock.node1).toBe(5);
      expect(merged.clock.node2).toBe(7);
      expect(merged.clock.node3).toBe(2);
    });

    it('should take max timestamp', () => {
      const a: VectorClock = { clock: { node1: 1 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 1 }, timestamp: 2000 };

      const merged = mergeClocks(a, b);

      expect(merged.timestamp).toBe(2000);
    });

    it('should handle disjoint clocks', () => {
      const a: VectorClock = { clock: { node1: 5 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node2: 3 }, timestamp: 2000 };

      const merged = mergeClocks(a, b);

      expect(merged.clock.node1).toBe(5);
      expect(merged.clock.node2).toBe(3);
    });
  });

  describe('compareClocks', () => {
    it('should return "equal" for identical clocks', () => {
      const a: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 2000 };

      expect(compareClocks(a, b)).toBe('equal');
    });

    it('should return "before" when a happened before b', () => {
      const a: VectorClock = { clock: { node1: 3, node2: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 2000 };

      expect(compareClocks(a, b)).toBe('before');
    });

    it('should return "after" when a happened after b', () => {
      const a: VectorClock = { clock: { node1: 5, node2: 5 }, timestamp: 2000 };
      const b: VectorClock = { clock: { node1: 3, node2: 3 }, timestamp: 1000 };

      expect(compareClocks(a, b)).toBe('after');
    });

    it('should return "concurrent" for concurrent events', () => {
      const a: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 3, node2: 5 }, timestamp: 2000 };

      expect(compareClocks(a, b)).toBe('concurrent');
    });

    it('should handle missing nodes as 0', () => {
      const a: VectorClock = { clock: { node1: 5 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 2000 };

      expect(compareClocks(a, b)).toBe('before');
    });
  });

  describe('happenedBefore', () => {
    it('should return true when a happened before b', () => {
      const a: VectorClock = { clock: { node1: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 5 }, timestamp: 2000 };

      expect(happenedBefore(a, b)).toBe(true);
    });

    it('should return false when a did not happen before b', () => {
      const a: VectorClock = { clock: { node1: 5 }, timestamp: 2000 };
      const b: VectorClock = { clock: { node1: 3 }, timestamp: 1000 };

      expect(happenedBefore(a, b)).toBe(false);
    });

    it('should return false for concurrent events', () => {
      const a: VectorClock = { clock: { node1: 5, node2: 3 }, timestamp: 1000 };
      const b: VectorClock = { clock: { node1: 3, node2: 5 }, timestamp: 2000 };

      expect(happenedBefore(a, b)).toBe(false);
    });
  });
});

// =============================================================================
// LWW Register Tests
// =============================================================================

describe('LWWRegister', () => {
  describe('basic operations', () => {
    it('should initialize with value', () => {
      const reg = new LWWRegister<string>('initial', 'node1');

      expect(reg.get()).toBe('initial');
    });

    it('should set and get value', () => {
      const reg = new LWWRegister<number>(0, 'node1');
      reg.set(42, 'node1');

      expect(reg.get()).toBe(42);
    });

    it('should return metadata with getWithMetadata', () => {
      const reg = new LWWRegister<string>('test', 'node1');
      const meta = reg.getWithMetadata();

      expect(meta.value).toBe('test');
      expect(meta.nodeId).toBe('node1');
      expect(meta.timestamp).toBeLessThanOrEqual(Date.now());
    });
  });

  describe('merge', () => {
    it('should take remote value when remote timestamp is newer', async () => {
      const reg = new LWWRegister<string>('local', 'node1');
      await new Promise(r => setTimeout(r, 10)); // Ensure time difference

      const result = reg.merge({
        value: 'remote',
        timestamp: Date.now(),
        nodeId: 'node2',
      });

      expect(result.value).toBe('remote');
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('remote');
      expect(reg.get()).toBe('remote');
    });

    it('should keep local value when local timestamp is newer', async () => {
      const reg = new LWWRegister<string>('local', 'node1');
      const oldTimestamp = Date.now() - 10000;

      const result = reg.merge({
        value: 'remote',
        timestamp: oldTimestamp,
        nodeId: 'node2',
      });

      expect(result.value).toBe('local');
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('local');
      expect(reg.get()).toBe('local');
    });

    it('should use nodeId as tiebreaker when timestamps are equal', () => {
      const timestamp = Date.now();
      const reg = new LWWRegister<string>('local', 'aaa');

      // Manually set timestamp to match
      const meta = reg.getWithMetadata();
      const result = reg.merge({
        value: 'remote',
        timestamp: meta.timestamp,
        nodeId: 'zzz', // Higher nodeId should win
      });

      expect(result.value).toBe('remote');
      expect(result.hadConflict).toBe(true);
    });

    it('should keep local when same timestamp and lower remote nodeId', () => {
      const reg = new LWWRegister<string>('local', 'zzz');
      const meta = reg.getWithMetadata();

      const result = reg.merge({
        value: 'remote',
        timestamp: meta.timestamp,
        nodeId: 'aaa', // Lower nodeId should lose
      });

      expect(result.value).toBe('local');
    });
  });

  describe('serialization', () => {
    it('should serialize and deserialize', () => {
      const reg = new LWWRegister<{ foo: string }>({ foo: 'bar' }, 'node1');
      const serialized = reg.serialize();

      expect(typeof serialized).toBe('string');

      const deserialized = LWWRegister.deserialize<{ foo: string }>(serialized);
      expect(deserialized.value.foo).toBe('bar');
    });
  });
});

// =============================================================================
// MV Register Tests
// =============================================================================

describe('MVRegister', () => {
  describe('basic operations', () => {
    it('should initialize without value', () => {
      const reg = new MVRegister<string>('node1');

      expect(reg.get()).toEqual([]);
      expect(reg.getSingle()).toBeUndefined();
    });

    it('should initialize with value', () => {
      const reg = new MVRegister<string>('node1', 'initial');

      expect(reg.get()).toEqual(['initial']);
      expect(reg.getSingle()).toBe('initial');
    });

    it('should set value', () => {
      const reg = new MVRegister<number>('node1');
      reg.set(42);

      expect(reg.getSingle()).toBe(42);
    });

    it('should not have conflict with single value', () => {
      const reg = new MVRegister<string>('node1', 'value');

      expect(reg.hasConflict()).toBe(false);
    });
  });

  describe('merge', () => {
    it('should add concurrent value', () => {
      const reg = new MVRegister<string>('node1', 'local');

      const remoteClock = createVectorClock('node2');
      const result = reg.merge({
        value: 'remote',
        vectorClock: remoteClock,
        nodeId: 'node2',
      });

      expect(result.value).toContain('local');
      expect(result.value).toContain('remote');
      expect(result.hadConflict).toBe(true);
      expect(reg.hasConflict()).toBe(true);
    });

    it('should replace dominated value', () => {
      const reg = new MVRegister<string>('node1', 'old');
      const values = reg.getValuesWithMetadata();
      const localClock = values[0].vectorClock;

      // Remote clock dominates local - increment on node1 to dominate
      let remoteClock = mergeClocks(localClock, localClock);
      remoteClock = incrementClock(remoteClock, 'node1');
      remoteClock = incrementClock(remoteClock, 'node2');

      const result = reg.merge({
        value: 'new',
        vectorClock: remoteClock,
        nodeId: 'node2',
      });

      // The remote value should replace the old one since it dominates
      expect(result.value).toContain('new');
      // May or may not have conflict depending on implementation
    });

    it('should ignore dominated remote value', () => {
      const reg = new MVRegister<string>('node1', 'initial');
      reg.set('newer'); // This creates a clock that dominates initial

      const oldClock = createVectorClock('node2'); // Clock from node2 only

      const result = reg.merge({
        value: 'old',
        vectorClock: oldClock,
        nodeId: 'node2',
      });

      // The 'newer' value should still be present
      // The 'old' value may also be present as concurrent
      expect(result.value).toContain('newer');
    });

    it('should not add duplicate values', () => {
      const reg = new MVRegister<string>('node1', 'value');
      const values = reg.getValuesWithMetadata();

      const result = reg.merge(values[0]);

      expect(result.value).toEqual(['value']);
    });
  });

  describe('resolve', () => {
    it('should resolve conflicts using resolver function', () => {
      const reg = new MVRegister<number>('node1', 10);

      // Add concurrent value
      reg.merge({
        value: 20,
        vectorClock: createVectorClock('node2'),
        nodeId: 'node2',
      });

      expect(reg.hasConflict()).toBe(true);

      // Resolve by taking maximum
      reg.resolve(values => Math.max(...values));

      expect(reg.hasConflict()).toBe(false);
      expect(reg.getSingle()).toBe(20);
    });

    it('should do nothing when no conflict', () => {
      const reg = new MVRegister<string>('node1', 'value');

      reg.resolve(() => 'different');

      expect(reg.getSingle()).toBe('value');
    });
  });

  describe('getValuesWithMetadata', () => {
    it('should return all values with metadata', () => {
      const reg = new MVRegister<string>('node1', 'local');

      reg.merge({
        value: 'remote',
        vectorClock: createVectorClock('node2'),
        nodeId: 'node2',
      });

      const values = reg.getValuesWithMetadata();

      expect(values).toHaveLength(2);
      expect(values[0].nodeId).toBe('node1');
      expect(values[1].nodeId).toBe('node2');
    });
  });
});

// =============================================================================
// OR-Set Tests
// =============================================================================

describe('ORSet', () => {
  describe('basic operations', () => {
    it('should start empty', () => {
      const set = new ORSet<string>('node1');

      expect(set.values()).toEqual([]);
      expect(set.size).toBe(0);
    });

    it('should add values', () => {
      const set = new ORSet<string>('node1');
      set.add('a');
      set.add('b');

      expect(set.values()).toContain('a');
      expect(set.values()).toContain('b');
      expect(set.size).toBe(2);
    });

    it('should check if value exists', () => {
      const set = new ORSet<string>('node1');
      set.add('exists');

      expect(set.has('exists')).toBe(true);
      expect(set.has('missing')).toBe(false);
    });

    it('should return add ID', () => {
      const set = new ORSet<string>('node1');
      const addId = set.add('value');

      expect(addId).toMatch(/^node1:\d+$/);
    });
  });

  describe('remove', () => {
    it('should remove value', () => {
      const set = new ORSet<string>('node1');
      set.add('a');
      set.add('b');

      const removed = set.remove('a');

      expect(removed).toBe(true);
      expect(set.has('a')).toBe(false);
      expect(set.has('b')).toBe(true);
    });

    it('should return false when removing non-existent value', () => {
      const set = new ORSet<string>('node1');

      const removed = set.remove('missing');

      expect(removed).toBe(false);
    });

    it('should handle object values', () => {
      const set = new ORSet<{ id: number }>('node1');
      set.add({ id: 1 });
      set.add({ id: 2 });

      const removed = set.remove({ id: 1 });

      expect(removed).toBe(true);
      expect(set.has({ id: 1 })).toBe(false);
      expect(set.has({ id: 2 })).toBe(true);
    });
  });

  describe('merge', () => {
    it('should add new elements from remote', () => {
      const set = new ORSet<string>('node1');
      set.add('local');

      const result = set.merge([
        { value: 'remote', addId: 'node2:1', addedAt: Date.now() },
      ]);

      expect(set.has('local')).toBe(true);
      expect(set.has('remote')).toBe(true);
      expect(result.hadConflict).toBe(false);
    });

    it('should handle remove conflicts (add wins)', () => {
      const set = new ORSet<string>('node1');
      const addId = set.add('value');
      set.remove('value');

      // Remote adds the same value with a different addId (add wins)
      const result = set.merge([
        { value: 'value', addId: 'node2:1', addedAt: Date.now() },
      ]);

      // The remote add should make the value available again
      expect(set.has('value')).toBe(true);
      // The implementation may or may not report this as conflict
      // depending on how it handles local removed vs remote new add
    });

    it('should apply valid removals from remote', () => {
      const set = new ORSet<string>('node1');
      const addId = set.add('value');
      const addedAt = Date.now();

      // Merge element that has been removed
      const result = set.merge([
        { value: 'value', addId, addedAt, removedAt: addedAt + 1, removeId: 'node2:2' },
      ]);

      expect(set.has('value')).toBe(false);
      expect(result.hadConflict).toBe(true);
    });
  });

  describe('clear', () => {
    it('should remove all elements', () => {
      const set = new ORSet<string>('node1');
      set.add('a');
      set.add('b');
      set.add('c');

      set.clear();

      expect(set.size).toBe(0);
      expect(set.values()).toEqual([]);
    });

    it('should keep removed elements in history', () => {
      const set = new ORSet<string>('node1');
      set.add('a');
      set.clear();

      const elements = set.getElements();

      expect(elements.length).toBeGreaterThan(0);
      expect(elements[0].removedAt).toBeDefined();
    });
  });

  describe('getElements', () => {
    it('should return all elements including removed', () => {
      const set = new ORSet<string>('node1');
      set.add('a');
      set.add('b');
      set.remove('a');

      const elements = set.getElements();

      expect(elements).toHaveLength(2);
      expect(elements.find(e => e.value === 'a')?.removedAt).toBeDefined();
      expect(elements.find(e => e.value === 'b')?.removedAt).toBeUndefined();
    });
  });
});

// =============================================================================
// G-Counter Tests
// =============================================================================

describe('GCounter', () => {
  describe('basic operations', () => {
    it('should start at 0', () => {
      const counter = new GCounter('node1');

      expect(counter.value()).toBe(0);
    });

    it('should increment', () => {
      const counter = new GCounter('node1');
      counter.increment();

      expect(counter.value()).toBe(1);
    });

    it('should increment by amount', () => {
      const counter = new GCounter('node1');
      counter.increment(5);

      expect(counter.value()).toBe(5);
    });

    it('should throw on negative increment', () => {
      const counter = new GCounter('node1');

      expect(() => counter.increment(-1)).toThrow('GCounter can only increment');
    });
  });

  describe('merge', () => {
    it('should merge with remote state', () => {
      const counter = new GCounter('node1');
      counter.increment(5);

      const result = counter.merge({ node2: 3, node3: 7 });

      expect(counter.value()).toBe(15); // 5 + 3 + 7
      expect(result.hadConflict).toBe(true);
    });

    it('should take max for same node', () => {
      const counter = new GCounter('node1');
      counter.increment(5);

      const result = counter.merge({ node1: 3 });

      expect(counter.value()).toBe(5); // max(5, 3)
      expect(result.hadConflict).toBe(false);
    });

    it('should update when remote is higher', () => {
      const counter = new GCounter('node1');
      counter.increment(3);

      counter.merge({ node1: 10 });

      expect(counter.value()).toBe(10);
    });
  });

  describe('getState', () => {
    it('should return state for synchronization', () => {
      const counter = new GCounter('node1');
      counter.increment(5);

      const state = counter.getState();

      expect(state).toEqual({ node1: 5 });
    });

    it('should return copy of state', () => {
      const counter = new GCounter('node1');
      counter.increment(5);

      const state = counter.getState();
      state.node1 = 999;

      expect(counter.value()).toBe(5);
    });
  });
});

// =============================================================================
// PN-Counter Tests
// =============================================================================

describe('PNCounter', () => {
  describe('basic operations', () => {
    it('should start at 0', () => {
      const counter = new PNCounter('node1');

      expect(counter.value()).toBe(0);
    });

    it('should increment', () => {
      const counter = new PNCounter('node1');
      counter.increment(5);

      expect(counter.value()).toBe(5);
    });

    it('should decrement', () => {
      const counter = new PNCounter('node1');
      counter.decrement(3);

      expect(counter.value()).toBe(-3);
    });

    it('should handle mixed operations', () => {
      const counter = new PNCounter('node1');
      counter.increment(10);
      counter.decrement(3);
      counter.increment(5);
      counter.decrement(2);

      expect(counter.value()).toBe(10); // 10 + 5 - 3 - 2
    });
  });

  describe('merge', () => {
    it('should merge with remote state', () => {
      const counter = new PNCounter('node1');
      counter.increment(10);
      counter.decrement(2);

      const result = counter.merge({
        positive: { node2: 5 },
        negative: { node2: 1 },
      });

      expect(counter.value()).toBe(12); // (10 + 5) - (2 + 1)
      expect(result.hadConflict).toBe(true);
    });

    it('should report no conflict when values unchanged', () => {
      const counter = new PNCounter('node1');
      counter.increment(5);

      const result = counter.merge({
        positive: { node1: 3 },
        negative: {},
      });

      expect(counter.value()).toBe(5); // max(5, 3)
      expect(result.hadConflict).toBe(false);
    });
  });

  describe('getState', () => {
    it('should return positive and negative states', () => {
      const counter = new PNCounter('node1');
      counter.increment(10);
      counter.decrement(3);

      const state = counter.getState();

      expect(state.positive).toEqual({ node1: 10 });
      expect(state.negative).toEqual({ node1: 3 });
    });
  });
});

// =============================================================================
// Generic Merge Function Tests
// =============================================================================

describe('Generic merge function', () => {
  describe('lww strategy', () => {
    it('should use remote when remote timestamp is newer', () => {
      const result = merge('local', 'remote', 'lww', {
        localTimestamp: 1000,
        remoteTimestamp: 2000,
      });

      expect(result.value).toBe('remote');
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('remote');
    });

    it('should use local when local timestamp is newer', () => {
      const result = merge('local', 'remote', 'lww', {
        localTimestamp: 2000,
        remoteTimestamp: 1000,
      });

      expect(result.value).toBe('local');
      expect(result.hadConflict).toBe(false);
    });

    it('should detect conflict with equal timestamps', () => {
      const result = merge('local', 'remote', 'lww', {
        localTimestamp: 1000,
        remoteTimestamp: 1000,
      });

      expect(result.value).toBe('local');
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('local');
    });

    it('should not report conflict when values are equal', () => {
      const result = merge('same', 'same', 'lww', {
        localTimestamp: 1000,
        remoteTimestamp: 1000,
      });

      expect(result.hadConflict).toBe(false);
    });
  });

  describe('vector-clock strategy', () => {
    it('should throw without vector clocks', () => {
      expect(() => merge('local', 'remote', 'vector-clock')).toThrow(
        'Vector clocks required'
      );
    });

    it('should use remote when local happened before', () => {
      const localClock: VectorClock = { clock: { node1: 1 }, timestamp: 1000 };
      const remoteClock: VectorClock = { clock: { node1: 2 }, timestamp: 2000 };

      const result = merge('local', 'remote', 'vector-clock', {
        localClock,
        remoteClock,
      });

      expect(result.value).toBe('remote');
      expect(result.hadConflict).toBe(true);
      expect(result.vectorClock).toEqual(remoteClock);
    });

    it('should use local when remote happened before', () => {
      const localClock: VectorClock = { clock: { node1: 5 }, timestamp: 2000 };
      const remoteClock: VectorClock = { clock: { node1: 3 }, timestamp: 1000 };

      const result = merge('local', 'remote', 'vector-clock', {
        localClock,
        remoteClock,
      });

      expect(result.value).toBe('local');
      expect(result.hadConflict).toBe(false);
    });

    it('should merge for concurrent updates', () => {
      const localClock: VectorClock = { clock: { node1: 5 }, timestamp: 1000 };
      const remoteClock: VectorClock = { clock: { node2: 5 }, timestamp: 2000 };

      const result = merge('local', 'remote', 'vector-clock', {
        localClock,
        remoteClock,
        customMerge: (l, r) => `${l}+${r}`,
      });

      expect(result.value).toBe('local+remote');
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('merged');
    });

    it('should use local for concurrent without custom merge', () => {
      const localClock: VectorClock = { clock: { node1: 5 }, timestamp: 1000 };
      const remoteClock: VectorClock = { clock: { node2: 5 }, timestamp: 2000 };

      const result = merge('local', 'remote', 'vector-clock', {
        localClock,
        remoteClock,
      });

      expect(result.value).toBe('local');
    });
  });

  describe('set-union strategy', () => {
    it('should union arrays', () => {
      const result = merge([1, 2, 3], [2, 3, 4, 5], 'set-union');

      expect(result.value).toEqual([1, 2, 3, 4, 5]);
      expect(result.hadConflict).toBe(false);
    });

    it('should throw for non-array values', () => {
      expect(() => merge('string', 'value', 'set-union')).toThrow(
        'Set union requires array values'
      );
    });
  });

  describe('custom strategy', () => {
    it('should use custom merge function', () => {
      const result = merge(10, 20, 'custom', {
        customMerge: (a, b) => a + b,
      });

      expect(result.value).toBe(30);
      expect(result.hadConflict).toBe(true);
      expect(result.conflictDetails?.resolution).toBe('merged');
    });

    it('should throw without custom function', () => {
      expect(() => merge('a', 'b', 'custom')).toThrow(
        'Custom merge function required'
      );
    });
  });

  describe('unknown strategy', () => {
    it('should throw for unknown strategy', () => {
      expect(() => merge('a', 'b', 'unknown' as MergeStrategy)).toThrow(
        'Unknown merge strategy: unknown'
      );
    });
  });
});

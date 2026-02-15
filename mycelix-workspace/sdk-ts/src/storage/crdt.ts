/**
 * UESS CRDT Merge Strategies
 *
 * Conflict-free Replicated Data Types for E0/E1 mutable data.
 * @see docs/architecture/uess/UESS-06-MUTABILITY.md
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Available CRDT merge strategies
 */
export type MergeStrategy =
  | 'lww'           // Last-Writer-Wins (timestamp-based)
  | 'vector-clock'  // Vector clock causality
  | 'set-union'     // Set union (add-wins)
  | 'counter'       // Grow-only counter
  | 'pn-counter'    // Positive-negative counter
  | 'or-set'        // Observed-Remove Set
  | 'mv-register'   // Multi-Value Register
  | 'custom';       // Custom merge function

/**
 * Vector clock for causal ordering
 */
export interface VectorClock {
  /** Node ID → logical time mapping */
  clock: Record<string, number>;

  /** Last update timestamp */
  timestamp: number;
}

/**
 * LWW Register value
 */
export interface LWWValue<T> {
  value: T;
  timestamp: number;
  nodeId: string;
}

/**
 * OR-Set element
 */
export interface ORSetElement<T> {
  value: T;
  addId: string;     // Unique add identifier
  addedAt: number;
  removedAt?: number;
  removeId?: string;
}

/**
 * MV-Register value (multi-value)
 */
export interface MVValue<T> {
  value: T;
  vectorClock: VectorClock;
  nodeId: string;
}

/**
 * Merge result
 */
export interface MergeResult<T> {
  /** Merged value */
  value: T;

  /** Whether a conflict occurred */
  hadConflict: boolean;

  /** Conflict details if any */
  conflictDetails?: {
    localValue: T;
    remoteValue: T;
    resolution: 'local' | 'remote' | 'merged';
  };

  /** New vector clock (if applicable) */
  vectorClock?: VectorClock;
}

// =============================================================================
// Vector Clock Operations
// =============================================================================

/**
 * Create a new vector clock
 */
export function createVectorClock(nodeId: string): VectorClock {
  return {
    clock: { [nodeId]: 1 },
    timestamp: Date.now(),
  };
}

/**
 * Increment a vector clock
 */
export function incrementClock(clock: VectorClock, nodeId: string): VectorClock {
  const newClock = { ...clock.clock };
  newClock[nodeId] = (newClock[nodeId] ?? 0) + 1;

  return {
    clock: newClock,
    timestamp: Date.now(),
  };
}

/**
 * Merge two vector clocks
 */
export function mergeClocks(a: VectorClock, b: VectorClock): VectorClock {
  const merged: Record<string, number> = { ...a.clock };

  for (const [nodeId, time] of Object.entries(b.clock)) {
    merged[nodeId] = Math.max(merged[nodeId] ?? 0, time);
  }

  return {
    clock: merged,
    timestamp: Math.max(a.timestamp, b.timestamp),
  };
}

/**
 * Compare two vector clocks
 * Returns:
 *  - 'before' if a happened before b
 *  - 'after' if a happened after b
 *  - 'concurrent' if neither happened before the other
 *  - 'equal' if they are the same
 */
export function compareClocks(
  a: VectorClock,
  b: VectorClock
): 'before' | 'after' | 'concurrent' | 'equal' {
  let aBeforeB = true;
  let bBeforeA = true;
  let equal = true;

  const allNodes = new Set([...Object.keys(a.clock), ...Object.keys(b.clock)]);

  for (const node of allNodes) {
    const aTime = a.clock[node] ?? 0;
    const bTime = b.clock[node] ?? 0;

    if (aTime !== bTime) {
      equal = false;
    }
    if (aTime > bTime) {
      aBeforeB = false;
    }
    if (bTime > aTime) {
      bBeforeA = false;
    }
  }

  if (equal) return 'equal';
  if (aBeforeB && !bBeforeA) return 'before';
  if (bBeforeA && !aBeforeB) return 'after';
  return 'concurrent';
}

/**
 * Check if clock a happened before clock b
 */
export function happenedBefore(a: VectorClock, b: VectorClock): boolean {
  return compareClocks(a, b) === 'before';
}

// =============================================================================
// LWW (Last-Writer-Wins) Register
// =============================================================================

/**
 * LWW Register - Simple timestamp-based conflict resolution
 */
export class LWWRegister<T> {
  private value: LWWValue<T>;

  constructor(initialValue: T, nodeId: string) {
    this.value = {
      value: initialValue,
      timestamp: Date.now(),
      nodeId,
    };
  }

  /**
   * Get current value
   */
  get(): T {
    return this.value.value;
  }

  /**
   * Get value with metadata
   */
  getWithMetadata(): LWWValue<T> {
    return { ...this.value };
  }

  /**
   * Set value (local update)
   */
  set(newValue: T, nodeId: string): void {
    this.value = {
      value: newValue,
      timestamp: Date.now(),
      nodeId,
    };
  }

  /**
   * Merge with remote value
   */
  merge(remote: LWWValue<T>): MergeResult<T> {
    const local = this.value;

    // Last writer wins based on timestamp
    if (remote.timestamp > local.timestamp) {
      this.value = remote;
      return {
        value: remote.value,
        hadConflict: true,
        conflictDetails: {
          localValue: local.value,
          remoteValue: remote.value,
          resolution: 'remote',
        },
      };
    } else if (remote.timestamp < local.timestamp) {
      return {
        value: local.value,
        hadConflict: true,
        conflictDetails: {
          localValue: local.value,
          remoteValue: remote.value,
          resolution: 'local',
        },
      };
    } else {
      // Same timestamp - use node ID as tiebreaker
      if (remote.nodeId > local.nodeId) {
        this.value = remote;
        return {
          value: remote.value,
          hadConflict: true,
          conflictDetails: {
            localValue: local.value,
            remoteValue: remote.value,
            resolution: 'remote',
          },
        };
      }
      return {
        value: local.value,
        hadConflict: remote.value !== local.value,
      };
    }
  }

  /**
   * Serialize for network transmission
   */
  serialize(): string {
    return JSON.stringify(this.value);
  }

  /**
   * Deserialize from network
   */
  static deserialize<T>(json: string): LWWValue<T> {
    return JSON.parse(json);
  }
}

// =============================================================================
// MV (Multi-Value) Register
// =============================================================================

/**
 * Multi-Value Register - Preserves concurrent values
 */
export class MVRegister<T> {
  private values: MVValue<T>[] = [];
  private nodeId: string;

  constructor(nodeId: string, initialValue?: T) {
    this.nodeId = nodeId;
    if (initialValue !== undefined) {
      this.values = [{
        value: initialValue,
        vectorClock: createVectorClock(nodeId),
        nodeId,
      }];
    }
  }

  /**
   * Get current values (may be multiple if concurrent)
   */
  get(): T[] {
    return this.values.map(v => v.value);
  }

  /**
   * Get single value (arbitrarily picks first)
   */
  getSingle(): T | undefined {
    return this.values[0]?.value;
  }

  /**
   * Check if there are concurrent values
   */
  hasConflict(): boolean {
    return this.values.length > 1;
  }

  /**
   * Set value (local update, supersedes all current values)
   */
  set(newValue: T): void {
    // Create new vector clock that dominates all current
    let newClock = createVectorClock(this.nodeId);
    for (const v of this.values) {
      newClock = mergeClocks(newClock, v.vectorClock);
    }
    newClock = incrementClock(newClock, this.nodeId);

    this.values = [{
      value: newValue,
      vectorClock: newClock,
      nodeId: this.nodeId,
    }];
  }

  /**
   * Merge with remote value
   */
  merge(remote: MVValue<T>): MergeResult<T[]> {
    const dominated: number[] = [];
    let dominatesRemote = false;

    for (let i = 0; i < this.values.length; i++) {
      const comparison = compareClocks(this.values[i].vectorClock, remote.vectorClock);

      if (comparison === 'before') {
        // Local value is dominated by remote
        dominated.push(i);
      } else if (comparison === 'after') {
        // Local value dominates remote
        dominatesRemote = true;
      }
      // If concurrent, keep both
    }

    // Remove dominated values
    this.values = this.values.filter((_, i) => !dominated.includes(i));

    // Add remote if not dominated
    if (!dominatesRemote) {
      // Check for duplicates
      const isDuplicate = this.values.some(
        v => compareClocks(v.vectorClock, remote.vectorClock) === 'equal'
      );
      if (!isDuplicate) {
        this.values.push(remote);
      }
    }

    return {
      value: this.get(),
      hadConflict: this.values.length > 1,
    };
  }

  /**
   * Resolve conflicts by picking one value
   */
  resolve(resolver: (values: T[]) => T): void {
    if (this.values.length <= 1) return;

    const resolved = resolver(this.get());
    this.set(resolved);
  }

  /**
   * Get values with metadata
   */
  getValuesWithMetadata(): MVValue<T>[] {
    return [...this.values];
  }
}

// =============================================================================
// OR-Set (Observed-Remove Set)
// =============================================================================

/**
 * Observed-Remove Set - Add-wins semantics
 */
export class ORSet<T> {
  private elements: Map<string, ORSetElement<T>> = new Map();
  private nodeId: string;
  private counter = 0;

  constructor(nodeId: string) {
    this.nodeId = nodeId;
  }

  /**
   * Get all values in the set
   */
  values(): T[] {
    return Array.from(this.elements.values())
      .filter(e => !e.removedAt)
      .map(e => e.value);
  }

  /**
   * Check if value is in set
   */
  has(value: T): boolean {
    for (const element of this.elements.values()) {
      if (!element.removedAt && this.valuesEqual(element.value, value)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Get set size
   */
  get size(): number {
    return this.values().length;
  }

  /**
   * Add value to set
   */
  add(value: T): string {
    const addId = `${this.nodeId}:${++this.counter}`;

    this.elements.set(addId, {
      value,
      addId,
      addedAt: Date.now(),
    });

    return addId;
  }

  /**
   * Remove value from set
   */
  remove(value: T): boolean {
    let removed = false;
    const removeId = `${this.nodeId}:${++this.counter}`;

    for (const element of this.elements.values()) {
      if (!element.removedAt && this.valuesEqual(element.value, value)) {
        element.removedAt = Date.now();
        element.removeId = removeId;
        removed = true;
      }
    }

    return removed;
  }

  /**
   * Merge with remote OR-Set elements
   */
  merge(remoteElements: ORSetElement<T>[]): MergeResult<T[]> {
    let hadConflict = false;

    for (const remote of remoteElements) {
      const local = this.elements.get(remote.addId);

      if (!local) {
        // New element, add it
        this.elements.set(remote.addId, { ...remote });
      } else {
        // Merge element states
        if (remote.removedAt && !local.removedAt) {
          // Remote removed, local didn't
          if (remote.addedAt <= (remote.removedAt ?? 0)) {
            // Valid removal
            local.removedAt = remote.removedAt;
            local.removeId = remote.removeId;
          }
          hadConflict = true;
        } else if (!remote.removedAt && local.removedAt) {
          // Local removed, remote didn't - add wins
          // Re-add with new ID
          this.add(remote.value);
          hadConflict = true;
        }
      }
    }

    return {
      value: this.values(),
      hadConflict,
    };
  }

  /**
   * Get all elements with metadata
   */
  getElements(): ORSetElement<T>[] {
    return Array.from(this.elements.values());
  }

  /**
   * Clear the set
   */
  clear(): void {
    const removeId = `${this.nodeId}:${++this.counter}`;
    const now = Date.now();

    for (const element of this.elements.values()) {
      if (!element.removedAt) {
        element.removedAt = now;
        element.removeId = removeId;
      }
    }
  }

  private valuesEqual(a: T, b: T): boolean {
    return JSON.stringify(a) === JSON.stringify(b);
  }
}

// =============================================================================
// G-Counter (Grow-only Counter)
// =============================================================================

/**
 * Grow-only Counter
 */
export class GCounter {
  private counts: Record<string, number> = {};
  private nodeId: string;

  constructor(nodeId: string) {
    this.nodeId = nodeId;
    this.counts[nodeId] = 0;
  }

  /**
   * Get current value
   */
  value(): number {
    return Object.values(this.counts).reduce((sum, n) => sum + n, 0);
  }

  /**
   * Increment counter
   */
  increment(amount = 1): void {
    if (amount < 0) {
      throw new Error('GCounter can only increment');
    }
    this.counts[this.nodeId] = (this.counts[this.nodeId] ?? 0) + amount;
  }

  /**
   * Merge with remote counter
   */
  merge(remote: Record<string, number>): MergeResult<number> {
    const localValue = this.value();

    for (const [nodeId, count] of Object.entries(remote)) {
      this.counts[nodeId] = Math.max(this.counts[nodeId] ?? 0, count);
    }

    return {
      value: this.value(),
      hadConflict: localValue !== this.value(),
    };
  }

  /**
   * Get state for synchronization
   */
  getState(): Record<string, number> {
    return { ...this.counts };
  }
}

// =============================================================================
// PN-Counter (Positive-Negative Counter)
// =============================================================================

/**
 * Positive-Negative Counter
 */
export class PNCounter {
  private positive: GCounter;
  private negative: GCounter;

  constructor(nodeId: string) {
    this.positive = new GCounter(nodeId);
    this.negative = new GCounter(nodeId);
  }

  /**
   * Get current value
   */
  value(): number {
    return this.positive.value() - this.negative.value();
  }

  /**
   * Increment counter
   */
  increment(amount = 1): void {
    this.positive.increment(amount);
  }

  /**
   * Decrement counter
   */
  decrement(amount = 1): void {
    this.negative.increment(amount);
  }

  /**
   * Merge with remote counter
   */
  merge(remote: { positive: Record<string, number>; negative: Record<string, number> }): MergeResult<number> {
    const localValue = this.value();

    this.positive.merge(remote.positive);
    this.negative.merge(remote.negative);

    return {
      value: this.value(),
      hadConflict: localValue !== this.value(),
    };
  }

  /**
   * Get state for synchronization
   */
  getState(): { positive: Record<string, number>; negative: Record<string, number> } {
    return {
      positive: this.positive.getState(),
      negative: this.negative.getState(),
    };
  }
}

// =============================================================================
// Generic Merge Function
// =============================================================================

/**
 * Generic merge function based on strategy
 */
export function merge<T>(
  local: T,
  remote: T,
  strategy: MergeStrategy,
  options?: {
    localClock?: VectorClock;
    remoteClock?: VectorClock;
    localTimestamp?: number;
    remoteTimestamp?: number;
    nodeId?: string;
    customMerge?: (local: T, remote: T) => T;
  }
): MergeResult<T> {
  switch (strategy) {
    case 'lww': {
      const localTs = options?.localTimestamp ?? Date.now();
      const remoteTs = options?.remoteTimestamp ?? Date.now();

      if (remoteTs > localTs) {
        return {
          value: remote,
          hadConflict: true,
          conflictDetails: {
            localValue: local,
            remoteValue: remote,
            resolution: 'remote',
          },
        };
      }
      return {
        value: local,
        hadConflict: remoteTs === localTs && local !== remote,
        conflictDetails: remoteTs === localTs && local !== remote ? {
          localValue: local,
          remoteValue: remote,
          resolution: 'local',
        } : undefined,
      };
    }

    case 'vector-clock': {
      if (!options?.localClock || !options?.remoteClock) {
        throw new Error('Vector clocks required for vector-clock strategy');
      }

      const comparison = compareClocks(options.localClock, options.remoteClock);

      if (comparison === 'before') {
        return {
          value: remote,
          hadConflict: true,
          vectorClock: options.remoteClock,
          conflictDetails: {
            localValue: local,
            remoteValue: remote,
            resolution: 'remote',
          },
        };
      } else if (comparison === 'after' || comparison === 'equal') {
        return {
          value: local,
          hadConflict: false,
          vectorClock: options.localClock,
        };
      } else {
        // Concurrent - need custom resolution
        const merged = options?.customMerge?.(local, remote) ?? local;
        return {
          value: merged,
          hadConflict: true,
          vectorClock: mergeClocks(options.localClock, options.remoteClock),
          conflictDetails: {
            localValue: local,
            remoteValue: remote,
            resolution: 'merged',
          },
        };
      }
    }

    case 'set-union': {
      if (!Array.isArray(local) || !Array.isArray(remote)) {
        throw new Error('Set union requires array values');
      }
      const union = [...new Set([...local, ...remote])];
      return {
        value: union as T,
        hadConflict: false,
      };
    }

    case 'custom': {
      if (!options?.customMerge) {
        throw new Error('Custom merge function required for custom strategy');
      }
      return {
        value: options.customMerge(local, remote),
        hadConflict: true,
        conflictDetails: {
          localValue: local,
          remoteValue: remote,
          resolution: 'merged',
        },
      };
    }

    default:
      throw new Error(`Unknown merge strategy: ${strategy}`);
  }
}

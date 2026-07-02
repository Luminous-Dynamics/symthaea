// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CRDT (Conflict-free Replicated Data Types)
 *
 * Real-time collaboration primitives:
 * - LWW Register (Last-Writer-Wins)
 * - G-Counter (Grow-only counter)
 * - PN-Counter (Positive-Negative counter)
 * - OR-Set (Observed-Remove Set)
 * - LWW-Map (Last-Writer-Wins Map)
 * - RGA (Replicated Growable Array) for sequences
 */

// ==================== Types ====================

export type NodeId = string;
export type Timestamp = number;
export type VectorClock = Map<NodeId, Timestamp>;

export interface CRDTMetadata {
  nodeId: NodeId;
  timestamp: Timestamp;
  vectorClock: VectorClock;
}

// ==================== Vector Clock ====================

export class VectorClockImpl {
  private clock: Map<NodeId, Timestamp>;

  constructor(initial?: VectorClock) {
    this.clock = new Map(initial);
  }

  increment(nodeId: NodeId): void {
    this.clock.set(nodeId, (this.clock.get(nodeId) || 0) + 1);
  }

  get(nodeId: NodeId): Timestamp {
    return this.clock.get(nodeId) || 0;
  }

  merge(other: VectorClock): void {
    for (const [nodeId, timestamp] of other) {
      this.clock.set(nodeId, Math.max(this.clock.get(nodeId) || 0, timestamp));
    }
  }

  compare(other: VectorClock): -1 | 0 | 1 | null {
    let dominated = false;
    let dominates = false;

    const allKeys = new Set([...this.clock.keys(), ...other.keys()]);

    for (const key of allKeys) {
      const thisValue = this.clock.get(key) || 0;
      const otherValue = other.get(key) || 0;

      if (thisValue < otherValue) dominated = true;
      if (thisValue > otherValue) dominates = true;
    }

    if (dominated && dominates) return null; // Concurrent
    if (dominated) return -1;
    if (dominates) return 1;
    return 0; // Equal
  }

  toMap(): VectorClock {
    return new Map(this.clock);
  }

  toJSON(): Record<string, number> {
    return Object.fromEntries(this.clock);
  }

  static fromJSON(json: Record<string, number>): VectorClockImpl {
    return new VectorClockImpl(new Map(Object.entries(json)));
  }
}

// ==================== LWW Register ====================

export class LWWRegister<T> {
  private value: T;
  private timestamp: Timestamp;
  private nodeId: NodeId;

  constructor(initialValue: T, nodeId: NodeId) {
    this.value = initialValue;
    this.timestamp = Date.now();
    this.nodeId = nodeId;
  }

  get(): T {
    return this.value;
  }

  set(value: T, timestamp?: Timestamp): void {
    const ts = timestamp || Date.now();
    if (ts > this.timestamp || (ts === this.timestamp && this.nodeId < this.nodeId)) {
      this.value = value;
      this.timestamp = ts;
    }
  }

  merge(other: { value: T; timestamp: Timestamp; nodeId: NodeId }): void {
    if (
      other.timestamp > this.timestamp ||
      (other.timestamp === this.timestamp && other.nodeId > this.nodeId)
    ) {
      this.value = other.value;
      this.timestamp = other.timestamp;
      this.nodeId = other.nodeId;
    }
  }

  getState(): { value: T; timestamp: Timestamp; nodeId: NodeId } {
    return { value: this.value, timestamp: this.timestamp, nodeId: this.nodeId };
  }
}

// ==================== G-Counter ====================

export class GCounter {
  private counts: Map<NodeId, number>;

  constructor() {
    this.counts = new Map();
  }

  increment(nodeId: NodeId, amount = 1): void {
    this.counts.set(nodeId, (this.counts.get(nodeId) || 0) + amount);
  }

  value(): number {
    let sum = 0;
    for (const count of this.counts.values()) {
      sum += count;
    }
    return sum;
  }

  merge(other: GCounter): void {
    for (const [nodeId, count] of other.counts) {
      this.counts.set(nodeId, Math.max(this.counts.get(nodeId) || 0, count));
    }
  }

  getState(): Map<NodeId, number> {
    return new Map(this.counts);
  }
}

// ==================== PN-Counter ====================

export class PNCounter {
  private positive: GCounter;
  private negative: GCounter;

  constructor() {
    this.positive = new GCounter();
    this.negative = new GCounter();
  }

  increment(nodeId: NodeId, amount = 1): void {
    this.positive.increment(nodeId, amount);
  }

  decrement(nodeId: NodeId, amount = 1): void {
    this.negative.increment(nodeId, amount);
  }

  value(): number {
    return this.positive.value() - this.negative.value();
  }

  merge(other: PNCounter): void {
    this.positive.merge(other.positive);
    this.negative.merge(other.negative);
  }

  getState(): { positive: Map<NodeId, number>; negative: Map<NodeId, number> } {
    return {
      positive: this.positive.getState(),
      negative: this.negative.getState(),
    };
  }
}

// ==================== OR-Set (Observed-Remove Set) ====================

interface ORSetElement<T> {
  value: T;
  unique: string;
  deleted: boolean;
}

export class ORSet<T> {
  private elements: Map<string, ORSetElement<T>>;
  private nodeId: NodeId;

  constructor(nodeId: NodeId) {
    this.elements = new Map();
    this.nodeId = nodeId;
  }

  private generateUnique(): string {
    return `${this.nodeId}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  add(value: T): string {
    const unique = this.generateUnique();
    this.elements.set(unique, { value, unique, deleted: false });
    return unique;
  }

  remove(value: T): void {
    for (const [unique, element] of this.elements) {
      if (element.value === value && !element.deleted) {
        element.deleted = true;
      }
    }
  }

  removeByUnique(unique: string): void {
    const element = this.elements.get(unique);
    if (element) {
      element.deleted = true;
    }
  }

  has(value: T): boolean {
    for (const element of this.elements.values()) {
      if (element.value === value && !element.deleted) {
        return true;
      }
    }
    return false;
  }

  values(): T[] {
    const result: T[] = [];
    for (const element of this.elements.values()) {
      if (!element.deleted) {
        result.push(element.value);
      }
    }
    return result;
  }

  merge(other: ORSet<T>): void {
    for (const [unique, otherElement] of other.elements) {
      const existing = this.elements.get(unique);
      if (!existing) {
        this.elements.set(unique, { ...otherElement });
      } else if (otherElement.deleted) {
        existing.deleted = true;
      }
    }
  }

  getState(): Map<string, ORSetElement<T>> {
    return new Map(this.elements);
  }
}

// ==================== LWW-Map ====================

export class LWWMap<K, V> {
  private entries: Map<K, LWWRegister<V | null>>;
  private nodeId: NodeId;

  constructor(nodeId: NodeId) {
    this.entries = new Map();
    this.nodeId = nodeId;
  }

  set(key: K, value: V): void {
    const existing = this.entries.get(key);
    if (existing) {
      existing.set(value);
    } else {
      this.entries.set(key, new LWWRegister(value, this.nodeId));
    }
  }

  get(key: K): V | undefined {
    const register = this.entries.get(key);
    const value = register?.get();
    return value === null ? undefined : value;
  }

  delete(key: K): void {
    const existing = this.entries.get(key);
    if (existing) {
      existing.set(null);
    }
  }

  has(key: K): boolean {
    const register = this.entries.get(key);
    return register !== undefined && register.get() !== null;
  }

  keys(): K[] {
    const result: K[] = [];
    for (const [key, register] of this.entries) {
      if (register.get() !== null) {
        result.push(key);
      }
    }
    return result;
  }

  values(): V[] {
    const result: V[] = [];
    for (const register of this.entries.values()) {
      const value = register.get();
      if (value !== null) {
        result.push(value);
      }
    }
    return result;
  }

  entries_(): [K, V][] {
    const result: [K, V][] = [];
    for (const [key, register] of this.entries) {
      const value = register.get();
      if (value !== null) {
        result.push([key, value]);
      }
    }
    return result;
  }

  merge(other: LWWMap<K, V>): void {
    for (const [key, otherRegister] of other.entries) {
      const existing = this.entries.get(key);
      if (existing) {
        existing.merge(otherRegister.getState());
      } else {
        const newRegister = new LWWRegister<V | null>(null, this.nodeId);
        newRegister.merge(otherRegister.getState());
        this.entries.set(key, newRegister);
      }
    }
  }

  toJSON(): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const [key, register] of this.entries) {
      const value = register.get();
      if (value !== null) {
        result[String(key)] = value;
      }
    }
    return result;
  }
}

// ==================== RGA (Replicated Growable Array) ====================

interface RGANode<T> {
  id: string;
  value: T | null; // null means tombstone
  timestamp: Timestamp;
  nodeId: NodeId;
  afterId: string | null;
}

export class RGA<T> {
  private nodes: Map<string, RGANode<T>>;
  private nodeId: NodeId;
  private head: string | null = null;

  constructor(nodeId: NodeId) {
    this.nodes = new Map();
    this.nodeId = nodeId;
  }

  private generateId(): string {
    return `${this.nodeId}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  private findPosition(afterId: string | null): number {
    if (afterId === null) return 0;

    const sequence = this.toArray();
    for (let i = 0; i < sequence.length; i++) {
      // Find the node and return position after it
      const nodeAtPosition = this.getNodeAtIndex(i);
      if (nodeAtPosition?.id === afterId) {
        return i + 1;
      }
    }
    return sequence.length;
  }

  private getNodeAtIndex(index: number): RGANode<T> | null {
    let currentId = this.head;
    let currentIndex = 0;

    while (currentId) {
      const node = this.nodes.get(currentId);
      if (!node) break;

      if (node.value !== null) {
        if (currentIndex === index) return node;
        currentIndex++;
      }

      // Find next node
      let nextId: string | null = null;
      for (const [id, n] of this.nodes) {
        if (n.afterId === currentId) {
          if (!nextId || this.compareNodes(n, this.nodes.get(nextId)!) > 0) {
            nextId = id;
          }
        }
      }
      currentId = nextId;
    }

    return null;
  }

  private compareNodes(a: RGANode<T>, b: RGANode<T>): number {
    if (a.timestamp !== b.timestamp) {
      return a.timestamp - b.timestamp;
    }
    return a.nodeId.localeCompare(b.nodeId);
  }

  insert(index: number, value: T): string {
    const id = this.generateId();
    const timestamp = Date.now();

    // Find the node to insert after
    let afterId: string | null = null;
    if (index > 0) {
      const beforeNode = this.getNodeAtIndex(index - 1);
      afterId = beforeNode?.id || null;
    }

    const node: RGANode<T> = {
      id,
      value,
      timestamp,
      nodeId: this.nodeId,
      afterId,
    };

    this.nodes.set(id, node);

    if (afterId === null && this.head === null) {
      this.head = id;
    }

    return id;
  }

  delete(index: number): void {
    const node = this.getNodeAtIndex(index);
    if (node) {
      node.value = null; // Tombstone
    }
  }

  deleteById(id: string): void {
    const node = this.nodes.get(id);
    if (node) {
      node.value = null;
    }
  }

  get(index: number): T | undefined {
    const node = this.getNodeAtIndex(index);
    return node?.value ?? undefined;
  }

  length(): number {
    let count = 0;
    for (const node of this.nodes.values()) {
      if (node.value !== null) count++;
    }
    return count;
  }

  toArray(): T[] {
    const result: T[] = [];
    const visited = new Set<string>();

    // Build adjacency list
    const children = new Map<string | null, RGANode<T>[]>();
    for (const node of this.nodes.values()) {
      const afterId = node.afterId;
      if (!children.has(afterId)) {
        children.set(afterId, []);
      }
      children.get(afterId)!.push(node);
    }

    // Sort children by timestamp and nodeId
    for (const nodeList of children.values()) {
      nodeList.sort((a, b) => this.compareNodes(a, b));
    }

    // DFS traversal
    const traverse = (afterId: string | null) => {
      const nodeList = children.get(afterId) || [];
      for (const node of nodeList) {
        if (visited.has(node.id)) continue;
        visited.add(node.id);

        if (node.value !== null) {
          result.push(node.value);
        }
        traverse(node.id);
      }
    };

    traverse(null);
    return result;
  }

  merge(other: RGA<T>): void {
    for (const [id, otherNode] of other.nodes) {
      const existing = this.nodes.get(id);
      if (!existing) {
        this.nodes.set(id, { ...otherNode });
      } else if (otherNode.value === null && existing.value !== null) {
        existing.value = null; // Apply tombstone
      }
    }

    // Update head if needed
    if (other.head && !this.head) {
      this.head = other.head;
    }
  }

  getState(): Map<string, RGANode<T>> {
    return new Map(this.nodes);
  }
}

// ==================== Document CRDT ====================

export interface DocumentOperation {
  type: 'insert' | 'delete' | 'update';
  path: string[];
  value?: unknown;
  index?: number;
  timestamp: Timestamp;
  nodeId: NodeId;
  id: string;
}

export class DocumentCRDT {
  private nodeId: NodeId;
  private root: LWWMap<string, unknown>;
  private arrays: Map<string, RGA<unknown>>;
  private vectorClock: VectorClockImpl;
  private pendingOps: DocumentOperation[] = [];
  private appliedOps: Set<string> = new Set();

  constructor(nodeId: NodeId) {
    this.nodeId = nodeId;
    this.root = new LWWMap(nodeId);
    this.arrays = new Map();
    this.vectorClock = new VectorClockImpl();
  }

  private getPath(path: string[]): unknown {
    let current: unknown = this.root.toJSON();
    for (const key of path) {
      if (current && typeof current === 'object') {
        current = (current as Record<string, unknown>)[key];
      } else {
        return undefined;
      }
    }
    return current;
  }

  set(path: string[], value: unknown): DocumentOperation {
    const op: DocumentOperation = {
      type: 'update',
      path,
      value,
      timestamp: Date.now(),
      nodeId: this.nodeId,
      id: `${this.nodeId}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };

    this.applyOperation(op);
    this.vectorClock.increment(this.nodeId);

    return op;
  }

  delete(path: string[]): DocumentOperation {
    const op: DocumentOperation = {
      type: 'delete',
      path,
      timestamp: Date.now(),
      nodeId: this.nodeId,
      id: `${this.nodeId}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };

    this.applyOperation(op);
    this.vectorClock.increment(this.nodeId);

    return op;
  }

  insertAt(path: string[], index: number, value: unknown): DocumentOperation {
    const op: DocumentOperation = {
      type: 'insert',
      path,
      index,
      value,
      timestamp: Date.now(),
      nodeId: this.nodeId,
      id: `${this.nodeId}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };

    this.applyOperation(op);
    this.vectorClock.increment(this.nodeId);

    return op;
  }

  applyOperation(op: DocumentOperation): boolean {
    if (this.appliedOps.has(op.id)) {
      return false;
    }

    this.appliedOps.add(op.id);

    switch (op.type) {
      case 'update': {
        const key = op.path.join('.');
        this.root.set(key, op.value);
        break;
      }
      case 'delete': {
        const key = op.path.join('.');
        this.root.delete(key);
        break;
      }
      case 'insert': {
        const arrayPath = op.path.join('.');
        let rga = this.arrays.get(arrayPath);
        if (!rga) {
          rga = new RGA(this.nodeId);
          this.arrays.set(arrayPath, rga);
        }
        rga.insert(op.index || 0, op.value);
        break;
      }
    }

    return true;
  }

  applyOperations(ops: DocumentOperation[]): void {
    // Sort by timestamp for causal ordering
    ops.sort((a, b) => a.timestamp - b.timestamp);

    for (const op of ops) {
      this.applyOperation(op);
    }
  }

  getDocument(): Record<string, unknown> {
    const doc = this.root.toJSON();

    // Merge in arrays
    for (const [path, rga] of this.arrays) {
      const keys = path.split('.');
      let current = doc;
      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) {
          current[keys[i]] = {};
        }
        current = current[keys[i]] as Record<string, unknown>;
      }
      current[keys[keys.length - 1]] = rga.toArray();
    }

    return doc;
  }

  getVectorClock(): VectorClock {
    return this.vectorClock.toMap();
  }

  merge(other: DocumentCRDT): void {
    this.root.merge(other.root);

    for (const [path, otherRga] of other.arrays) {
      const existing = this.arrays.get(path);
      if (existing) {
        existing.merge(otherRga);
      } else {
        const newRga = new RGA(this.nodeId);
        newRga.merge(otherRga);
        this.arrays.set(path, newRga);
      }
    }

    this.vectorClock.merge(other.vectorClock.toMap());
  }
}

export default {
  VectorClockImpl,
  LWWRegister,
  GCounter,
  PNCounter,
  ORSet,
  LWWMap,
  RGA,
  DocumentCRDT,
};

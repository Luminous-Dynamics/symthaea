// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Memory Backend (M0)
 *
 * In-memory storage for ephemeral data.
 * No persistence - data lost on restart.
 *
 * @see docs/architecture/uess/UESS-01-CORE.md §3.1
 */

import { hash } from '../../security/index.js';

import type { StorageMetadata } from '../types.js';

// =============================================================================
// Backend Interface
// =============================================================================

/**
 * Storage backend interface
 */
export interface StorageBackendAdapter {
  /** Get data by key */
  get<T>(key: string): Promise<{ data: T; metadata: StorageMetadata } | null>;

  /** Set data by key */
  set<T>(
    key: string,
    data: T,
    metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'>
  ): Promise<StorageMetadata>;

  /** Delete data by key */
  delete(key: string): Promise<boolean>;

  /** Check if key exists */
  has(key: string): Promise<boolean>;

  /** List all keys matching pattern */
  keys(pattern?: string): Promise<string[]>;

  /** Clear all data */
  clear(): Promise<void>;

  /** Get storage statistics */
  stats(): Promise<{
    itemCount: number;
    totalSizeBytes: number;
    oldestItem?: number;
    newestItem?: number;
  }>;
}

// =============================================================================
// Memory Storage Entry
// =============================================================================

interface MemoryEntry<T = unknown> {
  data: T;
  metadata: StorageMetadata;
  expiresAt?: number;
}

// =============================================================================
// Memory Backend Implementation
// =============================================================================

/**
 * In-memory storage backend for M0 ephemeral data
 */
export class MemoryBackend implements StorageBackendAdapter {
  private readonly store = new Map<string, MemoryEntry>();
  private readonly defaultTtlMs: number;
  private gcIntervalId?: ReturnType<typeof setInterval>;

  constructor(options: { defaultTtlMs?: number; gcIntervalMs?: number } = {}) {
    this.defaultTtlMs = options.defaultTtlMs ?? 60 * 60 * 1000; // 1 hour

    // Start garbage collection
    if (options.gcIntervalMs !== 0) {
      const gcInterval = options.gcIntervalMs ?? 60 * 1000; // 1 minute
      this.gcIntervalId = setInterval(() => this.gc(), gcInterval);
    }
  }

  /**
   * Get data by key
   */
  async get<T>(key: string): Promise<{ data: T; metadata: StorageMetadata } | null> {
    const entry = this.store.get(key) as MemoryEntry<T> | undefined;

    if (!entry) return null;

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(key);
      return null;
    }

    // Check tombstone
    if (entry.metadata.tombstone) {
      return null;
    }

    return {
      data: entry.data,
      metadata: entry.metadata,
    };
  }

  /**
   * Set data by key
   */
  async set<T>(
    key: string,
    data: T,
    metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'>
  ): Promise<StorageMetadata> {
    // Serialize and compute CID
    const serialized = JSON.stringify(data);
    const dataBytes = new TextEncoder().encode(serialized);
    const cid = await this.computeCid(dataBytes);

    // Compute expiry
    const now = Date.now();
    const expiresAt = metadata.expiresAt ?? now + this.defaultTtlMs;

    const fullMetadata: StorageMetadata = {
      ...metadata,
      cid,
      sizeBytes: dataBytes.length,
    };

    const entry: MemoryEntry<T> = {
      data,
      metadata: fullMetadata,
      expiresAt,
    };

    this.store.set(key, entry);

    return fullMetadata;
  }

  /**
   * Delete data by key
   */
  async delete(key: string): Promise<boolean> {
    return this.store.delete(key);
  }

  /**
   * Check if key exists
   */
  async has(key: string): Promise<boolean> {
    const entry = this.store.get(key);
    if (!entry) return false;

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(key);
      return false;
    }

    return !entry.metadata.tombstone;
  }

  /**
   * List keys matching pattern
   */
  async keys(pattern?: string): Promise<string[]> {
    const allKeys: string[] = [];
    const now = Date.now();

    for (const [key, entry] of this.store.entries()) {
      // Skip expired
      if (entry.expiresAt && entry.expiresAt < now) continue;
      // Skip tombstones
      if (entry.metadata.tombstone) continue;

      if (!pattern || this.matchPattern(key, pattern)) {
        allKeys.push(key);
      }
    }

    return allKeys;
  }

  /**
   * Clear all data
   */
  async clear(): Promise<void> {
    this.store.clear();
  }

  /**
   * Get storage statistics
   */
  async stats(): Promise<{
    itemCount: number;
    totalSizeBytes: number;
    oldestItem?: number;
    newestItem?: number;
  }> {
    let itemCount = 0;
    let totalSizeBytes = 0;
    let oldestItem: number | undefined;
    let newestItem: number | undefined;
    const now = Date.now();

    for (const entry of this.store.values()) {
      // Skip expired
      if (entry.expiresAt && entry.expiresAt < now) continue;
      // Skip tombstones
      if (entry.metadata.tombstone) continue;

      itemCount++;
      totalSizeBytes += entry.metadata.sizeBytes;

      if (!oldestItem || entry.metadata.storedAt < oldestItem) {
        oldestItem = entry.metadata.storedAt;
      }
      if (!newestItem || entry.metadata.storedAt > newestItem) {
        newestItem = entry.metadata.storedAt;
      }
    }

    return { itemCount, totalSizeBytes, oldestItem, newestItem };
  }

  /**
   * Garbage collection - remove expired entries
   */
  gc(): number {
    let removed = 0;
    const now = Date.now();

    for (const [key, entry] of this.store.entries()) {
      if (entry.expiresAt && entry.expiresAt < now) {
        this.store.delete(key);
        removed++;
      }
    }

    return removed;
  }

  /**
   * Stop garbage collection timer
   */
  dispose(): void {
    if (this.gcIntervalId) {
      clearInterval(this.gcIntervalId);
      this.gcIntervalId = undefined;
    }
  }

  /**
   * Simple pattern matching (supports * wildcard)
   */
  private matchPattern(key: string, pattern: string): boolean {
    const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
    return regex.test(key);
  }

  /**
   * Compute content identifier (hash)
   */
  private async computeCid(data: Uint8Array): Promise<string> {
    const hashBytes = await hash(data);
    // Convert to hex string with 'cid:' prefix
    return 'cid:' + Array.from(hashBytes).map(b => b.toString(16).padStart(2, '0')).join('');
  }
}

/**
 * Create a memory backend with default options
 */
export function createMemoryBackend(options?: {
  defaultTtlMs?: number;
  gcIntervalMs?: number;
}): MemoryBackend {
  return new MemoryBackend(options);
}

/**
 * Compute content identifier for data
 * Exported for use by other backends
 */
export function computeCID(data: unknown): string {
  // Use a simple hash-based CID for synchronous operation
  const serialized = JSON.stringify(data);
  let hash = 0;
  for (let i = 0; i < serialized.length; i++) {
    const char = serialized.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  // Use abs value and convert to hex
  const hexHash = Math.abs(hash).toString(16).padStart(8, '0');
  return `cid:${hexHash}`;
}

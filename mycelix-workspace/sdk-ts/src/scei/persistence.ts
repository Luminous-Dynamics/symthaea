// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk SCEI Persistence Layer
 *
 * Abstract persistence interface for SCEI modules.
 * Allows plugging in different storage backends:
 * - In-memory (default, for testing)
 * - IndexedDB (browser)
 * - SQLite (Node.js)
 * - Holochain DHT (production)
 *
 * @packageDocumentation
 * @module scei/persistence
 */

// ============================================================================
// STORAGE TYPES
// ============================================================================

export interface StorageKey {
  namespace: SCEINamespace;
  id: string;
}

export type SCEINamespace =
  | 'calibration:records'
  | 'calibration:reports'
  | 'metabolism:claims'
  | 'metabolism:tombstones'
  | 'propagation:state'
  | 'propagation:quarantine'
  | 'discovery:gaps'
  | 'discovery:claims';

export interface StorageOptions {
  /** Time-to-live in milliseconds (0 = no expiry) */
  ttl?: number;
  /** Tags for batch operations */
  tags?: string[];
}

export interface StorageMetadata {
  createdAt: number;
  updatedAt: number;
  expiresAt?: number;
  tags: string[];
  version: number;
}

export interface StoredItem<T> {
  key: StorageKey;
  value: T;
  metadata: StorageMetadata;
}

export interface QueryOptions {
  /** Prefix match on ID */
  idPrefix?: string;
  /** Filter by tags (all must match) */
  tags?: string[];
  /** Created after timestamp */
  createdAfter?: number;
  /** Created before timestamp */
  createdBefore?: number;
  /** Maximum results */
  limit?: number;
  /** Skip first N results */
  offset?: number;
  /** Sort field */
  sortBy?: 'createdAt' | 'updatedAt' | 'id';
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
}

// ============================================================================
// PERSISTENCE INTERFACE
// ============================================================================

/**
 * SCEI Persistence Interface
 *
 * Abstract storage interface that modules use for persistence.
 * Implementations can target different backends.
 *
 * @example
 * ```typescript
 * // Using in-memory storage (testing)
 * const storage = new InMemorySCEIStorage();
 *
 * // Save a calibration record
 * await storage.set(
 *   { namespace: 'calibration:records', id: 'pred-123' },
 *   { confidence: 0.8, domain: 'science', ... },
 *   { ttl: 86400000, tags: ['science'] }
 * );
 *
 * // Query records
 * const records = await storage.query('calibration:records', {
 *   tags: ['science'],
 *   limit: 100
 * });
 * ```
 */
export interface SCEIPersistence {
  /**
   * Get a single item by key
   */
  get<T>(key: StorageKey): Promise<StoredItem<T> | null>;

  /**
   * Set a single item
   */
  set<T>(key: StorageKey, value: T, options?: StorageOptions): Promise<void>;

  /**
   * Delete a single item
   */
  delete(key: StorageKey): Promise<boolean>;

  /**
   * Check if an item exists
   */
  exists(key: StorageKey): Promise<boolean>;

  /**
   * Query items by namespace with filters
   */
  query<T>(namespace: SCEINamespace, options?: QueryOptions): Promise<StoredItem<T>[]>;

  /**
   * Count items matching query
   */
  count(namespace: SCEINamespace, options?: QueryOptions): Promise<number>;

  /**
   * Batch get multiple items
   */
  batchGet<T>(keys: StorageKey[]): Promise<Map<string, StoredItem<T>>>;

  /**
   * Batch set multiple items
   */
  batchSet<T>(items: Array<{ key: StorageKey; value: T; options?: StorageOptions }>): Promise<void>;

  /**
   * Batch delete multiple items
   */
  batchDelete(keys: StorageKey[]): Promise<number>;

  /**
   * Delete all items in a namespace
   */
  clearNamespace(namespace: SCEINamespace): Promise<number>;

  /**
   * Delete all expired items
   */
  purgeExpired(): Promise<number>;

  /**
   * Get storage statistics
   */
  getStats(): Promise<StorageStats>;

  /**
   * Close the storage connection
   */
  close(): Promise<void>;
}

export interface StorageStats {
  totalItems: number;
  itemsByNamespace: Record<SCEINamespace, number>;
  totalSizeBytes?: number;
  oldestItem?: number;
  newestItem?: number;
}

// ============================================================================
// IN-MEMORY IMPLEMENTATION
// ============================================================================

/**
 * In-Memory SCEI Storage
 *
 * Simple in-memory implementation for testing and development.
 * Data is lost on process restart.
 */
export class InMemorySCEIStorage implements SCEIPersistence {
  private storage: Map<string, StoredItem<unknown>> = new Map();

  async get<T>(key: StorageKey): Promise<StoredItem<T> | null> {
    const fullKey = this.makeFullKey(key);
    const item = this.storage.get(fullKey);

    if (!item) return null;

    // Check expiry
    if (item.metadata.expiresAt && item.metadata.expiresAt < Date.now()) {
      this.storage.delete(fullKey);
      return null;
    }

    return item as StoredItem<T>;
  }

  async set<T>(key: StorageKey, value: T, options: StorageOptions = {}): Promise<void> {
    const fullKey = this.makeFullKey(key);
    const now = Date.now();

    const existing = this.storage.get(fullKey);
    const metadata: StorageMetadata = {
      createdAt: existing?.metadata.createdAt ?? now,
      updatedAt: now,
      expiresAt: options.ttl ? now + options.ttl : undefined,
      tags: options.tags ?? existing?.metadata.tags ?? [],
      version: (existing?.metadata.version ?? 0) + 1,
    };

    this.storage.set(fullKey, { key, value, metadata });
  }

  async delete(key: StorageKey): Promise<boolean> {
    const fullKey = this.makeFullKey(key);
    return this.storage.delete(fullKey);
  }

  async exists(key: StorageKey): Promise<boolean> {
    const item = await this.get(key);
    return item !== null;
  }

  async query<T>(namespace: SCEINamespace, options: QueryOptions = {}): Promise<StoredItem<T>[]> {
    const results: StoredItem<T>[] = [];
    const now = Date.now();

    for (const [fullKey, item] of this.storage) {
      // Skip if wrong namespace
      if (!fullKey.startsWith(`${namespace}:`)) continue;

      // Skip expired
      if (item.metadata.expiresAt && item.metadata.expiresAt < now) continue;

      // Filter by ID prefix
      if (options.idPrefix && !item.key.id.startsWith(options.idPrefix)) continue;

      // Filter by tags
      if (options.tags) {
        const hasAllTags = options.tags.every((tag) => item.metadata.tags.includes(tag));
        if (!hasAllTags) continue;
      }

      // Filter by creation time
      if (options.createdAfter && item.metadata.createdAt < options.createdAfter) continue;
      if (options.createdBefore && item.metadata.createdAt > options.createdBefore) continue;

      results.push(item as StoredItem<T>);
    }

    // Sort
    const sortBy = options.sortBy ?? 'createdAt';
    const sortOrder = options.sortOrder ?? 'desc';
    results.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortBy) {
        case 'id':
          aVal = a.key.id;
          bVal = b.key.id;
          break;
        case 'updatedAt':
          aVal = a.metadata.updatedAt;
          bVal = b.metadata.updatedAt;
          break;
        default:
          aVal = a.metadata.createdAt;
          bVal = b.metadata.createdAt;
      }

      if (sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });

    // Pagination
    const offset = options.offset ?? 0;
    const limit = options.limit ?? results.length;
    return results.slice(offset, offset + limit);
  }

  async count(namespace: SCEINamespace, options: QueryOptions = {}): Promise<number> {
    const results = await this.query(namespace, { ...options, limit: undefined });
    return results.length;
  }

  async batchGet<T>(keys: StorageKey[]): Promise<Map<string, StoredItem<T>>> {
    const results = new Map<string, StoredItem<T>>();

    for (const key of keys) {
      const item = await this.get<T>(key);
      if (item) {
        results.set(this.makeFullKey(key), item);
      }
    }

    return results;
  }

  async batchSet<T>(
    items: Array<{ key: StorageKey; value: T; options?: StorageOptions }>
  ): Promise<void> {
    for (const { key, value, options } of items) {
      await this.set(key, value, options);
    }
  }

  async batchDelete(keys: StorageKey[]): Promise<number> {
    let deleted = 0;
    for (const key of keys) {
      if (await this.delete(key)) {
        deleted++;
      }
    }
    return deleted;
  }

  async clearNamespace(namespace: SCEINamespace): Promise<number> {
    let deleted = 0;
    const prefix = `${namespace}:`;

    for (const key of this.storage.keys()) {
      if (key.startsWith(prefix)) {
        this.storage.delete(key);
        deleted++;
      }
    }

    return deleted;
  }

  async purgeExpired(): Promise<number> {
    let purged = 0;
    const now = Date.now();

    for (const [key, item] of this.storage) {
      if (item.metadata.expiresAt && item.metadata.expiresAt < now) {
        this.storage.delete(key);
        purged++;
      }
    }

    return purged;
  }

  async getStats(): Promise<StorageStats> {
    const itemsByNamespace: Partial<Record<SCEINamespace, number>> = {};
    let oldestItem: number | undefined;
    let newestItem: number | undefined;

    for (const item of this.storage.values()) {
      const ns = item.key.namespace;
      itemsByNamespace[ns] = (itemsByNamespace[ns] ?? 0) + 1;

      if (!oldestItem || item.metadata.createdAt < oldestItem) {
        oldestItem = item.metadata.createdAt;
      }
      if (!newestItem || item.metadata.createdAt > newestItem) {
        newestItem = item.metadata.createdAt;
      }
    }

    return {
      totalItems: this.storage.size,
      itemsByNamespace: itemsByNamespace as Record<SCEINamespace, number>,
      oldestItem,
      newestItem,
    };
  }

  async close(): Promise<void> {
    // No-op for in-memory
  }

  private makeFullKey(key: StorageKey): string {
    return `${key.namespace}:${key.id}`;
  }
}

// ============================================================================
// PERSISTENCE MANAGER
// ============================================================================

/**
 * SCEI Persistence Manager
 *
 * Manages storage backend and provides high-level operations.
 */
export class SCEIPersistenceManager {
  private storage: SCEIPersistence;
  private autoPurgeInterval?: ReturnType<typeof setInterval>;

  constructor(storage: SCEIPersistence, options: { autoPurgeIntervalMs?: number } = {}) {
    this.storage = storage;

    // Start auto-purge if requested
    if (options.autoPurgeIntervalMs && options.autoPurgeIntervalMs > 0) {
      this.autoPurgeInterval = setInterval(() => {
        this.storage.purgeExpired().catch((err) => {
          console.error('SCEI Persistence auto-purge error:', err);
        });
      }, options.autoPurgeIntervalMs);
    }
  }

  /**
   * Get the underlying storage backend
   */
  getStorage(): SCEIPersistence {
    return this.storage;
  }

  // ==========================================================================
  // CALIBRATION HELPERS
  // ==========================================================================

  async saveCalibrationRecord(
    id: string,
    record: unknown,
    options?: StorageOptions
  ): Promise<void> {
    await this.storage.set({ namespace: 'calibration:records', id }, record, options);
  }

  async getCalibrationRecord<T>(id: string): Promise<T | null> {
    const item = await this.storage.get<T>({ namespace: 'calibration:records', id });
    return item?.value ?? null;
  }

  async queryCalibrationRecords<T>(options?: QueryOptions): Promise<T[]> {
    const items = await this.storage.query<T>('calibration:records', options);
    return items.map((i) => i.value);
  }

  // ==========================================================================
  // METABOLISM HELPERS
  // ==========================================================================

  async saveClaim(id: string, claim: unknown, options?: StorageOptions): Promise<void> {
    await this.storage.set({ namespace: 'metabolism:claims', id }, claim, options);
  }

  async getClaim<T>(id: string): Promise<T | null> {
    const item = await this.storage.get<T>({ namespace: 'metabolism:claims', id });
    return item?.value ?? null;
  }

  async deleteClaim(id: string): Promise<boolean> {
    return this.storage.delete({ namespace: 'metabolism:claims', id });
  }

  async saveTombstone(id: string, tombstone: unknown): Promise<void> {
    await this.storage.set({ namespace: 'metabolism:tombstones', id }, tombstone);
  }

  async getTombstone<T>(id: string): Promise<T | null> {
    const item = await this.storage.get<T>({ namespace: 'metabolism:tombstones', id });
    return item?.value ?? null;
  }

  async queryTombstones<T>(options?: QueryOptions): Promise<T[]> {
    const items = await this.storage.query<T>('metabolism:tombstones', options);
    return items.map((i) => i.value);
  }

  // ==========================================================================
  // DISCOVERY HELPERS
  // ==========================================================================

  async saveGap(id: string, gap: unknown, options?: StorageOptions): Promise<void> {
    await this.storage.set({ namespace: 'discovery:gaps', id }, gap, options);
  }

  async getGap<T>(id: string): Promise<T | null> {
    const item = await this.storage.get<T>({ namespace: 'discovery:gaps', id });
    return item?.value ?? null;
  }

  async queryGaps<T>(options?: QueryOptions): Promise<T[]> {
    const items = await this.storage.query<T>('discovery:gaps', options);
    return items.map((i) => i.value);
  }

  // ==========================================================================
  // LIFECYCLE
  // ==========================================================================

  async close(): Promise<void> {
    if (this.autoPurgeInterval) {
      clearInterval(this.autoPurgeInterval);
    }
    await this.storage.close();
  }
}

// ============================================================================
// SINGLETON
// ============================================================================

let instance: SCEIPersistenceManager | null = null;

/**
 * Get the global SCEI persistence manager
 */
export function getSCEIPersistence(): SCEIPersistenceManager {
  if (!instance) {
    // Default to in-memory storage
    instance = new SCEIPersistenceManager(new InMemorySCEIStorage(), {
      autoPurgeIntervalMs: 60000, // Purge expired items every minute
    });
  }
  return instance;
}

/**
 * Set a custom persistence backend
 */
export function setSCEIPersistence(storage: SCEIPersistence): SCEIPersistenceManager {
  if (instance) {
    instance.close().catch(console.error);
  }
  instance = new SCEIPersistenceManager(storage);
  return instance;
}

/**
 * Reset the global persistence manager (for testing)
 */
export function resetSCEIPersistence(): void {
  if (instance) {
    instance.close().catch(console.error);
  }
  instance = null;
}

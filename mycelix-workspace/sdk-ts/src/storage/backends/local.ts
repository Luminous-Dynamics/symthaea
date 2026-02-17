/**
 * UESS Local Backend (M1)
 *
 * Local storage for temporal data.
 * Survives page refresh but not device change.
 *
 * Browser: Uses IndexedDB (with localStorage fallback)
 * Node.js: Uses file system persistence
 *
 * @see docs/architecture/uess/UESS-01-CORE.md §3.1
 */

import { createPersistenceAdapter, type PersistenceAdapter } from './persistence.js';
import { hash } from '../../security/index.js';

import type { StorageMetadata } from '../types.js';
import type { StorageBackendAdapter } from './memory.js';

// =============================================================================
// Local Storage Entry
// =============================================================================

interface LocalEntry<T = unknown> {
  data: T;
  metadata: StorageMetadata;
  expiresAt?: number;
}

// =============================================================================
// Local Backend Configuration
// =============================================================================

export interface LocalBackendOptions {
  /** Namespace for data isolation (default: 'uess') */
  namespace?: string;

  /** Default TTL in milliseconds (default: 7 days) */
  defaultTtlMs?: number;

  /** Garbage collection interval in ms (default: 1 hour, 0 to disable) */
  gcIntervalMs?: number;

  /** Disable automatic persistence (default: false) */
  disablePersistence?: boolean;

  /** Force a specific persistence adapter */
  forceAdapter?: 'indexeddb' | 'filesystem' | 'localstorage' | 'memory';

  /** Base directory for file persistence (Node.js only) */
  persistenceDir?: string;

  /** Debounce persistence writes by this many ms (default: 100) */
  persistenceDebounceMs?: number;

  /** @deprecated Use built-in persistence. Custom persist function */
  persist?: (data: Map<string, LocalEntry>) => Promise<void>;

  /** @deprecated Use built-in persistence. Custom load function */
  load?: () => Promise<Map<string, LocalEntry> | null>;
}

// =============================================================================
// Local Backend Implementation
// =============================================================================

const STORE_KEY = 'local-store';

/**
 * Local storage backend for M1 temporal data
 *
 * Automatically persists to disk/IndexedDB based on environment.
 * Data survives process restarts and page refreshes.
 */
export class LocalBackend implements StorageBackendAdapter {
  private readonly store = new Map<string, LocalEntry>();
  private readonly defaultTtlMs: number;
  private readonly namespace: string;
  private readonly persistenceAdapter: PersistenceAdapter<Map<string, LocalEntry>> | null;
  private readonly persistFn?: (data: Map<string, LocalEntry>) => Promise<void>;
  private readonly loadFn?: () => Promise<Map<string, LocalEntry> | null>;
  private readonly persistenceDebounceMs: number;
  private initialized = false;
  private gcIntervalId?: ReturnType<typeof setInterval>;
  private persistenceTimeout?: ReturnType<typeof setTimeout>;
  private persistencePending = false;

  constructor(options: LocalBackendOptions = {}) {
    this.namespace = options.namespace ?? 'uess';
    this.defaultTtlMs = options.defaultTtlMs ?? 7 * 24 * 60 * 60 * 1000; // 7 days
    this.persistenceDebounceMs = options.persistenceDebounceMs ?? 100;

    // Legacy custom persistence functions
    this.persistFn = options.persist;
    this.loadFn = options.load;

    // Set up automatic persistence (unless disabled or using legacy functions)
    if (!options.disablePersistence && !options.persist && !options.load) {
      this.persistenceAdapter = createPersistenceAdapter<Map<string, LocalEntry>>(
        `local-${this.namespace}`,
        {
          forceAdapter: options.forceAdapter,
          baseDir: options.persistenceDir,
        }
      );
    } else {
      this.persistenceAdapter = null;
    }

    // Start garbage collection
    if (options.gcIntervalMs !== 0) {
      const gcInterval = options.gcIntervalMs ?? 60 * 60 * 1000; // 1 hour
      this.gcIntervalId = setInterval(() => {
        void this.gc();
      }, gcInterval);
    }
  }

  /**
   * Initialize backend (load persisted data)
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    // Try loading from persistence adapter first
    if (this.persistenceAdapter) {
      try {
        const loaded = await this.persistenceAdapter.load(STORE_KEY);
        if (loaded instanceof Map) {
          for (const [key, entry] of loaded.entries()) {
            this.store.set(key, entry);
          }
        }
      } catch (error) {
        console.warn('UESS LocalBackend: Failed to load persisted data:', error);
      }
    }
    // Fall back to legacy load function
    else if (this.loadFn) {
      const loaded = await this.loadFn();
      if (loaded) {
        for (const [key, entry] of loaded.entries()) {
          this.store.set(key, entry);
        }
      }
    }

    this.initialized = true;
  }

  /**
   * Get data by key
   */
  async get<T>(key: string): Promise<{ data: T; metadata: StorageMetadata } | null> {
    await this.init();

    const fullKey = this.prefixKey(key);
    const entry = this.store.get(fullKey) as LocalEntry<T> | undefined;

    if (!entry) return null;

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(fullKey);
      await this.persist();
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
    await this.init();

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

    const entry: LocalEntry<T> = {
      data,
      metadata: fullMetadata,
      expiresAt,
    };

    const fullKey = this.prefixKey(key);
    this.store.set(fullKey, entry);

    await this.persist();

    return fullMetadata;
  }

  /**
   * Delete data by key
   */
  async delete(key: string): Promise<boolean> {
    await this.init();

    const fullKey = this.prefixKey(key);
    const result = this.store.delete(fullKey);

    if (result) {
      await this.persist();
    }

    return result;
  }

  /**
   * Check if key exists
   */
  async has(key: string): Promise<boolean> {
    await this.init();

    const fullKey = this.prefixKey(key);
    const entry = this.store.get(fullKey);

    if (!entry) return false;

    // Check expiration
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(fullKey);
      return false;
    }

    return !entry.metadata.tombstone;
  }

  /**
   * List keys matching pattern
   */
  async keys(pattern?: string): Promise<string[]> {
    await this.init();

    const allKeys: string[] = [];
    const now = Date.now();
    const prefix = this.namespace + ':';

    for (const [fullKey, entry] of this.store.entries()) {
      // Skip expired
      if (entry.expiresAt && entry.expiresAt < now) continue;
      // Skip tombstones
      if (entry.metadata.tombstone) continue;
      // Skip other namespaces
      if (!fullKey.startsWith(prefix)) continue;

      const key = fullKey.slice(prefix.length);

      if (!pattern || this.matchPattern(key, pattern)) {
        allKeys.push(key);
      }
    }

    return allKeys;
  }

  /**
   * Clear all data in namespace
   */
  async clear(): Promise<void> {
    await this.init();

    const prefix = this.namespace + ':';
    const keysToDelete: string[] = [];

    for (const key of this.store.keys()) {
      if (key.startsWith(prefix)) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      this.store.delete(key);
    }

    await this.persist();
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
    await this.init();

    let itemCount = 0;
    let totalSizeBytes = 0;
    let oldestItem: number | undefined;
    let newestItem: number | undefined;
    const now = Date.now();
    const prefix = this.namespace + ':';

    for (const [key, entry] of this.store.entries()) {
      if (!key.startsWith(prefix)) continue;
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
  async gc(): Promise<number> {
    await this.init();

    let removed = 0;
    const now = Date.now();
    const prefix = this.namespace + ':';

    for (const [key, entry] of this.store.entries()) {
      if (!key.startsWith(prefix)) continue;

      if (entry.expiresAt && entry.expiresAt < now) {
        this.store.delete(key);
        removed++;
      }
    }

    if (removed > 0) {
      await this.persist();
    }

    return removed;
  }

  /**
   * Stop garbage collection timer and flush pending persistence
   */
  async dispose(): Promise<void> {
    // Clear GC interval
    if (this.gcIntervalId) {
      clearInterval(this.gcIntervalId);
      this.gcIntervalId = undefined;
    }

    // Clear debounce timeout
    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
      this.persistenceTimeout = undefined;
    }

    // Flush any pending persistence immediately
    if (this.persistencePending) {
      await this.persistImmediate();
    }
  }

  /**
   * Force immediate persistence (bypassing debounce)
   */
  async flush(): Promise<void> {
    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
      this.persistenceTimeout = undefined;
    }
    await this.persistImmediate();
  }

  /**
   * Prefix key with namespace
   */
  private prefixKey(key: string): string {
    return `${this.namespace}:${key}`;
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
    return 'cid:' + Array.from(hashBytes).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  /**
   * Schedule persistence (debounced to avoid excessive writes)
   */
  private async persist(): Promise<void> {
    this.persistencePending = true;

    // Debounce persistence
    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
    }

    this.persistenceTimeout = setTimeout(() => {
      void this.persistImmediate();
    }, this.persistenceDebounceMs);
  }

  /**
   * Perform immediate persistence
   */
  private async persistImmediate(): Promise<void> {
    this.persistencePending = false;

    // Use built-in persistence adapter
    if (this.persistenceAdapter) {
      try {
        await this.persistenceAdapter.save(STORE_KEY, this.store);
      } catch (error) {
        console.warn('UESS LocalBackend: Failed to persist data:', error);
      }
      return;
    }

    // Fall back to legacy persist function
    if (this.persistFn) {
      await this.persistFn(this.store);
    }
  }
}

/**
 * Create a local backend with default options
 *
 * By default, automatically persists to:
 * - Browser: IndexedDB (with localStorage fallback)
 * - Node.js: File system (.uess-data/ directory)
 */
export function createLocalBackend(options?: LocalBackendOptions): LocalBackend {
  return new LocalBackend(options);
}

/**
 * Create a local backend with persistence explicitly disabled
 * (useful for testing or when persistence is handled externally)
 */
export function createEphemeralLocalBackend(options?: Omit<LocalBackendOptions, 'disablePersistence'>): LocalBackend {
  return new LocalBackend({ ...options, disablePersistence: true });
}

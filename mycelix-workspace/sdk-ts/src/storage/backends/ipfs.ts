// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS IPFS Backend
 *
 * Content-addressed storage via IPFS for N2+ public data.
 * Index is automatically persisted to survive restarts.
 *
 * @see docs/architecture/uess/UESS-01-CORE.md §3.4
 */

import { computeCID } from './memory.js';
import { createPersistenceAdapter, type PersistenceAdapter } from './persistence.js';

import type { StorageMetadata } from '../types.js';
import type { StorageBackendAdapter } from './memory.js';

// =============================================================================
// IPFS Backend Configuration
// =============================================================================

/**
 * Configuration for IPFS backend
 */
export interface IPFSBackendConfig {
  /** IPFS gateway URL for reads (default: 'https://ipfs.io') */
  gatewayUrl?: string;

  /** IPFS API URL for writes (default: 'http://localhost:5001') */
  apiUrl?: string;

  /** Whether to pin content (default: true) */
  pinContent?: boolean;

  /** Timeout for IPFS operations in ms (default: 30000) */
  timeoutMs?: number;

  /** Number of retries for failed operations (default: 3) */
  retries?: number;

  /** Enable local index for key→CID mapping */
  enableLocalIndex?: boolean;

  /** Local cache TTL in ms (default: 300000 = 5 min) */
  localCacheTtlMs?: number;

  /** Disable index persistence (default: false) */
  disableIndexPersistence?: boolean;

  /** Namespace for index persistence (default: 'ipfs') */
  persistenceNamespace?: string;
}

const DEFAULT_IPFS_CONFIG: Required<IPFSBackendConfig> = {
  gatewayUrl: 'https://ipfs.io',
  apiUrl: 'http://localhost:5001',
  pinContent: true,
  timeoutMs: 30000,
  retries: 3,
  enableLocalIndex: true,
  localCacheTtlMs: 300000,
  disableIndexPersistence: false,
  persistenceNamespace: 'ipfs',
};

// =============================================================================
// Types
// =============================================================================

/**
 * IPFS add response
 */
interface IPFSAddResponse {
  Hash: string;
  Size: string;
  Name?: string;
}

/**
 * Local index entry mapping key to IPFS CID
 */
interface IndexEntry {
  ipfsCid: string;
  localCid: string;
  metadata: StorageMetadata;
  addedAt: number;
}

/**
 * Cache entry for retrieved data
 */
interface CacheEntry<T> {
  data: T;
  metadata: StorageMetadata;
  cachedAt: number;
}

// =============================================================================
// IPFS Backend Implementation
// =============================================================================

const INDEX_STORE_KEY = 'ipfs-index';

/**
 * IPFS Backend - Content-addressed storage via IPFS
 *
 * Features:
 * - Native content addressing via IPFS CIDs
 * - Gateway reads for performance
 * - API writes with optional pinning
 * - Local index for key→CID mapping (automatically persisted)
 */
export class IPFSBackend implements StorageBackendAdapter {
  private readonly config: Required<IPFSBackendConfig>;
  private readonly index: Map<string, IndexEntry> = new Map();
  private readonly cache: Map<string, CacheEntry<unknown>> = new Map();
  private readonly indexPersistence: PersistenceAdapter<Map<string, IndexEntry>> | null;
  private disposed = false;
  private initialized = false;
  private indexDirty = false;
  private persistenceTimeout?: ReturnType<typeof setTimeout>;

  constructor(config: IPFSBackendConfig = {}) {
    this.config = { ...DEFAULT_IPFS_CONFIG, ...config };

    // Set up index persistence
    if (this.config.enableLocalIndex && !this.config.disableIndexPersistence) {
      this.indexPersistence = createPersistenceAdapter<Map<string, IndexEntry>>(
        `ipfs-index-${this.config.persistenceNamespace}`
      );
    } else {
      this.indexPersistence = null;
    }
  }

  /**
   * Initialize backend (load persisted index)
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    if (this.indexPersistence) {
      try {
        const loaded = await this.indexPersistence.load(INDEX_STORE_KEY);
        if (loaded instanceof Map) {
          for (const [key, entry] of loaded.entries()) {
            this.index.set(key, entry);
          }
        }
      } catch (error) {
        console.warn('UESS IPFSBackend: Failed to load persisted index:', error);
      }
    }

    this.initialized = true;
  }

  /**
   * Ensure backend is initialized before operations
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.init();
    }
  }

  /**
   * Store data in IPFS
   */
  async set<T>(
    key: string,
    data: T,
    metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'>
  ): Promise<StorageMetadata> {
    this.checkDisposed();
    await this.ensureInitialized();

    const serializedData = JSON.stringify(data);
    const sizeBytes = new TextEncoder().encode(serializedData).length;
    const localCid = computeCID(data);

    // Create wrapper with metadata for IPFS
    const wrapper = {
      data: serializedData,
      metadata: {
        ...metadata,
        localCid,
        storedAt: Date.now(),
      },
    };

    // Add to IPFS
    const ipfsCid = await this.addToIPFS(JSON.stringify(wrapper));

    const fullMetadata: StorageMetadata = {
      ...metadata,
      cid: localCid,
      sizeBytes,
      storedAt: Date.now(),
      // Store IPFS CID in custom field for retrieval
      ipfsCid,
    } as StorageMetadata & { ipfsCid: string };

    // Update local index
    if (this.config.enableLocalIndex) {
      this.index.set(key, {
        ipfsCid,
        localCid,
        metadata: fullMetadata,
        addedAt: Date.now(),
      });
      this.schedulePersistIndex();
    }

    // Update cache
    this.cache.set(key, {
      data,
      metadata: fullMetadata,
      cachedAt: Date.now(),
    });

    return fullMetadata;
  }

  /**
   * Retrieve data from IPFS
   */
  async get<T>(key: string): Promise<{ data: T; metadata: StorageMetadata } | null> {
    this.checkDisposed();
    await this.ensureInitialized();

    // Check cache first
    const cached = this.getCached<T>(key);
    if (cached) {
      return cached;
    }

    // Look up in index
    const indexEntry = this.index.get(key);
    if (!indexEntry) {
      return null;
    }

    // Check expiration
    if (indexEntry.metadata.expiresAt && indexEntry.metadata.expiresAt < Date.now()) {
      this.index.delete(key);
      this.schedulePersistIndex();
      return null;
    }

    // Check tombstone
    if (indexEntry.metadata.tombstone) {
      return null;
    }

    // Fetch from IPFS
    const wrapper = await this.fetchFromIPFS<{
      data: string;
      metadata: StorageMetadata;
    }>(indexEntry.ipfsCid);

    if (!wrapper) {
      return null;
    }

    const data = JSON.parse(wrapper.data) as T;

    // Update cache
    this.cache.set(key, {
      data,
      metadata: indexEntry.metadata,
      cachedAt: Date.now(),
    });

    return { data, metadata: indexEntry.metadata };
  }

  /**
   * Check if key exists in index
   */
  async has(key: string): Promise<boolean> {
    this.checkDisposed();
    await this.ensureInitialized();

    const indexEntry = this.index.get(key);
    if (!indexEntry) {
      return false;
    }

    // Check expiration
    if (indexEntry.metadata.expiresAt && indexEntry.metadata.expiresAt < Date.now()) {
      this.index.delete(key);
      this.schedulePersistIndex();
      return false;
    }

    // Check tombstone
    if (indexEntry.metadata.tombstone) {
      return false;
    }

    return true;
  }

  /**
   * Delete entry (creates tombstone in index, content remains in IPFS)
   */
  async delete(key: string): Promise<boolean> {
    this.checkDisposed();
    await this.ensureInitialized();

    const indexEntry = this.index.get(key);
    if (!indexEntry) {
      return false;
    }

    // Mark as tombstone (IPFS content is immutable, so we just mark index)
    indexEntry.metadata.tombstone = true;
    this.index.set(key, indexEntry);
    this.schedulePersistIndex();

    // Remove from cache
    this.cache.delete(key);

    return true;
  }

  /**
   * List keys in index
   */
  async keys(pattern?: string): Promise<string[]> {
    this.checkDisposed();
    await this.ensureInitialized();

    const allKeys = Array.from(this.index.keys());

    if (!pattern) {
      return allKeys.filter(key => {
        const entry = this.index.get(key);
        return entry && !entry.metadata.tombstone;
      });
    }

    const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
    return allKeys.filter(key => {
      const entry = this.index.get(key);
      return entry && !entry.metadata.tombstone && regex.test(key);
    });
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
    this.checkDisposed();
    await this.ensureInitialized();

    let totalSize = 0;
    let oldest: number | undefined;
    let newest: number | undefined;
    let count = 0;

    for (const entry of this.index.values()) {
      if (entry.metadata.tombstone) continue;

      count++;
      totalSize += entry.metadata.sizeBytes;

      const timestamp = entry.metadata.storedAt;
      if (oldest === undefined || timestamp < oldest) {
        oldest = timestamp;
      }
      if (newest === undefined || timestamp > newest) {
        newest = timestamp;
      }
    }

    return {
      itemCount: count,
      totalSizeBytes: totalSize,
      oldestItem: oldest,
      newestItem: newest,
    };
  }

  /**
   * Clear local index (IPFS content remains)
   */
  async clear(): Promise<void> {
    this.checkDisposed();
    await this.ensureInitialized();

    this.index.clear();
    this.cache.clear();
    this.schedulePersistIndex();
  }

  /**
   * Run garbage collection on cache and index
   */
  async gc(): Promise<number> {
    if (this.disposed) return 0;
    await this.ensureInitialized();

    let removed = 0;
    const now = Date.now();
    let indexModified = false;

    // Clean cache
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.cachedAt > this.config.localCacheTtlMs) {
        this.cache.delete(key);
        removed++;
      }
    }

    // Clean expired index entries
    for (const [key, entry] of this.index.entries()) {
      if (entry.metadata.expiresAt && entry.metadata.expiresAt < now) {
        this.index.delete(key);
        removed++;
        indexModified = true;
      }
    }

    if (indexModified) {
      this.schedulePersistIndex();
    }

    return removed;
  }

  /**
   * Clean up resources
   */
  async dispose(): Promise<void> {
    this.disposed = true;

    // Clear persistence timeout
    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
      this.persistenceTimeout = undefined;
    }

    // Flush pending index changes
    if (this.indexDirty) {
      await this.persistIndexImmediate();
    }

    this.index.clear();
    this.cache.clear();
  }

  /**
   * Force immediate index persistence
   */
  async flush(): Promise<void> {
    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
      this.persistenceTimeout = undefined;
    }
    await this.persistIndexImmediate();
  }

  /**
   * Schedule index persistence (debounced)
   */
  private schedulePersistIndex(): void {
    if (!this.indexPersistence) return;

    this.indexDirty = true;

    if (this.persistenceTimeout) {
      clearTimeout(this.persistenceTimeout);
    }

    this.persistenceTimeout = setTimeout(() => {
      void this.persistIndexImmediate();
    }, 100); // 100ms debounce
  }

  /**
   * Persist index immediately
   */
  private async persistIndexImmediate(): Promise<void> {
    if (!this.indexPersistence || !this.indexDirty) return;

    this.indexDirty = false;

    try {
      await this.indexPersistence.save(INDEX_STORE_KEY, this.index);
    } catch (error) {
      console.warn('UESS IPFSBackend: Failed to persist index:', error);
    }
  }

  // ===========================================================================
  // IPFS-Specific Methods
  // ===========================================================================

  /**
   * Get data directly by IPFS CID
   */
  async getByIPFSCid<T>(ipfsCid: string): Promise<T | null> {
    this.checkDisposed();

    const wrapper = await this.fetchFromIPFS<{
      data: string;
    }>(ipfsCid);

    if (!wrapper) {
      return null;
    }

    return JSON.parse(wrapper.data) as T;
  }

  /**
   * Pin content to ensure persistence
   */
  async pin(key: string): Promise<boolean> {
    this.checkDisposed();

    const indexEntry = this.index.get(key);
    if (!indexEntry) {
      return false;
    }

    return this.pinCID(indexEntry.ipfsCid);
  }

  /**
   * Unpin content
   */
  async unpin(key: string): Promise<boolean> {
    this.checkDisposed();

    const indexEntry = this.index.get(key);
    if (!indexEntry) {
      return false;
    }

    return this.unpinCID(indexEntry.ipfsCid);
  }

  /**
   * Get IPFS CID for a key
   */
  getIPFSCid(key: string): string | null {
    const indexEntry = this.index.get(key);
    return indexEntry?.ipfsCid ?? null;
  }

  /**
   * Resolve IPFS path/CID to data
   */
  async resolve<T>(ipfsPath: string): Promise<T | null> {
    this.checkDisposed();
    return this.fetchFromIPFS<T>(ipfsPath);
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private checkDisposed(): void {
    if (this.disposed) {
      throw new Error('IPFSBackend has been disposed');
    }
  }

  private getCached<T>(key: string): { data: T; metadata: StorageMetadata } | null {
    const cached = this.cache.get(key);
    if (!cached) {
      return null;
    }

    if (Date.now() - cached.cachedAt > this.config.localCacheTtlMs) {
      this.cache.delete(key);
      return null;
    }

    if (cached.metadata.expiresAt && cached.metadata.expiresAt < Date.now()) {
      this.cache.delete(key);
      return null;
    }

    if (cached.metadata.tombstone) {
      return null;
    }

    return {
      data: cached.data as T,
      metadata: cached.metadata,
    };
  }

  private async addToIPFS(content: string): Promise<string> {
    const formData = new FormData();
    formData.append('file', new Blob([content], { type: 'application/json' }));

    const url = `${this.config.apiUrl}/api/v0/add?pin=${this.config.pinContent}`;

    const response = await this.fetchWithRetry(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`IPFS add failed: ${response.statusText}`);
    }

    const result = await response.json() as IPFSAddResponse;
    return result.Hash;
  }

  private async fetchFromIPFS<T>(cid: string): Promise<T | null> {
    const url = `${this.config.gatewayUrl}/ipfs/${cid}`;

    try {
      const response = await this.fetchWithRetry(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`IPFS fetch failed: ${response.statusText}`);
      }

      return response.json() as Promise<T>;
    } catch (error) {
      console.error(`Failed to fetch from IPFS: ${cid}`, error);
      return null;
    }
  }

  private async pinCID(cid: string): Promise<boolean> {
    const url = `${this.config.apiUrl}/api/v0/pin/add?arg=${cid}`;

    try {
      const response = await this.fetchWithRetry(url, { method: 'POST' });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async unpinCID(cid: string): Promise<boolean> {
    const url = `${this.config.apiUrl}/api/v0/pin/rm?arg=${cid}`;

    try {
      const response = await this.fetchWithRetry(url, { method: 'POST' });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async fetchWithRetry(
    url: string,
    options: RequestInit
  ): Promise<Response> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.config.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

        const response = await fetch(url, {
          ...options,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        return response;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt < this.config.retries - 1) {
          await this.sleep(Math.pow(2, attempt) * 100);
        }
      }
    }

    throw lastError ?? new Error('IPFS operation failed');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create an IPFS backend instance
 */
export function createIPFSBackend(config?: IPFSBackendConfig): IPFSBackend {
  return new IPFSBackend(config);
}

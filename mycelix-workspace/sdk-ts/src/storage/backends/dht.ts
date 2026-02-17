/**
 * UESS DHT Backend
 *
 * Holochain DHT storage for M2/M3 persistent data.
 * Connects to the epistemic_storage zome for decentralized,
 * content-addressed storage with E/N/M classification.
 *
 * @see docs/architecture/uess/UESS-01-CORE.md §3.3
 */

import { computeCID } from './memory.js';

import type { StorageBackendAdapter } from './memory.js';
import type { MycelixClient } from '../../client/index.js';
import type { EpistemicClassification } from '../../epistemic/types.js';
import type { StorageMetadata, SchemaIdentity } from '../types.js';

// =============================================================================
// DHT Backend Configuration
// =============================================================================

/**
 * Generic Holochain client interface
 * Compatible with @holochain/client AppWebsocket
 */
export interface HolochainClient {
  callZome<T>(request: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

/**
 * Configuration for DHT backend
 */
export interface DHTBackendConfig {
  /** Mycelix client for Holochain access (or generic HolochainClient) */
  client: MycelixClient | HolochainClient;

  /** Zome name for storage operations (default: 'epistemic_storage') */
  zomeName?: string;

  /** DNA name/role (default: 'epistemic_storage') */
  dnaRole?: string;

  /** Agent ID for created_by field (default: 'anonymous') */
  agentId?: string;

  /** Timeout for DHT operations in ms (default: 30000) */
  timeoutMs?: number;

  /** Number of retries for failed operations (default: 3) */
  retries?: number;

  /** Local cache for read optimization */
  enableLocalCache?: boolean;

  /** Local cache TTL in ms (default: 60000) */
  localCacheTtlMs?: number;
}

const DEFAULT_DHT_CONFIG: Omit<Required<DHTBackendConfig>, 'client'> = {
  zomeName: 'epistemic_storage',
  dnaRole: 'epistemic_storage',
  agentId: 'anonymous',
  timeoutMs: 30000,
  retries: 3,
  enableLocalCache: true,
  localCacheTtlMs: 60000,
};

// =============================================================================
// Zome Input/Output Types (matching Rust zome)
// =============================================================================

/**
 * Zome storage metadata (snake_case to match Rust)
 */
interface ZomeStorageMetadata {
  cid: string;
  classification: {
    empirical: number;
    normative: number;
    materiality: number;
  };
  schema: {
    id: string;
    version: string;
    family: string | null;
  };
  stored_at: number;
  modified_at: number | null;
  version: number;
  expires_at: number | null;
  size_bytes: number;
  created_by: string;
  tombstone: boolean;
  retracted_by: string | null;
}

/**
 * Zome epistemic entry (matches Rust EpistemicEntry)
 */
interface ZomeEpistemicEntry {
  key: string;
  data: string;
  metadata: ZomeStorageMetadata;
}

/**
 * Zome storage stats (matches Rust StorageStats)
 */
interface ZomeStorageStats {
  item_count: number;
  total_size_bytes: number;
  oldest_item: number | null;
  newest_item: number | null;
}

/**
 * Zome replication status (matches Rust ReplicationStatus)
 */
interface ZomeReplicationStatus {
  key: string;
  holder_count: number;
  target_holders: number;
  is_replicated: boolean;
}

/**
 * Local cache entry
 */
interface CacheEntry<T> {
  data: T;
  metadata: StorageMetadata;
  cachedAt: number;
}

// =============================================================================
// DHT Backend Implementation
// =============================================================================

/**
 * DHT Backend - Stores data in Holochain DHT
 *
 * Features:
 * - Content-addressed storage via CID
 * - Automatic replication via DHT
 * - Local read cache for performance
 * - Retry logic for network failures
 */
export class DHTBackend implements StorageBackendAdapter {
  private readonly config: Required<DHTBackendConfig>;
  private readonly cache: Map<string, CacheEntry<unknown>> = new Map();
  private disposed = false;

  constructor(config: DHTBackendConfig) {
    this.config = { ...DEFAULT_DHT_CONFIG, ...config } as Required<DHTBackendConfig>;
  }

  /**
   * Store data in DHT with E/N/M classification
   */
  async set<T>(
    key: string,
    data: T,
    metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'>
  ): Promise<StorageMetadata> {
    this.checkDisposed();

    const serializedData = JSON.stringify(data);
    const sizeBytes = new TextEncoder().encode(serializedData).length;
    const cid = `cid:${computeCID(data)}`;
    const now = Date.now();

    // Build zome metadata (snake_case for Rust)
    const zomeMetadata: ZomeStorageMetadata = {
      cid,
      classification: {
        empirical: metadata.classification.empirical,
        normative: metadata.classification.normative,
        materiality: metadata.classification.materiality,
      },
      schema: {
        id: metadata.schema.id,
        version: metadata.schema.version,
        family: metadata.schema.family ?? null,
      },
      stored_at: metadata.storedAt ?? now,
      modified_at: metadata.modifiedAt ?? null,
      version: metadata.version ?? 1,
      expires_at: metadata.expiresAt ?? null,
      size_bytes: sizeBytes,
      created_by: metadata.createdBy ?? this.config.agentId,
      tombstone: metadata.tombstone ?? false,
      retracted_by: metadata.retractedBy ?? null,
    };

    // StoreInput format matching Rust zome
    await this.callZomeWithRetry('store_epistemic_entry', {
      key,
      entry: {
        data: serializedData,
        metadata: zomeMetadata,
      },
    });

    // Build SDK metadata (camelCase)
    const fullMetadata: StorageMetadata = {
      cid,
      classification: metadata.classification,
      schema: metadata.schema,
      storedAt: zomeMetadata.stored_at,
      modifiedAt: zomeMetadata.modified_at ?? undefined,
      version: zomeMetadata.version,
      expiresAt: zomeMetadata.expires_at ?? undefined,
      sizeBytes,
      createdBy: zomeMetadata.created_by,
      tombstone: zomeMetadata.tombstone,
      retractedBy: zomeMetadata.retracted_by ?? undefined,
    };

    // Update local cache
    if (this.config.enableLocalCache) {
      this.cache.set(key, {
        data,
        metadata: fullMetadata,
        cachedAt: Date.now(),
      });
    }

    return fullMetadata;
  }

  /**
   * Retrieve data from DHT
   */
  async get<T>(key: string): Promise<{ data: T; metadata: StorageMetadata } | null> {
    this.checkDisposed();

    // Check local cache first
    if (this.config.enableLocalCache) {
      const cached = this.getCached<T>(key);
      if (cached) {
        return cached;
      }
    }

    // Query DHT - returns ZomeEpistemicEntry or null
    const result = await this.callZomeWithRetry<ZomeEpistemicEntry | null>('get_epistemic_entry', { key });

    if (!result) {
      return null;
    }

    const data = JSON.parse(result.data) as T;
    const metadata = this.zomeMetadataToSdk(result.metadata);

    // Update local cache
    if (this.config.enableLocalCache) {
      this.cache.set(key, {
        data,
        metadata,
        cachedAt: Date.now(),
      });
    }

    return { data, metadata };
  }

  /**
   * Check if key exists in DHT
   */
  async has(key: string): Promise<boolean> {
    this.checkDisposed();

    // Check cache first
    if (this.config.enableLocalCache) {
      const cached = this.cache.get(key);
      if (cached && !this.isCacheExpired(cached)) {
        const meta = cached.metadata;
        if (!meta.tombstone && (!meta.expiresAt || meta.expiresAt > Date.now())) {
          return true;
        }
      }
    }

    const result = await this.callZomeWithRetry<boolean>('has_epistemic_entry', { key });
    return result ?? false;
  }

  /**
   * Delete entry from DHT (creates tombstone)
   *
   * Note: E3+ immutable entries cannot be deleted.
   * Returns false if the entry doesn't exist or cannot be deleted.
   */
  async delete(key: string): Promise<boolean> {
    this.checkDisposed();

    try {
      const result = await this.callZomeWithRetry<boolean>('delete_epistemic_entry', { key });

      // Remove from local cache
      if (this.config.enableLocalCache) {
        this.cache.delete(key);
      }

      return result ?? false;
    } catch (error) {
      // E3+ immutable entries throw an error when deletion is attempted
      if (error instanceof Error && this.isImmutabilityError(error)) {
        return false;
      }
      throw error;
    }
  }

  /**
   * List keys in DHT (with optional pattern filter)
   */
  async keys(pattern?: string): Promise<string[]> {
    this.checkDisposed();

    const allKeys = await this.callZomeWithRetry<string[]>('list_epistemic_keys', {
      pattern: pattern ?? null,
    });

    return allKeys ?? [];
  }

  /**
   * Get DHT storage statistics
   */
  async stats(): Promise<{
    itemCount: number;
    totalSizeBytes: number;
    oldestItem?: number;
    newestItem?: number;
  }> {
    this.checkDisposed();

    // Zome returns ZomeStorageStats with snake_case
    const dhtStats = await this.callZomeWithRetry<ZomeStorageStats>('get_storage_stats', null);

    return {
      itemCount: dhtStats?.item_count ?? 0,
      totalSizeBytes: dhtStats?.total_size_bytes ?? 0,
      oldestItem: dhtStats?.oldest_item ?? undefined,
      newestItem: dhtStats?.newest_item ?? undefined,
    };
  }

  /**
   * Clear all entries (admin operation)
   *
   * Note: This removes DHT links but doesn't delete entries
   * (they remain in DHT but are no longer indexed).
   */
  async clear(): Promise<void> {
    this.checkDisposed();

    await this.callZomeWithRetry('clear_epistemic_storage', null);

    // Clear local cache
    if (this.config.enableLocalCache) {
      this.cache.clear();
    }
  }

  /**
   * Run garbage collection
   */
  async gc(): Promise<number> {
    this.checkDisposed();

    // Garbage collection in DHT is handled by Holochain
    // This just cleans up local cache
    if (!this.config.enableLocalCache) {
      return 0;
    }

    let removed = 0;
    const now = Date.now();

    for (const [key, entry] of this.cache.entries()) {
      if (this.isCacheExpired(entry)) {
        this.cache.delete(key);
        removed++;
      } else if (entry.metadata.expiresAt && entry.metadata.expiresAt < now) {
        this.cache.delete(key);
        removed++;
      }
    }

    return removed;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.disposed = true;
    this.cache.clear();
  }

  // ===========================================================================
  // DHT-Specific Methods
  // ===========================================================================

  /**
   * Query entries by CID across DHT
   *
   * @param cid Content identifier (with or without 'cid:' prefix)
   */
  async getByCID<T>(cid: string): Promise<{ key: string; data: T; metadata: StorageMetadata } | null> {
    this.checkDisposed();

    // Normalize CID to include prefix
    const normalizedCid = cid.startsWith('cid:') ? cid : `cid:${cid}`;

    // Zome returns ZomeEpistemicEntry directly (includes key in entry)
    const result = await this.callZomeWithRetry<ZomeEpistemicEntry | null>('get_by_cid', {
      cid: normalizedCid,
    });

    if (!result) {
      return null;
    }

    const data = JSON.parse(result.data) as T;
    const metadata = this.zomeMetadataToSdk(result.metadata);

    return { key: result.key, data, metadata };
  }

  /**
   * Get replication status for a key
   */
  async getReplicationStatus(key: string): Promise<{
    holders: number;
    minRequired: number;
    isHealthy: boolean;
  }> {
    this.checkDisposed();

    // Zome returns ZomeReplicationStatus with snake_case
    const status = await this.callZomeWithRetry<ZomeReplicationStatus>('get_replication_status', { key });

    return {
      holders: status?.holder_count ?? 0,
      minRequired: status?.target_holders ?? 1,
      isHealthy: status?.is_replicated ?? false,
    };
  }

  /**
   * Force replication to ensure data durability
   */
  async ensureReplication(key: string, minHolders: number): Promise<boolean> {
    this.checkDisposed();

    const result = await this.callZomeWithRetry<boolean>('ensure_replication', {
      key,
      min_holders: minHolders,  // snake_case for Rust
    });

    return result ?? false;
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private checkDisposed(): void {
    if (this.disposed) {
      throw new Error('DHTBackend has been disposed');
    }
  }

  private getCached<T>(key: string): { data: T; metadata: StorageMetadata } | null {
    const cached = this.cache.get(key);
    if (!cached) {
      return null;
    }

    if (this.isCacheExpired(cached)) {
      this.cache.delete(key);
      return null;
    }

    // Check if actual data expired
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

  private isCacheExpired(entry: CacheEntry<unknown>): boolean {
    return Date.now() - entry.cachedAt > this.config.localCacheTtlMs;
  }

  private async callZomeWithRetry<R>(
    fnName: string,
    payload: unknown
  ): Promise<R | null> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.config.retries; attempt++) {
      try {
        const result = await this.callZome<R>(fnName, payload);
        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on validation errors - they will fail every time
        if (this.isValidationError(lastError)) {
          throw lastError;
        }

        // Wait before retry with exponential backoff
        if (attempt < this.config.retries - 1) {
          await this.sleep(Math.pow(2, attempt) * 100);
        }
      }
    }

    console.error(`DHT operation ${fnName} failed after ${this.config.retries} retries:`, lastError);
    throw lastError;
  }

  /**
   * Check if error is a validation error (should not retry)
   */
  private isValidationError(error: Error): boolean {
    const message = error.message.toLowerCase();
    return (
      message.includes('invalid') ||
      message.includes('validation') ||
      message.includes('immutable') ||
      message.includes('e3+') ||
      message.includes('cannot delete')
    );
  }

  /**
   * Check if error is an immutability error (E3+ entries)
   */
  private isImmutabilityError(error: Error): boolean {
    const message = error.message.toLowerCase();
    return (
      message.includes('e3+') ||
      message.includes('immutable') ||
      message.includes('cannot delete')
    );
  }

  private async callZome<R>(fnName: string, payload: unknown): Promise<R> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const result = await this.config.client.callZome<R>({
        role_name: this.config.dnaRole,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });

      return result;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Convert zome metadata (snake_case) to SDK metadata (camelCase)
   */
  private zomeMetadataToSdk(zome: ZomeStorageMetadata): StorageMetadata {
    return {
      cid: zome.cid,
      classification: {
        empirical: zome.classification.empirical,
        normative: zome.classification.normative,
        materiality: zome.classification.materiality,
      } as EpistemicClassification,
      schema: {
        id: zome.schema.id,
        version: zome.schema.version,
        family: zome.schema.family ?? undefined,
      } as SchemaIdentity,
      storedAt: zome.stored_at,
      modifiedAt: zome.modified_at ?? undefined,
      version: zome.version,
      expiresAt: zome.expires_at ?? undefined,
      sizeBytes: zome.size_bytes,
      createdBy: zome.created_by,
      tombstone: zome.tombstone,
      retractedBy: zome.retracted_by ?? undefined,
    };
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a DHT backend instance
 */
export function createDHTBackend(config: DHTBackendConfig): DHTBackend {
  return new DHTBackend(config);
}

/**
 * UESS Epistemic Storage Implementation
 *
 * Reference implementation of the Unified Epistemic Storage System.
 * Routes data to appropriate backends based on E/N/M classification.
 *
 * @see docs/architecture/uess/UESS-01-CORE.md §5
 */

import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClassification,
} from '../epistemic/types.js';
import { MycelixError, ErrorCode, ValidationError } from '../errors.js';
import { hash, secureUUID } from '../security/index.js';
import { DHTBackend, type DHTBackendConfig, type HolochainClient } from './backends/dht.js';
import { IPFSBackend, type IPFSBackendConfig } from './backends/ipfs.js';
import { LocalBackend } from './backends/local.js';
import { MemoryBackend, type StorageBackendAdapter } from './backends/memory.js';
import {
  StorageMetricsCollector,
  StorageTracer,
  StorageHealthChecker,
  createBackendHealthChecks,
  type BackendType,
  type StorageMetricsSnapshot,
  type HealthCheckResult,
} from './observability.js';
import { StorageRouter, type ClassificationRouter, DEFAULT_ROUTER_CONFIG } from './router.js';
import {
  type StorageReceipt,
  type StoredData,
  type StorageInfo,
  type StoreOptions,
  type RetrieveOptions,
  type UpdateOptions,
  type DeleteOptions,
  type AccessCapability,
  type ReclassificationReason,
  type EpistemicQuery,
  type QueryResult,
  type VerificationResult,
  type StorageStats,
  type StorageBackend,
  type StorageMetadata,
  type SchemaIdentity,
  MutabilityMode,
  AccessControlMode,
} from './types.js';

// =============================================================================
// Errors
// =============================================================================

/**
 * Storage-specific error
 */
export class StorageError extends MycelixError {
  constructor(
    message: string,
    code: ErrorCode = ErrorCode.STORAGE_ERROR,
    context?: Record<string, unknown>
  ) {
    super(message, code, context);
    this.name = 'StorageError';
  }
}

/**
 * Error when attempting to modify immutable data
 */
export class ImmutableError extends StorageError {
  constructor(key: string) {
    super(
      `Cannot modify immutable data: ${key}`,
      ErrorCode.STORAGE_IMMUTABLE,
      { key }
    );
    this.name = 'ImmutableError';
  }
}

/**
 * Error when access is denied
 */
export class AccessDeniedError extends StorageError {
  constructor(key: string, requiredPermission: string) {
    super(
      `Access denied to ${key}: requires ${requiredPermission}`,
      ErrorCode.ACCESS_DENIED,
      { key, requiredPermission }
    );
    this.name = 'AccessDeniedError';
  }
}

/**
 * Error when invariant is violated
 */
export class InvariantViolationError extends StorageError {
  constructor(invariant: string, details: string) {
    super(
      `Invariant violation [${invariant}]: ${details}`,
      ErrorCode.INVARIANT_VIOLATION,
      { invariant, details }
    );
    this.name = 'InvariantViolationError';
  }
}

// =============================================================================
// Epistemic Storage Interface
// =============================================================================

/**
 * Unified Epistemic Storage Interface
 *
 * All storage operations go through this interface.
 * The system routes to appropriate backends based on classification.
 */
export interface EpistemicStorage {
  /**
   * Store data with epistemic classification
   */
  store<T>(
    key: string,
    data: T,
    classification: EpistemicClassification,
    options: StoreOptions
  ): Promise<StorageReceipt>;

  /**
   * Retrieve data by key
   */
  retrieve<T>(
    key: string,
    capability?: AccessCapability,
    options?: RetrieveOptions
  ): Promise<StoredData<T> | null>;

  /**
   * Retrieve by content address (CID)
   */
  retrieveByCID<T>(
    cid: string,
    capability?: AccessCapability,
    options?: RetrieveOptions
  ): Promise<StoredData<T> | null>;

  /**
   * Update data (behavior depends on E-level mutability)
   */
  update<T>(
    key: string,
    data: T,
    capability: AccessCapability,
    options?: UpdateOptions
  ): Promise<StorageReceipt>;

  /**
   * Delete/retract data (behavior depends on E-level)
   */
  delete(
    key: string,
    capability: AccessCapability,
    options?: DeleteOptions
  ): Promise<void>;

  /**
   * Reclassify data (upgrade only per INV-2)
   */
  reclassify(
    key: string,
    newClassification: Partial<EpistemicClassification>,
    reason: ReclassificationReason,
    capability: AccessCapability
  ): Promise<StorageReceipt>;

  /**
   * Get current classification and storage info
   */
  getStorageInfo(key: string): Promise<StorageInfo | null>;

  /**
   * Query by classification and metadata
   */
  query<T>(query: EpistemicQuery): Promise<QueryResult<T>>;

  /**
   * Get storage statistics
   */
  getStats(): Promise<StorageStats>;

  /**
   * Verify data integrity
   */
  verify(key: string): Promise<VerificationResult>;

  /**
   * Check if key exists
   */
  exists(key: string): Promise<boolean>;
}

// =============================================================================
// Configuration
// =============================================================================

/**
 * Epistemic storage configuration
 */
export interface EpistemicStorageConfig {
  /** Agent ID of the current user */
  agentId: string;

  /** Router configuration */
  router?: Partial<typeof DEFAULT_ROUTER_CONFIG>;

  /** Network size provider (for replication calculation) */
  networkSizeProvider?: () => number;

  /** Memory backend options */
  memoryBackend?: {
    defaultTtlMs?: number;
    gcIntervalMs?: number;
  };

  /** Local backend options */
  localBackend?: {
    namespace?: string;
    defaultTtlMs?: number;
    gcIntervalMs?: number;
  };

  /** DHT backend options (enables Holochain DHT for M2+ data) */
  dhtBackend?: {
    /** Holochain client instance */
    client: HolochainClient;
    /** Zome configuration */
    zomeName?: string;
    dnaRole?: string;
    /** Performance options */
    timeoutMs?: number;
    retries?: number;
    /** Local cache for DHT reads */
    enableLocalCache?: boolean;
    localCacheTtlMs?: number;
  };

  /** Enable observability (metrics, tracing, health checks) */
  observability?: {
    /** Enable metrics collection (default: false) */
    enableMetrics?: boolean;
    /** Enable tracing (default: false) */
    enableTracing?: boolean;
    /** Enable periodic health checks */
    healthCheckIntervalMs?: number;
  };

  /** IPFS backend options (enables IPFS for M3 immutable data) */
  ipfsBackend?: {
    /** IPFS gateway URL for reads (default: 'https://ipfs.io') */
    gatewayUrl?: string;
    /** IPFS API URL for writes (default: 'http://localhost:5001') */
    apiUrl?: string;
    /** Whether to pin content (default: true) */
    pinContent?: boolean;
    /** Timeout for IPFS operations in ms (default: 30000) */
    timeoutMs?: number;
    /** Number of retries (default: 3) */
    retries?: number;
    /** Enable local index for key→CID mapping (default: true) */
    enableLocalIndex?: boolean;
    /** Local cache TTL in ms (default: 300000 = 5 min) */
    localCacheTtlMs?: number;
  };
}

// =============================================================================
// Key-to-CID Index
// =============================================================================

interface IndexEntry {
  key: string;
  cid: string;
  classification: EpistemicClassification;
  schema: SchemaIdentity;
  backend: StorageBackend;
  version: number;
  storedAt: number;
  createdBy: string;
}

// =============================================================================
// Epistemic Storage Implementation
// =============================================================================

/**
 * Reference implementation of Epistemic Storage
 */
export class EpistemicStorageImpl implements EpistemicStorage {
  private readonly agentId: string;
  private readonly router: ClassificationRouter;
  private readonly memoryBackend: MemoryBackend;
  private readonly localBackend: LocalBackend;
  private readonly dhtBackend: DHTBackend | null;
  private readonly ipfsBackend: IPFSBackend | null;

  // Observability
  private readonly metrics: StorageMetricsCollector | null;
  private readonly tracer: StorageTracer | null;
  private readonly healthChecker: StorageHealthChecker | null;

  // Key → IndexEntry mapping (INV-8: dual-index)
  private readonly keyIndex = new Map<string, IndexEntry>();
  // CID → Key mapping for reverse lookup
  private readonly cidIndex = new Map<string, string>();

  // Statistics
  private retrievalTimes: number[] = [];
  private cacheHits = 0;
  private cacheMisses = 0;

  constructor(config: EpistemicStorageConfig) {
    this.agentId = config.agentId;

    // Configure router with backend availability
    // enableIpfs should be true only if ipfsBackend is configured
    const routerConfig = {
      ...DEFAULT_ROUTER_CONFIG,
      ...config.router,
      enableIpfs: config.ipfsBackend !== undefined,
    };

    this.router = new StorageRouter(
      routerConfig,
      config.networkSizeProvider ?? (() => 100)
    );

    this.memoryBackend = new MemoryBackend(config.memoryBackend);
    this.localBackend = new LocalBackend(config.localBackend);

    // Initialize DHT backend if configured
    if (config.dhtBackend) {
      // Only pass defined values to let DHTBackend use its defaults
      const dhtConfig: DHTBackendConfig = {
        client: config.dhtBackend.client,
        agentId: config.agentId,
      };
      if (config.dhtBackend.zomeName !== undefined) dhtConfig.zomeName = config.dhtBackend.zomeName;
      if (config.dhtBackend.dnaRole !== undefined) dhtConfig.dnaRole = config.dhtBackend.dnaRole;
      if (config.dhtBackend.timeoutMs !== undefined) dhtConfig.timeoutMs = config.dhtBackend.timeoutMs;
      if (config.dhtBackend.retries !== undefined) dhtConfig.retries = config.dhtBackend.retries;
      if (config.dhtBackend.enableLocalCache !== undefined) dhtConfig.enableLocalCache = config.dhtBackend.enableLocalCache;
      if (config.dhtBackend.localCacheTtlMs !== undefined) dhtConfig.localCacheTtlMs = config.dhtBackend.localCacheTtlMs;

      this.dhtBackend = new DHTBackend(dhtConfig);
    } else {
      this.dhtBackend = null;
    }

    // Initialize IPFS backend if configured
    if (config.ipfsBackend) {
      const ipfsConfig: IPFSBackendConfig = {};
      if (config.ipfsBackend.gatewayUrl !== undefined) ipfsConfig.gatewayUrl = config.ipfsBackend.gatewayUrl;
      if (config.ipfsBackend.apiUrl !== undefined) ipfsConfig.apiUrl = config.ipfsBackend.apiUrl;
      if (config.ipfsBackend.pinContent !== undefined) ipfsConfig.pinContent = config.ipfsBackend.pinContent;
      if (config.ipfsBackend.timeoutMs !== undefined) ipfsConfig.timeoutMs = config.ipfsBackend.timeoutMs;
      if (config.ipfsBackend.retries !== undefined) ipfsConfig.retries = config.ipfsBackend.retries;
      if (config.ipfsBackend.enableLocalIndex !== undefined) ipfsConfig.enableLocalIndex = config.ipfsBackend.enableLocalIndex;
      if (config.ipfsBackend.localCacheTtlMs !== undefined) ipfsConfig.localCacheTtlMs = config.ipfsBackend.localCacheTtlMs;

      this.ipfsBackend = new IPFSBackend(ipfsConfig);
    } else {
      this.ipfsBackend = null;
    }

    // Initialize observability
    if (config.observability?.enableMetrics) {
      this.metrics = new StorageMetricsCollector();
    } else {
      this.metrics = null;
    }

    if (config.observability?.enableTracing) {
      this.tracer = new StorageTracer();
    } else {
      this.tracer = null;
    }

    if (config.observability?.healthCheckIntervalMs) {
      this.healthChecker = new StorageHealthChecker();
      createBackendHealthChecks(this.healthChecker, {
        memory: this.memoryBackend,
        local: this.localBackend,
        ...(this.dhtBackend ? { dht: this.dhtBackend } : {}),
        ...(this.ipfsBackend ? { ipfs: this.ipfsBackend } : {}),
      });
      this.healthChecker.startPeriodicChecks(config.observability.healthCheckIntervalMs);
    } else {
      this.healthChecker = null;
    }
  }

  /**
   * Store data with epistemic classification
   */
  async store<T>(
    key: string,
    data: T,
    classification: EpistemicClassification,
    options: StoreOptions
  ): Promise<StorageReceipt> {
    const startTime = Date.now();
    const now = startTime;

    // Validate classification
    this.validateClassification(classification);

    // Validate schema (required from Day 1)
    if (!options.schema) {
      throw new ValidationError('options.schema', 'Schema identity is required', options.schema);
    }

    // Route to tier
    const tier = this.router.route(classification);
    const backendType = tier.backend as BackendType;

    // Start trace span
    const span = this.tracer?.startSpan('storage.store', {
      key,
      backend: backendType,
      materiality: classification.materiality,
    });

    try {
      // Get backend
      const backend = this.getBackend(tier.backend);

      // Compute CID
      const serialized = JSON.stringify(data);
      const dataBytes = new TextEncoder().encode(serialized);
      const cid = await this.computeCid(dataBytes);

      // Check for duplicate key
      const existing = this.keyIndex.get(key);
      if (existing) {
        throw new StorageError(`Key already exists: ${key}`, ErrorCode.STORAGE_DUPLICATE_KEY, { key });
      }

      // Build metadata
      const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
        classification,
        schema: options.schema,
        storedAt: now,
        version: 1,
        createdBy: this.agentId,
        tombstone: false,
        expiresAt: tier.ttl ? now + tier.ttl : undefined,
      };

      // Store in backend
      await backend.set(key, data, metadata);

      // Update indices (INV-8)
      const indexEntry: IndexEntry = {
        key,
        cid,
        classification,
        schema: options.schema,
        backend: tier.backend,
        version: 1,
        storedAt: now,
        createdBy: this.agentId,
      };
      this.keyIndex.set(key, indexEntry);
      this.cidIndex.set(cid, key);

      // Build receipt
      const receipt: StorageReceipt = {
        key,
        cid,
        classification,
        schema: options.schema,
        storedAt: now,
        tier,
        indexLocations: [tier.backend, 'local'], // INV-8: always local + primary
        version: 1,
        shreddable: tier.encrypted,
        keyId: tier.encrypted ? await this.generateKeyId() : undefined,
      };

      // Record metrics
      const latency = Date.now() - startTime;
      this.metrics?.recordOperation('store', backendType);
      this.metrics?.recordLatency('store', latency);
      if (span) this.tracer?.endSpan(span.spanId, 'ok');

      return receipt;
    } catch (error) {
      const latency = Date.now() - startTime;
      this.metrics?.recordOperation('store', backendType);
      this.metrics?.recordLatency('store', latency);
      this.metrics?.recordError('store');
      if (span) this.tracer?.endSpan(span.spanId, 'error');
      throw error;
    }
  }

  /**
   * Retrieve data by key
   */
  async retrieve<T>(
    key: string,
    capability?: AccessCapability,
    options?: RetrieveOptions
  ): Promise<StoredData<T> | null> {
    const startTime = Date.now();

    // Check index first
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      this.cacheMisses++;
      this.metrics?.recordCacheAccess(false);
      this.metrics?.recordOperation('retrieve');
      this.metrics?.recordLatency('retrieve', Date.now() - startTime);
      return null;
    }

    this.cacheHits++;
    this.metrics?.recordCacheAccess(true);

    const backendType = indexEntry.backend as BackendType;
    const span = this.tracer?.startSpan('storage.retrieve', { key, backend: backendType });

    try {
      // Get classification and check access
      const tier = this.router.route(indexEntry.classification);

      // Check access control
      if (tier.accessControl !== AccessControlMode.PUBLIC) {
        if (!capability) {
          throw new AccessDeniedError(key, tier.accessControl);
        }
        this.validateCapability(capability, key, 'read');
      }

      // Get backend and retrieve
      const backend = this.getBackend(indexEntry.backend);
      const result = await backend.get<T>(key);

      if (!result) {
        // Data was in index but not in backend - inconsistency
        this.keyIndex.delete(key);
        this.cidIndex.delete(indexEntry.cid);
        if (span) this.tracer?.endSpan(span.spanId, 'ok');
        return null;
      }

      // Track retrieval time
      const latency = Date.now() - startTime;
      this.retrievalTimes.push(latency);
      if (this.retrievalTimes.length > 1000) {
        this.retrievalTimes.shift();
      }

      this.metrics?.recordOperation('retrieve', backendType);
      this.metrics?.recordLatency('retrieve', latency);
      if (span) this.tracer?.endSpan(span.spanId, 'ok');

      return {
        data: result.data,
        metadata: result.metadata,
        verified: options?.verify ? await this.verifyData(key, result.data, result.metadata) : true,
      };
    } catch (error) {
      this.metrics?.recordOperation('retrieve', backendType);
      this.metrics?.recordLatency('retrieve', Date.now() - startTime);
      this.metrics?.recordError('retrieve');
      if (span) this.tracer?.endSpan(span.spanId, 'error');
      throw error;
    }
  }

  /**
   * Retrieve by content address (CID)
   */
  async retrieveByCID<T>(
    cid: string,
    capability?: AccessCapability,
    options?: RetrieveOptions
  ): Promise<StoredData<T> | null> {
    // Look up key from CID
    const key = this.cidIndex.get(cid);
    if (!key) {
      return null;
    }

    return this.retrieve<T>(key, capability, options);
  }

  /**
   * Update data (behavior depends on E-level mutability)
   */
  async update<T>(
    key: string,
    data: T,
    capability: AccessCapability,
    options?: UpdateOptions
  ): Promise<StorageReceipt> {
    // Get existing entry
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      throw new StorageError(`Key not found: ${key}`, ErrorCode.STORAGE_NOT_FOUND, { key });
    }

    // Check access
    this.validateCapability(capability, key, 'write');

    // Get tier and check mutability
    const tier = this.router.route(indexEntry.classification);

    switch (tier.mutability) {
      case MutabilityMode.IMMUTABLE:
        throw new ImmutableError(key);

      case MutabilityMode.APPEND_ONLY:
        return this.appendVersion(key, data, indexEntry, options?.retractionReason);

      case MutabilityMode.MUTABLE_CRDT:
        return this.updateMutable(key, data, indexEntry, options?.mergeStrategy);

      default: {
        const _exhaustive: never = tier.mutability;
        throw new StorageError(`Unknown mutability mode: ${_exhaustive as string}`);
      }
    }
  }

  /**
   * Delete/retract data
   */
  async delete(
    key: string,
    capability: AccessCapability,
    options?: DeleteOptions
  ): Promise<void> {
    const startTime = Date.now();

    // Get existing entry
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      throw new StorageError(`Key not found: ${key}`, ErrorCode.STORAGE_NOT_FOUND, { key });
    }

    const backendType = indexEntry.backend as BackendType;
    const span = this.tracer?.startSpan('storage.delete', { key, backend: backendType });

    try {
      // Check access
      this.validateCapability(capability, key, 'delete');

      // Get tier and check mutability
      const tier = this.router.route(indexEntry.classification);

      if (tier.mutability === MutabilityMode.IMMUTABLE) {
        if (options?.hard && tier.encrypted) {
          // Cryptographic shredding - delete encryption key
          await this.shred(key);
        } else {
          throw new ImmutableError(key);
        }
      } else {
        // Delete from backend
        const backend = this.getBackend(indexEntry.backend);
        await backend.delete(key);

        // Remove from indices
        this.keyIndex.delete(key);
        this.cidIndex.delete(indexEntry.cid);
      }

      this.metrics?.recordOperation('delete', backendType);
      this.metrics?.recordLatency('delete', Date.now() - startTime);
      if (span) this.tracer?.endSpan(span.spanId, 'ok');
    } catch (error) {
      this.metrics?.recordOperation('delete', backendType);
      this.metrics?.recordLatency('delete', Date.now() - startTime);
      this.metrics?.recordError('delete');
      if (span) this.tracer?.endSpan(span.spanId, 'error');
      throw error;
    }
  }

  /**
   * Reclassify data (upgrade only per INV-2)
   */
  async reclassify(
    key: string,
    newClassification: Partial<EpistemicClassification>,
    reason: ReclassificationReason,
    capability: AccessCapability
  ): Promise<StorageReceipt> {
    // Get existing entry
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      throw new StorageError(`Key not found: ${key}`, ErrorCode.STORAGE_NOT_FOUND, { key });
    }

    // Check access
    this.validateCapability(capability, key, 'write');

    // Build full new classification
    const fullNewClassification: EpistemicClassification = {
      empirical: newClassification.empirical ?? indexEntry.classification.empirical,
      normative: newClassification.normative ?? indexEntry.classification.normative,
      materiality: newClassification.materiality ?? indexEntry.classification.materiality,
    };

    // Validate transition (INV-2: monotonically increasing)
    const validation = this.router.validateTransition(
      indexEntry.classification,
      fullNewClassification
    );
    if (!validation.valid) {
      throw new InvariantViolationError('INV-2', validation.error!);
    }

    // Check if migration required
    const needsMigration = this.router.requiresMigration(
      indexEntry.classification,
      fullNewClassification
    );

    if (needsMigration) {
      return this.migrateData(key, indexEntry, fullNewClassification, reason);
    } else {
      // Just update the index
      indexEntry.classification = fullNewClassification;
      this.keyIndex.set(key, indexEntry);

      return {
        key,
        cid: indexEntry.cid,
        classification: fullNewClassification,
        schema: indexEntry.schema,
        storedAt: indexEntry.storedAt,
        tier: this.router.route(fullNewClassification),
        indexLocations: [indexEntry.backend, 'local'],
        version: indexEntry.version,
        shreddable: this.router.route(fullNewClassification).encrypted,
      };
    }
  }

  /**
   * Get storage info for a key
   */
  async getStorageInfo(key: string): Promise<StorageInfo | null> {
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      return { exists: false };
    }

    const tier = this.router.route(indexEntry.classification);
    const backend = this.getBackend(indexEntry.backend);
    const result = await backend.get(key);

    return {
      exists: true,
      classification: indexEntry.classification,
      schema: indexEntry.schema,
      tier,
      version: indexEntry.version,
      sizeBytes: result?.metadata.sizeBytes,
      locations: [indexEntry.backend],
      tombstone: result?.metadata.tombstone,
      expiresAt: result?.metadata.expiresAt,
    };
  }

  /**
   * Query by classification and metadata
   */
  async query<T>(query: EpistemicQuery): Promise<QueryResult<T>> {
    const startTime = Date.now();
    const results: StoredData<T>[] = [];

    // Filter index entries
    for (const [key, entry] of this.keyIndex.entries()) {
      if (this.matchesQuery(entry, query)) {
        const result = await this.retrieve<T>(key);
        if (result) {
          results.push(result);
        }
      }
    }

    // Sort
    if (query.orderBy) {
      results.sort((a, b) => {
        let aVal: number, bVal: number;
        switch (query.orderBy) {
          case 'storedAt':
            aVal = a.metadata.storedAt;
            bVal = b.metadata.storedAt;
            break;
          case 'modifiedAt':
            aVal = a.metadata.modifiedAt ?? a.metadata.storedAt;
            bVal = b.metadata.modifiedAt ?? b.metadata.storedAt;
            break;
          case 'classification':
            aVal = a.metadata.classification.materiality * 100 +
                   a.metadata.classification.normative * 10 +
                   a.metadata.classification.empirical;
            bVal = b.metadata.classification.materiality * 100 +
                   b.metadata.classification.normative * 10 +
                   b.metadata.classification.empirical;
            break;
          default:
            return 0;
        }
        return query.orderDirection === 'desc' ? bVal - aVal : aVal - bVal;
      });
    }

    // Paginate
    const offset = query.offset ?? 0;
    const limit = query.limit ?? 100;
    const paginated = results.slice(offset, offset + limit);

    return {
      items: paginated,
      totalCount: results.length,
      hasMore: offset + limit < results.length,
      nextOffset: offset + limit < results.length ? offset + limit : undefined,
      executionTimeMs: Date.now() - startTime,
    };
  }

  /**
   * Get storage statistics
   */
  async getStats(): Promise<StorageStats> {
    const memStats = await this.memoryBackend.stats();
    const localStats = await this.localBackend.stats();
    const dhtStats = this.dhtBackend ? await this.dhtBackend.stats() : null;
    const ipfsStats = this.ipfsBackend ? await this.ipfsBackend.stats() : null;

    const itemsByBackend: Record<StorageBackend, number> = {
      memory: memStats.itemCount,
      local: localStats.itemCount,
      dht: dhtStats?.itemCount ?? 0,
      ipfs: ipfsStats?.itemCount ?? 0,
      filecoin: 0,
    };

    const itemsByMateriality: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    const itemsByNormative: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    const itemsByEmpirical: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };

    for (const entry of this.keyIndex.values()) {
      itemsByMateriality[entry.classification.materiality]++;
      itemsByNormative[entry.classification.normative]++;
      itemsByEmpirical[entry.classification.empirical]++;
    }

    const avgRetrievalTime = this.retrievalTimes.length > 0
      ? this.retrievalTimes.reduce((a, b) => a + b, 0) / this.retrievalTimes.length
      : 0;

    const cacheTotal = this.cacheHits + this.cacheMisses;
    const cacheHitRate = cacheTotal > 0 ? this.cacheHits / cacheTotal : 0;

    return {
      totalItems: this.keyIndex.size,
      totalSizeBytes: memStats.totalSizeBytes + localStats.totalSizeBytes + (dhtStats?.totalSizeBytes ?? 0) + (ipfsStats?.totalSizeBytes ?? 0),
      itemsByBackend,
      itemsByMateriality,
      itemsByNormative,
      itemsByEmpirical,
      cacheHitRate,
      avgRetrievalTimeMs: avgRetrievalTime,
      pendingMigrations: 0,
    };
  }

  /**
   * Verify data integrity
   */
  async verify(key: string): Promise<VerificationResult> {
    const indexEntry = this.keyIndex.get(key);
    if (!indexEntry) {
      return {
        key,
        verified: false,
        cidValid: false,
        backendStatus: {
          memory: 'unavailable',
          local: 'unavailable',
          dht: 'unavailable',
          ipfs: 'unavailable',
          filecoin: 'unavailable',
        },
        actualReplication: 0,
        expectedReplication: 0,
        errors: ['Key not found'],
        verifiedAt: Date.now(),
      };
    }

    const tier = this.router.route(indexEntry.classification);
    const backend = this.getBackend(indexEntry.backend);
    const result = await backend.get(key);

    const cidValid = result?.metadata.cid === indexEntry.cid;

    const backendStatus: Record<StorageBackend, 'available' | 'unavailable' | 'degraded'> = {
      memory: 'unavailable',
      local: 'unavailable',
      dht: 'unavailable',
      ipfs: 'unavailable',
      filecoin: 'unavailable',
    };
    backendStatus[indexEntry.backend] = result ? 'available' : 'unavailable';

    return {
      key,
      verified: cidValid && result !== null,
      cidValid,
      backendStatus,
      actualReplication: result ? 1 : 0,
      expectedReplication: tier.replication,
      errors: result ? [] : ['Data not found in backend'],
      verifiedAt: Date.now(),
    };
  }

  /**
   * Check if key exists
   */
  async exists(key: string): Promise<boolean> {
    return this.keyIndex.has(key);
  }

  // ===========================================================================
  // Private Helper Methods
  // ===========================================================================

  private getBackend(backend: StorageBackend): StorageBackendAdapter {
    switch (backend) {
      case 'memory':
        return this.memoryBackend;
      case 'local':
        return this.localBackend;
      case 'dht':
        // Use DHT backend if configured, otherwise fall back to local
        return this.dhtBackend ?? this.localBackend;
      case 'ipfs':
        // Use IPFS backend if configured, otherwise fall back to DHT or local
        return this.ipfsBackend ?? this.dhtBackend ?? this.localBackend;
      case 'filecoin':
        // Fall back to IPFS, DHT, or local
        return this.ipfsBackend ?? this.dhtBackend ?? this.localBackend;
      default:
        return this.localBackend;
    }
  }

  /**
   * Check if DHT backend is available
   */
  isDHTEnabled(): boolean {
    return this.dhtBackend !== null;
  }

  private validateClassification(classification: EpistemicClassification): void {
    if (classification.empirical < EmpiricalLevel.E0_Unverified || classification.empirical > EmpiricalLevel.E4_Consensus) {
      throw new ValidationError('classification.empirical', 'Must be 0-4', classification.empirical);
    }
    if (classification.normative < NormativeLevel.N0_Personal || classification.normative > NormativeLevel.N3_Universal) {
      throw new ValidationError('classification.normative', 'Must be 0-3', classification.normative);
    }
    if (classification.materiality < MaterialityLevel.M0_Ephemeral || classification.materiality > MaterialityLevel.M3_Immutable) {
      throw new ValidationError('classification.materiality', 'Must be 0-3', classification.materiality);
    }
  }

  private validateCapability(capability: AccessCapability, key: string, permission: string): void {
    // Check resource access
    if (capability.resourceKeys !== '*' && !capability.resourceKeys.includes(key)) {
      throw new AccessDeniedError(key, permission);
    }

    // Check permission
    if (!capability.permissions.includes(permission as 'read' | 'write' | 'delete' | 'delegate' | 'shred')) {
      throw new AccessDeniedError(key, permission);
    }

    // Check expiration
    if (capability.expiresAt && capability.expiresAt < Date.now()) {
      throw new AccessDeniedError(key, 'capability expired');
    }
  }

  private async computeCid(data: Uint8Array): Promise<string> {
    const hashBytes = await hash(data);
    return 'cid:' + Array.from(hashBytes).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  private async generateKeyId(): Promise<string> {
    return secureUUID();
  }

  private async verifyData<T>(_key: string, data: T, metadata: StorageMetadata): Promise<boolean> {
    const serialized = JSON.stringify(data);
    const dataBytes = new TextEncoder().encode(serialized);
    const computedCid = await this.computeCid(dataBytes);
    return computedCid === metadata.cid;
  }

  private async appendVersion<T>(
    key: string,
    data: T,
    indexEntry: IndexEntry,
    retractionReason?: string
  ): Promise<StorageReceipt> {
    const now = Date.now();
    const newVersion = indexEntry.version + 1;

    // Compute new CID
    const serialized = JSON.stringify(data);
    const dataBytes = new TextEncoder().encode(serialized);
    const newCid = await this.computeCid(dataBytes);

    // Build metadata
    const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
      classification: indexEntry.classification,
      schema: indexEntry.schema,
      storedAt: indexEntry.storedAt,
      modifiedAt: now,
      version: newVersion,
      createdBy: indexEntry.createdBy,
      tombstone: false,
      retractedBy: retractionReason,
    };

    // Store new version
    const backend = this.getBackend(indexEntry.backend);
    await backend.set(key, data, metadata);

    // Update index
    const oldCid = indexEntry.cid;
    indexEntry.cid = newCid;
    indexEntry.version = newVersion;
    this.keyIndex.set(key, indexEntry);
    this.cidIndex.delete(oldCid);
    this.cidIndex.set(newCid, key);

    const tier = this.router.route(indexEntry.classification);

    return {
      key,
      cid: newCid,
      classification: indexEntry.classification,
      schema: indexEntry.schema,
      storedAt: indexEntry.storedAt,
      tier,
      indexLocations: [indexEntry.backend, 'local'],
      version: newVersion,
      previousCid: oldCid,
      shreddable: tier.encrypted,
    };
  }

  private async updateMutable<T>(
    key: string,
    data: T,
    indexEntry: IndexEntry,
    _mergeStrategy?: string
  ): Promise<StorageReceipt> {
    const now = Date.now();

    // Compute new CID
    const serialized = JSON.stringify(data);
    const dataBytes = new TextEncoder().encode(serialized);
    const newCid = await this.computeCid(dataBytes);

    // Build metadata
    const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
      classification: indexEntry.classification,
      schema: indexEntry.schema,
      storedAt: indexEntry.storedAt,
      modifiedAt: now,
      version: indexEntry.version + 1,
      createdBy: indexEntry.createdBy,
      tombstone: false,
    };

    // Store (overwrite)
    const backend = this.getBackend(indexEntry.backend);
    await backend.set(key, data, metadata);

    // Update index
    const oldCid = indexEntry.cid;
    indexEntry.cid = newCid;
    indexEntry.version++;
    this.keyIndex.set(key, indexEntry);
    this.cidIndex.delete(oldCid);
    this.cidIndex.set(newCid, key);

    const tier = this.router.route(indexEntry.classification);

    return {
      key,
      cid: newCid,
      classification: indexEntry.classification,
      schema: indexEntry.schema,
      storedAt: indexEntry.storedAt,
      tier,
      indexLocations: [indexEntry.backend, 'local'],
      version: indexEntry.version,
      shreddable: tier.encrypted,
    };
  }

  private async migrateData(
    key: string,
    indexEntry: IndexEntry,
    newClassification: EpistemicClassification,
    _reason: ReclassificationReason
  ): Promise<StorageReceipt> {
    // Get data from old backend
    const oldBackend = this.getBackend(indexEntry.backend);
    const result = await oldBackend.get(key);

    if (!result) {
      throw new StorageError(`Data not found during migration: ${key}`);
    }

    // Get new tier
    const newTier = this.router.route(newClassification);
    const newBackend = this.getBackend(newTier.backend);

    // Store in new backend
    const metadata: Omit<StorageMetadata, 'cid' | 'sizeBytes'> = {
      ...result.metadata,
      classification: newClassification,
      modifiedAt: Date.now(),
      version: indexEntry.version + 1,
    };

    await newBackend.set(key, result.data, metadata);

    // Delete from old backend (after successful store)
    await oldBackend.delete(key);

    // Update index
    indexEntry.classification = newClassification;
    indexEntry.backend = newTier.backend;
    indexEntry.version++;
    this.keyIndex.set(key, indexEntry);

    return {
      key,
      cid: indexEntry.cid,
      classification: newClassification,
      schema: indexEntry.schema,
      storedAt: indexEntry.storedAt,
      tier: newTier,
      indexLocations: [newTier.backend, 'local'],
      version: indexEntry.version,
      shreddable: newTier.encrypted,
    };
  }

  private async shred(key: string): Promise<void> {
    // In a real implementation, this would delete the encryption key
    // For now, we just remove from indices
    const indexEntry = this.keyIndex.get(key);
    if (indexEntry) {
      this.keyIndex.delete(key);
      this.cidIndex.delete(indexEntry.cid);
    }
  }

  private matchesQuery(entry: IndexEntry, query: EpistemicQuery): boolean {
    // Classification filters
    if (query.classification) {
      const c = entry.classification;
      const qc = query.classification;

      if (qc.minEmpirical !== undefined && c.empirical < qc.minEmpirical) return false;
      if (qc.maxEmpirical !== undefined && c.empirical > qc.maxEmpirical) return false;
      if (qc.minNormative !== undefined && c.normative < qc.minNormative) return false;
      if (qc.maxNormative !== undefined && c.normative > qc.maxNormative) return false;
      if (qc.minMateriality !== undefined && c.materiality < qc.minMateriality) return false;
      if (qc.maxMateriality !== undefined && c.materiality > qc.maxMateriality) return false;
    }

    // Schema filters
    if (query.schema) {
      if (query.schema.id && entry.schema.id !== query.schema.id) return false;
      if (query.schema.family && entry.schema.family !== query.schema.family) return false;
      if (query.schema.version && entry.schema.version !== query.schema.version) return false;
    }

    // Time range
    if (query.timeRange) {
      if (query.timeRange.after && entry.storedAt < query.timeRange.after) return false;
      if (query.timeRange.before && entry.storedAt > query.timeRange.before) return false;
    }

    // Creator
    if (query.createdBy && entry.createdBy !== query.createdBy) return false;

    return true;
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.memoryBackend.dispose();
    this.localBackend.dispose();
    this.dhtBackend?.dispose();
    void this.ipfsBackend?.dispose();
    this.healthChecker?.stopPeriodicChecks();
  }

  /**
   * Get the DHT backend for direct access (advanced use cases)
   */
  getDHTBackend(): DHTBackend | null {
    return this.dhtBackend;
  }

  /**
   * Check if IPFS backend is available
   */
  isIPFSEnabled(): boolean {
    return this.ipfsBackend !== null;
  }

  /**
   * Get the IPFS backend for direct access
   */
  getIPFSBackend(): IPFSBackend | null {
    return this.ipfsBackend;
  }

  // ===========================================================================
  // Observability Accessors
  // ===========================================================================

  /**
   * Get metrics snapshot (returns null if metrics not enabled)
   */
  getMetricsSnapshot(): StorageMetricsSnapshot | null {
    return this.metrics?.getSnapshot() ?? null;
  }

  /**
   * Get metrics in Prometheus format
   */
  getPrometheusMetrics(): string | null {
    return this.metrics?.toPrometheus() ?? null;
  }

  /**
   * Get all completed trace spans
   */
  getTraceSpans(): import('./observability.js').TraceSpan[] {
    return this.tracer?.getAllSpans() ?? [];
  }

  /**
   * Run all health checks and return results
   */
  async runHealthChecks(): Promise<HealthCheckResult[]> {
    return this.healthChecker?.runAllChecks() ?? [];
  }

  /**
   * Get the metrics collector for direct access
   */
  getMetrics(): StorageMetricsCollector | null {
    return this.metrics;
  }

  /**
   * Get the tracer for direct access
   */
  getTracer(): StorageTracer | null {
    return this.tracer;
  }

  /**
   * Get the health checker for direct access
   */
  getHealthChecker(): StorageHealthChecker | null {
    return this.healthChecker;
  }
}

/**
 * Create an epistemic storage instance
 */
export function createEpistemicStorage(config: EpistemicStorageConfig): EpistemicStorageImpl {
  return new EpistemicStorageImpl(config);
}

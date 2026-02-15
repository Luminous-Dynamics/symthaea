/**
 * UESS (Unified Epistemic Storage System) Reference Implementation
 *
 * Routes data to appropriate backends based on E/N/M classification.
 *
 * ## Overview
 *
 * UESS classifies all data along three axes:
 * - **Empirical (E0-E4)**: How verifiable is the claim?
 * - **Normative (N0-N3)**: Who should have access?
 * - **Materiality (M0-M3)**: How long should it persist?
 *
 * Classification determines:
 * - Storage backend (memory, local, DHT, IPFS, Filecoin)
 * - Mutability (CRDT, append-only, immutable)
 * - Access control (owner, CapBAC, public)
 * - Replication factor
 * - Encryption requirements
 *
 * ## Quick Start
 *
 * ```typescript
 * import { storage } from '@mycelix/sdk';
 *
 * // Create storage instance
 * const store = storage.createEpistemicStorage({
 *   agentId: 'agent:alice',
 * });
 *
 * // Store ephemeral data (M0 → memory)
 * await store.store('session:123', { token: 'xyz' }, {
 *   empirical: storage.EmpiricalLevel.E0_Subjective,
 *   normative: storage.NormativeLevel.N0_Personal,
 *   materiality: storage.MaterialityLevel.M0_Ephemeral,
 * }, { schema: { id: 'session', version: '1.0.0' } });
 *
 * // Store persistent data (M2 → DHT)
 * await store.store('profile:alice', { name: 'Alice' }, {
 *   empirical: storage.EmpiricalLevel.E3_Cryptographic,
 *   normative: storage.NormativeLevel.N2_Network,
 *   materiality: storage.MaterialityLevel.M2_Persistent,
 * }, { schema: { id: 'profile', version: '1.0.0' } });
 *
 * // Retrieve data
 * const profile = await store.retrieve('profile:alice');
 * ```
 *
 * @see docs/architecture/uess/UESS-00-INDEX.md
 */

// Core types
export {
  // Storage backend
  type StorageBackend,
  MutabilityMode,
  AccessControlMode,

  // Storage tier
  type StorageTier,

  // Storage receipt
  type StorageReceipt,
  type StorageProof,

  // Stored data
  type StoredData,
  type StorageMetadata,
  type StorageInfo,

  // Schema identity
  type SchemaIdentity,

  // Access capability
  type AccessCapability,

  // Operation options
  type StoreOptions,
  type RetrieveOptions,
  type UpdateOptions,
  type DeleteOptions,

  // E4 reproducibility
  type E4ReproBundle,

  // Reclassification
  type ReclassificationReason,

  // Query
  type EpistemicQuery,
  type QueryResult,

  // Verification
  type VerificationResult,

  // Statistics
  type StorageStats,
} from './types.js';

// Router
export {
  type ClassificationRouter,
  type RouterConfig,
  DEFAULT_ROUTER_CONFIG,
  StorageRouter,
  createRouter,
  defaultRouter,
} from './router.js';

// Backends
export {
  type StorageBackendAdapter,
  MemoryBackend,
  createMemoryBackend,
  LocalBackend,
  createLocalBackend,
  DHTBackend,
  createDHTBackend,
  type DHTBackendConfig,
  type HolochainClient,
  IPFSBackend,
  createIPFSBackend,
  type IPFSBackendConfig,
} from './backends/index.js';

// Epistemic Storage
export {
  type EpistemicStorage,
  type EpistemicStorageConfig,
  EpistemicStorageImpl,
  createEpistemicStorage,
  StorageError,
  ImmutableError,
  AccessDeniedError,
  InvariantViolationError,
} from './epistemic-storage.js';

// Re-export epistemic levels for convenience
export {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '../epistemic/types.js';

// CapBAC (Capability-Based Access Control)
export {
  AccessRight,
  type CapabilityToken,
  type CapabilityConstraints,
  type ValidationResult,
  type ValidationContext,
  type CapBACConfig,
  CapBACManager,
  createCapBACManager,
  createCapBACManagerWithKey,
  createReadCapability,
  createFullAccessCapability,
} from './capbac.js';

// Cryptographic Shredding (GDPR)
export {
  type KeyEntry,
  type ShredAuditEntry,
  ShredReason,
  type ShredResult,
  type CryptoShreddingConfig,
  CryptoShreddingManager,
  createCryptoShreddingManager,
  executeGDPRErasure,
} from './crypto-shredding.js';

// E4 Reproducibility Bundles
export {
  type ReproBundle,
  type ReproClaim,
  type ReproInput,
  type ReproEnvironment,
  type ReproStep,
  type ReproOutput,
  type VerificationConfig,
  type ReproductionRecord,
  type VerificationStatus,
  ReproBundleManager,
  createReproBundleManager,
  createComputationBundle,
} from './repro-bundle.js';

// CRDT Merge Strategies
export {
  type MergeStrategy,
  type VectorClock,
  type LWWValue,
  type ORSetElement,
  type MVValue,
  type MergeResult,
  createVectorClock,
  incrementClock,
  mergeClocks,
  compareClocks,
  happenedBefore,
  LWWRegister,
  MVRegister,
  ORSet,
  GCounter,
  PNCounter,
  merge,
} from './crdt.js';

// Observability
export {
  type StorageOperation,
  type BackendType,
  type HealthStatus,
  type MetricValue,
  type HistogramBucket,
  type HistogramMetric,
  type TraceSpan,
  type HealthCheckResult,
  type StorageMetricsSnapshot,
  StorageMetricsCollector,
  StorageTracer,
  StorageHealthChecker,
  type HealthCheckFn,
  createMetricsCollector,
  createTracer,
  createHealthChecker,
  createBackendHealthChecks,
  instrumentedOperation,
} from './observability.js';

// Batch Operations
export {
  type BatchItem,
  type BatchResult,
  type BatchConfig,
  type BatchStats,
  type BatchProgressCallback,
  type StoreFn,
  type RetrieveFn,
  type DeleteFn,
  BatchExecutor,
  type BatchItemSource,
  StreamingBatchExecutor,
  createBatchExecutor,
  createStreamingBatchExecutor,
  createArraySource,
  batchStore,
  batchRetrieve,
  batchDelete,
} from './batch.js';

// Migration Tools
export {
  type MigrationDirection,
  type MigrationProgress,
  type MigrationProgressCallback,
  type MigrationItemResult,
  type MigrationResult,
  type MigrationOptions,
  type ExportBundle,
  type ExportItem,
  type SyncResult,
  migrate,
  exportBackend,
  importBundle,
  sync,
} from './migration.js';

// GraphQL API Layer
export {
  createStorageSchema,
  createStorageResolvers,
  type StorageResolvers,
} from './graphql.js';

// CLI
export { main as runStorageCLI } from './cli.js';

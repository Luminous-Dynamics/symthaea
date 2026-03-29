// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS (Unified Epistemic Storage System) - Core Types
 *
 * Reference implementation of the UESS specification.
 * @see docs/architecture/uess/UESS-01-CORE.md
 */

import type {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  EpistemicClassification,
} from '../epistemic/types.js';

// =============================================================================
// Schema Identity (required from Day 1 per UESS spec)
// =============================================================================

/**
 * Schema identity for stored data
 */
export interface SchemaIdentity {
  /** Unique schema identifier */
  id: string;

  /** Schema version (semver) */
  version: string;

  /** Schema family (allows multiple worldviews) */
  family?: string;
}

// =============================================================================
// Storage Backends
// =============================================================================

/**
 * Available storage backends
 */
export type StorageBackend = 'memory' | 'local' | 'dht' | 'ipfs' | 'filecoin';

/**
 * Mutability modes based on E-level
 */
export enum MutabilityMode {
  /** Full CRDT-based updates (E0-E1) */
  MUTABLE_CRDT = 'mutable_crdt',

  /** Append-only with retraction (E2) */
  APPEND_ONLY = 'append_only',

  /** Immutable, tombstone only (E3-E4) */
  IMMUTABLE = 'immutable',
}

/**
 * Access control modes based on N-level
 */
export enum AccessControlMode {
  /** Only owner capability (N0) */
  OWNER = 'owner',

  /** Capability-based delegation (N1) */
  CAPBAC = 'capbac',

  /** Public access (N2-N3) */
  PUBLIC = 'public',
}

// =============================================================================
// Storage Tier
// =============================================================================

/**
 * Resolved storage tier from classification
 */
export interface StorageTier {
  /** Primary backend */
  backend: StorageBackend;

  /** Additional backends (for redundancy) */
  additionalBackends: StorageBackend[];

  /** Replication factor */
  replication: number;

  /** Mutability mode */
  mutability: MutabilityMode;

  /** Access control mode */
  accessControl: AccessControlMode;

  /** Time-to-live in ms (undefined = permanent) */
  ttl?: number;

  /** Encryption required */
  encrypted: boolean;

  /** Content-addressed (CID) */
  contentAddressed: boolean;
}

// =============================================================================
// Storage Receipt
// =============================================================================

/**
 * Proof of storage on a backend
 */
export interface StorageProof {
  backend: StorageBackend;
  timestamp: number;
  proof: string;
  verifierSignature?: string;
}

/**
 * Receipt returned after successful storage operation
 */
export interface StorageReceipt {
  /** Unique key */
  key: string;

  /** Content identifier (hash of content) */
  cid: string;

  /** Classification at time of storage */
  classification: EpistemicClassification;

  /** Schema identity */
  schema: SchemaIdentity;

  /** When stored (ms since epoch) */
  storedAt: number;

  /** Resolved storage tier */
  tier: StorageTier;

  /** Index locations */
  indexLocations: StorageBackend[];

  /** Verification proofs (for E3+) */
  proofs?: StorageProof[];

  /** Version number (for append-only) */
  version?: number;

  /** Previous CID (for version chain) */
  previousCid?: string;

  /** Whether this data can be cryptographically shredded */
  shreddable: boolean;

  /** Key ID for encryption (N0/N1) */
  keyId?: string;
}

// =============================================================================
// Stored Data
// =============================================================================

/**
 * Metadata for stored content
 */
export interface StorageMetadata {
  /** Content identifier */
  cid: string;

  /** Current classification */
  classification: EpistemicClassification;

  /** Schema identity */
  schema: SchemaIdentity;

  /** When stored */
  storedAt: number;

  /** Last modified (for mutable data) */
  modifiedAt?: number;

  /** Version number */
  version: number;

  /** TTL expiry timestamp */
  expiresAt?: number;

  /** Size in bytes */
  sizeBytes: number;

  /** Creator agent ID */
  createdBy: string;

  /** Is this a tombstone? */
  tombstone: boolean;

  /** Retraction pointer (for E2 append-only) */
  retractedBy?: string;
}

/**
 * Stored data with metadata
 */
export interface StoredData<T> {
  /** The stored data */
  data: T;

  /** Storage metadata */
  metadata: StorageMetadata;

  /** Verification status */
  verified: boolean;
}

// =============================================================================
// Storage Info
// =============================================================================

/**
 * Storage information for a key
 */
export interface StorageInfo {
  /** Key exists */
  exists: boolean;

  /** Current classification */
  classification?: EpistemicClassification;

  /** Current schema */
  schema?: SchemaIdentity;

  /** Resolved tier */
  tier?: StorageTier;

  /** Current version */
  version?: number;

  /** Size in bytes */
  sizeBytes?: number;

  /** Storage locations */
  locations?: StorageBackend[];

  /** Is tombstoned */
  tombstone?: boolean;

  /** Expires at */
  expiresAt?: number;
}

// =============================================================================
// Access Capability
// =============================================================================

/**
 * Capability-based access token
 */
export interface AccessCapability {
  /** Unique capability ID */
  id: string;

  /** Resource key(s) this grants access to */
  resourceKeys: string[] | '*';

  /** Permissions granted */
  permissions: Array<'read' | 'write' | 'delete' | 'delegate' | 'shred'>;

  /** Issuer agent ID */
  issuedBy: string;

  /** Recipient agent ID */
  issuedTo: string;

  /** When issued */
  issuedAt: number;

  /** Expiration timestamp */
  expiresAt?: number;

  /** Can this capability be delegated? */
  delegable: boolean;

  /** Parent capability (for delegation chain) */
  parentCapabilityId?: string;

  /** Cryptographic signature */
  signature: string;
}

// =============================================================================
// Operation Options
// =============================================================================

/**
 * Options for store operation
 */
export interface StoreOptions {
  /** Schema identity (required) */
  schema: SchemaIdentity;

  /** Override TTL (must be within M-level limits) */
  ttl?: number;

  /** E4 reproducibility bundle */
  reproBundle?: E4ReproBundle;

  /** Skip validation (dangerous, for internal use) */
  skipValidation?: boolean;

  /** Custom metadata */
  customMetadata?: Record<string, unknown>;
}

/**
 * Options for retrieve operation
 */
export interface RetrieveOptions {
  /** Include full verification */
  verify?: boolean;

  /** Preferred backend */
  preferredBackend?: StorageBackend;

  /** Include version history */
  includeHistory?: boolean;

  /** Specific version to retrieve */
  version?: number;
}

/**
 * Options for update operation
 */
export interface UpdateOptions {
  /** CRDT merge strategy (for mutable data) */
  mergeStrategy?: 'lww' | 'vector_clock' | 'custom';

  /** Retraction reason (for append-only) */
  retractionReason?: string;
}

/**
 * Options for delete operation
 */
export interface DeleteOptions {
  /** Reason for deletion */
  reason?: string;

  /** Hard delete (shred) vs soft delete (tombstone) */
  hard?: boolean;
}

// =============================================================================
// E4 Reproducibility Bundle
// =============================================================================

/**
 * Complete specification for reproducible claims (E4)
 */
export interface E4ReproBundle {
  /** Content being claimed */
  content: {
    cid: string;
    data: Uint8Array;
  };

  /** How to reproduce this content */
  reproduction: {
    computation: {
      type: 'hash' | 'signature' | 'proof' | 'execution';
      algorithm: string;
      parameters?: Record<string, unknown>;
    };

    inputs: Array<{
      name: string;
      cid: string;
      type: 'data' | 'code' | 'environment';
    }>;

    environment?: {
      runtime: string;
      dependencies: Record<string, string>;
      containerImage?: string;
    };

    expectedOutput: {
      cid: string;
      format: string;
    };
  };

  witnesses: Array<{
    agentId: string;
    reproducedAt: number;
    outputCid: string;
    signature: string;
  }>;

  computedAt: number;
  author: string;
}

// =============================================================================
// Reclassification
// =============================================================================

/**
 * Reason for reclassification
 */
export interface ReclassificationReason {
  /** Why is this being reclassified */
  reason: string;

  /** Supporting evidence */
  evidence?: string[];

  /** Requested by agent */
  requestedBy: string;

  /** Timestamp */
  timestamp: number;
}

// =============================================================================
// Query Types
// =============================================================================

/**
 * Query for epistemic storage
 */
export interface EpistemicQuery {
  /** Filter by classification */
  classification?: {
    minEmpirical?: EmpiricalLevel;
    maxEmpirical?: EmpiricalLevel;
    minNormative?: NormativeLevel;
    maxNormative?: NormativeLevel;
    minMateriality?: MaterialityLevel;
    maxMateriality?: MaterialityLevel;
  };

  /** Filter by schema */
  schema?: {
    id?: string;
    family?: string;
    version?: string;
  };

  /** Filter by time range */
  timeRange?: {
    after?: number;
    before?: number;
  };

  /** Filter by creator */
  createdBy?: string;

  /** Full-text search (where supported) */
  text?: string;

  /** Custom metadata filters */
  metadata?: Record<string, unknown>;

  /** Pagination */
  limit?: number;
  offset?: number;

  /** Sort order */
  orderBy?: 'storedAt' | 'modifiedAt' | 'classification';
  orderDirection?: 'asc' | 'desc';
}

/**
 * Query result
 */
export interface QueryResult<T> {
  /** Matching items */
  items: Array<StoredData<T>>;

  /** Total count (may be estimate) */
  totalCount: number;

  /** Pagination info */
  hasMore: boolean;
  nextOffset?: number;

  /** Query execution time in ms */
  executionTimeMs: number;
}

// =============================================================================
// Verification
// =============================================================================

/**
 * Result of data verification
 */
export interface VerificationResult {
  /** Key verified */
  key: string;

  /** Overall verification status */
  verified: boolean;

  /** CID matches stored hash */
  cidValid: boolean;

  /** Signature valid (for E3+) */
  signatureValid?: boolean;

  /** Reproducible (for E4) */
  reproducible?: boolean;

  /** Backend availability */
  backendStatus: Record<StorageBackend, 'available' | 'unavailable' | 'degraded'>;

  /** Replication achieved */
  actualReplication: number;

  /** Expected replication */
  expectedReplication: number;

  /** Errors encountered */
  errors: string[];

  /** Verification timestamp */
  verifiedAt: number;
}

// =============================================================================
// Storage Statistics
// =============================================================================

/**
 * Storage statistics
 */
export interface StorageStats {
  /** Total items stored */
  totalItems: number;

  /** Total size in bytes */
  totalSizeBytes: number;

  /** Items by backend */
  itemsByBackend: Record<StorageBackend, number>;

  /** Items by M-level */
  itemsByMateriality: Record<number, number>;

  /** Items by N-level */
  itemsByNormative: Record<number, number>;

  /** Items by E-level */
  itemsByEmpirical: Record<number, number>;

  /** Cache hit rate */
  cacheHitRate: number;

  /** Average retrieval time in ms */
  avgRetrievalTimeMs: number;

  /** Pending migrations */
  pendingMigrations: number;

  /** Last garbage collection */
  lastGcAt?: number;
}

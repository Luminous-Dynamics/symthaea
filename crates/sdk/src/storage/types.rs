// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! UESS Core Types
//!
//! Type definitions for the Unified Epistemic Storage System.
//! Based on the TypeScript SDK implementation.

use crate::epistemic::{EmpiricalLevel, EpistemicClassification, MaterialityLevel, NormativeLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Storage backend types
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory ephemeral storage (M0)
    #[default]
    Memory,
    /// Local persistent storage (M1)
    Local,
    /// Distributed Hash Table via Holochain (M2)
    DHT,
    /// IPFS content-addressed storage (M3)
    IPFS,
    /// Filecoin for archival (M3+)
    Filecoin,
}

/// Mutability mode based on Empirical level
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutabilityMode {
    /// E0-E1: Full CRDT-based updates (LWW, vector clocks)
    #[default]
    MutableCRDT,
    /// E2: Append-only with retraction pointers
    AppendOnly,
    /// E3-E4: Immutable, tombstone-only deletion
    Immutable,
}

/// Access control mode based on Normative level
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessControlMode {
    /// N0: Owner-only capability
    #[default]
    Owner,
    /// N1: Capability-based access control (delegable)
    CapBAC,
    /// N2-N3: Public access
    Public,
}

/// Storage tier configuration derived from E/N/M classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageTier {
    /// Primary storage backend
    pub backend: StorageBackend,
    /// Additional backends for redundancy
    pub additional_backends: Vec<StorageBackend>,
    /// Replication factor
    pub replication: usize,
    /// Data mutability mode
    pub mutability: MutabilityMode,
    /// Access control mode
    pub access_control: AccessControlMode,
    /// Time-to-live in milliseconds (None = no expiry)
    pub ttl_ms: Option<u64>,
    /// Whether encryption is required
    pub encrypted: bool,
    /// Whether to use content-addressed storage
    pub content_addressed: bool,
}

impl Default for StorageTier {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Memory,
            additional_backends: vec![],
            replication: 1,
            mutability: MutabilityMode::MutableCRDT,
            access_control: AccessControlMode::Owner,
            ttl_ms: Some(3600 * 1000), // 1 hour default
            encrypted: true,
            content_addressed: false,
        }
    }
}

/// Schema identity for stored data
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct SchemaIdentity {
    /// Unique schema identifier (e.g., "user-profile-v1")
    pub id: String,
    /// Schema version (semver)
    pub version: String,
    /// Schema family for grouping related schemas
    pub family: Option<String>,
}

impl SchemaIdentity {
    /// Create a new schema identity
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            family: None,
        }
    }

    /// Add a schema family grouping
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }
}

/// Storage receipt returned after successful store operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageReceipt {
    /// Storage key
    pub key: String,
    /// Content identifier (SHA-256 hash)
    pub cid: String,
    /// Epistemic classification
    pub classification: EpistemicClassification,
    /// Data schema
    pub schema: SchemaIdentity,
    /// Timestamp when stored
    pub stored_at: u64,
    /// Storage tier used
    pub tier: StorageTier,
    /// All backends holding this data
    pub index_locations: Vec<StorageBackend>,
    /// Version number
    pub version: u32,
    /// Previous CID for version chains
    pub previous_cid: Option<String>,
    /// Whether data can be cryptographically shredded
    pub shreddable: bool,
    /// Encryption key ID (if encrypted)
    pub key_id: Option<String>,
}

/// Metadata stored alongside data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    /// Content identifier
    pub cid: String,
    /// Epistemic classification
    pub classification: EpistemicClassification,
    /// Data schema
    pub schema: SchemaIdentity,
    /// Timestamp when stored
    pub stored_at: u64,
    /// Timestamp when last modified
    pub modified_at: Option<u64>,
    /// Version number
    pub version: u32,
    /// Expiration timestamp
    pub expires_at: Option<u64>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Agent ID of creator
    pub created_by: String,
    /// Whether this entry is tombstoned
    pub tombstone: bool,
    /// Retraction reason (for E2 append-only)
    pub retracted_by: Option<String>,
}

impl StorageMetadata {
    /// Check if this entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            now > expires_at
        } else {
            false
        }
    }

    /// Check if this entry is accessible (not tombstoned, not expired)
    pub fn is_accessible(&self) -> bool {
        !self.tombstone && !self.is_expired()
    }
}

/// Stored data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredData<T> {
    /// The actual data
    pub data: T,
    /// Associated metadata
    pub metadata: StorageMetadata,
    /// Whether CID was verified
    pub verified: bool,
}

/// Storage options for store operation
#[derive(Debug, Clone, Default)]
pub struct StoreOptions {
    /// Schema identity (required)
    pub schema: SchemaIdentity,
    /// Custom TTL override
    pub ttl_ms: Option<u64>,
    /// Custom encryption key ID
    pub key_id: Option<String>,
    /// Agent ID performing the operation
    pub agent_id: String,
    /// Skip duplicate check
    pub allow_overwrite: bool,
}

/// Options for retrieve operation
#[derive(Debug, Clone, Default)]
pub struct RetrieveOptions {
    /// Whether to verify CID
    pub verify: bool,
    /// Decrypt data if encrypted
    pub decrypt: bool,
}

/// Options for update operation
#[derive(Debug, Clone, Default)]
pub struct UpdateOptions {
    /// CRDT merge strategy (for MutableCRDT mode)
    pub merge_strategy: Option<MergeStrategy>,
    /// Agent ID performing the operation
    pub agent_id: String,
}

/// CRDT merge strategies
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Last-writer-wins based on timestamp
    #[default]
    LastWriterWins,
    /// Vector clock based ordering
    VectorClock,
    /// OR-Set semantics for collections
    ORSet,
}

/// Options for delete operation
#[derive(Debug, Clone, Default)]
pub struct DeleteOptions {
    /// Whether to hard delete (if allowed) vs tombstone
    pub hard_delete: bool,
    /// Cryptographic shred (delete encryption key)
    pub shred: bool,
    /// Agent ID performing the operation
    pub agent_id: String,
}

/// Reason for reclassification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReclassificationReason {
    /// Human-readable reason
    pub reason: String,
    /// Agent who requested reclassification
    pub requested_by: String,
    /// Timestamp of request
    pub timestamp: u64,
}

/// Capability token for access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessCapability {
    /// Capability ID
    pub id: String,
    /// Resource key or pattern
    pub resource: String,
    /// Whether resource is a pattern (e.g., "user:*")
    pub is_pattern: bool,
    /// Granted rights
    pub rights: Vec<AccessRight>,
    /// Issuer agent ID
    pub issuer: String,
    /// Holder agent ID
    pub holder: String,
    /// Issue timestamp
    pub issued_at: u64,
    /// Expiration timestamp
    pub expires_at: Option<u64>,
    /// Maximum delegation depth
    pub delegation_depth: u8,
    /// Parent capability ID (for delegated caps)
    pub parent_id: Option<String>,
}

/// Access rights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessRight {
    /// Permission to read data
    Read,
    /// Permission to write/modify data
    Write,
    /// Permission to delete data
    Delete,
    /// Permission to delegate capabilities to others
    Delegate,
    /// Administrative access (implies all other rights)
    Admin,
}

impl AccessCapability {
    /// Check if this capability grants a specific right
    pub fn has_right(&self, right: AccessRight) -> bool {
        self.rights.contains(&right) || self.rights.contains(&AccessRight::Admin)
    }

    /// Check if this capability is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            now > expires_at
        } else {
            false
        }
    }

    /// Check if this capability matches a resource key
    pub fn matches_resource(&self, key: &str) -> bool {
        if self.is_pattern {
            // Simple glob matching (supports trailing *)
            if self.resource.ends_with('*') {
                let prefix = &self.resource[..self.resource.len() - 1];
                key.starts_with(prefix)
            } else {
                self.resource == key
            }
        } else {
            self.resource == key
        }
    }
}

/// Query for searching stored data
#[derive(Debug, Clone, Default)]
pub struct EpistemicQuery {
    /// Filter by empirical level range
    pub empirical_range: Option<(EmpiricalLevel, EmpiricalLevel)>,
    /// Filter by normative level range
    pub normative_range: Option<(NormativeLevel, NormativeLevel)>,
    /// Filter by materiality level range
    pub materiality_range: Option<(MaterialityLevel, MaterialityLevel)>,
    /// Filter by schema ID
    pub schema_id: Option<String>,
    /// Filter by schema family
    pub schema_family: Option<String>,
    /// Filter by time range (stored_at)
    pub time_range: Option<(u64, u64)>,
    /// Filter by creator
    pub created_by: Option<String>,
    /// Maximum results to return
    pub limit: Option<usize>,
    /// Results to skip
    pub offset: Option<usize>,
    /// Sort field
    pub order_by: Option<QueryOrderBy>,
    /// Sort direction
    pub order_direction: QueryDirection,
}

/// Field to order query results by
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryOrderBy {
    /// Order by storage timestamp
    StoredAt,
    /// Order by last modification timestamp
    ModifiedAt,
    /// Order by epistemic classification
    Classification,
}

/// Direction for ordering query results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryDirection {
    /// Ascending order (lowest to highest)
    #[default]
    Ascending,
    /// Descending order (highest to lowest)
    Descending,
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult<T> {
    /// Matching items
    pub items: Vec<StoredData<T>>,
    /// Total count (before pagination)
    pub total_count: usize,
    /// Whether there are more results
    pub has_more: bool,
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total items stored
    pub total_items: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Items per backend
    pub items_by_backend: HashMap<StorageBackend, usize>,
    /// Items per E-level
    pub items_by_empirical: HashMap<u8, usize>,
    /// Items per N-level
    pub items_by_normative: HashMap<u8, usize>,
    /// Items per M-level
    pub items_by_materiality: HashMap<u8, usize>,
    /// Cache hit rate (if applicable)
    pub cache_hit_rate: Option<f32>,
}

/// Storage information for a key
#[derive(Debug, Clone)]
pub struct StorageInfo {
    /// The storage key
    pub key: String,
    /// Current CID
    pub cid: String,
    /// Classification
    pub classification: EpistemicClassification,
    /// Schema
    pub schema: SchemaIdentity,
    /// Current tier
    pub tier: StorageTier,
    /// Version
    pub version: u32,
    /// Created timestamp
    pub created_at: u64,
    /// Modified timestamp
    pub modified_at: Option<u64>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether data is encrypted
    pub encrypted: bool,
}

/// Verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification passed
    pub valid: bool,
    /// CID matches
    pub cid_valid: bool,
    /// Data integrity check passed
    pub integrity_valid: bool,
    /// Signature valid (if signed)
    pub signature_valid: Option<bool>,
    /// Error message if verification failed
    pub error: Option<String>,
}

// =============================================================================
// Storage Location Tracking (TODO 2 from SECURITY_AUDIT)
// =============================================================================

/// Tracks the location of stored data including migration history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLocation {
    /// Current storage backend
    pub backend: StorageBackend,
    /// Path/key within the backend
    pub path: String,
    /// Previous location if data was migrated
    pub migrated_from: Option<Box<StorageLocation>>,
    /// Timestamp when data was stored at this location
    pub created_at: u64,
    /// Migration reason (if migrated)
    pub migration_reason: Option<String>,
}

impl StorageLocation {
    /// Create a new storage location
    pub fn new(backend: StorageBackend, path: impl Into<String>) -> Self {
        Self {
            backend,
            path: path.into(),
            migrated_from: None,
            created_at: current_timestamp_ms(),
            migration_reason: None,
        }
    }

    /// Create a migrated storage location
    pub fn migrated(
        backend: StorageBackend,
        path: impl Into<String>,
        from: StorageLocation,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            path: path.into(),
            migrated_from: Some(Box::new(from)),
            created_at: current_timestamp_ms(),
            migration_reason: Some(reason.into()),
        }
    }

    /// Check if this location was migrated
    pub fn was_migrated(&self) -> bool {
        self.migrated_from.is_some()
    }

    /// Get the migration history chain
    pub fn migration_history(&self) -> Vec<&StorageLocation> {
        let mut history = vec![self];
        let mut current = self;
        while let Some(ref prev) = current.migrated_from {
            history.push(prev.as_ref());
            current = prev.as_ref();
        }
        history
    }
}

/// Result of a migration operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    /// Key that was migrated
    pub key: String,
    /// Source backend
    pub from_backend: StorageBackend,
    /// Destination backend
    pub to_backend: StorageBackend,
    /// New storage location
    pub new_location: StorageLocation,
    /// Migration timestamp
    pub migrated_at: u64,
    /// Whether the source data was deleted
    pub source_deleted: bool,
    /// Any warnings during migration
    pub warnings: Vec<String>,
}

/// Get current timestamp in milliseconds
pub fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_identity() {
        let schema = SchemaIdentity::new("user-profile", "1.0.0").with_family("user-data");

        assert_eq!(schema.id, "user-profile");
        assert_eq!(schema.version, "1.0.0");
        assert_eq!(schema.family, Some("user-data".to_string()));
    }

    #[test]
    fn test_access_capability_matching() {
        let cap = AccessCapability {
            id: "cap-1".to_string(),
            resource: "user:alice:*".to_string(),
            is_pattern: true,
            rights: vec![AccessRight::Read, AccessRight::Write],
            issuer: "admin".to_string(),
            holder: "bob".to_string(),
            issued_at: 0,
            expires_at: None,
            delegation_depth: 0,
            parent_id: None,
        };

        assert!(cap.matches_resource("user:alice:profile"));
        assert!(cap.matches_resource("user:alice:settings"));
        assert!(!cap.matches_resource("user:bob:profile"));
        assert!(cap.has_right(AccessRight::Read));
        assert!(!cap.has_right(AccessRight::Delete));
    }

    #[test]
    fn test_storage_metadata_expiry() {
        use crate::epistemic::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let mut metadata = StorageMetadata {
            cid: "cid:test".to_string(),
            classification: EpistemicClassification::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
            ),
            schema: SchemaIdentity::new("test", "1.0"),
            stored_at: now,
            modified_at: None,
            version: 1,
            expires_at: Some(now - 1000), // Expired 1 second ago
            size_bytes: 100,
            created_by: "test".to_string(),
            tombstone: false,
            retracted_by: None,
        };

        assert!(metadata.is_expired());
        assert!(!metadata.is_accessible());

        metadata.expires_at = Some(now + 1000000); // Future
        assert!(!metadata.is_expired());
        assert!(metadata.is_accessible());

        metadata.tombstone = true;
        assert!(!metadata.is_accessible());
    }
}

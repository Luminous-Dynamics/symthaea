//! Universal Epistemic Storage System (UESS)
//!
//! A 3D classification-driven storage router that manages data across multiple
//! backends based on Empirical (E), Normative (N), and Materiality (M) levels.
//!
//! # Overview
//!
//! The UESS provides unified storage semantics regardless of underlying backend:
//!
//! | M-Level | Backend | Use Case |
//! |---------|---------|----------|
//! | M0 (Ephemeral) | Memory | Session tokens, caches |
//! | M1 (Temporal) | Local | User preferences, drafts |
//! | M2 (Persistent) | DHT | Profiles, documents |
//! | M3 (Foundational) | IPFS | Permanent records |
//!
//! # Routing Logic
//!
//! - **Backend**: Determined by Materiality level
//! - **Mutability**: Determined by Empirical level (E0-E1: CRDT, E2: append-only, E3+: immutable)
//! - **Access Control**: Determined by Normative level (N0: owner, N1: CapBAC, N2+: public)
//! - **Encryption**: Required for N0/N1 (private/communal)
//! - **Content Addressing**: Required for E3+ (cryptographic/reproducible)
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::storage::{EpistemicStorage, StorageConfig};
//! use mycelix_sdk::epistemic::{EmpiricalLevel, NormativeLevel, MaterialityLevel, EpistemicClassification};
//!
//! let storage = EpistemicStorage::new(StorageConfig::default());
//!
//! // Store ephemeral session data
//! let classification = EpistemicClassification {
//!     empirical: EmpiricalLevel::E0Null,
//!     normative: NormativeLevel::N0Personal,
//!     materiality: MaterialityLevel::M0Ephemeral,
//! };
//!
//! let receipt = storage.store("session:123", &session_data, classification, options).await?;
//! ```

pub mod backends;
mod router;
mod types;

pub use backends::{
    BackendStats, DHTBackend, DHTBackendConfig, DHTBackendStats, IPFSBackend, IPFSBackendConfig,
    IPFSBackendStats, LocalBackend, LocalBackendConfig, LocalBackendStats, MemoryBackend,
    MemoryBackendConfig, MemoryBackendStats, StorageBackendAdapter,
};
pub use router::{RouterConfig, StorageRouter, TransitionError};
pub use types::*;

use crate::epistemic::EpistemicClassification;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// UESS Storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Agent ID for this storage instance
    pub agent_id: String,
    /// Router configuration
    pub router: RouterConfig,
    /// Memory backend configuration (M0)
    pub memory: MemoryBackendConfig,
    /// Local backend configuration (M1)
    pub local: LocalBackendConfig,
    /// DHT backend configuration (M2)
    pub dht: DHTBackendConfig,
    /// IPFS backend configuration (M3)
    pub ipfs: IPFSBackendConfig,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            agent_id: "anonymous".to_string(),
            router: RouterConfig::default(),
            memory: MemoryBackendConfig::default(),
            local: LocalBackendConfig::default(),
            dht: DHTBackendConfig::default(),
            ipfs: IPFSBackendConfig::default(),
            enable_metrics: true,
        }
    }
}

/// Storage errors
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Key not found: {0}")]
    /// Key was not found in storage.
    NotFound(String),
    #[error("Access denied: {0}")]
    /// Access to the requested resource was denied.
    AccessDenied(String),
    #[error("Data is immutable and cannot be modified")]
    /// Attempted to modify immutable data.
    ImmutableData,
    #[error("Invalid classification transition: {0}")]
    /// Invalid epistemic classification transition.
    InvalidTransition(#[from] TransitionError),
    #[error("Serialization error: {0}")]
    /// Serialization or deserialization failed.
    SerializationError(String),
    #[error("Backend error: {0}")]
    /// Storage backend encountered an error.
    BackendError(String),
    #[error("Data already exists: {0}")]
    /// Key already exists in storage.
    DuplicateKey(String),
    #[error("Schema mismatch")]
    /// Schema does not match expected format.
    SchemaMismatch,
    #[error("CID verification failed")]
    /// Content identifier verification failed.
    CIDMismatch,
    #[error("Lock poisoned: {0}")]
    /// Internal lock was poisoned.
    LockPoisoned(String),
}

/// Index entry for key → metadata mapping
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct IndexEntry {
    key: String,
    cid: String,
    classification: EpistemicClassification,
    schema: SchemaIdentity,
    backend: StorageBackend,
    version: u32,
    stored_at: u64,
    created_by: String,
    /// Actual storage location with migration history (FIND-008/009 mitigation)
    location: StorageLocation,
}

/// Unified Epistemic Storage
///
/// Main entry point for UESS operations.
/// Routes data to appropriate backends based on E/N/M classification.
pub struct EpistemicStorage {
    /// Configuration
    config: StorageConfig,
    /// Storage router
    router: StorageRouter,
    /// Memory backend (M0 - Ephemeral)
    memory_backend: MemoryBackend,
    /// Local backend (M1 - Temporal)
    local_backend: LocalBackend,
    /// DHT backend (M2 - Persistent)
    dht_backend: DHTBackend,
    /// IPFS backend (M3 - Foundational)
    ipfs_backend: IPFSBackend,
    /// Key → Index mapping
    key_index: Arc<RwLock<HashMap<String, IndexEntry>>>,
    /// CID → Key mapping
    cid_index: Arc<RwLock<HashMap<String, String>>>,
    /// Metrics
    metrics: Arc<RwLock<StorageMetrics>>,
}

#[derive(Debug, Clone, Default)]
struct StorageMetrics {
    stores: u64,
    retrieves: u64,
    updates: u64,
    deletes: u64,
    reclassifications: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl EpistemicStorage {
    /// Create a new epistemic storage instance
    pub fn new(config: StorageConfig) -> Self {
        let router = StorageRouter::new(config.router.clone());
        let memory_backend = MemoryBackend::new(config.memory.clone());
        let local_backend = LocalBackend::with_config(config.local.clone());
        let dht_backend = DHTBackend::with_config(config.dht.clone());
        let ipfs_backend = IPFSBackend::with_config(config.ipfs.clone());

        Self {
            config,
            router,
            memory_backend,
            local_backend,
            dht_backend,
            ipfs_backend,
            key_index: Arc::new(RwLock::new(HashMap::new())),
            cid_index: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(StorageMetrics::default())),
        }
    }

    /// Create with default configuration
    pub fn default_storage() -> Self {
        Self::new(StorageConfig::default())
    }

    /// Store data with epistemic classification
    pub fn store<T: serde::Serialize>(
        &self,
        key: &str,
        data: &T,
        classification: EpistemicClassification,
        options: StoreOptions,
    ) -> Result<StorageReceipt, StorageError> {
        // Check for duplicate key
        if !options.allow_overwrite && self.exists(key) {
            return Err(StorageError::DuplicateKey(key.to_string()));
        }

        // Route to appropriate tier
        let tier = self.router.route(&classification);

        // Store in appropriate backend based on routed tier
        let metadata = match tier.backend {
            StorageBackend::Memory => self.memory_backend.set(
                key,
                data,
                classification,
                options.schema.clone(),
                &options.agent_id,
                options.ttl_ms.or(tier.ttl_ms),
            ),
            StorageBackend::Local => self.local_backend.set(
                key,
                data,
                classification,
                options.schema.clone(),
                &options.agent_id,
                options.ttl_ms.or(tier.ttl_ms),
            ),
            StorageBackend::DHT => self.dht_backend.set(
                key,
                data,
                classification,
                options.schema.clone(),
                &options.agent_id,
                options.ttl_ms.or(tier.ttl_ms),
            ),
            StorageBackend::IPFS => self.ipfs_backend.set(
                key,
                data,
                classification,
                options.schema.clone(),
                &options.agent_id,
                options.ttl_ms.or(tier.ttl_ms),
            ),
            StorageBackend::Filecoin => {
                // Filecoin uses IPFS as primary storage
                self.ipfs_backend.set(
                    key,
                    data,
                    classification,
                    options.schema.clone(),
                    &options.agent_id,
                    options.ttl_ms.or(tier.ttl_ms),
                )
            }
        }
        .ok_or_else(|| StorageError::SerializationError("Failed to serialize data".to_string()))?;

        // Create storage location for tracking
        let location = StorageLocation::new(tier.backend, key);

        // Update indices
        let index_entry = IndexEntry {
            key: key.to_string(),
            cid: metadata.cid.clone(),
            classification,
            schema: options.schema.clone(),
            backend: tier.backend,
            version: metadata.version,
            stored_at: metadata.stored_at,
            created_by: options.agent_id.clone(),
            location: location.clone(),
        };

        if let Ok(mut key_idx) = self.key_index.write() {
            key_idx.insert(key.to_string(), index_entry);
        }

        if let Ok(mut cid_idx) = self.cid_index.write() {
            cid_idx.insert(metadata.cid.clone(), key.to_string());
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.stores += 1;
        }

        // Collect all backends where data is stored (primary + additional for redundancy)
        let mut all_locations = vec![tier.backend];
        all_locations.extend(tier.additional_backends.iter().copied());

        let shreddable = tier.encrypted;
        Ok(StorageReceipt {
            key: key.to_string(),
            cid: metadata.cid,
            classification,
            schema: options.schema,
            stored_at: metadata.stored_at,
            tier,
            index_locations: all_locations,
            version: metadata.version,
            previous_cid: None,
            shreddable,
            key_id: options.key_id,
        })
    }

    /// Retrieve data by key
    pub fn retrieve<T: for<'de> serde::Deserialize<'de>>(
        &self,
        key: &str,
        capability: Option<&AccessCapability>,
        _options: RetrieveOptions,
    ) -> Result<StoredData<T>, StorageError> {
        // Get index entry
        let index_entry = {
            let key_idx = self
                .key_index
                .read()
                .map_err(|_| StorageError::BackendError("Lock error".to_string()))?;
            key_idx
                .get(key)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };

        // Check access control
        let tier = self.router.route(&index_entry.classification);
        if tier.access_control != AccessControlMode::Public {
            // Verify capability
            if let Some(cap) = capability {
                if !cap.matches_resource(key)
                    || !cap.has_right(AccessRight::Read)
                    || cap.is_expired()
                {
                    return Err(StorageError::AccessDenied(
                        "Invalid or insufficient capability".to_string(),
                    ));
                }
            } else {
                return Err(StorageError::AccessDenied(
                    "Capability required".to_string(),
                ));
            }
        }

        // Retrieve from backend
        let result: Option<StoredData<T>> = match index_entry.backend {
            StorageBackend::Memory => self.memory_backend.get(key),
            StorageBackend::Local => self.local_backend.get(key),
            StorageBackend::DHT => self.dht_backend.get(key),
            StorageBackend::IPFS | StorageBackend::Filecoin => self.ipfs_backend.get(key),
        };

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.retrieves += 1;
            if result.is_some() {
                metrics.cache_hits += 1;
            } else {
                metrics.cache_misses += 1;
            }
        }

        result.ok_or_else(|| StorageError::NotFound(key.to_string()))
    }

    /// Retrieve data by CID (content-addressed lookup)
    pub fn retrieve_by_cid<T: for<'de> serde::Deserialize<'de>>(
        &self,
        cid: &str,
        capability: Option<&AccessCapability>,
        options: RetrieveOptions,
    ) -> Result<StoredData<T>, StorageError> {
        // Look up key from CID
        let key = {
            let cid_idx = self
                .cid_index
                .read()
                .map_err(|_| StorageError::BackendError("Lock error".to_string()))?;
            cid_idx
                .get(cid)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(cid.to_string()))?
        };

        self.retrieve(&key, capability, options)
    }

    /// Update existing data
    pub fn update<T: serde::Serialize>(
        &self,
        key: &str,
        data: &T,
        capability: Option<&AccessCapability>,
        _options: UpdateOptions,
    ) -> Result<StorageReceipt, StorageError> {
        // Get index entry
        let index_entry = {
            let key_idx = self
                .key_index
                .read()
                .map_err(|_| StorageError::BackendError("Lock error".to_string()))?;
            key_idx
                .get(key)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };

        // Check mutability
        let tier = self.router.route(&index_entry.classification);
        if tier.mutability == MutabilityMode::Immutable {
            return Err(StorageError::ImmutableData);
        }

        // Check access control
        if tier.access_control != AccessControlMode::Public {
            if let Some(cap) = capability {
                if !cap.matches_resource(key)
                    || !cap.has_right(AccessRight::Write)
                    || cap.is_expired()
                {
                    return Err(StorageError::AccessDenied(
                        "Invalid or insufficient capability".to_string(),
                    ));
                }
            } else {
                return Err(StorageError::AccessDenied(
                    "Capability required".to_string(),
                ));
            }
        }

        // Update in backend (delete + set for backends without native update)
        let metadata = match index_entry.backend {
            StorageBackend::Memory => self.memory_backend.update(key, data, &self.config.agent_id),
            StorageBackend::Local => {
                // Local doesn't have native update, use delete + set
                self.local_backend.delete(key);
                self.local_backend.set(
                    key,
                    data,
                    index_entry.classification,
                    index_entry.schema.clone(),
                    &self.config.agent_id,
                    None,
                )
            }
            StorageBackend::DHT => {
                // DHT doesn't have native update, use delete + set
                self.dht_backend.delete(key);
                self.dht_backend.set(
                    key,
                    data,
                    index_entry.classification,
                    index_entry.schema.clone(),
                    &self.config.agent_id,
                    None,
                )
            }
            StorageBackend::IPFS | StorageBackend::Filecoin => {
                // IPFS is content-addressed, update creates new CID
                self.ipfs_backend.delete(key);
                self.ipfs_backend.set(
                    key,
                    data,
                    index_entry.classification,
                    index_entry.schema.clone(),
                    &self.config.agent_id,
                    None,
                )
            }
        }
        .ok_or_else(|| StorageError::BackendError("Update failed".to_string()))?;

        // Update indices
        if let Ok(mut key_idx) = self.key_index.write() {
            if let Some(entry) = key_idx.get_mut(key) {
                let old_cid = entry.cid.clone();
                entry.cid = metadata.cid.clone();
                entry.version = metadata.version;

                // Update CID index
                if let Ok(mut cid_idx) = self.cid_index.write() {
                    cid_idx.remove(&old_cid);
                    cid_idx.insert(metadata.cid.clone(), key.to_string());
                }
            }
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.updates += 1;
        }

        // Collect all backends where data is stored
        let mut all_locations = vec![tier.backend];
        all_locations.extend(tier.additional_backends.iter().copied());

        let shreddable = tier.encrypted;
        Ok(StorageReceipt {
            key: key.to_string(),
            cid: metadata.cid,
            classification: index_entry.classification,
            schema: index_entry.schema,
            stored_at: metadata.stored_at,
            tier,
            index_locations: all_locations,
            version: metadata.version,
            previous_cid: None,
            shreddable,
            key_id: None,
        })
    }

    /// Delete data
    pub fn delete(
        &self,
        key: &str,
        capability: Option<&AccessCapability>,
        options: DeleteOptions,
    ) -> Result<(), StorageError> {
        // Get index entry
        let index_entry = {
            let key_idx = self
                .key_index
                .read()
                .map_err(|_| StorageError::BackendError("Lock error".to_string()))?;
            key_idx
                .get(key)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };

        // Check access control
        let tier = self.router.route(&index_entry.classification);
        if tier.access_control != AccessControlMode::Public {
            if let Some(cap) = capability {
                if !cap.matches_resource(key)
                    || !cap.has_right(AccessRight::Delete)
                    || cap.is_expired()
                {
                    return Err(StorageError::AccessDenied(
                        "Invalid or insufficient capability".to_string(),
                    ));
                }
            } else {
                return Err(StorageError::AccessDenied(
                    "Capability required".to_string(),
                ));
            }
        }

        // Check if immutable (can only tombstone or shred)
        if tier.mutability == MutabilityMode::Immutable {
            if options.shred && tier.encrypted {
                // Cryptographic shredding - would delete encryption key
                // For now, just tombstone
                self.memory_backend.tombstone(key, &options.agent_id);
            } else if !options.hard_delete {
                // Tombstone only
                self.memory_backend.tombstone(key, &options.agent_id);
            } else {
                return Err(StorageError::ImmutableData);
            }
        } else {
            // Actually delete
            match index_entry.backend {
                StorageBackend::Memory => {
                    self.memory_backend.delete(key);
                }
                StorageBackend::Local => {
                    self.local_backend.delete(key);
                }
                StorageBackend::DHT => {
                    self.dht_backend.delete(key);
                }
                StorageBackend::IPFS | StorageBackend::Filecoin => {
                    self.ipfs_backend.delete(key);
                }
            }
        }

        // Remove from indices
        if let Ok(mut key_idx) = self.key_index.write() {
            key_idx.remove(key);
        }
        if let Ok(mut cid_idx) = self.cid_index.write() {
            cid_idx.remove(&index_entry.cid);
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.deletes += 1;
        }

        Ok(())
    }

    /// Reclassify data (upgrade only - INV-2)
    pub fn reclassify(
        &self,
        key: &str,
        new_classification: EpistemicClassification,
        _reason: ReclassificationReason,
        capability: Option<&AccessCapability>,
    ) -> Result<StorageReceipt, StorageError> {
        // Get index entry
        let index_entry = {
            let key_idx = self
                .key_index
                .read()
                .map_err(|_| StorageError::BackendError("Lock error".to_string()))?;
            key_idx
                .get(key)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };

        // Validate transition (INV-2: monotonic only)
        self.router
            .validate_transition(&index_entry.classification, &new_classification)?;

        // Check access control (need Admin right for reclassification)
        let tier = self.router.route(&index_entry.classification);
        if tier.access_control != AccessControlMode::Public {
            if let Some(cap) = capability {
                if !cap.matches_resource(key)
                    || !cap.has_right(AccessRight::Admin)
                    || cap.is_expired()
                {
                    return Err(StorageError::AccessDenied(
                        "Admin capability required for reclassification".to_string(),
                    ));
                }
            } else {
                return Err(StorageError::AccessDenied(
                    "Capability required".to_string(),
                ));
            }
        }

        // Check if migration is required
        let new_tier = self.router.route(&new_classification);
        let requires_migration = self
            .router
            .requires_migration(&index_entry.classification, &new_classification);

        // Track final location after potential migration
        let mut final_location = index_entry.location.clone();
        let mut migration_warnings: Vec<String> = Vec::new();

        if requires_migration {
            // Perform data migration between backends
            let migration_result =
                self.migrate_data(key, &index_entry, new_tier.backend, &_reason.reason);

            match migration_result {
                Ok(result) => {
                    final_location = result.new_location;
                    migration_warnings = result.warnings;
                }
                Err(e) => {
                    // Migration failed - return error
                    return Err(StorageError::BackendError(format!(
                        "Migration failed: {}",
                        e
                    )));
                }
            }
        }

        // Update index entry with new classification and location
        if let Ok(mut key_idx) = self.key_index.write() {
            if let Some(entry) = key_idx.get_mut(key) {
                entry.classification = new_classification;
                entry.backend = new_tier.backend;
                entry.location = final_location.clone();
            }
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.reclassifications += 1;
        }

        // Log migration warnings if any
        for warning in &migration_warnings {
            // In production, these would be logged via the logging framework
            let _ = warning; // Suppress unused warning
        }

        // Collect all backends where data is stored
        let mut all_locations = vec![new_tier.backend];
        all_locations.extend(new_tier.additional_backends.iter().copied());

        let shreddable = new_tier.encrypted;
        Ok(StorageReceipt {
            key: key.to_string(),
            cid: index_entry.cid,
            classification: new_classification,
            schema: index_entry.schema,
            stored_at: index_entry.stored_at,
            tier: new_tier,
            index_locations: all_locations,
            version: index_entry.version,
            previous_cid: None,
            shreddable,
            key_id: None,
        })
    }

    /// Check if a key exists
    pub fn exists(&self, key: &str) -> bool {
        if let Ok(key_idx) = self.key_index.read() {
            return key_idx.contains_key(key);
        }
        false
    }

    /// Get storage info for a key
    pub fn get_info(&self, key: &str) -> Option<StorageInfo> {
        let key_idx = self.key_index.read().ok()?;
        let entry = key_idx.get(key)?;
        let tier = self.router.route(&entry.classification);

        Some(StorageInfo {
            key: entry.key.clone(),
            cid: entry.cid.clone(),
            classification: entry.classification,
            schema: entry.schema.clone(),
            tier,
            version: entry.version,
            created_at: entry.stored_at,
            modified_at: None,
            size_bytes: 0, // Would need to query backend
            encrypted: self.router.route(&entry.classification).encrypted,
        })
    }

    /// Migrate data between storage backends
    ///
    /// This method handles the actual data movement during reclassification.
    /// It preserves the migration history for audit purposes.
    fn migrate_data(
        &self,
        key: &str,
        index_entry: &IndexEntry,
        target_backend: StorageBackend,
        reason: &str,
    ) -> Result<MigrationResult, StorageError> {
        let source_backend = index_entry.backend;
        let mut warnings: Vec<String> = Vec::new();

        // Skip if same backend
        if source_backend == target_backend {
            return Ok(MigrationResult {
                key: key.to_string(),
                from_backend: source_backend,
                to_backend: target_backend,
                new_location: index_entry.location.clone(),
                migrated_at: current_timestamp_ms(),
                source_deleted: false,
                warnings: vec!["No migration needed: same backend".to_string()],
            });
        }

        // Read data from source backend (as raw bytes for migration)
        let raw_data: Option<StoredData<serde_json::Value>> = match source_backend {
            StorageBackend::Memory => self.memory_backend.get(key),
            StorageBackend::Local => self.local_backend.get(key),
            StorageBackend::DHT => self.dht_backend.get(key),
            StorageBackend::IPFS | StorageBackend::Filecoin => self.ipfs_backend.get(key),
        };

        let stored_data = raw_data.ok_or_else(|| {
            StorageError::NotFound(format!(
                "Data not found in source backend {:?} for migration",
                source_backend
            ))
        })?;

        // Write to target backend
        let metadata = match target_backend {
            StorageBackend::Memory => self.memory_backend.set(
                key,
                &stored_data.data,
                index_entry.classification,
                index_entry.schema.clone(),
                &index_entry.created_by,
                None,
            ),
            StorageBackend::Local => self.local_backend.set(
                key,
                &stored_data.data,
                index_entry.classification,
                index_entry.schema.clone(),
                &index_entry.created_by,
                None,
            ),
            StorageBackend::DHT => self.dht_backend.set(
                key,
                &stored_data.data,
                index_entry.classification,
                index_entry.schema.clone(),
                &index_entry.created_by,
                None,
            ),
            StorageBackend::IPFS | StorageBackend::Filecoin => self.ipfs_backend.set(
                key,
                &stored_data.data,
                index_entry.classification,
                index_entry.schema.clone(),
                &index_entry.created_by,
                None,
            ),
        }
        .ok_or_else(|| {
            StorageError::BackendError("Failed to write to target backend".to_string())
        })?;

        // Verify data was written successfully by checking CID
        if metadata.cid != index_entry.cid {
            warnings.push(format!(
                "CID changed during migration: {} -> {}",
                index_entry.cid, metadata.cid
            ));
        }

        // Delete from source backend (cleanup)
        let source_deleted = match source_backend {
            StorageBackend::Memory => {
                self.memory_backend.delete(key);
                true
            }
            StorageBackend::Local => {
                self.local_backend.delete(key);
                true
            }
            StorageBackend::DHT => {
                self.dht_backend.delete(key);
                true
            }
            StorageBackend::IPFS | StorageBackend::Filecoin => {
                // IPFS is content-addressed - we don't delete from it during migration
                // as the content may be referenced elsewhere
                warnings.push("Source data retained in IPFS (content-addressed)".to_string());
                false
            }
        };

        // Create new location with migration history
        let new_location =
            StorageLocation::migrated(target_backend, key, index_entry.location.clone(), reason);

        Ok(MigrationResult {
            key: key.to_string(),
            from_backend: source_backend,
            to_backend: target_backend,
            new_location,
            migrated_at: current_timestamp_ms(),
            source_deleted,
            warnings,
        })
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        let memory_stats = self.memory_backend.stats();
        let local_stats = self.local_backend.stats();
        let dht_stats = self.dht_backend.stats();
        let ipfs_stats = self.ipfs_backend.stats();
        let metrics = self.metrics.read().ok();

        let total_items = memory_stats.item_count
            + local_stats.item_count
            + dht_stats.entry_count
            + ipfs_stats.item_count;

        let total_size = memory_stats.total_size_bytes
            + local_stats.total_size_bytes
            + dht_stats.bytes_stored
            + ipfs_stats.bytes_stored;

        let mut stats = StorageStats {
            total_items,
            total_size_bytes: total_size,
            items_by_backend: HashMap::new(),
            items_by_empirical: HashMap::new(),
            items_by_normative: HashMap::new(),
            items_by_materiality: HashMap::new(),
            cache_hit_rate: None,
        };

        stats
            .items_by_backend
            .insert(StorageBackend::Memory, memory_stats.item_count);
        stats
            .items_by_backend
            .insert(StorageBackend::Local, local_stats.item_count);
        stats
            .items_by_backend
            .insert(StorageBackend::DHT, dht_stats.entry_count);
        stats
            .items_by_backend
            .insert(StorageBackend::IPFS, ipfs_stats.item_count);

        // Calculate cache hit rate
        if let Some(m) = metrics {
            let total = m.cache_hits + m.cache_misses;
            if total > 0 {
                stats.cache_hit_rate = Some(m.cache_hits as f32 / total as f32);
            }
        }

        stats
    }

    /// Get the router
    pub fn router(&self) -> &StorageRouter {
        &self.router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

    fn test_classification() -> EpistemicClassification {
        EpistemicClassification {
            empirical: EmpiricalLevel::E1Testimonial,
            normative: NormativeLevel::N2Network, // Public access for easier testing
            materiality: MaterialityLevel::M0Ephemeral,
        }
    }

    fn test_options() -> StoreOptions {
        StoreOptions {
            schema: SchemaIdentity::new("test", "1.0"),
            agent_id: "test-agent".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_store_and_retrieve() {
        let storage = EpistemicStorage::default_storage();

        let receipt = storage
            .store(
                "key1",
                &"hello world".to_string(),
                test_classification(),
                test_options(),
            )
            .expect("Store should succeed");

        assert_eq!(receipt.key, "key1");
        assert!(!receipt.cid.is_empty());

        let result: StoredData<String> = storage
            .retrieve(
                "key1",
                None, // Public access
                RetrieveOptions::default(),
            )
            .expect("Retrieve should succeed");

        assert_eq!(result.data, "hello world");
    }

    #[test]
    fn test_retrieve_by_cid() {
        let storage = EpistemicStorage::default_storage();

        let receipt = storage
            .store(
                "key1",
                &"test data".to_string(),
                test_classification(),
                test_options(),
            )
            .expect("Store should succeed");

        let result: StoredData<String> = storage
            .retrieve_by_cid(&receipt.cid, None, RetrieveOptions::default())
            .expect("Retrieve by CID should succeed");

        assert_eq!(result.data, "test data");
    }

    #[test]
    fn test_update() {
        let storage = EpistemicStorage::default_storage();

        storage
            .store(
                "key1",
                &"v1".to_string(),
                test_classification(),
                test_options(),
            )
            .expect("Store should succeed");

        let receipt = storage
            .update("key1", &"v2".to_string(), None, UpdateOptions::default())
            .expect("Update should succeed");

        assert_eq!(receipt.version, 2);

        let result: StoredData<String> = storage
            .retrieve("key1", None, RetrieveOptions::default())
            .expect("Retrieve should succeed");
        assert_eq!(result.data, "v2");
    }

    #[test]
    fn test_delete() {
        let storage = EpistemicStorage::default_storage();

        storage
            .store(
                "key1",
                &"data".to_string(),
                test_classification(),
                test_options(),
            )
            .expect("Store should succeed");

        assert!(storage.exists("key1"));

        storage
            .delete("key1", None, DeleteOptions::default())
            .expect("Delete should succeed");

        assert!(!storage.exists("key1"));
    }

    #[test]
    fn test_reclassify_valid() {
        let storage = EpistemicStorage::default_storage();

        let from = EpistemicClassification {
            empirical: EmpiricalLevel::E1Testimonial,
            normative: NormativeLevel::N2Network,
            materiality: MaterialityLevel::M1Temporal,
        };

        storage
            .store("key1", &"data".to_string(), from, test_options())
            .expect("Store should succeed");

        // Valid upgrade: M1 → M2
        let to = EpistemicClassification {
            empirical: EmpiricalLevel::E1Testimonial,
            normative: NormativeLevel::N2Network,
            materiality: MaterialityLevel::M2Persistent,
        };

        let reason = ReclassificationReason {
            reason: "Upgrade to persistent".to_string(),
            requested_by: "admin".to_string(),
            timestamp: 0,
        };

        let receipt = storage
            .reclassify("key1", to, reason, None)
            .expect("Reclassify should succeed");

        assert_eq!(
            receipt.classification.materiality,
            MaterialityLevel::M2Persistent
        );
    }

    #[test]
    fn test_reclassify_invalid_downgrade() {
        let storage = EpistemicStorage::default_storage();

        let from = EpistemicClassification {
            empirical: EmpiricalLevel::E2PrivateVerify,
            normative: NormativeLevel::N2Network,
            materiality: MaterialityLevel::M2Persistent,
        };

        storage
            .store("key1", &"data".to_string(), from, test_options())
            .expect("Store should succeed");

        // Invalid: E2 → E1 (downgrade)
        let to = EpistemicClassification {
            empirical: EmpiricalLevel::E1Testimonial,
            normative: NormativeLevel::N2Network,
            materiality: MaterialityLevel::M2Persistent,
        };

        let reason = ReclassificationReason {
            reason: "Downgrade".to_string(),
            requested_by: "admin".to_string(),
            timestamp: 0,
        };

        let result = storage.reclassify("key1", to, reason, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_immutable_update_fails() {
        let storage = EpistemicStorage::default_storage();

        // E3 = Immutable
        let classification = EpistemicClassification {
            empirical: EmpiricalLevel::E3Cryptographic,
            normative: NormativeLevel::N2Network,
            materiality: MaterialityLevel::M0Ephemeral,
        };

        storage
            .store("key1", &"data".to_string(), classification, test_options())
            .expect("Store should succeed");

        let result = storage.update(
            "key1",
            &"new data".to_string(),
            None,
            UpdateOptions::default(),
        );
        assert!(matches!(result, Err(StorageError::ImmutableData)));
    }

    #[test]
    fn test_duplicate_key_error() {
        let storage = EpistemicStorage::default_storage();

        storage
            .store(
                "key1",
                &"data".to_string(),
                test_classification(),
                test_options(),
            )
            .expect("First store should succeed");

        let result = storage.store(
            "key1",
            &"data2".to_string(),
            test_classification(),
            test_options(),
        );
        assert!(matches!(result, Err(StorageError::DuplicateKey(_))));
    }

    #[test]
    fn test_stats() {
        let storage = EpistemicStorage::default_storage();

        storage
            .store(
                "key1",
                &"data".to_string(),
                test_classification(),
                test_options(),
            )
            .unwrap();
        storage
            .store(
                "key2",
                &"data".to_string(),
                test_classification(),
                test_options(),
            )
            .unwrap();

        let stats = storage.stats();
        // Memory backend should have 2 items (test_classification uses M0Ephemeral)
        assert_eq!(
            *stats
                .items_by_backend
                .get(&StorageBackend::Memory)
                .unwrap_or(&0),
            2
        );
        // Total should include at least these 2 items
        assert!(stats.total_items >= 2);
    }

    #[test]
    fn test_storage_error_display() {
        // Test Display implementation for all StorageError variants

        let err = StorageError::NotFound("key123".to_string());
        assert!(err.to_string().contains("not found"));
        assert!(err.to_string().contains("key123"));

        let err = StorageError::AccessDenied("insufficient permissions".to_string());
        assert!(err.to_string().contains("Access denied"));
        assert!(err.to_string().contains("insufficient permissions"));

        let err = StorageError::ImmutableData;
        assert!(err.to_string().contains("immutable"));

        let err = StorageError::SerializationError("json error".to_string());
        assert!(err.to_string().contains("Serialization error"));
        assert!(err.to_string().contains("json error"));

        let err = StorageError::BackendError("connection failed".to_string());
        assert!(err.to_string().contains("Backend error"));
        assert!(err.to_string().contains("connection failed"));

        let err = StorageError::DuplicateKey("key456".to_string());
        assert!(err.to_string().contains("already exists"));
        assert!(err.to_string().contains("key456"));

        let err = StorageError::SchemaMismatch;
        assert!(err.to_string().contains("Schema mismatch"));

        let err = StorageError::CIDMismatch;
        assert!(err.to_string().contains("CID verification failed"));

        let err = StorageError::LockPoisoned("RwLock poisoned".to_string());
        assert!(err.to_string().contains("Lock poisoned"));
        assert!(err.to_string().contains("RwLock poisoned"));
    }

    #[test]
    fn test_storage_with_poisoned_index_locks() {
        // Test that EpistemicStorage handles poisoned index locks gracefully
        use std::panic;

        let storage = EpistemicStorage::default_storage();

        // Store some data
        let receipt = storage
            .store(
                "key1",
                &"test data".to_string(),
                test_classification(),
                test_options(),
            )
            .unwrap();

        // Poison the key_index lock
        let key_index_ref = Arc::clone(&storage.key_index);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = key_index_ref.write().unwrap();
            panic!("poison key_index");
        }));

        // Verify lock is poisoned
        assert!(key_index_ref.read().is_err());

        // Operations should fail gracefully with BackendError (not panic)
        // Note: Current implementation uses .ok() pattern which silently degrades
        // In production code, we might want to propagate LockPoisoned errors more explicitly

        // retrieve should fail with NotFound (because it can't read the index)
        let result: Result<StoredData<String>, StorageError> =
            storage.retrieve("key1", None, RetrieveOptions::default());
        assert!(
            matches!(result, Err(StorageError::BackendError(_))),
            "retrieve should fail when key_index is poisoned"
        );

        // exists should return false (can't read index)
        assert!(
            !storage.exists("key1"),
            "exists should return false when index is poisoned"
        );

        // get_info should return None
        assert!(
            storage.get_info("key1").is_none(),
            "get_info should return None when index is poisoned"
        );

        // Now test with poisoned cid_index
        let storage2 = EpistemicStorage::default_storage();
        let receipt2 = storage2
            .store(
                "key2",
                &"test data".to_string(),
                test_classification(),
                test_options(),
            )
            .unwrap();

        let cid_index_ref = Arc::clone(&storage2.cid_index);
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = cid_index_ref.write().unwrap();
            panic!("poison cid_index");
        }));

        // retrieve_by_cid should fail
        let result: Result<StoredData<String>, StorageError> =
            storage2.retrieve_by_cid(&receipt2.cid, None, RetrieveOptions::default());
        assert!(
            matches!(result, Err(StorageError::BackendError(_))),
            "retrieve_by_cid should fail when cid_index is poisoned"
        );
    }
}
